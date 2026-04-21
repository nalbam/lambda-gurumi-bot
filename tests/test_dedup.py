import time

import boto3
import pytest

try:
    from moto import mock_aws
except ImportError:  # pragma: no cover
    pytest.skip("moto not installed", allow_module_level=True)

from src.dedup import ConversationStore, DedupStore


TABLE = "lambda-slack-bot-test"
REGION = "us-east-1"


def _create_table():
    client = boto3.client("dynamodb", region_name=REGION)
    client.create_table(
        TableName=TABLE,
        KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
        AttributeDefinitions=[
            {"AttributeName": "id", "AttributeType": "S"},
            {"AttributeName": "user", "AttributeType": "S"},
            {"AttributeName": "expire_at", "AttributeType": "N"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "user-index",
                "KeySchema": [
                    {"AttributeName": "user", "KeyType": "HASH"},
                    {"AttributeName": "expire_at", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "KEYS_ONLY"},
                "ProvisionedThroughput": {"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
            }
        ],
        ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
    )


@mock_aws
def test_dedup_reserve_first_call_succeeds():
    _create_table()
    store = DedupStore(table_name=TABLE, region=REGION)
    assert store.reserve("abc") is True


@mock_aws
def test_dedup_reserve_second_call_returns_false():
    _create_table()
    store = DedupStore(table_name=TABLE, region=REGION)
    assert store.reserve("abc") is True
    assert store.reserve("abc") is False


@mock_aws
def test_dedup_different_keys_independent():
    _create_table()
    store = DedupStore(table_name=TABLE, region=REGION)
    assert store.reserve("a") is True
    assert store.reserve("b") is True


@mock_aws
def test_count_user_active_ignores_expired():
    _create_table()
    store = DedupStore(table_name=TABLE, region=REGION)
    store.reserve("fresh", user="U1", ttl_seconds=3600)
    # Manually insert an expired record for the same user.
    boto3.resource("dynamodb", region_name=REGION).Table(TABLE).put_item(
        Item={"id": "dedup:old", "user": "U1", "expire_at": int(time.time()) - 10}
    )
    assert store.count_user_active("U1") == 1


@mock_aws
def test_count_user_active_unknown_user_zero():
    _create_table()
    store = DedupStore(table_name=TABLE, region=REGION)
    assert store.count_user_active("nobody") == 0


@mock_aws
def test_conversation_put_and_get_roundtrip():
    _create_table()
    convo = ConversationStore(table_name=TABLE, region=REGION)
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    convo.put("T1", "U1", msgs)
    assert convo.get("T1") == msgs


@mock_aws
def test_conversation_get_missing_returns_empty():
    _create_table()
    convo = ConversationStore(table_name=TABLE, region=REGION)
    assert convo.get("unseen") == []


@mock_aws
def test_conversation_truncate_to_chars():
    _create_table()
    convo = ConversationStore(table_name=TABLE, region=REGION)
    msgs = [{"role": "user", "content": "x" * 1000} for _ in range(10)]
    convo.put("T1", "U1", msgs, max_chars=3000)
    stored = convo.get("T1")
    import json
    assert len(json.dumps(stored, ensure_ascii=False)) <= 3000
    assert len(stored) < len(msgs)


def test_conversation_truncate_helper_direct():
    msgs = [{"role": "user", "content": "x" * 500} for _ in range(5)]
    trimmed = ConversationStore.truncate_to_chars(msgs, max_chars=1200)
    import json
    assert len(json.dumps(trimmed, ensure_ascii=False)) <= 1200
    assert len(trimmed) < len(msgs)


def test_conversation_truncate_keeps_newest_messages():
    """Truncation drops the oldest entries — the most recent turn must survive
    as long as it fits."""
    msgs = [
        {"role": "user", "content": f"msg-{i}"}
        for i in range(20)
    ]
    trimmed = ConversationStore.truncate_to_chars(msgs, max_chars=200)
    assert trimmed, "should keep at least some messages"
    # The newest message must be in the kept suffix.
    assert trimmed[-1]["content"] == "msg-19"


def test_conversation_truncate_budget_matches_exact_dumps_length():
    """The fast cumulative-size algorithm must agree with the naive
    json.dumps(kept) size, within a single byte."""
    import json

    msgs = [{"role": "user", "content": "a" * 17 + str(i)} for i in range(8)]
    for budget in (50, 80, 120, 200, 300, 500, 1000, 5000):
        trimmed = ConversationStore.truncate_to_chars(msgs, max_chars=budget)
        assert len(json.dumps(trimmed, ensure_ascii=False)) <= budget, (
            f"budget={budget}, kept={len(trimmed)}, "
            f"actual={len(json.dumps(trimmed, ensure_ascii=False))}"
        )


def test_conversation_truncate_single_large_msg_overflows_budget():
    """If every individual message exceeds the budget, return an empty list
    rather than partial garbage."""
    msgs = [{"role": "user", "content": "x" * 1000}]
    trimmed = ConversationStore.truncate_to_chars(msgs, max_chars=50)
    assert trimmed == []
