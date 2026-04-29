"""Tests for the receiver / worker split in app.lambda_handler.

We only exercise the routing + enqueue logic here — `_process` itself is
covered transitively by tests/test_agent.py and friends. All external
dependencies (boto3, slack_sdk.WebClient, Bolt) are mocked so the tests
don't need real credentials.
"""
import json

import pytest


@pytest.fixture
def app_module():
    """Import the app module fresh.

    `app.Settings.from_env()` reads env at import time but does not
    validate Slack credentials, so this is safe without setting
    SLACK_BOT_TOKEN / SLACK_SIGNING_SECRET.
    """
    import app

    return app


# --------------------------------------------------------------------------- #
# lambda_handler routing
# --------------------------------------------------------------------------- #


def test_lambda_handler_routes_worker_flag_to_process_worker(app_module, monkeypatch):
    """`event["_worker"] is True` must skip Slack / Bolt entirely and call the worker."""
    received = {}

    def fake_worker(payload):
        received["payload"] = payload

    monkeypatch.setattr(app_module, "_process_worker", fake_worker)

    def boom_bolt():
        raise AssertionError("_get_bolt_app must not be called on the worker path")

    monkeypatch.setattr(app_module, "_get_bolt_app", boom_bolt)

    event = {"_worker": True, "slack_event": {"channel": "C1", "text": "hi"}, "is_dm": False}
    result = app_module.lambda_handler(event, None)

    assert result == {"statusCode": 200, "body": ""}
    assert received["payload"] is event


def test_lambda_handler_short_circuits_slack_retry_without_bolt(app_module, monkeypatch):
    """Receiver path: an X-Slack-Retry-Num header means Slack is re-delivering.
    We already dispatched the first try to a worker; swallow the retry."""

    def boom_bolt():
        raise AssertionError("Bolt must not be invoked on a retried delivery")

    monkeypatch.setattr(app_module, "_get_bolt_app", boom_bolt)

    event = {"headers": {"X-Slack-Retry-Num": "1"}, "body": "..."}
    assert app_module.lambda_handler(event, None) == {"statusCode": 200, "body": ""}


def test_lambda_handler_short_circuits_slack_retry_lowercase_header(app_module, monkeypatch):
    """API Gateway may lowercase the header name. Our guard normalizes."""
    monkeypatch.setattr(
        app_module,
        "_get_bolt_app",
        lambda: (_ for _ in ()).throw(AssertionError("should not hit Bolt")),
    )
    event = {"headers": {"x-slack-retry-num": "3"}}
    assert app_module.lambda_handler(event, None) == {"statusCode": 200, "body": ""}


def test_lambda_handler_delegates_normal_request_to_bolt(app_module, monkeypatch):
    """A normal Slack HTTP event (no retry header, no _worker flag) must
    reach the Bolt SlackRequestHandler."""
    calls = []

    class FakeHandler:
        def __init__(self, bolt_app):
            calls.append(("init", bolt_app))

        def handle(self, event, context):
            calls.append(("handle", event, context))
            return {"statusCode": 200, "body": "bolt"}

    monkeypatch.setattr(app_module, "SlackRequestHandler", FakeHandler)
    monkeypatch.setattr(app_module, "_get_bolt_app", lambda: "fake-bolt-app")

    event = {"headers": {"Content-Type": "application/json"}, "body": "{}"}
    result = app_module.lambda_handler(event, "ctx")

    assert result == {"statusCode": 200, "body": "bolt"}
    assert calls[0] == ("init", "fake-bolt-app")
    assert calls[1] == ("handle", event, "ctx")


def test_lambda_handler_worker_flag_false_takes_receiver_path(app_module, monkeypatch):
    """`_worker=False` should fall through to the receiver path, not be
    treated as a worker marker."""

    def boom_worker(payload):
        raise AssertionError("_process_worker must not run when _worker is falsy")

    monkeypatch.setattr(app_module, "_process_worker", boom_worker)

    class FakeHandler:
        def __init__(self, bolt_app):
            pass

        def handle(self, event, context):
            return {"statusCode": 202}

    monkeypatch.setattr(app_module, "SlackRequestHandler", FakeHandler)
    monkeypatch.setattr(app_module, "_get_bolt_app", lambda: object())

    assert app_module.lambda_handler({"_worker": False, "headers": {}}, None) == {"statusCode": 202}


# --------------------------------------------------------------------------- #
# _enqueue_worker — receiver → worker bridge
# --------------------------------------------------------------------------- #


def test_enqueue_worker_runs_inline_when_not_in_lambda(app_module, monkeypatch):
    """Without AWS_LAMBDA_FUNCTION_NAME (local dev / tests), the
    receiver must execute the worker path inline rather than trying to
    issue a boto3.invoke that would fail against real AWS."""
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)

    def boom_client():
        raise AssertionError("boto3 must not be touched off-Lambda")

    monkeypatch.setattr(app_module, "_get_lambda_client", boom_client)

    captured = []
    monkeypatch.setattr(app_module, "_process_worker", lambda payload: captured.append(payload))

    event = {"channel": "C1", "text": "hello"}
    app_module._enqueue_worker(event, is_dm=False)

    assert captured == [{"slack_event": event, "is_dm": False}]


def test_enqueue_worker_fires_async_invoke_in_lambda(app_module, monkeypatch):
    """When AWS_LAMBDA_FUNCTION_NAME is set, issue an async Lambda invoke
    with the correct payload shape — and NOT run the worker inline."""
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "gurumi-mention")

    invocations = []

    class FakeLambdaClient:
        def invoke(self, **kwargs):
            invocations.append(kwargs)
            return {"StatusCode": 202}

    monkeypatch.setattr(app_module, "_get_lambda_client", lambda: FakeLambdaClient())

    def boom_inline(payload):
        raise AssertionError("inline worker must not run when invoke succeeds")

    monkeypatch.setattr(app_module, "_process_worker", boom_inline)

    event = {"channel": "D1", "text": "안녕", "user": "U1"}
    app_module._enqueue_worker(event, is_dm=True)

    assert len(invocations) == 1
    call = invocations[0]
    assert call["FunctionName"] == "gurumi-mention"
    assert call["InvocationType"] == "Event"
    payload = json.loads(call["Payload"].decode("utf-8"))
    assert payload == {"_worker": True, "slack_event": event, "is_dm": True}


def test_enqueue_worker_falls_back_to_inline_on_invoke_failure(app_module, monkeypatch):
    """If boto3.invoke raises (IAM denied / throttle / network), fall back
    to inline execution so the user's message isn't silently dropped."""
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "gurumi-mention")

    class BrokenClient:
        def invoke(self, **_kwargs):
            raise RuntimeError("network unreachable")

    monkeypatch.setattr(app_module, "_get_lambda_client", lambda: BrokenClient())

    captured = []
    monkeypatch.setattr(app_module, "_process_worker", lambda payload: captured.append(payload))

    event = {"channel": "C1", "text": "hi"}
    app_module._enqueue_worker(event, is_dm=False)

    assert captured == [{"slack_event": event, "is_dm": False}]


def test_enqueue_worker_payload_preserves_non_ascii(app_module, monkeypatch):
    """Korean / emoji in the Slack event must survive the JSON round-trip."""
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "gurumi-mention")

    invocations = []

    class FakeClient:
        def invoke(self, **kwargs):
            invocations.append(kwargs)

    monkeypatch.setattr(app_module, "_get_lambda_client", lambda: FakeClient())
    monkeypatch.setattr(app_module, "_process_worker", lambda _p: None)

    event = {"text": "이미지 그려줘 🎨", "channel": "C1"}
    app_module._enqueue_worker(event, is_dm=False)

    payload = json.loads(invocations[0]["Payload"].decode("utf-8"))
    assert payload["slack_event"]["text"] == "이미지 그려줘 🎨"


# --------------------------------------------------------------------------- #
# _process_worker — the async Lambda entrypoint
# --------------------------------------------------------------------------- #


def test_process_worker_builds_webclient_from_bot_token(app_module, monkeypatch):
    """Bolt's injected WebClient is gone by the time the worker invocation
    runs — it lived in the receiver process. The worker must mint a fresh
    one from the bot token on `settings`.

    `Settings` is a frozen dataclass, so we replace the whole module-level
    `settings` with a dataclasses.replace() copy that carries the token
    we want to observe.
    """
    import dataclasses

    override = dataclasses.replace(app_module.settings, slack_bot_token="xoxb-test-token")
    monkeypatch.setattr(app_module, "settings", override)

    created = []

    class FakeWeb:
        def __init__(self, token):
            created.append(token)
            self.token = token

    monkeypatch.setattr(app_module, "WebClient", FakeWeb)

    captured = {}

    def fake_process(event, client, say, is_dm):
        captured["event"] = event
        captured["client"] = client
        captured["say"] = say
        captured["is_dm"] = is_dm

    monkeypatch.setattr(app_module, "_process", fake_process)

    payload = {"slack_event": {"channel": "C1", "text": "hi"}, "is_dm": True}
    app_module._process_worker(payload)

    assert created == ["xoxb-test-token"]
    assert captured["event"] == {"channel": "C1", "text": "hi"}
    assert captured["is_dm"] is True
    assert isinstance(captured["client"], FakeWeb)
    assert callable(captured["say"])


def test_process_worker_say_callable_posts_to_event_channel(app_module, monkeypatch):
    """The `say` closure passed to _process must post to the channel the
    original Slack event came from — not some default — and must forward
    thread_ts when provided."""
    posts = []

    class FakeWeb:
        def __init__(self, token):
            pass

        def chat_postMessage(self, **kwargs):
            posts.append(kwargs)

    monkeypatch.setattr(app_module, "WebClient", FakeWeb)

    captured_say = {}

    def fake_process(event, client, say, is_dm):
        captured_say["fn"] = say

    monkeypatch.setattr(app_module, "_process", fake_process)

    payload = {"slack_event": {"channel": "C-origin", "text": "x"}, "is_dm": False}
    app_module._process_worker(payload)

    say = captured_say["fn"]
    say("hello world")
    say("threaded reply", thread_ts="1700000000.000100")

    assert posts == [
        {"channel": "C-origin", "text": "hello world"},
        {"channel": "C-origin", "text": "threaded reply", "thread_ts": "1700000000.000100"},
    ]


def test_process_worker_tolerates_missing_fields(app_module, monkeypatch):
    """Defensive: a malformed payload (no slack_event, no is_dm) must
    still reach _process with sane defaults rather than crash."""

    class FakeWeb:
        def __init__(self, token):
            pass

    monkeypatch.setattr(app_module, "WebClient", FakeWeb)

    seen = {}

    def fake_process(event, client, say, is_dm):
        seen["event"] = event
        seen["is_dm"] = is_dm

    monkeypatch.setattr(app_module, "_process", fake_process)

    app_module._process_worker({})

    assert seen == {"event": {}, "is_dm": False}


# --------------------------------------------------------------------------- #
# Channel allowlist — block reply with first-channel substitution
# --------------------------------------------------------------------------- #


class _FakeDedup:
    """Minimal DedupStore stand-in: reserve always succeeds, no throttle."""

    def reserve(self, key, user="system"):
        return True

    def count_user_active(self, user):
        return 0


def test_process_blocked_channel_substitutes_first_allowed_channel(app_module, monkeypatch):
    """비허용 채널 응답의 `{}` 는 ALLOWED_CHANNEL_IDS 의 첫 번째 채널로 치환되며,
    Slack 채널 멘션 형식(`<#ID>`)으로 감싸 클릭 가능한 링크로 렌더되어야 한다."""
    import dataclasses

    override = dataclasses.replace(
        app_module.settings,
        allowed_channel_ids=["C04PPA399CP", "C08A9550X"],
        allowed_channel_message="구루미에게 질문은 {} 채널을 이용해 주세요~",
    )
    monkeypatch.setattr(app_module, "settings", override)
    monkeypatch.setattr(app_module, "_get_dedup", lambda: _FakeDedup())

    posts = []

    def fake_say(text, thread_ts=None):
        posts.append({"text": text, "thread_ts": thread_ts})

    event = {
        "channel": "C-BLOCKED",
        "ts": "1700000000.000100",
        "text": "hi",
        "user": "U1",
        "client_msg_id": "msg-block-1",
    }
    app_module._process(event, client=object(), say=fake_say, is_dm=False)

    assert posts == [
        {
            "text": "구루미에게 질문은 <#C04PPA399CP> 채널을 이용해 주세요~",
            "thread_ts": "1700000000.000100",
        }
    ]


def test_process_blocked_channel_message_without_placeholder_unchanged(app_module, monkeypatch):
    """`{}` 가 없는 메시지는 가공 없이 그대로 전송되어야 한다."""
    import dataclasses

    override = dataclasses.replace(
        app_module.settings,
        allowed_channel_ids=["C04PPA399CP"],
        allowed_channel_message="허용되지 않은 채널입니다.",
    )
    monkeypatch.setattr(app_module, "settings", override)
    monkeypatch.setattr(app_module, "_get_dedup", lambda: _FakeDedup())

    posts = []
    app_module._process(
        {
            "channel": "C-X",
            "ts": "1.1",
            "text": "hi",
            "user": "U1",
            "client_msg_id": "msg-block-2",
        },
        client=object(),
        say=lambda text, thread_ts=None: posts.append({"text": text, "thread_ts": thread_ts}),
        is_dm=False,
    )

    assert posts == [{"text": "허용되지 않은 채널입니다.", "thread_ts": "1.1"}]


def test_process_blocked_channel_no_message_when_unset(app_module, monkeypatch):
    """ALLOWED_CHANNEL_MESSAGE 가 비어 있으면 차단된 채널에서 아무 응답도 가지 않는다."""
    import dataclasses

    override = dataclasses.replace(
        app_module.settings,
        allowed_channel_ids=["C04PPA399CP"],
        allowed_channel_message="",
    )
    monkeypatch.setattr(app_module, "settings", override)
    monkeypatch.setattr(app_module, "_get_dedup", lambda: _FakeDedup())

    posts = []
    app_module._process(
        {
            "channel": "C-X",
            "ts": "1.1",
            "text": "hi",
            "user": "U1",
            "client_msg_id": "msg-block-3",
        },
        client=object(),
        say=lambda text, thread_ts=None: posts.append({"text": text, "thread_ts": thread_ts}),
        is_dm=False,
    )

    assert posts == []
