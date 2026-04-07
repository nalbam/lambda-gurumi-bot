#!/usr/bin/env bash
set -euo pipefail

# Sync AWS documentation repos to S3 for Knowledge Base ingestion
# Usage: ./sync.sh <s3-bucket> [repos-file]

S3_BUCKET="${1:?Usage: sync.sh <s3-bucket> [repos-file]}"
REPOS_FILE="${2:-$(dirname "$0")/repos.txt}"
BUILD_DIR="build/awsdocs"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

# Read repos list, skip comments and empty lines
grep -v '^\s*#' "${REPOS_FILE}" | grep -v '^\s*$' | while read -r repo; do
  repo_name=$(basename "${repo}")
  echo "=== Cloning ${repo} ==="
  git clone --depth 1 "https://github.com/${repo}.git" "${BUILD_DIR}/${repo_name}"

  # Find doc_source directory (standard awsdocs structure)
  doc_dir="${BUILD_DIR}/${repo_name}/doc_source"
  if [ ! -d "${doc_dir}" ]; then
    echo "No doc_source/ found in ${repo_name}, skipping"
    rm -rf "${BUILD_DIR}/${repo_name}"
    continue
  fi

  # Keep only markdown files
  find "${doc_dir}" -type f ! -name '*.md' -delete
  find "${doc_dir}" -type d -empty -delete

  md_count=$(find "${doc_dir}" -name '*.md' | wc -l | tr -d ' ')
  echo "Found ${md_count} markdown files in ${repo_name}"

  # Sync to S3
  echo "Syncing ${repo_name} to s3://${S3_BUCKET}/documents/${repo_name}/"
  aws s3 sync --delete \
    "${doc_dir}/" \
    "s3://${S3_BUCKET}/documents/${repo_name}/"

  # Cleanup
  rm -rf "${BUILD_DIR}/${repo_name}"
  echo ""
done

echo "=== All repos synced ==="
