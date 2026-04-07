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

  # Collect all markdown files into a flat output directory
  out_dir="${BUILD_DIR}/out/${repo_name}"
  mkdir -p "${out_dir}"
  find "${BUILD_DIR}/${repo_name}" -name '*.md' \
    ! -name 'README.md' ! -name 'CONTRIBUTING.md' ! -name 'LICENSE*' \
    ! -path '*/.github/*' \
    -exec cp {} "${out_dir}/" \;

  md_count=$(find "${out_dir}" -name '*.md' | wc -l | tr -d ' ')
  if [ "${md_count}" -eq 0 ]; then
    echo "No markdown docs found in ${repo_name}, skipping"
    rm -rf "${BUILD_DIR}/${repo_name}" "${out_dir}"
    continue
  fi

  echo "Found ${md_count} markdown files in ${repo_name}"

  # Sync to S3
  echo "Syncing ${repo_name} to s3://${S3_BUCKET}/documents/${repo_name}/"
  aws s3 sync --delete \
    "${out_dir}/" \
    "s3://${S3_BUCKET}/documents/${repo_name}/"

  # Cleanup
  rm -rf "${BUILD_DIR}/${repo_name}" "${out_dir}"
  echo ""
done

echo "=== All repos synced ==="
