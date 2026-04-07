#!/usr/bin/env bash
set -euo pipefail

# Sync AWS documentation repos to S3 for Knowledge Base ingestion
# Usage: ./sync.sh <s3-bucket> [repos-file]
#
# repos.txt format: org/repo[:branch]
#   awsdocs/amazon-s3-userguide          (defaults to main)
#   awsdocs/amazon-eks-user-guide:mainline

S3_BUCKET="${1:?Usage: sync.sh <s3-bucket> [repos-file]}"
REPOS_FILE="${2:-$(dirname "$0")/repos.txt}"
BUILD_DIR="build/awsdocs"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

# Read repos list, skip comments and empty lines
grep -v '^\s*#' "${REPOS_FILE}" | grep -v '^\s*$' | while read -r entry; do
  # Parse repo and optional branch (org/repo:branch)
  repo="${entry%%:*}"
  branch="${entry#*:}"
  if [ "${branch}" = "${entry}" ]; then
    branch="main"
  fi

  repo_name=$(basename "${repo}")
  echo "=== Cloning ${repo} (branch: ${branch}) ==="
  git clone --depth 1 --branch "${branch}" "https://github.com/${repo}.git" "${BUILD_DIR}/${repo_name}"

  # Collect documentation files (.md, .adoc) into a flat output directory
  out_dir="${BUILD_DIR}/out/${repo_name}"
  mkdir -p "${out_dir}"
  find "${BUILD_DIR}/${repo_name}" \( -name '*.md' -o -name '*.adoc' \) \
    ! -name 'README.md' ! -name 'README-*.md' \
    ! -name 'CONTRIBUTING.md' ! -name 'LICENSE*' \
    ! -name 'CHANGELOG*' ! -name 'CODE_OF_CONDUCT*' \
    ! -path '*/.github/*' \
    -exec cp {} "${out_dir}/" \;

  doc_count=$(find "${out_dir}" \( -name '*.md' -o -name '*.adoc' \) | wc -l | tr -d ' ')
  if [ "${doc_count}" -eq 0 ]; then
    echo "No documentation files found in ${repo_name}, skipping"
    rm -rf "${BUILD_DIR}/${repo_name}" "${out_dir}"
    continue
  fi

  echo "Found ${doc_count} documentation files in ${repo_name}"

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
