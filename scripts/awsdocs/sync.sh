#!/usr/bin/env bash
set -euo pipefail

# Sync AWS official PDF documentation to S3 for Knowledge Base ingestion
# Usage: ./sync.sh <s3-bucket> [docs-file]
#
# docs.txt format: name url
#   s3 https://docs.aws.amazon.com/pdfs/AmazonS3/latest/userguide/s3-userguide.pdf

S3_BUCKET="${1:?Usage: sync.sh <s3-bucket> [docs-file]}"
DOCS_FILE="${2:-$(dirname "$0")/docs.txt}"
BUILD_DIR="build/awsdocs"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

# Read docs list, skip comments and empty lines
grep -v '^\s*#' "${DOCS_FILE}" | grep -v '^\s*$' | while read -r name url; do
  echo "=== Downloading ${name} ==="
  out_dir="${BUILD_DIR}/${name}"
  mkdir -p "${out_dir}"

  curl -sL -o "${out_dir}/${name}.pdf" "${url}"
  size=$(du -h "${out_dir}/${name}.pdf" | cut -f1)
  echo "Downloaded ${name}.pdf (${size})"

  # Sync to S3
  echo "Syncing to s3://${S3_BUCKET}/documents/${name}/"
  aws s3 sync --delete \
    "${out_dir}/" \
    "s3://${S3_BUCKET}/documents/${name}/"

  rm -rf "${out_dir}"
  echo ""
done

echo "=== All docs synced ==="
