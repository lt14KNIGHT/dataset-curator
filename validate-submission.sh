#!/usr/bin/env bash
# validate-submission.sh — run AFTER deploying to HF Spaces
# Usage: ./validate-submission.sh https://YOUR_USERNAME-dataset-curator.hf.space .

set -uo pipefail

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  echo "Usage: $0 <hf_space_url> [repo_dir]"
  echo "Example: $0 https://johndoe-dataset-curator.hf.space ."
  exit 1
fi

PING_URL="${PING_URL%/}"
PASS=0

echo ""
echo "========================================"
echo "  OpenEnv Submission Validator"
echo "========================================"
echo "Space URL : $PING_URL"
echo "Repo dir  : $REPO_DIR"
echo ""

# Step 1 — ping /reset
echo "[1/3] Pinging $PING_URL/reset ..."
HTTP=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>/dev/null || echo "000")

if [ "$HTTP" = "200" ]; then
  echo "  PASSED — HF Space is live and responds to /reset"
  PASS=$((PASS+1))
else
  echo "  FAILED — /reset returned HTTP $HTTP (expected 200)"
  echo "  Hint: make sure the Space has finished building in HF dashboard"
  exit 1
fi

# Step 2 — docker build
echo ""
echo "[2/3] Running docker build ..."
if ! command -v docker &>/dev/null; then
  echo "  FAILED — docker not found. Install: https://docs.docker.com/get-docker/"
  exit 1
fi

if docker build "$REPO_DIR" -t dataset-curator-validate 2>&1 | tail -3; then
  echo "  PASSED — Docker build succeeded"
  PASS=$((PASS+1))
else
  echo "  FAILED — Docker build failed"
  exit 1
fi

# Step 3 — openenv validate
echo ""
echo "[3/3] Running openenv validate ..."
if ! command -v openenv &>/dev/null; then
  echo "  FAILED — openenv not found. Run: pip install openenv-core"
  exit 1
fi

if (cd "$REPO_DIR" && openenv validate 2>&1); then
  echo "  PASSED — openenv validate passed"
  PASS=$((PASS+1))
else
  echo "  FAILED — openenv validate failed"
  exit 1
fi

echo ""
echo "========================================"
echo "  All $PASS/3 checks passed!"
echo "  Your submission is ready."
echo "========================================"
echo ""
