#!/usr/bin/env bash
# Docker smoke test — builds the image, starts it with a fake LM Studio upstream,
# and asserts health endpoints + models proxying.
#
# Requires: docker, curl, python3
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="lms-passthrough:smoke-test"
CONTAINER="lms-passthrough-smoke-$$"
MOCK_PID=""
STATE_DIR=""

cleanup() {
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    [ -n "$MOCK_PID" ] && kill "$MOCK_PID" 2>/dev/null || true
    [ -n "$STATE_DIR" ] && rm -rf "$STATE_DIR"
}
trap cleanup EXIT

# Use a repo-local directory so Docker Desktop (macOS) can mount it.
STATE_DIR="$REPO_ROOT/.smoke-test-state-$$"
mkdir -p "$STATE_DIR"

# ---- Build image ----
echo "==> Building image $IMAGE"
docker build -t "$IMAGE" "$REPO_ROOT"

# ---- Start mock LM Studio upstream on the host ----
echo "==> Starting mock upstream on port 9876"
python3 "$REPO_ROOT/tests/docker/mock_upstream.py" &
MOCK_PID=$!
sleep 1

# ---- Start container ----
docker run -d --name "$CONTAINER" \
    --add-host=host.docker.internal:host-gateway \
    -p 8000:8000 \
    -v "$REPO_ROOT/tests/docker/config.yaml:/app/config.yaml:ro" \
    -v "$STATE_DIR:/app/state" \
    "$IMAGE"

# Wait for the proxy to be ready (health/live returns 200)
echo "==> Waiting for proxy"
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health/live >/dev/null 2>&1; then
        echo "    proxy ready after ${i}s"
        break
    fi
    sleep 1
done

PASS=0
FAIL=0
assert() {
    local label="$1" actual="$2" expected="$3"
    if [ "$actual" = "$expected" ]; then
        echo "  PASS: $label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $label (got '$actual', expected '$expected')"
        FAIL=$((FAIL + 1))
    fi
}

assert_contains() {
    local label="$1" haystack="$2" needle="$3"
    if echo "$haystack" | grep -q "$needle"; then
        echo "  PASS: $label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $label ('$needle' not found)"
        FAIL=$((FAIL + 1))
    fi
}

json_field() {
    local url="$1" path="$2"
    python3 -c "import sys,json; print(json.load(sys.stdin)${path})" < <(curl -s "$url")
}

# ---- /health/live ----
echo "==> Checking /health/live"
STATUS=$(curl -sf -o /dev/null -w '%{http_code}' http://localhost:8000/health/live)
assert "/health/live returns 200" "$STATUS" "200"
assert "/health/live body is ok" "$(json_field http://localhost:8000/health/live "['status']")" "ok"

# ---- /health/ready ----
echo "==> Checking /health/ready"
STATUS=$(curl -sf -o /dev/null -w '%{http_code}' http://localhost:8000/health/ready)
assert "/health/ready returns 200" "$STATUS" "200"
assert "/health/ready body is ok" "$(json_field http://localhost:8000/health/ready "['status']")" "ok"
assert "/health/ready probed lm-studio as ready" \
    "$(json_field http://localhost:8000/health/ready "['providers']['lm-studio']")" "True"

# ---- /api/v1/models through the proxy ----
echo "==> Checking /api/v1/models"
STATUS=$(curl -sf -o /dev/null -w '%{http_code}' http://localhost:8000/api/v1/models)
assert "/api/v1/models returns 200" "$STATUS" "200"
BODY=$(curl -sf http://localhost:8000/api/v1/models)
assert_contains "models response contains test-model" "$BODY" "test-model"

# ---- Mounted state volume ----
echo "==> Checking state volume"
if [ -f "$STATE_DIR/lms-passthrough.sqlite3" ]; then
    echo "  PASS: state SQLite file created"
    PASS=$((PASS + 1))
else
    echo "  FAIL: state SQLite file not found at $STATE_DIR"
    FAIL=$((FAIL + 1))
fi

# ---- Readiness is a real signal: losing the upstream makes the proxy unready ----
# Runs last: every other check above needs the mock upstream.
echo "==> Checking /health/ready with the upstream down"
kill "$MOCK_PID" 2>/dev/null || true
MOCK_PID=""
STATUS=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health/ready)
assert "/health/ready returns 503 without a reachable provider" "$STATUS" "503"
assert "/health/ready body is unavailable" \
    "$(json_field http://localhost:8000/health/ready "['status']")" "unavailable"
assert "/health/ready probed lm-studio as not ready" \
    "$(json_field http://localhost:8000/health/ready "['providers']['lm-studio']")" "False"

# ---- Compose validation ----
echo "==> Validating compose.yml"
if docker compose -f "$REPO_ROOT/compose.yml" config --quiet 2>/dev/null; then
    echo "  PASS: compose file is valid"
    PASS=$((PASS + 1))
else
    echo "  FAIL: compose file validation failed"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"
if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
