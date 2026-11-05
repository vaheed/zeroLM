#!/usr/bin/env bash
# End-to-end check of the Docker image: one-shot, persistence across containers,
# tag isolation, and refusal of a second container on a tag that is in use.
# Usage: scripts/docker_smoke.sh   (needs docker compose; builds the image)
set -euo pipefail
cd "$(dirname "$0")/.."

# One compose project per run, so this never touches a user's real volume.
export COMPOSE_PROJECT_NAME="zerolm-smoke-$$"
cleanup() {
  docker rm -f "zl-smoke-a-$$" >/dev/null 2>&1 || true
  docker compose --profile api --profile daemon down -v >/dev/null 2>&1 || true
}
trap cleanup EXIT

run() { docker compose run --rm -T zerolm "$@"; }

docker compose build

echo "== one-shot answer"
run "does it use training" | grep -q "no training"

echo "== memory persists across containers"
printf 'remember my name is Vahid\n' | run >/dev/null
printf 'what is my name\n' | run | grep -q "my name is Vahid\."

echo "== tags are isolated"
if printf 'what is my name\n' | run live --tag other | grep -q "Vahid"; then
  echo "tag 'other' saw the default tag's memory" >&2; exit 1
fi

echo "== JSON-lines mode"
printf '{"text":"what license does it use"}\n' | run live --jsonl --tag daemon | grep -q '"intent":"license"'

echo "== a second container on a tag in use is refused"
docker compose run -d --name "zl-smoke-a-$$" zerolm live --tag busy >/dev/null
sleep 2
if printf 'hello\n' | run live --tag busy >/dev/null 2>&1; then
  echo "second container was not refused" >&2; exit 1
fi

echo "== killed container leaves the tag usable"
docker kill "zl-smoke-a-$$" >/dev/null
printf 'hello\n' | run live --tag busy | grep -q "bot>"

echo "== HTTP API container: refuses to start without a token"
if ZEROLM_TOKEN= docker compose run --rm -T --no-deps zerolm-api serve --addr 0.0.0.0:8080 >/dev/null 2>&1; then
  echo "the API container started without a token" >&2; exit 1
fi

echo "== HTTP API container: health check, auth, chat completions, persistence"
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
export ZEROLM_PORT="$PORT" ZEROLM_TOKEN="smoke-test-token-$$-0123456789"
docker compose up -d zerolm-api >/dev/null
for _ in $(seq 1 40); do
  [ "$(docker inspect -f '{{.State.Health.Status}}' "$(docker compose ps -q zerolm-api)")" = healthy ] && break
  sleep 1
done
[ "$(docker inspect -f '{{.State.Health.Status}}' "$(docker compose ps -q zerolm-api)")" = healthy ] \
  || { echo "API container never became healthy" >&2; docker compose logs zerolm-api >&2; exit 1; }
URL="http://127.0.0.1:$PORT"
curl -fsS "$URL/v1/health" | grep -q '"status":"ok"'
[ "$(curl -s -o /dev/null -w '%{http_code}' "$URL/v1/models")" = 401 ]
AUTH="Authorization: Bearer $ZEROLM_TOKEN"
curl -fsS -H "$AUTH" -d '{"model":"zerolm","messages":[{"role":"user","content":"what license does it use"}]}' \
  "$URL/v1/chat/completions" | grep -q 'MIT license'
curl -fsS -H "$AUTH" -d '{"model":"zerolm-live","user":"api","messages":[{"role":"user","content":"remember my name is Vahid"}]}' \
  "$URL/v1/chat/completions" >/dev/null
docker compose restart zerolm-api >/dev/null
for _ in $(seq 1 40); do curl -fsS "$URL/v1/health" >/dev/null 2>&1 && break; sleep 1; done
curl -fsS -H "$AUTH" -d '{"model":"zerolm-live","user":"api","messages":[{"role":"user","content":"what is my name"}]}' \
  "$URL/v1/chat/completions" | grep -q 'my name is Vahid'
echo "== the CLI cannot take a session the API container holds"
if printf 'hello\n' | docker compose run --rm -T zerolm live --tag api >/dev/null 2>&1; then
  echo "CLI container shared a session with the API container" >&2; exit 1
fi

echo "docker smoke test passed"
