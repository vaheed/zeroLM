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
  docker compose down -v >/dev/null 2>&1 || true
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

echo "docker smoke test passed"
