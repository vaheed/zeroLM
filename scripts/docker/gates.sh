#!/usr/bin/env bash
# Runs the project's checks inside the container and prints a pass/fail table.
# Usage: gates.sh [quick]   quick = audit + tests + clippy only (no benchmarks)
# ZEROLM_VERBOSE=1 also prints test output (--nocapture) and lists report files.
set -uo pipefail
cd /zerolm
MODE="${1:-full}"
VERBOSE="${ZEROLM_VERBOSE:-0}"
NOCAP=(); [[ "$VERBOSE" != 0 ]] && NOCAP=(-- --nocapture)
declare -a NAMES RESULTS

# Phase 6 hidden seed. A git checkout derives it from HEAD; the image has no
# .git, so use the commit baked in at build time (same seed as CI for that
# commit). Without one, draw a random seed: never a developer-chosen value.
if [[ -z "${ZEROLM_HIDDEN_SEED:-}" ]]; then
  if [[ "${ZEROLM_COMMIT:-unknown}" =~ ^[0-9a-f]{40}$ ]]; then
    export ZEROLM_HIDDEN_SEED="$ZEROLM_COMMIT"
    echo "Phase 6 hidden seed: from image commit $ZEROLM_COMMIT"
  else
    export ZEROLM_HIDDEN_SEED="random-$(od -An -N8 -tx8 /dev/urandom | tr -d ' ')"
    echo "Phase 6 hidden seed: random ($ZEROLM_HIDDEN_SEED); build with ZEROLM_COMMIT=\$(git rev-parse HEAD) to match CI"
  fi
fi

gate() {
  local name="$1"; shift
  printf '\n\033[1m== %s ==\033[0m\n$ %s\n' "$name" "$*"
  local t0=$SECONDS
  if "$@"; then RESULTS+=("PASS ($((SECONDS - t0)) s)"); else RESULTS+=("FAIL ($((SECONDS - t0)) s)"); fi
  NAMES+=("$name")
}

gate "Phase 0: policy audit" zerolm-audit --json
gate "Unit and integration tests" cargo test --workspace --locked "${NOCAP[@]}"
gate "Clippy (warnings are errors)" cargo clippy --workspace --all-targets --locked -- -D warnings
if [[ "$MODE" != quick ]]; then
  gate "Phase 2: architecture bake-off (~2 min)" zerolm-bakeoff --check model/specs/architecture.json
  gate "Phase 4: online learning (~15 s)" zerolm-online-lm
  gate "Phase 5: weight-resident memory (~65 s)" zerolm-memory
  gate "Phase 6: compositional reasoning (~90 s)" zerolm-reasoning
fi

printf '\n\033[1m== Summary ==\033[0m\n'
fail=0
for i in "${!NAMES[@]}"; do
  printf '  %-45s %s\n' "${NAMES[$i]}" "${RESULTS[$i]}"
  [[ "${RESULTS[$i]}" == FAIL* ]] && fail=1
done
printf '  %-45s %s\n' "Phase 1: baseline provenance" "NOT RUN (needs a clean git checkout; run on the host)"
[[ "$MODE" == quick ]] && printf '  %-45s %s\n' "Phases 2, 4, 5, 6 benchmarks" "NOT RUN (quick mode)"
if [[ "$VERBOSE" != 0 ]]; then
  printf '\nReports (also on the host under ./reports):\n'; find reports -type f | sort | sed 's/^/  /'
fi
exit $fail
