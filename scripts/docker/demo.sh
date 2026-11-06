#!/usr/bin/env bash
# Guided walkthrough of what ZeroLM can do today, with the internals visible.
# Usage: demo.sh            (verbose on: shows parsing, tensor writes, premises)
#        ZEROLM_VERBOSE=0 demo.sh   (replies only)
# State lives in $ZEROLM_DATA (default /data/demo), wiped at the start.
set -euo pipefail
cd /zerolm 2>/dev/null || true
export ZEROLM_VERBOSE="${ZEROLM_VERBOSE:-1}"
DATA="${ZEROLM_DATA:-/data}/demo"
rm -rf "$DATA"; mkdir -p "$DATA"

step() { printf '\n\033[1m== %s ==\033[0m\n%s\n\n' "$1" "$2"; }

step "1. The model at birth" \
"No pretrained weights: initialization is seeded and pinned by SHA-256."
zerolm info

step "2. Reasoning over stored statements" \
"Statements become bound vectors in tensor memory. Questions are answered by
composing them; the verbose lines show the parse, the tensors written, and the
premises behind each derived answer.  Input: examples/reasoning.txt"
zerolm reason --dir "$DATA/reason" < examples/reasoning.txt 2>&1

step "3. A new process: memory persists in the checkpoint" \
"Nothing is replayed; the answer is read from the saved tensors.
Input: examples/reasoning-after-restart.txt"
zerolm reason --dir "$DATA/reason" < examples/reasoning-after-restart.txt 2>&1

step "4. All eight reasoning operators" \
"Inheritance with exceptions, comparison, temporal order, conditionals.
Input: examples/reasoning-more.txt"
zerolm reason < examples/reasoning-more.txt 2>&1

step "5. Weight-resident fact memory" \
"teach/ask/correct/forget write and read checkpoint tensors; the frozen core
stays byte-identical (see 'core unchanged=true').
Input: examples/memory.txt, then examples/memory-after-restart.txt in a new process"
zerolm memory --dir "$DATA/memory" < examples/memory.txt 2>&1
zerolm memory --dir "$DATA/memory" < examples/memory-after-restart.txt 2>&1

step "6. Online learning from a stream (about 15 s)" \
"One chronological pass; held-out bits per byte before and after, against baselines."
zerolm-online-lm --out "$DATA/online-lm" 2>&1

step "Done" \
"Try it yourself:  docker compose run --rm reason   (or: memory)
Run every gate:    docker compose run --rm gates
Honest limits:     docs/claims-and-limitations.md"
