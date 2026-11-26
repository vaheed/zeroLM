#!/usr/bin/env bash
# Fetch the evaluation-only sets: BLiMP (CC BY 4.0, github.com/alexwarstadt/blimp,
# 67 paradigm files concatenated in name order) and the LAMBADA test passages
# (CC BY 4.0, zenodo.org/records/2630551). Output: $1/eval/blimp.jsonl and
# $1/eval/lambada_test.txt (default $1 = data). Their SHA-256 values are
# recorded in research/corpus/manifest.json and checked by `zerolm-data verify`.
# The LAMBADA archive is 334 MB; only its test file is kept.
set -euo pipefail
OUT="${1:-data}/eval"
mkdir -p "$OUT"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
names="$(curl -sf -m 60 https://api.github.com/repos/alexwarstadt/blimp/contents/data | jq -r '.[].name' | grep '\.jsonl$' | LC_ALL=C sort)"
[ "$(echo "$names" | wc -l)" -eq 67 ] || { echo "expected 67 BLiMP files" >&2; exit 1; }
: > "$OUT/blimp.jsonl"
for n in $names; do
  curl -sf -m 60 -o "$tmp/$n" "https://raw.githubusercontent.com/alexwarstadt/blimp/master/data/$n"
  cat "$tmp/$n" >> "$OUT/blimp.jsonl"
done
curl -sf -L -m 1800 -o "$tmp/lambada.tar.gz" https://zenodo.org/records/2630551/files/lambada-dataset.tar.gz
f="$(tar tzf "$tmp/lambada.tar.gz" | grep 'lambada_test_plain_text.txt$' | head -1)"
tar xzf "$tmp/lambada.tar.gz" -C "$tmp" "$f"
cp "$tmp/$f" "$OUT/lambada_test.txt"
sha256sum "$OUT/blimp.jsonl" "$OUT/lambada_test.txt"; wc -c "$OUT/blimp.jsonl" "$OUT/lambada_test.txt"
