#!/usr/bin/env bash
# Fetch a fixed list of US public-domain books from a Project Gutenberg mirror,
# cut the Gutenberg header and footer, and write one shard of documents
# separated by byte 0x1E. Output: $1/gutenberg/books.bin (default data/).
# The shard's SHA-256 is recorded in research/corpus/manifest.json.
set -euo pipefail
OUT="${1:-data}/gutenberg"
MIRROR="https://gutenberg.pglaf.org"
IDS="11 16 35 36 43 45 55 74 76 84 98 120 158 161 174 215 219 345 514 768 910 1232 1260 1400 1661 2701 5200 1080 2591 1998 205 4300 2600 1727 2680 3207 3296 2147 30254 408"
mkdir -p "$OUT"
: > "$OUT/books.bin"
for id in $IDS; do
  path=""; n=${#id}
  if [ "$n" -eq 1 ]; then path="0"; else for ((i=0;i<n-1;i++)); do path="$path${path:+/}${id:$i:1}"; done; fi
  tmp="$(mktemp)"
  ok=0
  for f in "$id-0.txt" "$id.txt" "$id-8.txt"; do
    if curl -sf -m 60 -L -o "$tmp" "$MIRROR/$path/$id/$f"; then ok=1; break; fi
  done
  if [ "$ok" = 0 ]; then echo "skip $id (not found)" >&2; rm -f "$tmp"; continue; fi
  # Keep only the text between the START and END markers; drop non-UTF-8 files.
  if ! iconv -f utf-8 -t utf-8 "$tmp" >/dev/null 2>&1; then echo "skip $id (not UTF-8)" >&2; rm -f "$tmp"; continue; fi
  awk '/^\*\*\* ?START OF/{on=1;next} /^\*\*\* ?END OF/{on=0} on' "$tmp" | tr -d '\r' | tr -d '\036' >> "$OUT/books.bin"
  printf '\036' >> "$OUT/books.bin"
  rm -f "$tmp"; sleep 1
done
sha256sum "$OUT/books.bin"; wc -c "$OUT/books.bin"
