#!/usr/bin/env bash
# PROVENANCE ONLY. This script is how the shards were first built, but it is not reproducible: the mirror
# times out on large files (falling back to other editions) and some files have changed since. The exact
# bytes are archived in research/corpus/archive/ and unpacked by unpack-archive.sh; use that.
# Usage: fetch-gutenberg.sh [OUT=data] [GROUP=fiction|nonfiction|fresh-fiction|fresh-nonfiction]
# GROUP fresh-fiction / fresh-nonfiction write gutenberg/fresh-fiction.bin and fresh-nonfiction.bin: books held back entirely as the fresh held-out role of the
# Gate A rerun (never in exposure, probe, or development, and never seen by any earlier run).
# GROUP nonfiction writes gutenberg/nonfiction.bin (essays, science, history, philosophy), the
# second source of the retention test; fiction writes gutenberg/books.bin.
# Fetch a fixed list of US public-domain books from a Project Gutenberg mirror,
# cut the Gutenberg header and footer, and write one shard of documents
# separated by byte 0x1E. Output: $1/gutenberg/books.bin (default data/).
# The shard's SHA-256 is recorded in research/corpus/manifest.json.
set -euo pipefail
GROUP="${2:-fiction}"  # fiction | nonfiction | fresh-fiction | fresh-nonfiction
OUT="${1:-data}/gutenberg"
MIRROR="https://gutenberg.pglaf.org"
if [ "$GROUP" = fresh-fiction ]; then
  FILE=fresh-fiction.bin
  IDS="1342 829 2554 28054 600 2814 1257"
elif [ "$GROUP" = fresh-nonfiction ]; then
  FILE=fresh-nonfiction.bin
  IDS="4363 7370 5827"
elif [ "$GROUP" = nonfiction ]; then
  FILE=nonfiction.bin
  IDS="1228 2300 944 148 1497 8438 3300 34901 11224 4280 9662 10615 1404 147 2376 10378 5001 59 132 25717 674 2707 7142 3600"
else
  FILE=books.bin
  IDS="11 16 35 36 43 45 55 74 76 84 98 120 158 161 174 215 219 345 514 768 910 1232 1260 1400 1661 2701 5200 1080 2591 1998 205 4300 2600 1727 2680 3207 3296 2147 30254 408"
fi
mkdir -p "$OUT"
: > "$OUT/$FILE"
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
  awk '/^\*\*\* ?START OF/{on=1;next} /^\*\*\* ?END OF/{on=0} on' "$tmp" | tr -d '\r' | tr -d '\036' >> "$OUT/$FILE"
  printf '\036' >> "$OUT/$FILE"
  rm -f "$tmp"; sleep 1
done
sha256sum "$OUT/$FILE"; wc -c "$OUT/$FILE"
