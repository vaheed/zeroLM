#!/usr/bin/env bash
# Unpack the archived Project Gutenberg shards into $1/gutenberg/ (default data/) and verify them
# against research/corpus/manifest.json. These four shards were first built by fetch-gutenberg.sh, but
# the mirror does not give the same bytes twice: large files time out and fall back to other
# editions, and some files changed. The archive (xz, public-domain text, about 10 MB) pins the exact
# bytes every result was computed on; the manifest's SHA-256 verifies them.
set -euo pipefail
OUT="${1:-data}/gutenberg"
HERE="$(dirname "$0")"
mkdir -p "$OUT"
for f in books nonfiction fresh-fiction fresh-nonfiction; do
  xz -dc "$HERE/archive/$f.bin.xz" > "$OUT/$f.bin"
done
ls -l "$OUT"
