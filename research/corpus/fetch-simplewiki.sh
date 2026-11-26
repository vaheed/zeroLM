#!/usr/bin/env bash
# Fetch the Simple English Wikipedia dump of 2026-11-01 (CC BY-SA 4.0, Wikipedia contributors) and
# clean it into one shard of articles separated by byte 0x1E: $1/wikipedia/simplewiki.bin (default
# data/). The shard's SHA-256 is recorded in research/corpus/manifest.json. The dump is 360 MB
# compressed; Wikimedia removes old dumps, so if this URL is gone the pinned hash cannot be reproduced
# and the manifest must be re-pinned deliberately.
set -euo pipefail
OUT="${1:-data}/wikipedia"
DUMP="simplewiki-20261101-pages-articles.xml.bz2"
mkdir -p "$OUT"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
curl -sf -L -m 3000 -o "$tmp/$DUMP" "https://dumps.wikimedia.org/simplewiki/20261101/$DUMP"
python3 -I "$(dirname "$0")/wiki_clean.py" "$tmp/$DUMP" "$OUT/simplewiki.bin"
sha256sum "$OUT/simplewiki.bin"; wc -c "$OUT/simplewiki.bin"
