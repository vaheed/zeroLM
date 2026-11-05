#!/usr/bin/env python3
"""Check that the C and Rust runtimes agree, on the test prompts and on many damaged copies of them.

Runs `--verbose` on both binaries for every prompt in tests/*.tsv and examples/, then for seeded
typo variants (swap/drop/double/substitute) and a few hand-picked edge cases (odd spacing, non-ASCII,
over-long words, inflections). Any difference in output is printed and the exit code is 1.
"""
from __future__ import annotations
import argparse
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robustness import damage, rows  # noqa: E402

EDGE = ["recalling licensing contributors testers running benchmarks", "  Weird   spacing,, and Ünïcode wörds  ",
        "a" * 300 + " licence", "classes processes passes guesses glasses boxes churches dishes addresses",
        "x", "!!!", "TESTS, TESTING; Tested?", "fine-tune fine tuned fine tuning", "queries queried querying"]


def run(binary: str, model: str, text: str) -> str:
    return subprocess.run([binary, "--model", model, "--verbose", text], capture_output=True, text=True).stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c", default=str(ROOT / "zerolm"))
    ap.add_argument("--rust", default=str(ROOT / "rust/target/release/zerolm"))
    ap.add_argument("--model", default=str(ROOT / "weights.zlm"))
    ap.add_argument("--variants", type=int, default=3, help="damaged copies per prompt")
    args = ap.parse_args()

    prompts = [t for f in sorted((ROOT / "tests").glob("*.tsv")) for _, t in rows(f)]
    prompts += [l.split("\t")[-1].strip() for f in (ROOT / "examples").glob("*.tsv") for l in f.read_text().splitlines() if l.strip() and not l.startswith("#")]
    texts = list(prompts) + EDGE
    for seed in range(args.variants):
        rng = random.Random(seed)
        texts += [d[0] for t in prompts if (d := damage(t, rng))]
    bad = 0
    for t in texts:
        a, b = run(args.c, args.model, t), run(args.rust, args.model, t)
        if a != b:
            bad += 1
            print(f"DIFF {t[:70]!r}\n  C:    {a.splitlines()[:1]}\n  Rust: {b.splitlines()[:1]}")
    print(f"{len(texts)} inputs compared, {bad} differences")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
