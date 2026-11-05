#!/usr/bin/env python3
"""Does a bigger model help? Accuracy as a function of how many phrases the weights contain.

Keeps a random fraction of the FEATURE lines of a weight file (seeded, several repeats), always leaving
at least one phrase per intent, and measures accuracy on labelled prompts. The responses and the decoder
are untouched. This is an ablation of the hand-written model, not training; it shows whether adding
phrases still pays off on held-out text. Output is JSON plus a table.
"""
from __future__ import annotations
import argparse
import json
import random
import statistics
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_cases(path: Path):
    return [tuple(l.split("\t", 1)) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#") and "\t" in l]


def accuracy(binary, model, cases):
    p = subprocess.run([binary, "classify", "--model", model], input="\n".join(t for _, t in cases) + "\n",
                       capture_output=True, text=True, check=True, cwd=ROOT)
    got = [json.loads(l)["intent"] for l in p.stdout.splitlines()]
    return sum(g == w.strip() for g, (w, _) in zip(got, cases)) / len(cases)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=ROOT / "weights.zlm")
    ap.add_argument("--binary", default=str(ROOT / "rust/target/release/zerolm"))
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    lines = args.model.read_text(encoding="utf-8").splitlines()
    feat_idx = [i for i, l in enumerate(lines) if l.startswith("FEATURE ")]
    intent_of = {i: int(lines[i].split()[1]) for i in feat_idx}
    sets = {"dev": read_cases(ROOT / "tests/heldout.tsv"), "test": read_cases(ROOT / "tests/heldout_test.tsv"),
            "regression": read_cases(ROOT / "tests/regression.tsv")}
    results = []
    for frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        per = {k: [] for k in sets}
        for rep in range(args.repeats if frac < 1.0 else 1):
            rng = random.Random(1000 * rep + int(frac * 100))
            keep = set(rng.sample(feat_idx, max(1, round(frac * len(feat_idx)))))
            for intent in set(intent_of.values()):  # never leave an intent without a phrase
                mine = [i for i in feat_idx if intent_of[i] == intent]
                if not keep.intersection(mine):
                    keep.add(rng.choice(mine))
            out = [l for i, l in enumerate(lines) if not l.startswith("FEATURE ") or i in keep]
            with tempfile.NamedTemporaryFile("w", suffix=".zlm", delete=False) as f:
                f.write("\n".join(out) + "\n")
            for k, cases in sets.items():
                per[k].append(accuracy(args.binary, f.name, cases))
            Path(f.name).unlink()
        row = {"fraction": frac, "phrases": round(frac * len(feat_idx))}
        for k, v in per.items():
            row[k] = {"mean": statistics.mean(v), "min": min(v), "max": max(v)}
        results.append(row)
        print(f"{frac:4.0%} {row['phrases']:4d} phrases   dev {row['dev']['mean']:.1%}   test {row['test']['mean']:.1%}"
              f"  (range {row['test']['min']:.1%}-{row['test']['max']:.1%})   regression {row['regression']['mean']:.1%}")
    if args.json:
        args.json.write_text(json.dumps({"phrases_total": len(feat_idx), "repeats": args.repeats, "rows": results}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
