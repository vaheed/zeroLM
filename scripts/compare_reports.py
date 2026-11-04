#!/usr/bin/env python3
"""Compare two ZeroLM evaluation reports and flag regressions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    base = json.loads(args.baseline.read_text(encoding="utf-8"))
    cand = json.loads(args.candidate.read_text(encoding="utf-8"))
    metrics = ("regression_accuracy", "unknown_recall", "overall_accuracy")
    failed = False
    for metric in metrics:
        delta = float(cand[metric]) - float(base[metric])
        print(f"{metric}: {base[metric]:.1%} -> {cand[metric]:.1%} ({delta:+.1%})")
        if delta < 0:
            failed = True
    print("Result: REGRESSION DETECTED" if failed else "Result: no measured regression")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
