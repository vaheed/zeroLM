#!/usr/bin/env python3
"""Measure a ZeroLM intent router on labelled data (intent<TAB>text per line).

Reports accuracy, per-intent precision/recall/F1, the most common confusions, and a
score-threshold sweep (accuracy and coverage if you only accept answers scoring at least t).
Pure measurement: it never changes weights. Use data the weights were NOT written from,
otherwise the numbers only show that your own phrases match themselves.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def find_binary(explicit: str | None) -> Path:
    for c in ([Path(explicit)] if explicit else []) + [ROOT / "rust/target/release/zerolm", ROOT / "rust/target/debug/zerolm"]:
        if c.exists():
            return c
    raise SystemExit("no zerolm binary found; run `cargo build --release` in rust/")


def read_cases(path: Path):
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2 or not parts[1].strip():
            raise SystemExit(f"{path}:{n}: expected intent<TAB>text")
        rows.append((parts[0].strip(), parts[1].strip()))
    if not rows:
        raise SystemExit(f"{path}: no cases")
    return rows


def classify(binary: Path, model: Path, texts: list[str]):
    proc = subprocess.run([str(binary), "classify", "--model", str(model), "--top", "1"],
                          input="\n".join(texts) + "\n", text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"classify failed: {proc.stderr.strip()}")
    try:
        out = [json.loads(l) for l in proc.stdout.splitlines()]
    except json.JSONDecodeError:
        raise SystemExit(f"{binary} has no `classify` command; rebuild it with `cargo build --release` in rust/")
    if len(out) != len(texts):
        raise SystemExit("classify returned a different number of results than inputs")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=ROOT / "weights.zlm")
    ap.add_argument("--data", type=Path, required=True, help="labelled TSV: intent<TAB>text")
    ap.add_argument("--binary")
    ap.add_argument("--json", type=Path, help="write the full report here")
    ap.add_argument("--min-accuracy", type=float, help="exit 1 if accuracy is below this (0-1)")
    args = ap.parse_args()

    cases = read_cases(args.data)
    results = classify(find_binary(args.binary), args.model, [t for _, t in cases])

    labels = sorted({l for l, _ in cases} | {r["intent"] for r in results})
    tp, fp, fn = Counter(), Counter(), Counter()
    confusion = Counter()
    correct = 0
    for (want, text), r in zip(cases, results):
        got = r["intent"]
        if got == want:
            correct += 1
            tp[want] += 1
        else:
            fp[got] += 1
            fn[want] += 1
            confusion[(want, got)] += 1
    total = len(cases)
    per = {}
    for l in labels:
        p = tp[l] / (tp[l] + fp[l]) if tp[l] + fp[l] else None
        r_ = tp[l] / (tp[l] + fn[l]) if tp[l] + fn[l] else None
        f1 = 2 * p * r_ / (p + r_) if p and r_ else (0.0 if p is not None and r_ is not None else None)
        per[l] = {"support": tp[l] + fn[l], "precision": p, "recall": r_, "f1": f1}

    # Threshold sweep over the best raw score. Rows the model abstained on (score < 0.75) count
    # as `unknown`; raising t above 0.75 turns more low-score answers into `unknown`.
    sweep = []
    for t in (0.75, 1.5, 2.0, 2.5, 3.0, 3.5):
        acc_n = ok = 0
        for (want, _), r in zip(cases, results):
            got = r["intent"] if r["score"] >= t else "unknown"
            if got != "unknown":
                acc_n += 1
            ok += got == want
        sweep.append({"min_score": t, "accuracy": ok / total, "answered": acc_n / total})

    abstained = sum(r["abstained"] for r in results)
    report = {"model": str(args.model), "data": str(args.data), "cases": total, "accuracy": correct / total,
              "abstained": abstained / total, "per_intent": per, "sweep": sweep,
              "confusions": [{"expected": a, "got": b, "count": c} for (a, b), c in confusion.most_common(15)],
              "misses": [{"expected": w, "got": r["intent"], "score": r["score"], "text": t}
                         for (w, t), r in zip(cases, results) if r["intent"] != w][:50]}
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"{args.data}: {correct}/{total} correct = {correct / total:.1%}  (abstained on {abstained / total:.1%})\n")
    print(f"{'intent':<22}{'support':>8}{'precision':>11}{'recall':>9}{'f1':>7}")
    fmt = lambda v: "   -  " if v is None else f"{v:6.2f}"
    for l in labels:
        s = per[l]
        print(f"{l:<22}{s['support']:>8}{fmt(s['precision']):>11}{fmt(s['recall']):>9}{fmt(s['f1']):>7}")
    print("\nscore threshold sweep (accept an answer only if score >= t):")
    for s in sweep:
        print(f"  t={s['min_score']:<4}  accuracy {s['accuracy']:6.1%}   answered {s['answered']:6.1%}")
    if report["misses"]:
        print("\nmisses:")
        for m in report["misses"][:15]:
            print(f"  want {m['expected']:<12} got {m['got']:<12} score {m['score']:<5} {m['text']!r}")
    if args.min_accuracy is not None and correct / total < args.min_accuracy:
        print(f"\nFAIL: accuracy {correct / total:.1%} is below {args.min_accuracy:.1%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
