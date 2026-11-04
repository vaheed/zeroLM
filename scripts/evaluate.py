#!/usr/bin/env python3
"""Run deterministic intent and abstention evaluation for ZeroLM."""
from __future__ import annotations
import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / ("zerolm.exe" if sys.platform == "win32" else "zerolm")
REPORT_DIR = ROOT / "reports"


def read_cases(path: Path, expected: str | None = None):
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if expected is None:
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_no}: expected intent<TAB>prompt")
            label, prompt = parts
        else:
            label, prompt = expected, line.strip()
        rows.append((label, prompt))
    return rows


def ask(prompt: str):
    proc = subprocess.run([str(BINARY), "--verbose", prompt], text=True,
                          capture_output=True, check=False, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"model failed for {prompt!r}: {proc.stderr or proc.stdout}")
    match = re.search(r"\[intent=([a-z0-9_]+) score=([0-9.]+) emitted_tokens=(\d+)\]", proc.stdout)
    if not match:
        raise RuntimeError(f"could not parse model diagnostics for {prompt!r}: {proc.stdout}")
    response = proc.stdout[match.end():].strip()
    return match.group(1), float(match.group(2)), int(match.group(3)), response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    if not BINARY.exists():
        raise SystemExit("binary not found; run `make` first")

    regression = read_cases(ROOT / "tests" / "regression.tsv")
    unknown = read_cases(ROOT / "tests" / "unknown.tsv", "unknown")
    rows = []
    confusion = defaultdict(Counter)
    started = time.perf_counter()
    for expected, prompt in regression + unknown:
        actual, score, count, response = ask(prompt)
        ok = actual == expected and bool(response) and count > 0
        rows.append({"expected": expected, "actual": actual, "prompt": prompt,
                     "score": score, "tokens": count, "passed": ok,
                     "response": response})
        confusion[expected][actual] += 1
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    reg_rows = rows[:len(regression)]
    unk_rows = rows[len(regression):]
    reg_pass = sum(r["passed"] for r in reg_rows)
    unk_pass = sum(r["passed"] for r in unk_rows)
    total_pass = reg_pass + unk_pass
    report = {
        "project": "ZeroLM",
        "version": "0.2.0",
        "training": "none",
        "regression_cases": len(regression),
        "regression_passed": reg_pass,
        "regression_accuracy": reg_pass / max(1, len(regression)),
        "unknown_cases": len(unknown),
        "unknown_abstained": unk_pass,
        "unknown_recall": unk_pass / max(1, len(unknown)),
        "total_cases": len(rows),
        "total_passed": total_pass,
        "overall_accuracy": total_pass / max(1, len(rows)),
        "wall_time_ms": round(elapsed_ms, 3),
        "mean_case_ms": round(elapsed_ms / max(1, len(rows)), 3),
        "confusion": {k: dict(v) for k, v in sorted(confusion.items())},
        "failures": [r for r in rows if not r["passed"]],
    }
    print(f"Regression: {reg_pass}/{len(regression)} ({report['regression_accuracy']:.1%})")
    print(f"Unknown abstention: {unk_pass}/{len(unknown)} ({report['unknown_recall']:.1%})")
    print(f"Overall: {total_pass}/{len(rows)} ({report['overall_accuracy']:.1%})")
    print(f"Mean CLI case time: {report['mean_case_ms']:.3f} ms")
    for failure in report["failures"]:
        print(f"FAIL expected={failure['expected']} actual={failure['actual']} prompt={failure['prompt']!r}")

    if args.write_report:
        REPORT_DIR.mkdir(exist_ok=True)
        (REPORT_DIR / "latest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        md = ["# ZeroLM Evaluation Report", "", f"- Version: `{report['version']}`",
              f"- Regression: **{reg_pass}/{len(regression)}** ({report['regression_accuracy']:.1%})",
              f"- Unknown abstention: **{unk_pass}/{len(unknown)}** ({report['unknown_recall']:.1%})",
              f"- Overall: **{total_pass}/{len(rows)}** ({report['overall_accuracy']:.1%})",
              f"- Mean CLI case time: {report['mean_case_ms']:.3f} ms", "",
              "## Failures", ""]
        if report["failures"]:
            md.extend(f"- Expected `{r['expected']}`, got `{r['actual']}`: `{r['prompt']}`" for r in report["failures"])
        else:
            md.append("No failures.")
        (REPORT_DIR / "latest.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print("Wrote reports/latest.json and reports/latest.md")

    # The thresholds are explicit and intentionally strict for the included cases.
    return 0 if report["regression_accuracy"] >= 0.95 and report["unknown_recall"] >= 0.95 else 1


if __name__ == "__main__":
    raise SystemExit(main())
