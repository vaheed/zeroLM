#!/usr/bin/env python3
"""Run documented ZeroLM CLI examples in one-shot and interactive modes."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / ("zerolm.exe" if sys.platform == "win32" else "zerolm")
DEFAULT_CASES = ROOT / "examples" / "prompts.tsv"
DIAGNOSTIC = re.compile(
    r"\[intent=([a-z0-9_]+) score=([0-9.]+) emitted_tokens=(\d+)\]"
)


def load_cases(path: Path) -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        fields = raw.split("\t", 1)
        if len(fields) != 2 or not fields[0].strip() or not fields[1].strip():
            raise ValueError(f"{path}:{line_no}: expected intent<TAB>prompt")
        cases.append((fields[0].strip(), fields[1].strip()))
    if not cases:
        raise ValueError(f"no examples found in {path}")
    return cases


def run_one_shot(expected: str, prompt: str) -> dict[str, object]:
    proc = subprocess.run(
        [str(BINARY), "--verbose", prompt],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output = proc.stdout.strip()
    match = DIAGNOSTIC.search(output)
    actual = match.group(1) if match else None
    response = output[match.end():].strip() if match else ""
    passed = proc.returncode == 0 and actual == expected and bool(response)
    return {
        "prompt": prompt,
        "expected_intent": expected,
        "actual_intent": actual,
        "passed": passed,
        "response": response,
        "stderr": proc.stderr.strip(),
        "returncode": proc.returncode,
    }


def run_interactive(cases: list[tuple[str, str]]) -> dict[str, object]:
    payload = "\n".join([prompt for _, prompt in cases] + ["/quit", ""])
    proc = subprocess.run(
        [str(BINARY)],
        cwd=ROOT,
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )
    reply_count = proc.stdout.count("bot> ")
    passed = proc.returncode == 0 and reply_count == len(cases)
    return {
        "passed": passed,
        "expected_replies": len(cases),
        "actual_replies": reply_count,
        "returncode": proc.returncode,
        "stderr": proc.stderr.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--report", type=Path, help="write a machine-readable JSON report")
    args = parser.parse_args()

    if not BINARY.is_file():
        raise SystemExit(f"binary not found: {BINARY}; run `make` first")

    cases = load_cases(args.cases)
    results = [run_one_shot(expected, prompt) for expected, prompt in cases]
    interactive = run_interactive(cases)
    passed = sum(bool(row["passed"]) for row in results)
    all_passed = passed == len(results) and bool(interactive["passed"])
    report = {
        "example_cases": len(results),
        "example_passed": passed,
        "example_failed": len(results) - passed,
        "interactive": interactive,
        "passed": all_passed,
        "cases": results,
    }

    print("ZeroLM documented examples")
    for row in results:
        status = "PASS" if row["passed"] else "FAIL"
        print(
            f"{status} expected={row['expected_intent']} "
            f"actual={row['actual_intent']} prompt={row['prompt']!r}"
        )
        if not row["passed"]:
            if row["response"]:
                print(f"  response: {row['response']}")
            if row["stderr"]:
                print(f"  stderr: {row['stderr']}")
    print(
        f"Interactive mode: {'PASS' if interactive['passed'] else 'FAIL'} "
        f"({interactive['actual_replies']}/{interactive['expected_replies']} replies)"
    )
    print(f"Examples: {passed}/{len(results)} passed")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.report}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
