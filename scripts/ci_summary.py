#!/usr/bin/env python3
"""Create a concise GitHub Actions job summary from ZeroLM CI reports."""
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def read_json(name: str) -> dict[str, Any]:
    path = REPORTS / name
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def outcome(value: str) -> str:
    value = (value or "not run").strip().lower()
    if value in {"success", "skipped"}:
        return f"✅ {value}"
    if value in {"failure", "cancelled", "timed_out"}:
        return f"❌ {value}"
    return f"⚪ {value or 'not run'}"


def fmt_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def fmt_number(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def build_summary() -> str:
    evaluation = read_json("latest.json")
    audit = read_json("weight-audit.json")
    benchmark = read_json("benchmark.json")
    examples = read_json("examples.json")

    repo = os.getenv("GITHUB_REPOSITORY", "local checkout")
    sha = os.getenv("GITHUB_SHA", "")[:12] or "local"
    ref = os.getenv("GITHUB_REF_NAME", "local")
    run_id = os.getenv("GITHUB_RUN_ID", "")
    run_url = ""
    if run_id and os.getenv("GITHUB_SERVER_URL"):
        run_url = f"{os.environ['GITHUB_SERVER_URL']}/{repo}/actions/runs/{run_id}"

    job_status = os.getenv("JOB_STATUS", "unknown")
    pipeline_status = outcome(os.getenv("PIPELINE_OUTCOME", "not run"))
    examples_status = outcome(os.getenv("EXAMPLES_OUTCOME", "not run"))
    memory_status = outcome(os.getenv("MEMORY_OUTCOME", "not run"))

    lines = [
        "# ZeroLM CI — execution report",
        "",
        f"**Job result:** {outcome(job_status)}  ",
        f"**Repository:** `{repo}`  ",
        f"**Commit:** `{sha}`  ",
        f"**Ref:** `{ref}`  ",
        f"**Runner:** `{os.getenv('RUNNER_OS', platform.system())}` / `{platform.machine()}`  ",
        f"**Python:** `{platform.python_version()}`  ",
    ]
    if run_url:
        lines.append(f"**Workflow run:** [Open run]({run_url})  ")

    lines.extend([
        "",
        "## Job stages",
        "",
        "| Stage | Result |",
        "|---|---|",
        f"| Full build, smoke tests, regression evaluation, weight audit and benchmark | {pipeline_status} |",
        f"| Documented one-shot and interactive examples | {examples_status} |",
        f"| Linux 2 GiB virtual-memory-limit smoke test | {memory_status} |",
        "",
        "## Model and quality results",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Regression cases | {evaluation.get('regression_passed', 'n/a')} / {evaluation.get('regression_cases', 'n/a')} |",
        f"| Regression accuracy on included cases | {fmt_percent(evaluation.get('regression_accuracy'))} |",
        f"| Unknown-input abstention | {evaluation.get('unknown_abstained', 'n/a')} / {evaluation.get('unknown_cases', 'n/a')} |",
        f"| Unknown-input recall | {fmt_percent(evaluation.get('unknown_recall'))} |",
        f"| Overall included-case score | {evaluation.get('total_passed', 'n/a')} / {evaluation.get('total_cases', 'n/a')} ({fmt_percent(evaluation.get('overall_accuracy'))}) |",
        f"| Demo examples | {examples.get('example_passed', 'n/a')} / {examples.get('example_cases', 'n/a')} |",
        f"| Interactive demo replies | {examples.get('interactive', {}).get('actual_replies', 'n/a')} / {examples.get('interactive', {}).get('expected_replies', 'n/a')} |",
        "",
        "## Model structure",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Intents | {audit.get('intents', 'n/a')} |",
        f"| Input features | {audit.get('features', 'n/a')} |",
        f"| Recurrent states | {audit.get('states', 'n/a')} |",
        f"| Output vocabulary | {audit.get('vocabulary', 'n/a')} |",
        f"| Dense-equivalent weight slots | {audit.get('dense_equivalent_slots', 'n/a')} |",
        f"| Serialized weight file | {audit.get('weight_file_bytes', 'n/a')} bytes |",
        "",
        "## Runtime benchmark",
        "",
        "| Measurement | Result |",
        "|---|---:|",
        f"| Iterations | {benchmark.get('iterations', 'n/a')} |",
        f"| Total elapsed | {fmt_number(benchmark.get('elapsed_seconds'), 4)} s |",
        f"| Mean interaction | {fmt_number(benchmark.get('mean_interaction_ms'), 3)} ms |",
        f"| Throughput | {fmt_number(benchmark.get('interactions_per_second'), 2)} interactions/s |",
        "",
        "## Artifacts",
        "",
        "The `zerolm-ci-reports-*` workflow artifact contains available build logs, smoke-test output, evaluation JSON/Markdown, example results, weight audit, benchmark data, and environment details.",
        "",
        "## Interpretation and limits",
        "",
        "- These scores describe only the finite, hand-authored prompts included in this repository. They are not a general language-understanding benchmark.",
        "- The evaluator checks expected intent selection and non-empty output; human review is still required to judge semantic correctness and factual quality.",
        "- The benchmark includes process startup and terminal I/O. Compare results only on equivalent runners and toolchains.",
        "- The Linux memory check limits virtual address space to 2 GiB; it is not a measurement or guarantee of physical RAM consumption.",
        "- No training, fine-tuning, distillation, or gradient-based weight optimization is performed by these CI steps.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    content = build_summary()
    report_path = REPORTS / "ci-summary.md"
    report_path.write_text(content, encoding="utf-8")

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as stream:
            stream.write(content)
    else:
        print(content)
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
