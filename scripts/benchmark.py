#!/usr/bin/env python3
"""Measure interactive inference throughput using repeated prompts."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / ("zerolm.exe" if sys.platform == "win32" else "zerolm")
PROMPTS = [
    "hello", "who are you", "what can you do", "explain the architecture",
    "show the weight file", "does it use training", "what are your limitations",
    "run the benchmark pipeline", "tell me about weight-space programming",
    "what is the weather in Paris",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1 or args.iterations > 100000:
        raise SystemExit("iterations must be between 1 and 100000")
    if not BINARY.exists():
        raise SystemExit("binary not found; run `make` first")
    inputs = [(PROMPTS[i % len(PROMPTS)]) for i in range(args.iterations)]
    payload = "\n".join(inputs + ["/quit", ""])
    started = time.perf_counter()
    proc = subprocess.run([str(BINARY)], input=payload, text=True, capture_output=True, cwd=ROOT)
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise SystemExit(proc.stderr or proc.stdout)
    replies = proc.stdout.count("bot> ")
    if replies != args.iterations:
        raise SystemExit(f"expected {args.iterations} replies, got {replies}")
    result = {
        "iterations": args.iterations,
        "elapsed_seconds": round(elapsed, 6),
        "mean_interaction_ms": round(elapsed * 1000 / args.iterations, 3),
        "interactions_per_second": round(args.iterations / max(elapsed, 1e-9), 2),
        "note": "Includes process startup and terminal I/O; compare runs on the same machine.",
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Iterations: {result['iterations']}")
        print(f"Elapsed: {result['elapsed_seconds']:.6f} s")
        print(f"Mean interaction: {result['mean_interaction_ms']:.3f} ms")
        print(f"Throughput: {result['interactions_per_second']:.2f} interactions/s")
        print(result['note'])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
