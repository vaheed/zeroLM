#!/usr/bin/env python3
"""Benchmark the C runtime, the Rust port and ZeroLM Live; write reports/benchmarks.{json,md}.

Needs both binaries built:  make  &&  (cd rust && cargo build --release)

Everything here is wall-clock on the current machine, so only compare numbers
from the same run. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C_BIN = ROOT / ("zerolm.exe" if sys.platform == "win32" else "zerolm")
RUST_BIN = ROOT / "rust" / "target" / "release" / ("zerolm.exe" if sys.platform == "win32" else "zerolm")
MODEL = ROOT / "weights.zlm"
REPORT_DIR = ROOT / "reports"

PROMPTS = [
    "hello", "who are you", "what can you do", "explain the architecture",
    "show the weight file", "does it use training", "what are your limitations",
    "run the benchmark pipeline", "what license does it use", "how do I install ZeroLM",
    "what is the weather in Paris", "is it a transformer",
]


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def summarize(ms: list[float]) -> dict:
    return {
        "n": len(ms),
        "mean_ms": round(statistics.fmean(ms), 4),
        "p50_ms": round(pct(ms, 50), 4),
        "p95_ms": round(pct(ms, 95), 4),
        "p99_ms": round(pct(ms, 99), 4),
        "max_ms": round(max(ms), 4),
    }


def one_shot(binary: Path, n: int, load_only: bool = False) -> dict:
    """Spawn the binary n times: start-up + answer + exit (or start-up + --info if load_only)."""
    times = []
    for i in range(n):
        tail = ["--info"] if load_only else [PROMPTS[i % len(PROMPTS)]]
        t = time.perf_counter()
        p = subprocess.Popen([str(binary), "--model", str(MODEL), *tail],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if p.wait() != 0:
            raise SystemExit(f"{binary} failed")
        times.append((time.perf_counter() - t) * 1000)
    return summarize(times)


def vm_hwm_kib(pid: int) -> int:
    for row in Path(f"/proc/{pid}/status").read_text().splitlines():
        if row.startswith("VmHWM:"):
            return int(row.split()[1])
    return 0


def resident_memory(binary: Path, args: list[str]) -> int:
    """Peak RSS (KiB) of an interactive process after it has answered one prompt.

    Read from the process's own /proc entry while it is alive. (wait4's ru_maxrss
    is not usable: it inherits the benchmark parent's size across fork.)"""
    p = subprocess.Popen([str(binary), *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    assert p.stdin and p.stdout
    p.stdin.write("hello\n")
    p.stdin.flush()
    while "bot>" not in (p.stdout.readline() or "bot>"):
        pass
    hwm = vm_hwm_kib(p.pid)
    p.stdin.write("/quit\n")
    p.stdin.close()
    p.wait(timeout=10)
    return hwm


def batch(binary: Path, args: list[str], n: int, cwd: Path) -> dict:
    """Pipe n prompts through one process: steady-state speed without start-up."""
    payload = "\n".join(PROMPTS[i % len(PROMPTS)] for i in range(n)) + "\n/quit\n"
    t = time.perf_counter()
    p = subprocess.run([str(binary), *args], input=payload, text=True, capture_output=True, cwd=cwd)
    dt = time.perf_counter() - t
    if p.returncode != 0:
        raise SystemExit(p.stderr)
    return {"n": n, "seconds": round(dt, 4), "turns_per_second": round(n / dt, 1)}


def lockstep_live(life_dir: Path, tag: str, n: int, extra: list[str] | None = None,
                  prelude: list[str] | None = None, text_for=None) -> dict:
    """One live session in --jsonl mode: send a line, wait for its reply, repeat.

    Measures real per-turn round-trip latency including the life-file save."""
    cmd = [str(RUST_BIN), "live", "--jsonl", "--model", str(MODEL), "--tag", tag, *(extra or [])]
    p = subprocess.Popen(cmd, cwd=life_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    assert p.stdin and p.stdout

    last: dict = {}

    def turn(text: str) -> float:
        t = time.perf_counter()
        p.stdin.write(json.dumps({"text": text}) + "\n")
        p.stdin.flush()
        line = p.stdout.readline()
        dt = (time.perf_counter() - t) * 1000
        if not line.startswith("{"):
            raise SystemExit(f"bad reply: {line!r}")
        last.update(json.loads(line))
        return dt

    for line in prelude or []:
        turn(line)
    pick = text_for or (lambda i: PROMPTS[i % len(PROMPTS)])
    ms = [turn(pick(i)) for i in range(n)]
    hwm = vm_hwm_kib(p.pid)
    memories_at_end = last.get("memories", 0)
    p.stdin.close()
    p.wait(timeout=10)
    out = summarize(ms)
    out["turns_per_second"] = round(1000 / statistics.fmean(ms), 1)
    out["peak_rss_kib"] = hwm
    out["memories_at_end"] = memories_at_end
    life = life_dir / f"life.{tag}.zlm"
    out["life_file_bytes"] = life.stat().st_size if life.exists() else 0
    return out


def tool_version(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001 - informational only
        return "unknown"


def cpu_model() -> str:
    try:
        for row in Path("/proc/cpuinfo").read_text().splitlines():
            if row.startswith("model name"):
                return row.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def to_markdown(r: dict) -> str:
    env, o = r["environment"], r["one_shot"]
    L = [
        "# ZeroLM benchmark results",
        "",
        f"- Date: {env['date']}",
        f"- Machine: {env['cpu']} ({env['cpus']} logical CPUs), {env['kernel']}",
        f"- Tools: {env['rustc']}; {env['cc']}",
        f"- Model: {r['model']['intents']} intents, `weights.zlm` {r['model']['bytes']:,} bytes",
        "",
        "All numbers are wall-clock on this machine, in one run. Compare rows with each other, "
        "not with numbers from other machines.",
        "",
        "## 1. One-shot process (start-up + answer + exit)",
        "",
        f"{o['c']['n']} spawns per binary.",
        "",
        "| Runtime | mean ms | p50 ms | p95 ms | p99 ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, key in (("C, answer a prompt", "c"), ("Rust, answer a prompt", "rust"),
                      ("C, load model only (`--info`)", "c_load"), ("Rust, load model only (`--info`)", "rust_load")):
        v = o[key]
        L.append(f"| {name} | {v['mean_ms']} | {v['p50_ms']} | {v['p95_ms']} | {v['p99_ms']} |")
    b = r["batch"]
    L += [
        "",
        "## 2. Steady-state throughput (one process, prompts piped in, no start-up)",
        "",
        f"{b['c']['n']} prompts.",
        "",
        "| Runtime / mode | turns/s |",
        "| --- | ---: |",
        f"| C, interactive | {b['c']['turns_per_second']:,} |",
        f"| Rust, interactive, `--faithful` (same softmax math as C) | {b['rust_faithful']['turns_per_second']:,} |",
        f"| Rust, interactive (default pointer-walk decoder) | {b['rust']['turns_per_second']:,} |",
        "",
        "## 3. Resident memory (peak RSS of a running process after one answer)",
        "",
        "| Process | peak RSS KiB |",
        "| --- | ---: |",
        f"| C, interactive | {r['memory']['c']:,} |",
        f"| Rust, interactive | {r['memory']['rust']:,} |",
    ]
    L += [
        "",
        "## 4. ZeroLM Live round-trip latency (one request, wait for reply)",
        "",
        "`--jsonl` session; each turn also saves the life file atomically with an fsync.",
        "",
        "| Scenario | n | mean ms | p50 ms | p95 ms | p99 ms | turns/s | peak RSS KiB | facts at end | life file B |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, v in r["live"].items():
        L.append(f"| {label} | {v['n']} | {v['mean_ms']} | {v['p50_ms']} | {v['p95_ms']} | {v['p99_ms']} | "
                 f"{v['turns_per_second']:,} | {v['peak_rss_kib']} | {v['memories_at_end']} | {v['life_file_bytes']} |")
    L += ["", "## Reading these numbers", ""]
    L += [f"- {line}" for line in r["notes"]]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spawns", type=int, default=200, help="one-shot spawns per binary")
    ap.add_argument("--batch", type=int, default=5000, help="prompts for the throughput test")
    ap.add_argument("--live-turns", type=int, default=500, help="round trips per live scenario")
    ap.add_argument("--no-write", action="store_true", help="print only; do not write reports/")
    args = ap.parse_args()

    for b in (C_BIN, RUST_BIN):
        if not b.exists():
            raise SystemExit(f"missing {b}; build with `make` and `cargo build --release` in rust/")

    info = subprocess.run([str(C_BIN), "--info", "--model", str(MODEL)], capture_output=True, text=True).stdout
    intents = next(int(w.split("=")[1]) for w in info.split() if w.startswith("intents="))

    result: dict = {
        "environment": {
            "date": time.strftime("%Y-%m-%d"),
            "cpu": cpu_model(),
            "cpus": os.cpu_count(),
            "kernel": f"{platform.system()} {platform.release()}",
            "rustc": tool_version(["rustc", "--version"]),
            "cc": tool_version(["cc", "--version"]),
        },
        "model": {"intents": intents, "bytes": MODEL.stat().st_size},
    }

    print("1/3 one-shot spawns ...", file=sys.stderr)
    one_shot(C_BIN, 10), one_shot(RUST_BIN, 10)  # warm the page cache
    result["one_shot"] = {
        "c": one_shot(C_BIN, args.spawns), "rust": one_shot(RUST_BIN, args.spawns),
        "c_load": one_shot(C_BIN, args.spawns, load_only=True),
        "rust_load": one_shot(RUST_BIN, args.spawns, load_only=True),
    }

    print("2/3 steady-state throughput ...", file=sys.stderr)
    result["batch"] = {
        "c": batch(C_BIN, ["--model", str(MODEL)], args.batch, ROOT),
        "rust": batch(RUST_BIN, ["--model", str(MODEL)], args.batch, ROOT),
        "rust_faithful": batch(RUST_BIN, ["--faithful", "--model", str(MODEL)], args.batch, ROOT),
    }
    result["memory"] = {"c": resident_memory(C_BIN, ["--model", str(MODEL)]),
                        "rust": resident_memory(RUST_BIN, ["--model", str(MODEL)])}

    print("3/3 ZeroLM Live latency ...", file=sys.stderr)
    live: dict = {}
    facts = [f"remember fact{i} is value{i}" for i in range(64)]
    # Facts fade after ~100 unused turns, so the "full" runs recall them in rotation
    # (every fact is asked about every 64 turns) to keep all 64 alive throughout.
    recall = lambda i: f"what is fact{i % 64}"  # noqa: E731
    disk = Path(tempfile.mkdtemp(prefix="zerolm-bench-"))
    try:
        live["life file on disk, empty memory"] = lockstep_live(disk, "empty", args.live_turns)
        live["life file on disk, 64 facts, recalling them"] = lockstep_live(disk, "full", args.live_turns, prelude=facts, text_for=recall)
        live["life file on disk, tiny context budget (64 B)"] = lockstep_live(
            disk, "ctx", args.live_turns, extra=["--context", "64"])
    finally:
        shutil.rmtree(disk, ignore_errors=True)
    shm = Path("/dev/shm")
    if shm.is_dir() and os.access(shm, os.W_OK):
        ram = Path(tempfile.mkdtemp(prefix="zerolm-bench-", dir=shm))
        try:
            live["life file on tmpfs, 64 facts, recalling them"] = lockstep_live(
                ram, "full", args.live_turns, prelude=facts, text_for=recall)
        finally:
            shutil.rmtree(ram, ignore_errors=True)
    result["live"] = live

    for label, v in live.items():
        if "64 facts" in label and v["memories_at_end"] != 64:
            raise SystemExit(f"invalid run: '{label}' ended with {v['memories_at_end']} facts, expected 64")
    disk_ms = live["life file on disk, 64 facts, recalling them"]["mean_ms"]
    tmp = live.get("life file on tmpfs, 64 facts, recalling them")
    o, b = result["one_shot"], result["batch"]
    result["notes"] = [
        f"One-shot: C {o['c']['mean_ms']} ms vs Rust {o['rust']['mean_ms']} ms mean. Loading the model alone "
        f"(`--info`) takes C {o['c_load']['mean_ms']} ms and Rust {o['rust_load']['mean_ms']} ms, so start-up and "
        "model loading, not decoding, are what a one-shot call pays for.",
        f"Steady state: the C runtime and `rust --faithful` run the same softmax decoder "
        f"({b['c']['turns_per_second']:,} vs {b['rust_faithful']['turns_per_second']:,} turns/s). The default Rust "
        f"decoder walks successor pointers instead, which is mathematically identical here "
        f"(output is diffed against C in CI) and gives {b['rust']['turns_per_second']:,} turns/s.",
        f"ZeroLM Live: mean {disk_ms} ms per turn with the life file on disk. "
        + (f"On tmpfs it is {tmp['mean_ms']} ms, so nearly all of the disk figure is the per-turn fsync of the "
           "life file, not the model or the memory lookup." if tmp else ""),
        "Memory size barely matters: compare the empty and 64-fact rows. Facts are kept alive in the 64-fact rows "
        "by asking about each in rotation (unused facts fade after about 100 turns); the run aborts if fewer than "
        "64 remain at the end.",
        "Latency rows are lock-step round trips (send one line, wait for its reply), not pipelined throughput.",
    ]

    md = to_markdown(result)
    if not args.no_write:
        REPORT_DIR.mkdir(exist_ok=True)
        (REPORT_DIR / "benchmarks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (REPORT_DIR / "benchmarks.md").write_text(md, encoding="utf-8")
        print(f"wrote {REPORT_DIR/'benchmarks.json'} and {REPORT_DIR/'benchmarks.md'}", file=sys.stderr)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
