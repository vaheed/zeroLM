#!/usr/bin/env python3
"""Write or verify tests/next/heldout_manifest.json: seeds, counts and sha256 of the generated holdout files.

  manifest.py check     regenerate each listed file into a temp dir and compare sha256 (CI)
  manifest.py write     recompute (used only when the generator or grammar changes on purpose)
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAN = ROOT / "tests" / "next" / "heldout_manifest.json"
GEN = ROOT / "research" / "benchmark" / "generate.py"
DEV = ROOT / "tests" / "next" / "dev.jsonl"


def gen(profile, seed, out):
    subprocess.run([sys.executable, str(GEN), "--profile", profile, "--seed", str(seed), "--dev", str(DEV), "--out", str(out)], check=True, capture_output=True)
    return hashlib.sha256(Path(out).read_bytes()).hexdigest(), len(Path(out).read_text().splitlines())


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    man = json.loads(MAN.read_text())
    bad = 0
    with tempfile.TemporaryDirectory() as td:
        for entry in man["files"]:
            sha, n = gen(entry["profile"], entry["seed"], Path(td) / "x.jsonl")
            if mode == "write":
                entry["sha256"], entry["cases"] = sha, n
            elif sha != entry["sha256"]:
                print(f"MISMATCH seed {entry['seed']}: {sha} != {entry['sha256']}")
                bad = 1
            else:
                print(f"ok seed {entry['seed']} ({n} cases) sha256 {sha[:16]}")
    man["dev_sha256"] = hashlib.sha256(DEV.read_bytes()).hexdigest()
    man["generator_sha256"] = hashlib.sha256(GEN.read_bytes()).hexdigest() if mode == "write" else man.get("generator_sha256")
    if mode == "write":
        MAN.write_text(json.dumps(man, indent=1) + "\n")
    return bad


sys.exit(main())
