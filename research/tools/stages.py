#!/usr/bin/env python3
"""Evaluate every construction stage (base + patches 001..N) on one case file -> reports/next/patch_stages.json.

Shows what each manual patch changed. Stage 0 is the base construction; it is rebuilt from the same builder
with `--upto`, so nothing is stored by hand.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "benchmark"))
import evaluate as EV  # noqa: E402

cases_path = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "tests/next/final.jsonl")
cases = EV.load(cases_path)
adv = EV.load(ROOT / "tests/next/adversarial.jsonl")
n_patches = len(list((ROOT / "model/next/patches").glob("*.json")))
out = []
with tempfile.TemporaryDirectory() as td:
    for upto in range(n_patches + 1):
        w = Path(td) / f"w{upto}.zln"
        subprocess.run([sys.executable, str(ROOT / "research/build_next.py"), "--upto", str(upto), "--out", str(w)], check=True, capture_output=True)
        row = {"stage": upto, "patches": upto}
        for name, cs in ((cases_path.name, cases), ("adversarial.jsonl", adv)):
            m = EV.score(cs, EV.run_next(cs, (), w))
            row[name] = {k: m.get(k) for k in ("exact_match", "semantic_accuracy", "false_acceptance", "false_rejection")}
            if name != "adversarial.jsonl":
                row[name]["compositional_exact_match"] = m["compositional_all"]["exact_match"]
                row[name]["by_category_exact"] = {k: v.get("exact_match") for k, v in m["by_category"].items() if "exact_match" in v}
                row[name]["ambiguous_false_acceptance"] = m["by_category"].get("ambiguous", {}).get("false_acceptance")
        out.append(row)
        print(upto, row[cases_path.name]["exact_match"], row[cases_path.name]["compositional_exact_match"], row["adversarial.jsonl"]["exact_match"], row["adversarial.jsonl"]["false_acceptance"])
(ROOT / "reports/next/patch_stages.json").write_text(json.dumps({"cases": cases_path.name, "stages": out}, indent=1) + "\n")
