#!/usr/bin/env python3
"""Evaluate ZeroLM on the public CLINC150 benchmark (Larson et al., 2019).

Downloads the 'small' CLINC150 split and reports:
  * in-scope accuracy - CLINC intents with a ZeroLM equivalent (see MAPPING)
  * abstention rate   - out-of-scope and unsupported-intent queries that
                        ZeroLM correctly answers with `unknown`
ZeroLM is a tiny hand-wired model, so this measures honest generalisation,
not a leaderboard score. Network failures exit 0 with a skip note unless
--require-network is given.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import ask, BINARY, ROOT  # noqa: E402

URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_small.json"
CACHE = ROOT / "reports" / "clinc150_small.json"
MAPPING = {
    "greeting": "greeting", "goodbye": "farewell", "thank_you": "thanks",
    "what_can_i_ask_you": "capabilities", "who_made_you": "identity",
    "what_is_your_name": "identity", "are_you_a_bot": "identity",
}


def load():
    try:
        with urllib.request.urlopen(URL, timeout=30) as r:
            raw = r.read()
        CACHE.write_bytes(raw)
    except Exception as exc:  # noqa: BLE001
        if CACHE.exists():
            raw = CACHE.read_bytes()
        else:
            raise RuntimeError(f"cannot download CLINC150: {exc}") from exc
    return json.loads(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=600, help="max sampled queries per group")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--require-network", action="store_true")
    ap.add_argument("--min-abstain", type=float, default=0.0)
    ap.add_argument("--min-accuracy", type=float, default=0.0)
    a = ap.parse_args()
    if not BINARY.exists():
        raise SystemExit("binary not found; run `make` first")
    try:
        data = load()
    except RuntimeError as exc:
        print(f"SKIPPED: {exc}")
        raise SystemExit(1 if a.require_network else 0)

    rng = random.Random(a.seed)
    test = data["test"] + data["val"]
    groups = {
        "in_scope": [(q, MAPPING[lab]) for q, lab in test if lab in MAPPING],
        "unsupported": [(q, "unknown") for q, lab in test if lab not in MAPPING],
        "out_of_scope": [(q, "unknown") for q, _ in data["oos_test"]],
    }
    res, errors = {}, []
    for name, rows in groups.items():
        rng.shuffle(rows)
        rows = rows[: a.limit]
        ok = 0
        for q, exp in rows:
            got = ask(q)[0]
            if got == exp:
                ok += 1
            elif len(errors) < 15:
                errors.append({"group": name, "query": q, "expected": exp, "got": got})
        res[name] = {"n": len(rows), "correct": ok, "rate": round(ok / max(1, len(rows)), 4)}
    abst_n = res["unsupported"]["n"] + res["out_of_scope"]["n"]
    abst_ok = res["unsupported"]["correct"] + res["out_of_scope"]["correct"]
    summary = {"dataset": "CLINC150-small", "groups": res,
               "in_scope_accuracy": res["in_scope"]["rate"],
               "abstention_rate": round(abst_ok / max(1, abst_n), 4),
               "sample_errors": errors}
    if a.json:
        print(json.dumps(summary, indent=2))
    else:
        print("CLINC150 online benchmark")
        for k, v in res.items():
            print(f"  {k:13s} {v['correct']}/{v['n']} = {v['rate']:.1%}")
        print(f"  in-scope accuracy : {summary['in_scope_accuracy']:.1%}")
        print(f"  abstention rate   : {summary['abstention_rate']:.1%}")
        for e in errors[:8]:
            print(f"  miss [{e['group']}] {e['query']!r}: expected {e['expected']}, got {e['got']}")
    if summary["abstention_rate"] < a.min_abstain or summary["in_scope_accuracy"] < a.min_accuracy:
        raise SystemExit("online benchmark below threshold")


if __name__ == "__main__":
    main()
