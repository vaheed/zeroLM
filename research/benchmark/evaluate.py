#!/usr/bin/env python3
"""Evaluate systems on a ZeroLM Next case file and write machine-readable reports.

Systems: next (Rust network, optional --ablate), symbolic, keyword, lookup, legacy (C v0.2).
Metrics are computed here, in Python, from the gold fields written by generate.py.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from scenarios import canon_world, execute  # noqa: E402
from baselines import symbolic, simple  # noqa: E402

RUST = ROOT / "rust" / "target" / "release" / "zerolm"
MODEL = ROOT / "model" / "next" / "weights.zln"
LEGACY = ROOT / "zerolm"


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


# ------------------------------------------------------------------ systems
def run_next(cases, ablate=(), model=MODEL, summary=None):
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "in.jsonl"
        inp.write_text("\n".join(json.dumps({"id": c["id"], "text": c["text"], **({"world": c["world"]} if c.get("world") else {})}) for c in cases) + "\n")
        sfile = Path(td) / "summary.json"
        cmd = [str(RUST), "next", "--model", str(model), "--summary", str(sfile)]
        for a in ablate:
            cmd += ["--ablate", a]
        r = subprocess.run(cmd + ["batch", str(inp)], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"rust batch failed: {r.stderr}")
        out = {}
        for line in r.stdout.splitlines():
            d = json.loads(line)
            out[d["id"]] = d
        if summary is not None:
            summary.update(json.loads(sfile.read_text()))
    preds = []
    for c in cases:
        d = out[c["id"]]
        preds.append({"accepted": d["accepted"], "ops": d["ops"], "world_after": d.get("world_after"), "errors": d["errors"]})
    return preds


def run_py(fn, cases):
    return [dict(zip(("accepted", "ops"), fn(c["text"]))) for c in cases]


def run_legacy(cases):
    def one(c):
        r = subprocess.run([str(LEGACY), "--verbose", c["text"] or " "], capture_output=True, text=True)
        head = r.stdout.splitlines()[0] if r.stdout else ""
        intent = head.split("intent=")[1].split()[0] if "intent=" in head else "unknown"
        return {"accepted": intent != "unknown", "ops": [], "intent": intent}

    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, cases))


# ------------------------------------------------------------------ scoring
def pred_world(case, p):
    if p.get("world_after") is not None:
        return p["world_after"]
    if not p["accepted"] or not case.get("world"):
        return None
    try:
        w, _ = execute(case["world"], p["ops"])
        return canon_world(w)
    except Exception:
        return None


def score(cases, preds):
    S = [(c, p) for c, p in zip(cases, preds) if c["expect"] == "accept"]
    U = [(c, p) for c, p in zip(cases, preds) if c["expect"] == "reject"]
    m = {"cases": len(cases), "supported": len(S), "unsupported": len(U)}

    def frac(a, b):
        return round(a / b, 5) if b else None

    exact = sum(p["accepted"] and p["ops"] == c["gold_ops"] for c, p in S)
    perop_hit = perop_n = 0
    sem = 0
    for c, p in S:
        for i, g in enumerate(c["gold_ops"]):
            perop_n += 1
            perop_hit += p["accepted"] and i < len(p["ops"]) and p["ops"][i] == g
        sem += pred_world(c, p) == c.get("gold_world")
    m.update(exact_match=frac(exact, len(S)), per_operation_accuracy=frac(perop_hit, perop_n), semantic_accuracy=frac(sem, len(S)),
             false_rejection=frac(sum(not p["accepted"] for _, p in S), len(S)),
             false_acceptance=frac(sum(p["accepted"] for _, p in U), len(U)),
             correct_rejection=frac(sum(not p["accepted"] for _, p in U), len(U)),
             abstention_rate=frac(sum(not p["accepted"] for p in preds), len(preds)))
    # per category
    cat = collections.defaultdict(list)
    mixed = {c["category"] for c in cases if c["expect"] == "accept"} & {c["category"] for c in cases if c["expect"] == "reject"}
    for c, p in zip(cases, preds):
        cat[c["category"] + ("/reject" if c["category"] in mixed and c["expect"] == "reject" else "")].append((c, p))
    m["by_category"] = {}
    for k, items in sorted(cat.items()):
        if items[0][0]["expect"] == "accept":
            m["by_category"][k] = {"n": len(items), "exact_match": frac(sum(p["accepted"] and p["ops"] == c["gold_ops"] for c, p in items), len(items)),
                                   "semantic_accuracy": frac(sum(pred_world(c, p) == c["gold_world"] for c, p in items), len(items))}
        else:
            m["by_category"][k] = {"n": len(items), "false_acceptance": frac(sum(p["accepted"] for _, p in items), len(items))}
    # seen / unseen / length
    def sub(pred_fn):
        it = [(c, p) for c, p in S if pred_fn(c)]
        return {"n": len(it), "exact_match": frac(sum(p["accepted"] and p["ops"] == c["gold_ops"] for c, p in it), len(it))}

    comp = lambda c: c["category"] in ("single", "comp2", "comp3", "multi", "long", "binding", "tracking")
    if S and "seen_sequence" in S[0][0]:
        m["seen_sequence"] = sub(lambda c: comp(c) and c.get("seen_sequence"))
        m["unseen_sequence"] = sub(lambda c: comp(c) and not c.get("seen_sequence"))
        m["unseen_primitive_combination"] = sub(lambda c: comp(c) and c.get("unseen_ops", 0) > 0)
    m["compositional_all"] = sub(comp)
    m["longer_than_dev"] = sub(lambda c: c["n_ops"] > 3)
    m["multi_operation_execution"] = {"n": sum(1 for c, _ in S if c["n_ops"] >= 2),
                                      "semantic_accuracy": frac(sum(pred_world(c, p) == c["gold_world"] for c, p in S if c["n_ops"] >= 2), sum(1 for c, _ in S if c["n_ops"] >= 2))}
    prim = [(c, p) for c, p in S if c["category"] == "primitive"]
    m["primitive_interpretation"] = {"n": len(prim), "exact_match": frac(sum(p["accepted"] and p["ops"] == c["gold_ops"] for c, p in prim), len(prim))}
    # counterfactual pairs
    pairs = collections.defaultdict(dict)
    for c, p in zip(cases, preds):
        if c.get("pair"):
            pairs[c["pair"]["id"]][c["pair"]["role"]] = (c, p)
    strict = diffc = 0
    for v in pairs.values():
        (ca, pa), (cb, pb) = v["a"], v["b"]
        ok = pa["accepted"] and pb["accepted"] and pa["ops"] == ca["gold_ops"] and pb["ops"] == cb["gold_ops"]
        strict += ok
        gd = ca["gold_ops"] != cb["gold_ops"]
        diffc += ok or (pa["accepted"] and pb["accepted"] and (pa["ops"] != pb["ops"]) == gd and False)
    m["counterfactual_pairs"] = {"n": len(pairs), "consistency": frac(strict, len(pairs))}
    # order sensitivity
    groups = collections.defaultdict(list)
    for c, p in zip(cases, preds):
        if c.get("group"):
            groups[c["group"]].append((c, p))
    ok = 0
    for items in groups.values():
        ok += all(p["accepted"] and p["ops"] == c["gold_ops"] for c, p in items)
    m["order_groups"] = {"n": len(groups), "all_variants_correct": frac(ok, len(groups))}
    # executor agreement (Rust executor vs independent Python executor on the predicted ops)
    agree = n = 0
    for c, p in S:
        if p.get("world_after") is not None and c.get("world"):
            n += 1
            try:
                w, _ = execute(c["world"], p["ops"])
                agree += canon_world(w) == p["world_after"]
            except (KeyError, TypeError):  # malformed operations from an ablated network: counts as disagreement
                pass
    m["executor_agreement"] = {"n": n, "rate": frac(agree, n)} if n else None
    return m


def fmt(m):
    g = lambda k: m.get(k)
    lines = [f"cases={m['cases']} supported={m['supported']} unsupported={m['unsupported']}",
             f"exact_match={g('exact_match')} per_op={g('per_operation_accuracy')} semantic={g('semantic_accuracy')}",
             f"false_rejection={g('false_rejection')} false_acceptance={g('false_acceptance')} abstention={g('abstention_rate')}"]
    for k in ("primitive_interpretation", "compositional_all", "seen_sequence", "unseen_sequence", "unseen_primitive_combination", "longer_than_dev",
              "multi_operation_execution", "counterfactual_pairs", "order_groups"):
        if m.get(k):
            lines.append(f"{k}: {m[k]}")
    return "\n".join(lines)


def make_preds(system, cases, args, summary=None):
    if system == "next":
        return run_next(cases, args.ablate, Path(args.model), summary)
    if system == "symbolic":
        return run_py(symbolic.parse, cases)
    if system == "keyword":
        return run_py(simple.keyword, cases)
    if system == "lookup":
        return run_py(simple.Lookup(args.dev), cases)
    if system == "legacy":
        return run_legacy(cases)
    raise SystemExit(f"unknown system {system}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--system", default="next")
    ap.add_argument("--ablate", action="append", default=[])
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--dev", default=str(ROOT / "tests" / "next" / "dev.jsonl"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dump-failures", type=Path)
    ap.add_argument("--min", action="append", default=[], help="metric=threshold, e.g. exact_match=0.95 (fails the run)")
    args = ap.parse_args()
    cases = load(args.cases)
    summary = {}
    t0 = time.time()
    preds = make_preds(args.system, cases, args, summary)
    m = score(cases, preds)
    m.update(system=args.system, ablations=args.ablate, wall_s=round(time.time() - t0, 2), runtime=summary or None)
    print(fmt(m))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(m, indent=1) + "\n")
    if args.dump_failures:
        with args.dump_failures.open("w") as f:
            for c, p in zip(cases, preds):
                bad = (not p["accepted"] or p["ops"] != c["gold_ops"]) if c["expect"] == "accept" else p["accepted"]
                if bad:
                    f.write(json.dumps({"id": c["id"], "category": c["category"], "text": c["text"], "gold": c["gold_ops"], "pred": p["ops"],
                                        "accepted": p["accepted"], "errors": p.get("errors")}) + "\n")
    rc = 0
    for spec in args.min:
        k, v = spec.split("=")
        cur = m
        for part in k.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        if cur is None or cur < float(v):
            print(f"FAIL: {k}={cur} < {v}")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
