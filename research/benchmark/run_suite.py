#!/usr/bin/env python3
"""Run the full ZeroLM Next experiment suite and write reports/next/*.json and SUMMARY.md.

  baseline.json    ZeroLM Next vs legacy v0.2, symbolic parser, keyword classifier, phrase lookup, random weights
  composition.json ZeroLM Next metrics by category / seen-unseen / length, runtime, environment
  ablation.json    causal ablations of the designed network
  scaling.json     length scaling, latency and memory
Exit status is non-zero if an acceptance floor is missed (see FLOORS).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import evaluate as EV  # noqa: E402
import generate as GEN  # noqa: E402
from scenarios import Rng  # noqa: E402

KEYS = ("cases", "supported", "unsupported", "exact_match", "per_operation_accuracy", "semantic_accuracy", "false_rejection",
        "false_acceptance", "abstention_rate")
# Acceptance floors from the research brief (kept at the brief's values, never lowered).
FLOORS = {"primitive_interpretation.exact_match": 0.99, "unseen_sequence.exact_match": 0.95, "multi_operation_execution.semantic_accuracy": 0.90,
          "counterfactual_pairs.consistency": 0.95, "correct_rejection": 0.99}


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except OSError:
        return "unavailable"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def environment(cases_path, seed):
    return {"date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "platform": platform.platform(), "python": platform.python_version(),
            "rustc": sh(["rustc", "--version"]), "cc": sh(["cc", "--version"]).splitlines()[0] if sh(["cc", "--version"]) else None,
            "git_commit": sh(["git", "rev-parse", "HEAD"]), "git_dirty": bool(sh(["git", "status", "--porcelain"])),
            "benchmark_seed": seed, "cases_file": str(Path(cases_path).name), "cases_sha256": sha256(cases_path),
            "weights_sha256": sha256(EV.MODEL), "generator_sha256": sha256(HERE / "generate.py"), "grammar_sha256": sha256(HERE / "grammar.json")}


def brief(m):
    out = {k: m.get(k) for k in KEYS}
    for k in ("primitive_interpretation", "compositional_all", "seen_sequence", "unseen_sequence", "unseen_primitive_combination", "longer_than_dev",
              "multi_operation_execution", "counterfactual_pairs", "order_groups", "executor_agreement"):
        out[k] = m.get(k)
    out["by_category"] = m["by_category"]
    return out


def run(system, cases, ablate=(), dev=None, model=None):
    class A:
        pass
    a = A()
    a.ablate, a.model, a.dev = list(ablate), str(model or EV.MODEL), dev
    summary = {}
    preds = EV.make_preds(system, cases, a, summary)
    m = EV.score(cases, preds)
    m["runtime"] = summary or None
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, default=ROOT / "tests/next/final.jsonl")
    ap.add_argument("--seed", type=int, default=20260701)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/next")
    ap.add_argument("--dev", default=str(ROOT / "tests/next/dev.jsonl"))
    ap.add_argument("--skip-legacy", action="store_true")
    ap.add_argument("--random-seeds", type=int, default=5)
    ap.add_argument("--parts", default="composition,baseline,ablation,scaling", help="comma list; skipped parts are loaded from existing reports")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cases = EV.load(a.cases)
    env = environment(a.cases, a.seed)
    t0 = time.time()

    parts = set(a.parts.split(","))
    load = lambda n: json.loads((a.out / n).read_text())
    # ---- composition (always recomputed: it is cheap and every other part needs the reference numbers)
    nxt = run("next", cases)
    adv = run("next", EV.load(ROOT / "tests/next/adversarial.jsonl"))
    comp = {"environment": env, "system": "ZeroLM Next", "metrics": brief(nxt), "adversarial_handwritten": brief(adv), "runtime": nxt["runtime"]}
    (a.out / "composition.json").write_text(json.dumps(comp, indent=1) + "\n")

    # ---- baselines
    if "baseline" not in parts:
        base = load("baseline.json")
    else:
        base = {"environment": env, "systems": {"zerolm_next": brief(nxt)}}
        for name in ("symbolic", "keyword", "lookup"):
            base["systems"][name] = brief(run(name, cases, dev=a.dev))
        if not a.skip_legacy:
            legacy = brief(run("legacy", cases))
            legacy["note"] = "v0.2 is an intent classifier: it cannot emit operations, so exact match is 0 by construction; rejection behaviour is the only comparable signal."
            base["systems"]["legacy_v0.2_c"] = legacy
        rand = []
        for s in range(1, a.random_seeds + 1):
            m = run("next", cases, [f"random:{s}"])
            rand.append({"seed": s, **{k: m.get(k) for k in ("exact_match", "semantic_accuracy", "false_acceptance", "abstention_rate")}})
        base["random_weights_same_architecture"] = {"runs": rand, "max_exact_match": max(r["exact_match"] or 0 for r in rand)}
        shuf = []
        for s in range(1, a.random_seeds + 1):
            m = run("next", cases, [f"shuffle:{s}"])
            shuf.append({"seed": s, **{k: m.get(k) for k in ("exact_match", "semantic_accuracy", "false_acceptance", "abstention_rate")}})
        base["shuffled_weights_same_values"] = {"runs": shuf, "max_exact_match": max(r["exact_match"] or 0 for r in shuf)}
        (a.out / "baseline.json").write_text(json.dumps(base, indent=1) + "\n")

    # ---- ablations
    if "ablation" not in parts:
        abl = load("ablation.json")
    else:
        abl = {"environment": env, "reference": {k: nxt.get(k) for k in ("exact_match", "semantic_accuracy", "false_acceptance")}, "runs": []}
        specs = [["recurrence"], ["state_read"], ["binding"], ["pointer_shift"], ["time_order"], ["zero:A_w"], ["zero:A_o"], ["zero:B_w"], ["zero:B_o"],
                 ["zero:S"], ["zero:E"], ["zero:V_w"], ["zero:V_o"], ["zero:B_e"], ["zero:ex_w"], ["zero:tf_w"], ["zero:rank_eps"], ["zero:Gm_w"], ["zero:Pe_w"],
                 ["permute:E:1"], ["permute:B_o:1"], ["permute:A_w:1"], ["permute:B_w:1"], ["permute:V_w:1"], ["permute:dec.1.w:1"], ["permute:S:1"],
                 ["shuffle:1"], ["random:1"]]
        for sp in specs:
            m = run("next", cases, sp)
            abl["runs"].append({"ablation": sp[0], **{k: m.get(k) for k in ("exact_match", "semantic_accuracy", "false_acceptance", "false_rejection", "abstention_rate")},
                                "drop_exact_match": round((nxt["exact_match"] or 0) - (m["exact_match"] or 0), 5)})
        (a.out / "ablation.json").write_text(json.dumps(abl, indent=1) + "\n")

    # ---- scaling
    if "scaling" not in parts:
        sc = load("scaling.json")
    else:
        sc = {"environment": env, "rows_capacity": 24, "points": []}
        prof = GEN.Profile("final")
        rng = Rng(424242)
        for n in (1, 2, 4, 8, 12, 16, 20, 22, 23, 24, 25, 28):
            cs = []
            for i in range(300):
                b = GEN.build(rng, prof, n, structure=[("seq", [k]) for k in range(n)])
                cs.append(GEN.make_case(f"sc{n}-{i}", "final", "scaling", b))
            m = run("next", cs)
            rt = m["runtime"] or {}
            sc["points"].append({"operations": n, "cases": len(cs), "exact_match": m["exact_match"], "false_rejection": m["false_rejection"],
                                 "parse_us_mean": rt.get("parse_us_mean"), "parse_us_p99": rt.get("parse_us_p99"), "peak_rss_kb": rt.get("peak_rss_kb")})
        sc["weights_bytes"] = (nxt["runtime"] or {}).get("weights_bytes")
        (a.out / "scaling.json").write_text(json.dumps(sc, indent=1) + "\n")

    # ---- floors + summary
    fails = []
    for k, floor in FLOORS.items():
        cur = nxt
        for part in k.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        if cur is None or cur < floor:
            fails.append(f"{k}={cur} < {floor}")
    if adv["exact_match"] != 1.0 or adv["false_acceptance"] != 0.0:
        fails.append(f"adversarial suite: exact_match={adv['exact_match']} false_acceptance={adv['false_acceptance']}")
    ablation_dead = [r["ablation"] for r in abl["runs"] if r["ablation"] in ("recurrence", "state_read", "binding", "pointer_shift", "time_order", "zero:B_w", "zero:B_o", "zero:A_w", "zero:E", "random:1", "shuffle:1")
                     and r["drop_exact_match"] < 0.05]
    if ablation_dead:
        fails.append(f"ablations with no measurable effect (possible bypass): {ablation_dead}")
    if base["random_weights_same_architecture"]["max_exact_match"] > 0.5:
        fails.append("random weights score > 0.5")
    md = ["# ZeroLM Next - experiment summary", "", f"cases `{a.cases.name}` seed {a.seed}, {len(cases)} cases, commit `{env['git_commit'][:10]}`, weights sha256 `{env['weights_sha256'][:16]}`", "",
          "| system | exact match | semantic acc. | false acceptance | abstention |", "|---|---:|---:|---:|---:|"]
    for k, v in base["systems"].items():
        md.append(f"| {k} | {v['exact_match']} | {v['semantic_accuracy']} | {v['false_acceptance']} | {v['abstention_rate']} |")
    md.append(f"| random weights (max of {a.random_seeds} seeds) | {base['random_weights_same_architecture']['max_exact_match']} | | | |")
    md.append(f"| shuffled weights (max of {a.random_seeds} seeds) | {base['shuffled_weights_same_values']['max_exact_match']} | | | |")
    md += ["", "| ablation | exact match | drop |", "|---|---:|---:|"] + [f"| {r['ablation']} | {r['exact_match']} | {r['drop_exact_match']} |" for r in abl["runs"]]
    md += ["", "| ops | exact match | mean parse us | p99 us |", "|---:|---:|---:|---:|"] + [f"| {p['operations']} | {p['exact_match']} | {p['parse_us_mean']} | {p['parse_us_p99']} |" for p in sc["points"]]
    md += ["", "**Acceptance floors:** " + ("all met" if not fails else "FAILED - " + "; ".join(fails))]
    (a.out / "SUMMARY.md").write_text("\n".join(md) + "\n")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nsuite wall time {time.time() - t0:.1f}s")
    for f in fails:
        print("FAIL:", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
