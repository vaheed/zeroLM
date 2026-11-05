#!/usr/bin/env python3
"""Weight-space programming workflow for ZeroLM Next (no optimizer, no search).

  wsp.py inspect [--unit NAME]          architecture / unit weights
  wsp.py diagnose --text T [--gold F]   failed-inference trace: which operation wrote or advanced wrongly
  wsp.py validate PATCH.json            format, names and tensor dimensions (dry-run build + Rust load)
  wsp.py apply PATCH.json               backup, apply, evaluate before/after on targets and regression
                                        sets, record the observed results, auto-revert on regression
  wsp.py revert                         restore the most recent backup
  wsp.py compare A.zln B.zln --cases F  before/after metrics for two weight files

A patch is a human-reasoned JSON document (see experiments/next-patches/). Applying it appends it to
model/next/patches/, from which research/build_next.py regenerates weights.zln deterministically.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "benchmark"))
import build_next as B  # noqa: E402
import evaluate as EV  # noqa: E402

WEIGHTS = B.OUT
PATCHES = B.PATCH_DIR
BACKUPS = ROOT / "model" / "next" / "backups"
RUST = EV.RUST
REQUIRED = ["id", "title", "hypothesis", "affected", "mathematical_effect", "expected", "targets", "regression", "edits"]


def spec():
    return json.loads(B.SPEC.read_text())


def cmd_inspect(a):
    M = B.construct(spec())
    for name, p in B.load_patches(PATCHES):
        B.apply_patch(M, p, name)
    t, lists, dec, dims = B.densify(M)
    print(f"dims {dims}; features {len(M.features)}; fields {len(M.fields)}")
    for layer in "ABV":
        units = getattr(M, layer)
        print(f"layer {layer}: {len(units)} units")
        for u in units:
            if a.unit and u.name != a.unit:
                continue
            print(f"  {u.name:34s} bias={u.bias:+g} inputs={len(u.w)}" + (f" out={u.out}" if u.out else "") + (" [error]" if u.err else "")
                  + (f" adv={u.adv:g} rev={u.rev:g}" if layer == "A" else ""))
            if a.unit:
                for k, v in sorted(u.w.items()):
                    print(f"      {k:20s} {v:+g}")
    nz = sum(1 for m in t.values() for row in m for v in (row if isinstance(row, list) else [row]) if v)
    print(f"tensors: {len(t)}; non-zero weights: {nz}")


def trace(text, model):
    r = subprocess.run([str(RUST), "next", "--model", str(model), "trace", text], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stderr)
    return json.loads(r.stdout)


def cmd_diagnose(a):
    tr = trace(a.text, a.model)
    print(f"accepted={tr['accepted']} error_total={tr['error_total']} errors={list(tr['error_units'])}")
    gold = json.loads(Path(a.gold).read_text()) if a.gold else None
    pred = tr["ops"]
    if gold is not None:
        print(f"gold ops: {len(gold)}  predicted ops: {len(pred)}")
        for i, (g, p) in enumerate(zip(gold, pred)):
            for k in sorted(set(g) | set(p)):
                if g.get(k) != p.get(k):
                    print(f"  op[{i}].{k}: gold={g.get(k)} predicted={p.get(k)}")
    print("row boundaries (pointer advances) and the units that fired:")
    for i, s in enumerate(tr["steps"]):
        if s["advance"]:
            units = {k: v for k, v in s["boundary_units"].items()}
            print(f"  step {i} token {s['token']!r}: advance={s['advance']} reverse={s['reverse']} via {units} -> row {s['pointer_row']}")
    print("writes per token (row, fields written):")
    for i, s in enumerate(tr["steps"]):
        w = {k: v for k, v in s["write_vector"].items() if k not in ("TIME",)}
        if w:
            print(f"  step {i} {s['token']!r:14s} row {s['pointer_row']:>2}  {w}  via {list(s['write_units'])}")
    for r in tr["rows"]:
        if r["validity_errors"]:
            print(f"  row {r['row']} validity errors: {r['validity_errors']}")


def build_to(path, extra=(), upto=None):
    t, lists, dec, dims, applied = B.build(spec(), PATCHES, upto, extra)
    path.write_text(B.serialize(spec(), t, lists, dec, dims))
    return applied


def load_patch(path):
    p = json.loads(Path(path).read_text())
    missing = [k for k in REQUIRED if k not in p]
    if missing:
        raise SystemExit(f"patch lacks required keys: {missing}")
    if not isinstance(p["edits"], list) or not p["edits"]:
        raise SystemExit("patch needs a non-empty edits list")
    return p


def cmd_validate(a):
    p = load_patch(a.patch)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "w.zln"
        build_to(out, [(Path(a.patch).name, p)], upto=int(p["id"]) - 1)
        r = subprocess.run([str(RUST), "next", "--model", str(out), "info"], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"Rust rejects the resulting weights: {r.stderr}")
        print(r.stdout.strip())
    print(f"OK: patch {p['id']} is well-formed, names resolve, tensor shapes are consistent")


def metrics(weights, files):
    out = {}
    for f in files:
        cases = EV.load(ROOT / f)
        m = EV.score(cases, EV.run_next(cases, (), Path(weights)))
        out[f] = {k: m.get(k) for k in ("cases", "supported", "unsupported", "exact_match", "semantic_accuracy", "false_rejection", "false_acceptance")}
    return out


def cmd_apply(a):
    p = load_patch(a.patch)
    cmd_validate(a)
    BACKUPS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bdir = BACKUPS / f"{stamp}-before-{p['id']}"
    bdir.mkdir()
    shutil.copy(WEIGHTS, bdir / "weights.zln")
    if PATCHES.is_dir():
        shutil.copytree(PATCHES, bdir / "patches")
    PATCHES.mkdir(parents=True, exist_ok=True)
    dest = PATCHES / Path(a.patch).name
    if dest.exists():
        raise SystemExit(f"{dest} already applied")
    files = list(dict.fromkeys(p["targets"] + p["regression"]))
    before = metrics(bdir / "weights.zln", files)
    shutil.copy(a.patch, dest)
    build_to(WEIGHTS)
    after = metrics(WEIGHTS, files)
    regress = []
    for f in p["regression"]:
        b, c = before[f], after[f]
        if (c["exact_match"] or 0.0) < (b["exact_match"] or 0.0) - 1e-12 or (c["false_acceptance"] or 0.0) > (b["false_acceptance"] or 0.0) + 1e-12:
            regress.append(f)
    def better(b, c):
        if c["supported"] and (c["exact_match"] or 0.0) > (b["exact_match"] or 0.0):
            return True
        return bool(c["unsupported"]) and (c["false_acceptance"] if c["false_acceptance"] is not None else 1.0) < (b["false_acceptance"] if b["false_acceptance"] is not None else 0.0)

    improved = all(better(before[f], after[f]) for f in p["targets"])
    print(json.dumps({"before": before, "after": after}, indent=1))
    if regress or not improved:
        shutil.copy(bdir / "weights.zln", WEIGHTS)
        dest.unlink()
        print(f"REVERTED: {'regression on ' + str(regress) if regress else 'targets did not improve'}")
        return 1
    p["observed"] = {"before": {f: before[f] for f in p["targets"]}, "after": {f: after[f] for f in p["targets"]}}
    p["regression_result"] = {"before": {f: before[f] for f in p["regression"]}, "after": {f: after[f] for f in p["regression"]}}
    dest.write_text(json.dumps(p, indent=1) + "\n")
    Path(a.patch).write_text(json.dumps(p, indent=1) + "\n")
    print(f"APPLIED {p['id']}: weights.zln regenerated; backup in {bdir}")
    return 0


def cmd_revert(a):
    last = sorted(BACKUPS.glob("*-before-*"))[-1:]
    if not last:
        raise SystemExit("no backup")
    b = last[0]
    shutil.copy(b / "weights.zln", WEIGHTS)
    if PATCHES.is_dir():
        shutil.rmtree(PATCHES)
    if (b / "patches").is_dir():
        shutil.copytree(b / "patches", PATCHES)
    print(f"restored {b.name}")


def cmd_compare(a):
    cases = EV.load(a.cases)
    for w in (a.before, a.after):
        m = EV.score(cases, EV.run_next(cases, (), Path(w)))
        print(w, {k: m.get(k) for k in ("exact_match", "semantic_accuracy", "false_acceptance", "false_rejection")})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inspect"); s.add_argument("--unit"); s.set_defaults(f=cmd_inspect)
    s = sub.add_parser("diagnose"); s.add_argument("--text", required=True); s.add_argument("--gold"); s.add_argument("--model", default=str(WEIGHTS)); s.set_defaults(f=cmd_diagnose)
    s = sub.add_parser("validate"); s.add_argument("patch"); s.set_defaults(f=cmd_validate)
    s = sub.add_parser("apply"); s.add_argument("patch"); s.set_defaults(f=cmd_apply)
    s = sub.add_parser("revert"); s.set_defaults(f=cmd_revert)
    s = sub.add_parser("compare"); s.add_argument("before"); s.add_argument("after"); s.add_argument("--cases", required=True); s.set_defaults(f=cmd_compare)
    a = ap.parse_args()
    return a.f(a) or 0


if __name__ == "__main__":
    sys.exit(main())
