#!/usr/bin/env python3
"""Construct ZeroLM Next weights by closed-form design. No training, no search.

Input : model/next/spec.json      (lexicon + constants, human-authored)
        model/next/patches/NNN-*.json (ordered, human-reasoned weight patches; see research/tools/wsp.py)
Output: model/next/weights.zln    (text tensors read by rust/src/next)

Every number written here is either 0, a structural +/-1, a documented constant
(time offset, big-M gate, comparator slope) or a 0.5 decoder bias. The derivation of each
tensor is in research/ARCHITECTURE.md. Nothing is fitted to examples.

Usage: build_next.py [--check] [--no-patches]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "model" / "next" / "spec.json"
PATCH_DIR = ROOT / "model" / "next" / "patches"
OUT = ROOT / "model" / "next" / "weights.zln"


# ---------------------------------------------------------------- naming
def feature_names(spec):
    tf = spec["token_features"]
    names = []
    for block in ("ACT", "TYP", "COL", "DIR", "CNT"):
        names += [f"{block}.{v}" for v in tf[block]]
    names += tf["SINGLE"]
    return names


def field_blocks(spec):
    """Ordered (block, labels or None for a single field)."""
    tf = spec["token_features"]
    return [
        ("ACT", tf["ACT"]), ("TYP", tf["TYP"]), ("COL", tf["COL"]), ("NCOL", tf["COL"]),
        ("DIR", tf["DIR"]), ("CNT", tf["CNT"]), ("NEG", None),
        ("CTYP", tf["TYP"]), ("CCOL", tf["COL"]), ("CFLAG", None), ("CNEG", None),
        ("CCL", None), ("PB", None), ("TIME", None),
    ]


def field_names(spec):
    out = []
    for block, labels in field_blocks(spec):
        out += [f"{block}.{l}" for l in labels] if labels else [block]
    return out


def block_of(spec, block):
    return [n for n in field_names(spec) if n == block or n.startswith(block + ".")]


# ---------------------------------------------------------------- layer helper
class Layer:
    """Hidden ReLU layer: h = relu(W z + b), with named inputs and named units."""

    def __init__(self, inputs):
        self.inputs = list(inputs)
        self.idx = {n: i for i, n in enumerate(self.inputs)}
        self.units = []  # (name, {input: w}, bias, {output: w})

    def unit(self, name, w, bias=0.0, out=None):
        for k in w:
            if k not in self.idx:
                raise KeyError(f"unit {name}: unknown input {k}")
        self.units.append((name, dict(w), float(bias), dict(out or {})))

    def matrix(self):
        return [[u[1].get(n, 0.0) for n in self.inputs] for u in self.units]

    def bias(self):
        return [u[2] for u in self.units]

    def names(self):
        return [u[0] for u in self.units]


def add(*dicts):
    out = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + v
    return out


def neg(d):
    return {k: -v for k, v in d.items()}


def scale(d, s):
    return {k: v * s for k, v in d.items()}


# ---------------------------------------------------------------- symbolic model
MARKERS0 = {"TIME"}  # fields that do not make a row "exist" (bookkeeping only)


class Unit:
    def __init__(self, name, w, bias=0.0, out=None, adv=0.0, rev=0.0, err=False):
        self.name, self.w, self.bias = name, dict(w), float(bias)
        self.out = dict(out or {})   # B units: {field: weight}
        self.adv, self.rev = adv, rev  # A units: weight into the (advance, reverse) outputs
        self.err = err                 # B units: contributes to the sticky error mass


class Model:
    def __init__(self, spec):
        self.spec = spec
        self.c = spec["constants"]
        self.features = feature_names(spec)
        self.fields = field_names(spec)
        self.markers = set(MARKERS0)
        self.lexicon = {w: list(v) for w, v in spec["lexicon"].items()}
        self.neutral = list(spec["neutral_words"])
        self.A, self.B, self.V = [], [], []
        self.overrides = []  # (tensor, row, col, old, new), applied after densification
        self.tf_field = "TIME"
        self.cfg = {}


def add(*dicts):
    out = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + v
    return out


def neg(d):
    return {k: -v for k, v in d.items()}


def scale(d, s):
    return {k: v * s for k, v in d.items()}


def construct(spec):
    """The base (revision 0) construction: every unit below is derived in research/ARCHITECTURE.md."""
    M = Model(spec)
    c = M.c
    BIG = float(c["big"])
    feats, fields = M.features, M.fields
    tf = spec["token_features"]

    def X(block):
        return {f"x.{n}": 1.0 for n in feats if n == block or n.startswith(block + ".")}

    def R(block, p="r"):
        return {f"{p}.{n}": 1.0 for n in fields if n == block or n.startswith(block + ".")}

    # ---- layer A: row-boundary detector (inputs: token x, current row r)
    hasACT, PB = R("ACT"), {"r.PB": 1.0}
    for cls, xs in (("SEQ", {"x.SEQ": 1.0}), ("BEFORE", {"x.BEFORE": 1.0}), ("ACT", X("ACT")), ("NEG", {"x.NEG": 1.0})):
        M.A.append(Unit(f"bnd_{cls}", add(xs, hasACT, neg(PB)), -1.0, adv=1.0))
        M.A.append(Unit(f"bnd_{cls}_pb", add(xs, hasACT, PB), -2.0, adv=1.0, rev=1.0))
    M.A.append(Unit("bnd_AFTER", add({"x.AFTER": 1.0}, hasACT), -1.0, adv=1.0, rev=1.0))

    # ---- layer B: role-gated writes, timing, error detectors
    cm = add({"rp.CFLAG": 1.0}, {"rp.CCL": -1.0})  # in-condition-clause indicator (exactly 0 or 1)
    hTYP, hCOL, hDIR = R("TYP", "rp"), R("COL", "rp"), R("DIR", "rp")

    def wr(name, w, bias, *targets):
        M.B.append(Unit(name, w, bias, out={tg: 1.0 for tg in targets}))

    for a in tf["ACT"]:
        wr(f"w_ACT.{a}", {f"x.ACT.{a}": 1.0}, 0, f"ACT.{a}")
    for ty in tf["TYP"]:
        wr(f"w_TYP.{ty}", add({f"x.TYP.{ty}": 1.0}, neg(cm)), 0, f"TYP.{ty}")
        wr(f"w_CTYP.{ty}", add({f"x.TYP.{ty}": 1.0}, cm), -1, f"CTYP.{ty}")
    for co in tf["COL"]:
        xc = {f"x.COL.{co}": 1.0}
        wr(f"w_COL.{co}", add(xc, neg(cm), neg(hTYP)), 0, f"COL.{co}")
        wr(f"w_NCOL.{co}", add(xc, hTYP, neg(cm)), -1, f"NCOL.{co}")
        wr(f"w_CCOL.{co}", add(xc, cm), -1, f"CCOL.{co}")
    for d in tf["DIR"]:
        wr(f"w_DIR.{d}", {f"x.DIR.{d}": 1.0}, 0, f"DIR.{d}")
    for n in tf["CNT"]:
        wr(f"w_CNT.{n}", {f"x.CNT.{n}": 1.0}, 0, f"CNT.{n}")
    wr("w_NEG", {"x.NEG": 1.0}, 0, "NEG")
    wr("w_CFLAG", add({"x.IF": 1.0, "x.UNLESS": 1.0}, {"rp.CFLAG": -1.0}), 0, "CFLAG")
    wr("w_CNEG", add({"x.UNLESS": 1.0}, {"rp.CFLAG": -1.0}), 0, "CNEG")
    wr("w_CCL_close", add({"x.CLOSE": 1.0}, cm), -1, "CCL")
    wr("w_CCL_seq", add({"x.SEQ": 1.0}, cm), -1, "CCL")
    wr("w_CCL_act", add(X("ACT"), cm), -1, "CCL")
    wr("w_PB", add({"x.BEFORE": 1.0}, neg(R("ACT", "r"))), 0, "PB")
    wr("time_rev", add({"r.TIME": 1.0}, scale({"s.rev": 1.0}, BIG)), -0.5 - BIG, "TIME")
    wr("time_fwd", add({"s.m": 1.0}, scale({"s.adv": 1.0}, BIG), scale({"s.rev": 1.0}, -BIG)), 1.0 - BIG, "TIME")
    errs = [
        ("err_unknown_token", {"x.UNK": 1.0}, 0),
        ("err_direction_without_object", add(X("DIR"), neg(hTYP)), 0),
        ("err_direction_in_condition", add(X("DIR"), cm), -1),
        ("err_count_without_direction", add(X("CNT"), neg(hDIR)), 0),
        ("err_count_in_condition", add(X("CNT"), cm), -1),
        ("err_action_after_object", add(X("ACT"), hTYP), -1),
        ("err_action_after_color", add(X("ACT"), hCOL), -1),
        ("err_second_condition", add({"x.IF": 1.0, "x.UNLESS": 1.0}, {"rp.CFLAG": 1.0}), -1),
    ]
    for name, w, b in errs:
        M.B.append(Unit(name, w, b, err=True))

    # ---- row validity layer V (shared across rows). Inputs: row fields, ex, tf.
    r_ = lambda b: R(b, "T")
    Tn = lambda n: {f"T.{n}": 1.0}
    verr = []
    for blk in ("ACT", "TYP", "COL", "NCOL", "DIR", "CNT", "CTYP", "CCOL"):
        verr.append((f"overfull_{blk}", r_(blk), -1))
    for blk in ("NEG", "CNEG"):
        verr.append((f"overfull_{blk}", Tn(blk), -1))
    verr += [
        ("row_without_action", add({"ex": 1.0}, neg(r_("ACT"))), 0),
        ("move_needs_object", add(Tn("ACT.move"), neg(r_("TYP"))), 0),
        ("move_needs_direction", add(Tn("ACT.move"), neg(r_("DIR"))), 0),
        ("move_forbids_new_color", add(Tn("ACT.move"), r_("NCOL")), -1),
        ("paint_needs_object", add(Tn("ACT.paint"), neg(r_("TYP"))), 0),
        ("paint_needs_new_color", add(Tn("ACT.paint"), neg(r_("NCOL"))), 0),
        ("paint_forbids_direction", add(Tn("ACT.paint"), r_("DIR")), -1),
        ("paint_forbids_count", add(Tn("ACT.paint"), r_("CNT")), -1),
        ("remove_needs_object", add(Tn("ACT.remove"), neg(r_("TYP"))), 0),
        ("remove_forbids_direction", add(Tn("ACT.remove"), r_("DIR")), -1),
        ("remove_forbids_count", add(Tn("ACT.remove"), r_("CNT")), -1),
        ("remove_forbids_new_color", add(Tn("ACT.remove"), r_("NCOL")), -1),
        ("count_needs_direction", add(r_("CNT"), neg(r_("DIR"))), 0),
        ("condition_needs_object", add(Tn("CFLAG"), neg(r_("CTYP"))), 0),
        ("dangling_connector", add({"tf": 1.0}, {"ex": -1.0}), 0),
    ]
    for name, w, b in verr:
        M.V.append(Unit(name, w, b))
    return M


# ---------------------------------------------------------------- patches (symbolic, auditable)
def find_unit(M, layer, name):
    for u in getattr(M, layer):
        if u.name == name:
            return u
    raise SystemExit(f"patch: no unit {name} in layer {layer}")


def apply_patch(M, patch, label=""):
    """Patch ops: add_feature, add_field, set_lexicon, remove_neutral, add_unit, set_unit_weight, set_override."""
    for op in patch["edits"]:
        kind = op["op"]
        if kind == "add_feature":
            if op["name"] in M.features:
                raise SystemExit(f"{label}: feature {op['name']} exists")
            M.features.append(op["name"])
        elif kind == "add_field":
            if op["name"] in M.fields:
                raise SystemExit(f"{label}: field {op['name']} exists")
            M.fields.append(op["name"])
            if op.get("marker"):
                M.markers.add(op["name"])
        elif kind == "set_lexicon":
            for f in op["features"]:
                if f not in M.features:
                    raise SystemExit(f"{label}: unknown feature {f}")
            M.lexicon[op["word"]] = list(op["features"])
        elif kind == "remove_neutral":
            if op["word"] not in M.neutral:
                raise SystemExit(f"{label}: {op['word']} is not a neutral word")
            M.neutral.remove(op["word"])
        elif kind == "add_unit":
            layer = op["layer"]
            if any(u.name == op["name"] for u in getattr(M, layer)):
                raise SystemExit(f"{label}: unit {op['name']} exists in {layer}")
            u = Unit(op["name"], op["inputs"], op.get("bias", 0.0), out=op.get("out"), adv=op.get("adv", 0.0), rev=op.get("rev", 0.0),
                     err=op.get("err", False))
            getattr(M, layer).append(u)
        elif kind == "set_unit_weight":
            u = find_unit(M, op["layer"], op["unit"])
            old = u.w.get(op["input"], 0.0)
            if abs(old - op["old"]) > 1e-12:
                raise SystemExit(f"{label}: {op['unit']}.{op['input']} is {old}, patch expects {op['old']}")
            u.w[op["input"]] = float(op["new"])
        elif kind == "set_override":  # edit one entry of a derived tensor by row/col names
            M.overrides.append((op["tensor"], op["row"], op["col"], op["old"], op["new"], label))
        else:
            raise SystemExit(f"{label}: unknown patch op {kind}")


def tensor_axes(M):
    """Row and column names of the tensors that patches may address."""
    x = [f"x.{n}" for n in M.features]
    return {
        "tf_w": (["0"], list(M.fields)), "ex_w": (["0"], list(M.fields)), "time_sel": (["0"], list(M.fields)),
        "E": (words_of(M), list(M.features)),
    }


def words_of(M):
    return list(M.lexicon) + M.neutral + ["<unk>"]


# ---------------------------------------------------------------- densification
def densify(M):
    spec, c = M.spec, M.c
    N, OFF = c["rows"], float(c["time_offset"])
    feats, fields = M.features, M.fields
    D, K = len(feats), len(fields)
    fidx = {n: i for i, n in enumerate(fields)}
    t, lists = {}, {}

    words = words_of(M)
    seen = set()
    for w in words:
        if w in seen or " " in w:
            raise ValueError(f"lexicon key {w!r} must be a unique single token (no phrase entries)")
        seen.add(w)
    E = []
    for w in words:
        row = [0.0] * D
        if w == "<unk>":
            row[feats.index("UNK")] = 1.0
        else:
            for f in M.lexicon.get(w, []):
                row[feats.index(f)] = 1.0
        E.append(row)
    lists.update(vocab=words, features=feats, fields=fields)
    t["E"] = E

    def dense(units, inputs, getter=lambda u: u.w):
        idx = {n: i for i, n in enumerate(inputs)}
        rows = []
        for u in units:
            row = [0.0] * len(inputs)
            for k, v in getter(u).items():
                if k not in idx:
                    raise SystemExit(f"unit {u.name}: unknown input {k}")
                row[idx[k]] = v
            rows.append(row)
        return rows

    a_in = [f"x.{n}" for n in feats] + [f"r.{n}" for n in fields]
    t["A_w"], t["A_b"] = dense(M.A, a_in), [u.bias for u in M.A]
    t["A_o"] = [[u.adv for u in M.A], [u.rev for u in M.A]]
    lists["A_units"] = [u.name for u in M.A]

    b_in = ([f"x.{n}" for n in feats] + [f"rp.{n}" for n in fields] + [f"r.{n}" for n in fields] + ["s.adv", "s.rev", "s.m"])
    t["B_w"], t["B_b"] = dense(M.B, b_in), [u.bias for u in M.B]
    B_o = [[0.0] * len(M.B) for _ in range(K)]
    for j, u in enumerate(M.B):
        for tgt, w in u.out.items():
            if tgt not in fidx:
                raise SystemExit(f"unit {u.name}: unknown output field {tgt}")
            B_o[fidx[tgt]][j] = w
    t["B_o"] = B_o
    t["B_e"] = [[1.0 if u.err else 0.0 for u in M.B]]
    lists["B_units"] = [u.name for u in M.B]

    t["S"] = [[1.0 if i == j + 1 else 0.0 for j in range(N)] for i in range(N)]
    T0 = [[0.0] * K for _ in range(N)]
    T0[0][fidx["TIME"]] = OFF
    t["T0"] = T0
    t["p0"] = [[1.0] + [0.0] * (N - 1)]
    t["m0"] = [[OFF]]
    gm = [0.0] * (K + 1)
    gm[fidx["TIME"]], gm[K] = 1.0, -1.0
    t["Gm_w"], t["Gm_b"] = [gm], [[0.0]]
    t["Pe_w"], t["Pe_b"] = [[-1.0] * N], [[1.0]]

    v_in = [f"T.{n}" for n in fields] + ["ex", "tf"]
    t["V_w"], t["V_b"] = dense(M.V, v_in), [u.bias for u in M.V]
    t["V_o"] = [[1.0] * len(M.V)]
    lists["V_units"] = [u.name for u in M.V]
    t["ex_w"] = [[0.0 if n in M.markers else 1.0 for n in fields]]
    t["tf_w"] = [[1.0 if n == M.tf_field else 0.0 for n in fields]]
    t["time_sel"] = [[1.0 if n == "TIME" else 0.0 for n in fields]]
    t["empty_w"], t["empty_b"] = [[-1.0] * N], [[1.0]]
    t["rank_eps"] = [[(0.25 if j < i else -0.25) for j in range(N)] for i in range(N)]
    t["rank_slope"] = [[float(c["rank_slope"])]]
    t["reject_threshold"] = [[float(c["reject_threshold"])]]

    # ---- output decoder: logits = Wd T_row + bd, argmax; class "none" omits the key
    dec = []
    for i, d in enumerate(spec["decode"]):
        cols = [n for n in fields if n == d["block"] or n.startswith(d["block"] + ".")]
        names = [n.split(".", 1)[1] if "." in n else n for n in cols]
        if d["kind"] == "bool" and d.get("gate"):
            classes = ["none", "false", "true"]
            W = [[0.0] * K for _ in classes]
            W[1][fidx[d["gate"]]], W[2][fidx[cols[0]]] = 0.5, 1.0
            b = [0.25, 0.0, 0.0]
        elif d["kind"] == "bool":
            classes, W, b = ["false", "true"], [[0.0] * K, [0.0] * K], [0.5, 0.0]
            W[1][fidx[cols[0]]] = 1.0
        else:
            classes = ["none"] + (["1"] if d.get("gate") else []) + names
            W = [[0.0] * K for _ in classes]
            b = [0.25 if d.get("gate") else 0.5] + [0.0] * (len(classes) - 1)
            if d.get("gate"):
                W[1][fidx[d["gate"]]] = 0.5
            for cn in cols:
                W[classes.index(cn.split(".", 1)[1], 2 if d.get("gate") else 1)][fidx[cn]] = 1.0
        t[f"dec.{i}.w"], t[f"dec.{i}.b"] = W, [b]
        lists[f"dec.{i}.classes"] = classes
        dec.append(d)
    lists["decode_paths"] = [d["path"] for d in dec]
    lists["decode_kinds"] = [d["kind"] for d in dec]

    # ---- point edits of derived tensors (by names), checked against the value they replace
    axes = {
        "tf_w": (["0"], list(fields)), "ex_w": (["0"], list(fields)), "time_sel": (["0"], list(fields)),
        "E": (words, list(feats)), "A_o": (["adv", "rev"], lists["A_units"]),
    }
    for tname, row, col, old, new, label in M.overrides:
        rn, cn = axes[tname]
        r, cidx = rn.index(str(row)), cn.index(col)
        if abs(t[tname][r][cidx] - old) > 1e-12:
            raise SystemExit(f"{label}: {tname}[{row}][{col}] is {t[tname][r][cidx]}, patch expects {old}")
        t[tname][r][cidx] = float(new)
    return t, lists, dec, dict(D=D, K=K, N=N, OFF=OFF)


def load_patches(patch_dir, upto=None):
    if not patch_dir.is_dir():
        return []
    out = []
    for p in sorted(patch_dir.glob("*.json")):
        if upto is not None and int(p.name.split("-")[0]) > upto:
            continue
        out.append((p.name, json.loads(p.read_text())))
    return out


def build(spec, patch_dir=PATCH_DIR, upto=None, extra_patches=()):
    M = construct(spec)
    applied = []
    for name, patch in list(load_patches(patch_dir, upto)) + list(extra_patches):
        apply_patch(M, patch, name)
        applied.append(name)
    t, lists, dec, dims = densify(M)
    return t, lists, dec, dims, applied


# ---------------------------------------------------------------- serialization
def fmt(x):
    if x == int(x):
        return str(int(x))
    return repr(float(x))


def serialize(spec, t, lists, dec, dims):
    out = ["ZLN 1", f"meta name {spec['name']}", f"meta version {spec['version']}",
           "meta training none"]
    for k, v in dims.items():
        out.append(f"meta {k} {fmt(v)}")
    for name in sorted(lists):
        out.append(f"list {name} {len(lists[name])} " + " ".join(lists[name]))
    for name in sorted(t):
        m = t[name]
        if m and not isinstance(m[0], list):
            m = [m]
        out.append(f"tensor {name} {len(m)} {len(m[0])}")
        for row in m:
            out.append(" ".join(fmt(v) for v in row))
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if weights.zln differs from the construction")
    ap.add_argument("--upto", type=int, default=None, help="apply only patches numbered <= N (0 = base construction)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--patch-dir", type=Path, default=PATCH_DIR)
    a = ap.parse_args()
    spec = json.loads(SPEC.read_text())
    t, lists, dec, dims, applied = build(spec, a.patch_dir, a.upto)
    text = serialize(spec, t, lists, dec, dims)
    if a.check:
        cur = a.out.read_text() if a.out.exists() else ""
        if cur != text:
            print(f"FAIL: {a.out} is not reproduced by build_next.py", file=sys.stderr)
            return 1
        print(f"OK: {a.out} reproduced exactly ({len(text)} bytes, patches: {applied or 'none'})")
        return 0
    a.out.write_text(text)
    print(f"wrote {a.out} ({len(text)} bytes, patches: {applied or 'none'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
