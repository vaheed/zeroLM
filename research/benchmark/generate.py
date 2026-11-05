#!/usr/bin/env python3
"""Seeded benchmark generator for ZeroLM Next.

Cases are composed from research/benchmark/grammar.json. The gold semantics (operation list in
execution order, final world) are computed from the composition itself, never by running any model
and never from model/next/spec.json.

    generate.py --profile dev   --seed 1001 --out tests/next/dev.jsonl
    generate.py --profile final --seed 20260601 --dev tests/next/dev.jsonl --out tests/next/final.jsonl
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scenarios import Rng, DIRS, apply_op, canon_world, execute, match, random_world  # noqa: E402

G = json.loads((Path(__file__).resolve().parent / "grammar.json").read_text())
NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


class Profile:
    def __init__(self, name):
        self.name = name
        d = G["dev_subset"]
        self.dev = name == "dev"
        self.types = d["types"] if self.dev else list(G["types"])
        self.colors = d["colors"] if self.dev else G["colors"]
        self.max_clauses = d["max_clauses"] if self.dev else 12
        self.joiners = [", then ", " and ", " then "] if self.dev else G["joiners"]
        self.groups = ["after", "before"] if self.dev else ["after", "before", "pre_after", "pre_before"]

    def syn(self, rng, xs):
        return xs[0] if self.dev else rng.choice(xs)


def world_for(rng, prof):
    g = copy.deepcopy(G)
    g["types"] = {t: v for t, v in G["types"].items() if t in prof.types}
    g["colors"] = prof.colors
    n = max(4, min(len(prof.types) * len(prof.colors), 7))
    g["world"] = dict(G["world"], min_objects=4, max_objects=n)
    return random_world(rng, g)


# ------------------------------------------------------------------ gold operations
def ref(rng, sim, obj, prof):
    r = {"type": obj["type"]}
    unique_type = len(match(sim, obj["type"])) == 1
    if not unique_type or rng.chance(0.75):
        r["color"] = obj["color"]
    return r


def make_op(rng, sim, prof, modifiers, prev_touched=None, action=None):
    objs = sim["objects"]
    acts = ["move", "paint", "remove"] if len(objs) > 2 else ["move", "paint"]
    if action is None:
        x = rng.below(100)
        action = "move" if x < 55 else ("paint" if x < 80 else "remove")
        if action not in acts:
            action = "move"
    obj = prev_touched if (prev_touched and prev_touched in objs) else rng.choice(objs)
    op = {"action": action, "object": ref(rng, sim, obj, prof)}
    if action == "move":
        opts = []
        for d, (dx, dy) in DIRS.items():
            mx = 0
            while 0 <= obj["x"] + dx * (mx + 1) < sim["width"] and 0 <= obj["y"] + dy * (mx + 1) < sim["height"] and mx < 9:
                mx += 1
            if mx:
                opts.append((d, mx))
        d, mx = rng.choice(opts)
        op["direction"], op["count"] = d, 1 + rng.below(mx)
    elif action == "paint":
        free = [c for c in prof.colors if c != obj["color"] and not any(o["type"] == obj["type"] and o["color"] == c for o in objs)]
        if not free:
            return make_op(rng, sim, prof, modifiers, None, "move")
        op["to_color"] = rng.choice(free)
    op["negated"] = bool(modifiers and rng.chance(0.15))
    if modifiers and rng.chance(0.12):
        if rng.chance(0.5):
            o2 = rng.choice(objs)
            c = {"type": o2["type"]}
            if rng.chance(0.7):
                c["color"] = o2["color"]
        else:
            c = {"type": rng.choice(prof.types)}
            if rng.chance(0.7):
                c["color"] = rng.choice(prof.colors)
        c["negated"] = rng.chance(0.5)
        op["condition"] = c
    return op


# ------------------------------------------------------------------ surface realization
def new_style(rng, prof):
    return {"verb": rng.below(4), "det": rng.below(2), "noun": rng.below(3), "dir": rng.below(3), "cnt": rng.below(4),
            "cnt_omit": rng.chance(0.4), "neg": rng.below(3), "cond_pos": rng.below(3), "cond_det": rng.below(2),
            "gerund": rng.below(4)}


def pick(prof, xs, i):
    return xs[0] if prof.dev else xs[i % len(xs)]


def np_text(prof, st, ref_, det_idx=None):
    words = []
    det = "the" if (prof.dev or (st["det"] if det_idx is None else det_idx) == 0) else "a"
    noun = pick(prof, G["types"][ref_["type"]], st["noun"])
    color = ref_.get("color")
    if det == "a" and color and color[0] in "aeiou":
        det = "an"
    return " ".join([det] + ([color] if color else []) + [noun])


def count_text(prof, st, n):
    if n == 1 and st["cnt_omit"] and not prof.dev:
        return ""
    opts = G["count_words"][str(n)]
    return opts[0] if prof.dev else opts[st["cnt"] % len(opts)]


def clause_text(prof, op, st, gerund=False):
    verbs = G["verbs"][op["action"]]["gerund" if gerund else "imperative"]
    v = pick(prof, verbs, st["gerund"] if gerund else st["verb"])
    parts = [v, np_text(prof, st, op["object"])]
    if op["action"] == "move":
        parts.append(pick(prof, G["directions"][op["direction"]], st["dir"]))
        c = count_text(prof, st, op["count"])
        if c:
            parts.append(c)
    elif op["action"] == "paint":
        parts.append(op["to_color"])
    s = " ".join(parts)
    if op["negated"]:
        s = pick(prof, G["negation"], st["neg"]) + " " + s
    cond = op.get("condition")
    if cond:
        c = {"type": cond["type"]}
        if "color" in cond:
            c["color"] = cond["color"]
        kw = "unless" if cond["negated"] else "if"
        ctext = f"{kw} there is {np_text(prof, st, c, st['cond_det'])}"
        pos = st["cond_pos"] if not prof.dev else 0
        s = f"{ctext}, {s}" if pos == 0 else (f"{s} {ctext}" if pos == 1 else f"{s}, {ctext}")
    return s


def render(prof, structure, ops, styles, rng):
    """structure: list of (kind, [slot indices in text order]); ops/styles indexed by slot."""
    out = ""
    for gi, (kind, slots) in enumerate(structure):
        if kind == "seq":
            g = clause_text(prof, ops[slots[0]], styles[slots[0]])
        else:
            a, b = slots
            if kind in ("after", "before"):
                g = f"{clause_text(prof, ops[a], styles[a])} {kind} {clause_text(prof, ops[b], styles[b], True)}"
            else:
                k = kind.split("_")[1]
                g = f"{k} {clause_text(prof, ops[a], styles[a], True)}, {clause_text(prof, ops[b], styles[b])}"
        out = g if gi == 0 else out + prof.joiners[rng.below(len(prof.joiners))] + g if not out.endswith(" ") else out + g
    return out


def finish(rng, prof, text):
    if not prof.dev and rng.chance(0.5):
        text = text[0].upper() + text[1:]
    if not prof.dev and rng.chance(0.5) and not text.endswith("."):
        text += "."
    return text


# ------------------------------------------------------------------ structure sampling
def sample_structure(rng, prof, n):
    structure, slot, rem = [], 0, n
    while rem:
        if rem >= 2 and rng.chance(0.3):
            kind = rng.choice(prof.groups)
            structure.append((kind, [slot, slot + 1]))
            slot += 2
            rem -= 2
        else:
            structure.append(("seq", [slot]))
            slot += 1
            rem -= 1
    return structure


def exec_order(structure):
    order = []
    for kind, s in structure:
        if kind in ("after", "pre_before"):
            order += [s[1], s[0]]
        else:
            order += s
    return order


def build(rng, prof, n, track=False, distinct=False, structure=None):
    world = world_for(rng, prof)
    structure = structure or sample_structure(rng, prof, n)
    order = exec_order(structure)
    in_group = {s for k, sl in structure if k != "seq" for s in sl}
    sim = copy.deepcopy(world)
    ops = [None] * n
    prev = None
    for pos, slot in enumerate(order):
        mod = slot not in in_group
        action = None
        if track and pos > 0 and prev is not None and prev[0] == "paint":
            action = "move" if rng.chance(0.6) else "remove"
        if track and pos == 0:
            action = "paint"
        op = make_op(rng, sim, prof, mod, prev[1] if (track and prev) else None, action)
        if track and pos > 0 and prev and prev[0] == "paint":
            # reference the repainted object by its new colour (state tracking)
            op["object"] = {"type": prev[1]["type"], "color": prev[1]["color"]}
        apply_op(sim, op)
        if op["action"] == "paint" and not op["negated"] and "condition" not in op:
            tgt = [o for o in sim["objects"] if o["type"] == op["object"]["type"] and o["color"] == op["to_color"]]
            prev = ("paint", tgt[0]) if tgt else None
        else:
            prev = None
        ops[slot] = op
    styles = [new_style(rng, prof) for _ in range(n)]
    text = finish(rng, prof, render(prof, structure, ops, styles, rng))
    gold_ops = [ops[s] for s in order]
    return dict(world=world, structure=structure, ops=ops, styles=styles, text=text, gold_ops=gold_ops)


def op_sig(op):
    return json.dumps(op, sort_keys=True)


def make_case(cid, split, category, b, **extra):
    gw, st = execute(b["world"], b["gold_ops"])
    case = {"id": cid, "split": split, "category": category, "text": b["text"], "expect": "accept", "gold_ops": b["gold_ops"],
            "world": b["world"], "gold_world": canon_world(gw), "gold_status": st, "n_ops": len(b["gold_ops"])}
    case.update(extra)
    return case


# ------------------------------------------------------------------ unsupported / ambiguous
def corrupt(rng, prof, kind):
    """Return (text, subtype) for a sentence outside the supported grammar."""
    U = G["unsupported"]
    b = build(rng, prof, 1)
    op, st = b["ops"][0], b["styles"][0]
    op = dict(op, negated=False)
    op.pop("condition", None)
    if op["action"] != "move":
        op = make_op(rng, b["world"], prof, False, None, "move")
    plain = {"action": "move", "object": op["object"], "direction": op["direction"], "count": op["count"], "negated": False}
    np_ = np_text(prof, st, op["object"])
    v = pick(prof, G["verbs"]["move"]["imperative"], st["verb"])
    d = pick(prof, G["directions"][op["direction"]], st["dir"])
    o2 = {"type": rng.choice(prof.types), "color": rng.choice(prof.colors)}
    np2 = np_text(prof, st, o2)
    other = rng.choice([x for x in DIRS if x != op["direction"]])
    table = {
        "unknown_verb": lambda: f"{rng.choice(U['unknown_verbs'])} {np_} {d}",
        "unknown_type": lambda: f"{v} the {op['object'].get('color', 'red')} {rng.choice(U['unknown_types'])} {d}",
        "unknown_color": lambda: f"{v} the {rng.choice(U['unknown_colors'])} {op['object']['type']} {d}",
        "unknown_direction": lambda: f"{v} {np_} {rng.choice(U['unknown_directions'])}",
        "missing_direction": lambda: f"{v} {np_}" if rng.chance(0.5) else f"{v} {np_} {count_text(prof, st, 2)}",
        "missing_object": lambda: f"{v} {d}",
        "direction_first": lambda: f"{v} {d} {np_}",
        "pronoun": lambda: f"{v} it {d}",
        "coordination": lambda: f"{v} {np_} and {np2} {d}",
        "double_direction": lambda: f"{v} {np_} {d} {other}",
        "double_count": lambda: f"{v} {np_} {d} twice thrice",
        "count_without_direction": lambda: f"{v} {np_} twice",
        "count_before_direction": lambda: f"{v} {np_} twice {d}",
        "trailing_connector": lambda: f"{v} {np_} {d} then",
        "stray_word": lambda: f"{v} {np_} quickly {d}",
        "count_out_of_range": lambda: f"{v} {np_} {d} {10 + rng.below(90)} times",
        "condition_only": lambda: f"if there is {np2}",
        "double_condition": lambda: f"if there is {np2}, if there is {np_}, {v} {np_} {d}",
        "paint_without_color": lambda: f"paint {np_}",
        "paint_with_direction": lambda: f"paint {np_} {d}",
        "object_only": lambda: np_,
        "chatter": lambda: rng.choice(U["chatter"]),
        "second_object_missing": lambda: f"{v} {np_} {d}, then {v} {d}",
    }
    return table[kind](), kind


UNSUP = ["unknown_verb", "unknown_type", "unknown_color", "unknown_direction", "missing_direction", "missing_object",
         "direction_first", "pronoun", "coordination", "double_direction", "double_count", "count_without_direction",
         "count_before_direction", "trailing_connector", "stray_word", "count_out_of_range", "condition_only",
         "double_condition", "paint_without_color", "paint_with_direction", "object_only", "chatter", "second_object_missing"]


def ambiguous(rng, prof, i):
    """Chains of before/after connectors: not covered by the grammar (groups hold exactly two clauses)."""
    b = build(rng, prof, 3, structure=[("seq", [0]), ("seq", [1]), ("seq", [2])])
    v = [clause_text(prof, b["ops"][k], b["styles"][k], g) for k, g in ((0, False), (1, True), (2, True))]
    c = [("after", "after"), ("before", "after"), ("after", "before"), ("before", "before")][i % 4]
    return f"{v[0]} {c[0]} {v[1]} {c[1]} {v[2]}"


# ------------------------------------------------------------------ counterfactual pairs
def mutate(rng, prof, op, what, sim_world):
    o = copy.deepcopy(op)
    if what == "color":
        o["object"]["color"] = rng.choice([c for c in prof.colors if c != op["object"].get("color")])
    elif what == "type":
        o["object"]["type"] = rng.choice([t for t in prof.types if t != op["object"]["type"]])
    elif what == "direction":
        o["direction"] = rng.choice([d for d in DIRS if d != op["direction"]])
    elif what == "count":
        o["count"] = rng.choice([c for c in range(1, 10) if c != op["count"]])
    elif what == "to_color":
        o["to_color"] = rng.choice([c for c in prof.colors if c != op["to_color"] and c != op["object"].get("color")])
    elif what == "negated":
        o["negated"] = not op["negated"]
    elif what == "cond_polarity":
        o["condition"]["negated"] = not o["condition"]["negated"]
    elif what == "action":
        if op["action"] == "move":
            for k in ("direction", "count"):
                o.pop(k)
            o["action"] = "remove"
        else:
            o.pop("to_color")
            o["action"] = "remove"
    return o


def pair(rng, prof, what):
    for _ in range(50):
        b = build(rng, prof, 1)
        op = b["ops"][0]
        op = dict(op, negated=False)
        op.pop("condition", None)
        ok = {"color": "color" in op["object"], "type": True, "direction": op["action"] == "move", "count": op["action"] == "move",
              "to_color": op["action"] == "paint", "negated": True, "cond_polarity": True, "action": op["action"] != "remove"}
        if what == "cond_polarity":
            op["condition"] = {"type": rng.choice(prof.types), "negated": rng.chance(0.5)}
        if not ok[what]:
            continue
        if what == "action" and op["action"] == "paint":
            pass
        op2 = mutate(rng, prof, op, what, b["world"])
        st = b["styles"][0]
        t1 = finish(rng, prof, clause_text(prof, op, st))
        # reuse the same style choices so that the sentences differ only in the mutated component
        t2 = finish(Rng(0), prof, clause_text(prof, op2, st)) if False else None
        rng_state = rng.r.getstate()
        t1 = clause_text(prof, op, st)
        t2 = clause_text(prof, op2, st)
        cap = rng.chance(0.5)
        t1, t2 = (t1[0].upper() + t1[1:], t2[0].upper() + t2[1:]) if cap else (t1, t2)
        return b["world"], (op, t1), (op2, t2)
    raise RuntimeError("pair generation failed")


def order_variants(rng, prof):
    b = build(rng, prof, 2, structure=[("seq", [0]), ("seq", [1])])
    out = []
    for kind in ("seq", "after", "before", "pre_after", "pre_before"):
        struct = [("seq", [0, 1])] if False else ([("seq", [0]), ("seq", [1])] if kind == "seq" else [(kind, [0, 1])])
        order = exec_order(struct)
        bb = dict(b)
        bb["ops"] = [dict(o, negated=False) for o in b["ops"]]
        for o in bb["ops"]:
            o.pop("condition", None)
        sim = copy.deepcopy(b["world"])
        text = finish(Rng(rng.below(10 ** 6)), prof, render(prof, struct, bb["ops"], b["styles"], rng))
        out.append((kind, order, text, bb["ops"]))
    return b["world"], out


# ------------------------------------------------------------------ driver
def generate(profile, seed, counts):
    prof = Profile(profile)
    rng = Rng(seed)
    cases = []
    cid = 0

    def nid(prefix):
        nonlocal cid
        cid += 1
        return f"{prefix}-{cid:06d}"

    # primitives: deterministic enumeration of every surface word of the grammar in a minimal frame
    base_w = {"width": 8, "height": 8, "objects": [
        {"id": 1, "type": "block", "color": "red", "x": 4, "y": 4}, {"id": 2, "type": "circle", "color": "blue", "x": 3, "y": 3},
        {"id": 3, "type": "triangle", "color": "green", "x": 5, "y": 5}, {"id": 4, "type": "star", "color": "yellow", "x": 1, "y": 1},
        {"id": 5, "type": "block", "color": "purple", "x": 6, "y": 2}, {"id": 6, "type": "circle", "color": "orange", "x": 2, "y": 6}]}

    def prim(text, ops, tag):
        b = {"world": base_w, "gold_ops": ops, "text": text}
        cases.append(make_case(nid("prim"), "primitive", "primitive", b, tags={"primitive": tag}))

    mv = lambda t="block", c="red", d="left", n=1: {"action": "move", "object": {"type": t, "color": c}, "direction": d, "count": n, "negated": False}
    if profile == "final":
        for a, kind in (("move", "imperative"), ("paint", "imperative"), ("remove", "imperative")):
            for v in G["verbs"][a][kind]:
                if a == "move":
                    prim(f"{v} the red block left", [mv()], f"verb:{v}")
                elif a == "paint":
                    prim(f"{v} the red block blue", [{"action": "paint", "object": {"type": "block", "color": "red"}, "to_color": "blue", "negated": False}], f"verb:{v}")
                else:
                    prim(f"{v} the red block", [{"action": "remove", "object": {"type": "block", "color": "red"}, "negated": False}], f"verb:{v}")
        for t, words in G["types"].items():
            col = {"block": "red", "circle": "blue", "triangle": "green", "star": "yellow"}[t]
            for w in words:
                prim(f"move the {col} {w} left", [mv(t, col)], f"type:{w}")
        for c in G["colors"]:
            tgt = {"red": ("block", "red"), "blue": ("circle", "blue"), "green": ("triangle", "green"), "yellow": ("star", "yellow"), "purple": ("block", "purple"), "orange": ("circle", "orange")}[c]
            prim(f"move the {c} {tgt[0]} left", [mv(tgt[0], c)], f"color:{c}")
            prim(f"paint the red block {c}", [{"action": "paint", "object": {"type": "block", "color": "red"}, "to_color": c, "negated": False}], f"newcolor:{c}")
        for d, words in G["directions"].items():
            for w in words:
                prim(f"move the red block {w}", [mv(d=d)], f"direction:{w}")
        for n, words in G["count_words"].items():
            for w in words:
                prim(f"move the red block left {w}", [mv(n=int(n))], f"count:{w}")
        for w in G["negation"]:
            prim(f"{w} move the red block left", [dict(mv(), negated=True)], f"negation:{w}")
        prim("if there is a blue circle, move the red block left", [dict(mv(), condition={"type": "circle", "color": "blue", "negated": False})], "word:if")
        prim("move the red block left unless there is a blue circle", [dict(mv(), condition={"type": "circle", "color": "blue", "negated": True})], "word:unless")
        for j in (", then", "and then", "and", "then", "next", "afterwards", ";", "."):
            prim(f"move the red block left {j} move the blue circle up", [mv(), mv("circle", "blue", "up")], f"joiner:{j}")
        prim("move the red block left after moving the blue circle up", [mv("circle", "blue", "up"), mv()], "word:after")
        prim("move the red block left before moving the blue circle up", [mv(), mv("circle", "blue", "up")], "word:before")
        prim("after moving the red block left, move the blue circle up", [mv(), mv("circle", "blue", "up")], "word:after-prefix")
        prim("before moving the red block left, move the blue circle up", [mv("circle", "blue", "up"), mv()], "word:before-prefix")

    def add_comp(category, n_cases, lo, hi, **kw):
        for _ in range(n_cases):
            n = lo + rng.below(hi - lo + 1)
            b = build(rng, prof, n, **kw)
            cases.append(make_case(nid(category[:4]), profile, category, b))

    add_comp("single", counts["single"], 1, 1)
    add_comp("comp2", counts["comp2"], 2, 2)
    add_comp("comp3", counts["comp3"], 3, 3)
    if profile == "final":
        add_comp("multi", counts["multi"], 4, 5)
        add_comp("long", counts["long"], 6, 12)
        add_comp("binding", counts["binding"], 2, 4, distinct=True)
        add_comp("tracking", counts["tracking"], 2, 4, track=True)
    # order
    for k in range(counts["order"]):
        w, variants = order_variants(rng, prof)
        for kind, order, text, ops in variants:
            b = {"world": w, "gold_ops": [ops[i] for i in order], "text": text}
            cases.append(make_case(nid("ordr"), profile, "order", b, group=f"order{k}", tags={"connector": kind}))
    # counterfactual pairs
    whats = ["color", "type", "direction", "count", "to_color", "negated", "cond_polarity", "action"]
    for k in range(counts["pairs"]):
        what = whats[k % len(whats)]
        w, (o1, t1), (o2, t2) = pair(rng, prof, what)
        for role, (op, text) in (("a", (o1, t1)), ("b", (o2, t2))):
            cases.append(make_case(nid("cfac"), profile, "counterfactual", {"world": w, "gold_ops": [op], "text": text},
                                   pair={"id": f"pair{k}", "role": role, "changed": what}))
    # unsupported + ambiguous
    for k in range(counts["unsupported"]):
        text, sub = corrupt(rng, prof, UNSUP[k % len(UNSUP)])
        cases.append({"id": nid("unsp"), "split": profile, "category": "unsupported", "text": text, "expect": "reject",
                      "gold_ops": [], "world": None, "tags": {"kind": sub}})
    for k in range(counts["ambiguous"]):
        cases.append({"id": nid("ambg"), "split": profile, "category": "ambiguous", "text": ambiguous(rng, prof, k), "expect": "reject",
                      "gold_ops": [], "world": None, "tags": {"kind": "chained_order_connectors"}})
    return cases


COUNTS = {
    "dev": dict(single=120, comp2=120, comp3=60, order=20, pairs=40, unsupported=46, ambiguous=0),
    "final": dict(single=2000, comp2=3000, comp3=3000, multi=2000, long=1500, binding=1000, tracking=1000, order=600,
                  pairs=1600, unsupported=1500, ambiguous=300),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["dev", "final"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dev", type=Path, help="dev cases used to mark seen/unseen combinations")
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    counts = {k: max(1, int(v * a.scale)) for k, v in COUNTS[a.profile].items()}
    cases = generate(a.profile, a.seed, counts)
    if a.dev:
        dev = [json.loads(l) for l in a.dev.read_text().splitlines() if l.strip()]
        seq_seen = {json.dumps(c["gold_ops"], sort_keys=True) for c in dev}
        op_seen = {op_sig(o) for c in dev for o in c["gold_ops"]}
        for c in cases:
            if c["expect"] == "accept":
                c["seen_sequence"] = json.dumps(c["gold_ops"], sort_keys=True) in seq_seen
                c["unseen_ops"] = sum(op_sig(o) not in op_seen for o in c["gold_ops"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(c, sort_keys=True) for c in cases) + "\n"
    a.out.write_text(body)
    print(f"{a.out}: {len(cases)} cases, seed {a.seed}, sha256 {hashlib.sha256(body.encode()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
