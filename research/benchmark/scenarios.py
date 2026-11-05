"""Reference semantics for the benchmark: world model, executor, scenario simulation.

This is an independent Python implementation of the executor contract in rust/src/next/world.rs.
Coordinates: origin top-left, x right, y down; "up" is y-1.
"""
from __future__ import annotations

import copy
import random

DIRS = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}


class Rng:
    """Deterministic across Python versions: built on Random.random() only."""

    def __init__(self, seed):
        self.r = random.Random(seed)

    def below(self, n):
        return int(self.r.random() * n)

    def choice(self, xs):
        return xs[self.below(len(xs))]

    def chance(self, p):
        return self.r.random() < p

    def shuffle(self, xs):
        for i in range(len(xs) - 1, 0, -1):
            j = self.below(i + 1)
            xs[i], xs[j] = xs[j], xs[i]


def match(world, typ, color=None):
    return [o for o in world["objects"] if o["type"] == typ and (color is None or o["color"] == color)]


def apply_op(world, op):
    """Mutates world; returns a status string."""
    if op.get("negated"):
        return "skipped_negated"
    cond = op.get("condition")
    if cond:
        exists = bool(match(world, cond["type"], cond.get("color")))
        if exists == cond["negated"]:
            return "skipped_condition"
    found = match(world, op["object"]["type"], op["object"].get("color"))
    if not found:
        return "no_such_object"
    if len(found) > 1:
        return "ambiguous_object"
    o = found[0]
    if op["action"] == "move":
        dx, dy = DIRS[op["direction"]]
        n = op.get("count", 1)
        nx, ny = o["x"] + dx * n, o["y"] + dy * n
        if not (0 <= nx < world["width"] and 0 <= ny < world["height"]):
            return "out_of_bounds"
        o["x"], o["y"] = nx, ny
    elif op["action"] == "paint":
        o["color"] = op["to_color"]
    elif op["action"] == "remove":
        world["objects"].remove(o)
    return "ok"


def execute(world, ops):
    w = copy.deepcopy(world)
    st = [apply_op(w, op) for op in ops]
    return w, st


def canon_world(w):
    return {"width": w["width"], "height": w["height"],
            "objects": sorted(({k: o[k] for k in ("id", "type", "color", "x", "y")} for o in w["objects"]), key=lambda o: o["id"])}


def random_world(rng, g):
    """Objects with unique (type, color) pairs on distinct cells."""
    types, colors = list(g["types"].keys()), g["colors"]
    n = g["world"]["min_objects"] + rng.below(g["world"]["max_objects"] - g["world"]["min_objects"] + 1)
    pairs = [(t, c) for t in types for c in colors]
    rng.shuffle(pairs)
    cells = [(x, y) for x in range(g["world"]["width"]) for y in range(g["world"]["height"])]
    rng.shuffle(cells)
    objs = [{"id": i + 1, "type": pairs[i][0], "color": pairs[i][1], "x": cells[i][0], "y": cells[i][1]} for i in range(n)]
    return {"width": g["world"]["width"], "height": g["world"]["height"], "objects": objs}
