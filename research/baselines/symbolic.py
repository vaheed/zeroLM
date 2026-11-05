"""Baseline: an explicit recursive-descent parser for the same instruction domain.

This is the strong, honest comparison for ZeroLM Next: a conventional symbolic parser written
directly against the grammar. It has its own lexicon (words copied from the benchmark grammar, not
from the model) and shares no code with the network. It rejects chained before/after connectors
because its group rule allows exactly two clauses.
"""
from __future__ import annotations

import re

VERBS = {"move": "move", "moves": "move", "moving": "move", "shift": "move", "shifting": "move", "slide": "move", "sliding": "move",
         "push": "move", "pushing": "move", "paint": "paint", "painting": "paint", "color": "paint", "colour": "paint",
         "recolor": "paint", "dye": "paint", "remove": "remove", "removing": "remove", "delete": "remove", "deleting": "remove",
         "erase": "remove", "erasing": "remove"}
TYPES = {"block": "block", "cube": "block", "box": "block", "circle": "circle", "disc": "circle", "ball": "circle",
         "triangle": "triangle", "pyramid": "triangle", "star": "star"}
COLORS = {"red", "blue", "green", "yellow", "purple", "orange"}
DIRS = {"left": "left", "leftward": "left", "west": "left", "right": "right", "rightward": "right", "east": "right",
        "up": "up", "upward": "up", "upwards": "up", "north": "up", "down": "down", "downward": "down", "downwards": "down", "south": "down"}
NUMS = {"once": 1, "twice": 2, "thrice": 3, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
NUMS.update({str(i): i for i in range(1, 10)})
NEG = {"not", "dont", "never"}
DET = {"the", "a", "an"}
FILL = {"to", "by", "times", "time", "steps", "step", "cells", "cell", "space", "spaces", "please", "that", "finally", "first", "do"}
SEQ = {"then", "and", "next", "afterwards"}
CLOSE = {",", ".", ";"}


class Reject(Exception):
    pass


def tokenize(text):
    t = text.lower().replace("'", "").replace("\u2019", "")
    return re.findall(r"[^\W_]+|[,.;]|[^\w\s!?:\"()\-\u201c\u201d]", t)


class P:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def skip(self, words):
        while self.peek() in words:
            self.i += 1

    def np(self):
        self.skip(DET | {"there", "is"})
        ref = {}
        if self.peek() in COLORS:
            ref["color"] = self.take()
        if self.peek() not in TYPES:
            raise Reject("noun")
        ref["type"] = TYPES[self.take()]
        return ref

    def cond(self):
        kw = self.take()
        self.skip({"there", "is"})
        c = self.np()
        c["negated"] = kw == "unless"
        return c

    def clause(self):
        cond = None
        if self.peek() in ("if", "unless"):
            cond = self.cond()
            self.skip(CLOSE)
        neg = False
        self.skip({"do"})
        if self.peek() in NEG:
            self.take()
            neg = True
        if self.peek() not in VERBS:
            raise Reject("verb")
        act = VERBS[self.take()]
        op = {"action": act, "object": self.np()}
        if act == "move":
            if self.peek() not in DIRS:
                raise Reject("direction")
            op["direction"] = DIRS[self.take()]
            op["count"] = 1
            if self.peek() in NUMS:
                op["count"] = NUMS[self.take()]
            self.skip(FILL)
        elif act == "paint":
            if self.peek() not in COLORS:
                raise Reject("color")
            op["to_color"] = self.take()
        op["negated"] = neg
        j = self.i
        # a comma may precede a suffix condition; a full stop or semicolon ends the sentence, so a following
        # "if" belongs to the NEXT clause
        if self.peek() == "," and self.t[self.i + 1:self.i + 2] and self.t[self.i + 1] in ("if", "unless"):
            self.i += 1
        if self.peek() in ("if", "unless"):
            if cond:
                raise Reject("two conditions")
            cond = self.cond()
        else:
            self.i = j
        if cond:
            op["condition"] = cond
        return op


def parse(text):
    toks = tokenize(text)
    p = P(toks)
    ops = []
    try:
        if not toks:
            raise Reject("empty")
        p.skip(CLOSE | SEQ)
        while p.peek() is not None:
            if p.peek() in ("after", "before"):
                kind = p.take()
                a = p.clause()
                p.skip(CLOSE)
                b = p.clause()
                ops += [a, b] if kind == "after" else [b, a]
            else:
                a = p.clause()
                if p.peek() in ("after", "before"):
                    kind = p.take()
                    b = p.clause()
                    ops += [b, a] if kind == "after" else [a, b]
                else:
                    ops.append(a)
            if p.peek() is None:
                break
            if p.peek() not in CLOSE | SEQ:
                raise Reject("junk")
            p.skip(CLOSE | SEQ)
            if p.peek() is None and toks[-1] in SEQ:
                raise Reject("dangling connector")
        if not ops:
            raise Reject("no operation")
        return True, ops
    except Reject:
        return False, []
