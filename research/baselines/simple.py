"""Two deliberately simple comparison baselines, reported as such.

keyword: a bag-of-keywords intent classifier that emits ONE operation from the first keyword of each kind.
lookup : memorises the development sentences and answers only for exact matches (a phrase table).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import symbolic as S


def keyword(text):
    toks = S.tokenize(text)
    act = next((S.VERBS[t] for t in toks if t in S.VERBS), None)
    typ = next((S.TYPES[t] for t in toks if t in S.TYPES), None)
    cols = [t for t in toks if t in S.COLORS]
    d = next((S.DIRS[t] for t in toks if t in S.DIRS), None)
    n = next((S.NUMS[t] for t in toks if t in S.NUMS), 1)
    if not act or not typ:
        return False, []
    op = {"action": act, "object": {"type": typ}, "negated": any(t in S.NEG for t in toks)}
    if cols:
        op["object"]["color"] = cols[0]
    if act == "move":
        if not d:
            return False, []
        op["direction"], op["count"] = d, n
    if act == "paint":
        if len(cols) < 2:
            return False, []
        op["object"]["color"], op["to_color"] = cols[0], cols[-1]
    return True, [op]


class Lookup:
    def __init__(self, dev_path):
        self.table = {}
        for line in Path(dev_path).read_text().splitlines():
            c = json.loads(line)
            if c["expect"] == "accept":
                self.table[re.sub(r"\W+", " ", c["text"].lower()).strip()] = c["gold_ops"]

    def __call__(self, text):
        k = re.sub(r"\W+", " ", text.lower()).strip()
        return (True, self.table[k]) if k in self.table else (False, [])
