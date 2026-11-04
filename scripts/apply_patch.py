#!/usr/bin/env python3
"""Apply a small, explicit, non-training patch to model/spec.json.

Patch operations: add_feature, set_feature_weight, replace_response.
A backup is written before any non-dry-run change.
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("patch", type=Path)
    parser.add_argument("--spec", type=Path, default=ROOT / "model" / "spec.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    patch = json.loads(args.patch.read_text(encoding="utf-8"))
    intents = {item["name"]: item for item in spec["intents"]}
    operations = patch.get("operations")
    if not isinstance(operations, list) or not operations:
        raise SystemExit("patch must contain a non-empty operations array")

    for number, operation in enumerate(operations, 1):
        op = operation.get("op")
        name = operation.get("intent")
        if name not in intents:
            raise SystemExit(f"operation {number}: unknown intent {name!r}")
        item = intents[name]
        if op == "add_feature":
            phrase, weight = operation.get("phrase"), operation.get("weight")
            if not isinstance(phrase, str) or not phrase.strip():
                raise SystemExit(f"operation {number}: phrase is required")
            if not isinstance(weight, (int, float)) or not 0.1 <= float(weight) <= 20:
                raise SystemExit(f"operation {number}: weight must be in [0.1, 20]")
            if any(existing[0] == phrase for existing in item["features"]):
                raise SystemExit(f"operation {number}: phrase already exists")
            item["features"].append([phrase, float(weight)])
        elif op == "set_feature_weight":
            phrase, weight = operation.get("phrase"), operation.get("weight")
            if not isinstance(weight, (int, float)) or not 0.1 <= float(weight) <= 20:
                raise SystemExit(f"operation {number}: weight must be in [0.1, 20]")
            matches = [feature for feature in item["features"] if feature[0] == phrase]
            if len(matches) != 1:
                raise SystemExit(f"operation {number}: expected one matching phrase")
            matches[0][1] = float(weight)
        elif op == "replace_response":
            response = operation.get("text")
            if not isinstance(response, str) or not response.strip() or "\n" in response:
                raise SystemExit(f"operation {number}: response must be a non-empty single line")
            item["response"] = response
        else:
            raise SystemExit(f"operation {number}: unsupported op {op!r}")

    # Reuse the canonical spec validator before touching the file.
    names = [i["name"] for i in spec["intents"]]
    if len(names) > 32 or len(set(names)) != len(names) or names[-1] != "unknown":
        raise SystemExit("intent list failed validation")
    if sum(len(i["features"]) for i in spec["intents"]) > 256:
        raise SystemExit("feature count exceeds runtime limit")
    if sum(len(i["response"].split()) + 1 for i in spec["intents"]) > 2048:
        raise SystemExit("state count exceeds runtime limit")
    for item in spec["intents"]:
        if (not isinstance(item.get("response"), str) or not item["response"].strip()
                or "\n" in item["response"] or len(item["response"].split()) + 1 > 256):
            raise SystemExit(f"invalid response for intent {item['name']}")
        for feature in item["features"]:
            if (len(feature) != 2 or not isinstance(feature[0], str) or not feature[0].strip()
                    or len(feature[0]) > 180 or not isinstance(feature[1], (int, float))
                    or not 0.1 <= float(feature[1]) <= 20):
                raise SystemExit(f"invalid feature in intent {item['name']}")

    if args.dry_run:
        print(f"Patch valid: {len(operations)} operations; no files changed.")
        return 0
    backup = args.spec.with_suffix(args.spec.suffix + ".bak")
    shutil.copy2(args.spec, backup)
    args.spec.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"Applied {len(operations)} operations. Backup: {backup}")
    print("Next: run `make pipeline` and keep the patch only if the evaluation improves without regressions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
