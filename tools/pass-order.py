#!/usr/bin/env python3
"""Write one CSV: every pass in the pipeline, in execution order, once per parent.

The trace records one `pass-execution` event per pass per IR unit: a large model
has 1,043 dispatch functions, so each codegen pass fires 1,043 times and the raw
event count runs into the millions. What a reader wants is the pipeline itself:
which pass ran first and how long it took, which passes ran inside it, and so on.

This tool therefore keeps the pipeline *structure*, not the raw events. A pass
that runs many times under the same parent becomes one row whose `executions`
says how many times it ran and whose `total_s` / `self_s` are the sums. The
result for this model is a few hundred rows, one per distinct step of the
pipeline, in execution order:

  seq                     row number in reading order, 1 = first
  depth                   0 = top-level pipeline step, 1.. = nested inside the row above
  parent_seq              seq of the enclosing row, 0 at depth 0
  stage                   pipeline stage (marker written by the driver)
  pass                    pass name
  executions              how many executions this row aggregates
  start_s                 seconds since the first event of the trace
  total_s                 wall time of the step, nested work included (sum over executions)
  self_s                  the step's own time: total minus all nested spans
  other_s                 time in nested non-pass actions (pattern walks,
                          conversion drivers), so that
                          total_s = self_s + Σ (child total_s) + other_s
  ops_before / ops_after  operation count of the scope, at the first and the
                          last execution of the row

Usage:
  pass-order.py --trace traces/hal.json --csv pass-order.csv
"""

import argparse
import json
import sys

ACTION = '"cat":"compile-analysis"'
STAGE_EVENT = '"name":"stage"'
PASS = "pass-execution"


def short(name):
    return name.split("::")[-1] or name


def stage_label(text):
    for suffix in ("-transformation-pipeline", "-pipeline"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


class Node:
    """One distinct step of the pipeline under one parent."""

    __slots__ = ("pass_name", "stage", "depth", "first_ts", "executions",
                 "total_us", "self_us", "other_us", "children", "ops_before",
                 "ops_after", "seq")

    def __init__(self, pass_name, stage, depth, first_ts):
        self.pass_name = pass_name
        self.stage = stage
        self.depth = depth
        self.first_ts = first_ts
        self.executions = 0
        self.total_us = 0
        self.self_us = 0
        self.other_us = 0
        self.children = {}
        self.ops_before = ""
        self.ops_after = ""
        self.seq = 0


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args(argv[1:])

    roots = []
    stack = []              # open spans: dict(node, ts, child_us, is_pass)
    stage = "?"
    origin = None

    with open(args.trace, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if STAGE_EVENT in line:
                event = json.loads(line.strip().rstrip(","))
                stage = stage_label((event.get("args") or {}).get("stage", "?"))
                continue
            if ACTION not in line:
                continue
            event = json.loads(line.strip().rstrip(","))
            phase = event.get("ph")
            ts = event.get("ts", 0)
            if origin is None:
                origin = ts
            if phase == "B":
                fields = event.get("args") or {}
                is_pass = event.get("name") == PASS
                node = None
                if stack:
                    # A pass may sit under a non-pass span (a pattern walk, a
                    # conversion driver); attach it to the nearest pass ancestor
                    # so nothing is silently dropped from the table.
                    parent = None
                    for frame in reversed(stack):
                        if frame["node"] is not None:
                            parent = frame["node"]
                            break
                    if is_pass and parent is not None:
                        name = short(fields.get("pass", "?"))
                        node = parent.children.get(name)
                        if node is None:
                            node = Node(name, stage, parent.depth + 1, ts)
                            node.ops_before = fields.get("opsBefore", "")
                            parent.children[name] = node
                elif is_pass:
                    name = short(fields.get("pass", "?"))
                    node = Node(name, stage, 0, ts)
                    node.ops_before = fields.get("opsBefore", "")
                    roots.append(node)
                stack.append({"node": node, "ts": ts, "child_us": 0,
                              "is_pass": is_pass})
                continue
            if phase != "E" or not stack:
                continue
            frame = stack.pop()
            total = max(0, ts - frame["ts"])
            if stack:
                stack[-1]["child_us"] += total
            node = frame["node"]
            # Only direct children: a non-pass span nested in another non-pass
            # span is already inside that span's total.
            if not frame["is_pass"] and stack and stack[-1]["is_pass"]:
                parent = stack[-1]["node"]
                if parent is not None:
                    parent.other_us += total
            if frame["is_pass"] and node is not None:
                node.executions += 1
                node.total_us += total
                node.self_us += max(0, total - frame["child_us"])
                node.ops_after = (event.get("args") or {}).get("opsAfter", "")

    header = ["seq", "depth", "parent_seq", "stage", "pass", "executions",
              "start_s", "total_s", "self_s", "other_s", "ops_before",
              "ops_after"]
    written = 0
    with open(args.csv, "w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(header) + "\n")
        counter = 0

        def emit(node, parent_seq):
            nonlocal counter, written
            counter += 1
            node.seq = counter
            handle.write(",".join([
                str(node.seq),
                str(node.depth),
                str(parent_seq),
                node.stage,
                node.pass_name,
                str(node.executions),
                f"{(node.first_ts - origin) / 1e6:.3f}",
                f"{node.total_us / 1e6:.3f}",
                f"{node.self_us / 1e6:.3f}",
                f"{node.other_us / 1e6:.3f}",
                str(node.ops_before),
                str(node.ops_after),
            ]) + "\n")
            written += 1
            for child in node.children.values():
                emit(child, node.seq)

        for root in roots:
            emit(root, 0)

    print(f"{args.csv}: {written} rows, {len(roots)} top-level steps")


if __name__ == "__main__":
    main(sys.argv)
