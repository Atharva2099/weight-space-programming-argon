"""Rescore saved benchmark JSONL files: answer accuracy and exit type.

Exit types (judged on the saved 180-char answer sample after the first </think>):
  clean   - no further </think>
  stutter - extra </think>, but an answer was still extracted
  loop    - extra </think> and no answer extracted
Correct = saved `pred` equals `gt` as a number (after removing , and $), else exact text match ignoring spaces.
Runs that never exit get no answer, so they always score 0. Text inside <think> was not saved and is not scored.
"""

import json
from collections import Counter, defaultdict

FILES = ["benchmark_results_3way.jsonl", "phase2_benchmark_results.jsonl"]


def num(s):
    try:
        return float(str(s).replace(",", "").replace("$", "").strip().rstrip("."))
    except ValueError:
        return None


def correct(pred, gt):
    p, g = num(pred), num(gt)
    if p is not None and g is not None:
        return abs(p - g) < 1e-6
    return pred != "" and pred.replace(" ", "") == gt.replace(" ", "")


def exit_type(r):
    if not r["exited"]:
        return "no_exit"
    if r["resp_sample"].count("</think>") == 0:
        return "clean"
    return "stutter" if r["pred"] else "loop"


summary = defaultdict(Counter)
for f in FILES:
    for line in open(f):
        r = json.loads(line)
        c = summary[f"{r['condition']}/{r['dataset']}"]
        c["n"] += 1
        c[exit_type(r)] += 1
        c["correct"] += correct(r["pred"], r["gt"])

with open("rescore_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
for k, c in summary.items():
    print(k, dict(c))
