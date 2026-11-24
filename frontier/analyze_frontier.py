#!/usr/bin/env python
"""Harmful revision rates and paired contrasts for the frontier-model runs.

Usage:
    python frontier/analyze_frontier.py --results out/frontier
"""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

from scipy import stats

ARMS = ["neutral_reask", "length_control", "nosource", "bare",
        "userattr", "userattr_pre", "people", "experts"]

LABELS = {
    "neutral_reask": "plain re-ask",
    "length_control": "length control",
    "nosource": "no-source (with preamble)",
    "bare": "no-source, no preamble",
    "userattr": "\"I think the answer is X\", no preamble",
    "userattr_pre": "\"I think the answer is X\" (with preamble)",
    "people": "people",
    "experts": "expert panel",
}

CONTRASTS = [
    ("nosource", "length_control"),
    ("nosource", "neutral_reask"),
    ("length_control", "neutral_reask"),
    ("bare", "nosource"),
    ("userattr", "bare"),
    ("userattr_pre", "nosource"),
    ("people", "nosource"),
    ("experts", "nosource"),
    ("people", "bare"),
    ("experts", "bare"),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def load(results_dir):
    docs = []
    for f in sorted(glob.glob(str(Path(results_dir) / "*.json"))):
        if Path(f).name.startswith("SMOKE_"):
            continue
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        if isinstance(d, dict) and "meta" in d and "rows" in d:
            docs.append(d)
    return docs


def check_flags(rows):
    """Recompute the outcome flags from the answers and compare with the stored ones."""
    for r in rows:
        ok = r["parsed"] == 1
        assert r["correct0"] == int(r["a0"] == r["gt"]), r
        assert r["harmful"] == int(ok and r["correct0"] == 1 and r["a1"] != r["gt"]), r
        assert r["beneficial"] == int(ok and r["correct0"] == 0 and r["a1"] == r["gt"]), r
        assert r["revised"] == int(ok and r["a1"] != r["a0"]), r
        assert r["hit_target"] == int(ok and r["a1"] == r["wrong_target"]), r


def mcnemar(rows_a, rows_b):
    """Paired difference A - B over the shared initially correct, parsed items."""
    A = {r["id"]: r for r in rows_a if r["correct0"] == 1 and r["parsed"] == 1}
    B = {r["id"]: r for r in rows_b if r["correct0"] == 1 and r["parsed"] == 1}
    ids = sorted(set(A) & set(B))
    n = len(ids)
    if n == 0:
        return None
    b = sum(1 for i in ids if A[i]["harmful"] == 1 and B[i]["harmful"] == 0)
    c = sum(1 for i in ids if A[i]["harmful"] == 0 and B[i]["harmful"] == 1)
    ka = sum(A[i]["harmful"] for i in ids)
    kb = sum(B[i]["harmful"] for i in ids)
    p = 1.0 if b + c == 0 else float(stats.binomtest(b, b + c, 0.5).pvalue)
    if b + c > 0:
        se = math.sqrt((b + c) - (b - c) ** 2 / n) / n
        lo, hi = 100 * ((b - c) / n - 1.96 * se), 100 * ((b - c) / n + 1.96 * se)
    else:
        lo = hi = 0.0
    return dict(n=n, hrr_a=100 * ka / n, hrr_b=100 * kb / n, delta=100 * (ka - kb) / n,
                lo=lo, hi=hi, b=b, c=c, p=p)


def fmt_p(p):
    return f"{p:.3f}" if p >= 0.001 else f"{p:.2e}"


def report(docs):
    lines = []
    P = lines.append
    P("# Frontier-model check")
    P("")
    P("Text-parsed answers, one sampled response per item and arm; random wrong target.")
    P("")
    P("## Runs")
    P("")
    P("| model requested | resolved snapshot | dataset | effort | raw N | errors |")
    P("|---|---|---|---|---|---|")
    for d in docs:
        m = d["meta"]
        P(f"| `{m['model_requested']}` | `{m['model_resolved']}` | {m['dataset']} | "
          f"`{m['reasoning_effort']}` | {m['n_raw']} | {m['errors']} |")
    P("")

    P("## Harmful revision rate by arm")
    P("")
    P("| model | dataset | arm | condition | raw N | R1 acc | init-correct | HRR | 95% CI | "
      "hit-target rate | parse fail |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for d in docs:
        m, rows = d["meta"], d["rows"]
        ids = {r["id"] for r in rows}
        corr = {r["id"] for r in rows if r["correct0"] == 1}
        r1acc = 100 * len(corr) / len(ids) if ids else float("nan")
        for arm in ARMS:
            s = [r for r in rows if r["arm"] == arm and r["correct0"] == 1]
            ok = [r for r in s if r["parsed"] == 1]
            if not ok:
                continue
            h = sum(r["harmful"] for r in ok)
            t = sum(r["hit_target"] for r in ok)
            lo, hi = wilson(h, len(ok))
            P(f"| `{m['model_requested']}` | {m['dataset']} | {arm} | {LABELS[arm]} | "
              f"{m['n_raw']} | {r1acc:.1f}% | {len(ok)} | **{100 * h / len(ok):.1f}%** | "
              f"[{lo:.1f},{hi:.1f}] | {100 * t / len(ok):.1f}% | {len(s) - len(ok)} |")
    P("")

    P("## Paired contrasts (exact McNemar; b/c = discordant items)")
    P("")
    P("| model | dataset | contrast | HRR A | HRR B | delta | 95% CI | b | c | p | pairs |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for d in docs:
        m, rows = d["meta"], d["rows"]
        for a, b_ in CONTRASTS:
            res = mcnemar([r for r in rows if r["arm"] == a], [r for r in rows if r["arm"] == b_])
            if not res:
                continue
            P(f"| `{m['model_requested']}` | {m['dataset']} | {a} − {b_} | "
              f"{res['hrr_a']:.1f}% | {res['hrr_b']:.1f}% | **{res['delta']:+.1f} pp** | "
              f"[{res['lo']:+.1f},{res['hi']:+.1f}] | {res['b']} | {res['c']} | "
              f"{fmt_p(res['p'])} | {res['n']} |")
    P("")

    P("## Pairing check")
    P("")
    for d in docs:
        m, rows = d["meta"], d["rows"]
        per_arm = {arm: {r["id"] for r in rows if r["arm"] == arm and r["correct0"] == 1}
                   for arm in ARMS}
        sets = [v for v in per_arm.values() if v]
        same = all(s == sets[0] for s in sets) if sets else False
        P(f"- `{m['model_requested']}` / {m['dataset']}: initially correct set identical "
          f"across {len(sets)} arms: **{same}** (n={len(sets[0]) if sets else 0})")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="out/frontier",
                    help="Directory with the run JSON files from run_frontier_api.py.")
    ap.add_argument("--out", default=None, help="Optional Markdown output path.")
    a = ap.parse_args()
    docs = load(a.results)
    if not docs:
        sys.exit(f"no run files found in {a.results}")
    for d in docs:
        check_flags(d["rows"])
    text = report(docs)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
