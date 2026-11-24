#!/usr/bin/env python
"""Pool per-run long CSVs into the paper's tables.

Reads every *_long.csv under --runs (written by run_arbitration.py) and writes
table1_mechanism_contrasts.csv, table2_crossmodel.csv, table3_role_control.csv,
table4_dose_curve.csv and table6_controls.csv to --out, in the same format as results/.

HRR is pooled: harmful flips / initially-correct trials, over all models and
datasets present, order variant 0. table6_controls.csv also gives the average of
per-model-dataset rates ("cell average"), which the paper uses for the placebo,
container and off-ceiling rows.

Usage:
    python make_tables.py --runs out --out tables
"""
import argparse
import csv
import glob
import os
import statistics

from speaker_free.metrics import wilson

TABLE1_CONDITIONS = [
    ("neutral_reask", "baseline", "neutral"),
    ("length_control", "baseline", "length"),
] + [
    (f"{s}_{f}", s, f)
    for s in ("all_wrong", "mixed_3_3", "all_correct")
    for f in ("nosource", "people", "richpeers", "experts")
]
TABLE2_CONDITIONS = [
    "all_wrong_experts", "all_wrong_nosource", "all_wrong_people",
    "mixed_3_3_experts", "mixed_3_3_nosource",
]
ROLES = ["nosource", "numbered_people", "named_people", "students", "online_participants", "experts"]


def load(runs_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "**", "*_long.csv"), recursive=True)):
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if str(r.get("variant", "0")) != "0":
                    continue
                for k in ("correct0", "harmful", "beneficial"):
                    r[k] = int(r[k])
                for k in ("delta_target_prob", "delta_gt_prob", "init_conf"):
                    r[k] = float(r[k])
                r.setdefault("target_mode", "top")
                rows.append(r)
    return rows


def hrr(sub):
    c = [r for r in sub if r["correct0"] == 1]
    k = sum(r["harmful"] for r in c)
    p, lo, hi = wilson(k, len(c))
    return p, lo, hi, len(c)


def r4(x):
    return round(x, 4)


def write(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def table1(rows):
    out = []
    for cond, struct, framing in TABLE1_CONDITIONS:
        sub = [r for r in rows if r["condition"] == cond]
        if not sub:
            continue
        p, lo, hi, n = hrr(sub)
        w = [r for r in sub if r["correct0"] == 0]
        ben = sum(r["beneficial"] for r in w) / max(1, len(w))
        # the pushed option is the wrong target, except under all-correct pressure (gold)
        key = "delta_gt_prob" if struct == "all_correct" else "delta_target_prob"
        delta = statistics.mean(r[key] for r in sub)
        out.append({"structure": struct, "framing": framing, "condition": cond, "N_items": n,
                    "HRR": r4(p), "HRR_lo": r4(lo), "HRR_hi": r4(hi),
                    "BenR": r4(ben), "delta_target_prob": r4(delta)})
    return out


def table2(rows):
    out = []
    for model in sorted({r["model"] for r in rows}):
        row = {"model": model}
        for cond in TABLE2_CONDITIONS:
            sub = [r for r in rows if r["model"] == model and r["condition"] == cond]
            row[cond] = r4(hrr(sub)[0]) if sub else ""
        out.append(row)
    return out


def table3(rows):
    out = []
    for struct in ("all_wrong", "mixed_3_3"):
        for role in ROLES:
            sub = [r for r in rows if r["condition"] == f"{struct}_{role}"]
            if not sub:
                continue
            p, lo, hi, n = hrr(sub)
            out.append({"structure": struct, "role": role, "HRR": r4(p),
                        "HRR_lo": r4(lo), "HRR_hi": r4(hi), "N": n})
    return out


def table4(rows):
    out = []
    for cond in sorted({r["condition"] for r in rows}):
        p, lo, hi, n = hrr([r for r in rows if r["condition"] == cond])
        out.append({"condition": cond, "HRR": r4(p), "HRR_lo": r4(lo), "HRR_hi": r4(hi), "N": n})
    return out


def cell_average(sub):
    cells = {}
    for r in sub:
        if r["correct0"] == 1:
            k = cells.setdefault((r["model"], r["dataset"]), [0, 0])
            k[0] += r["harmful"]
            k[1] += 1
    rates = [h / n for h, n in cells.values() if n]
    return (sum(rates) / len(rates) if rates else float("nan")), len(rates)


def table6(rows):
    """Single-turn controls: pooled and cell-averaged all-wrong HRR per condition."""
    groups = []
    top = [r for r in rows if r["target_mode"] == "top"]
    mech = [r for r in top if r["grid"] == "mechanism"]
    groups.append(("mechanism", "all_wrong_nosource",
                   [r for r in mech if r["condition"] == "all_wrong_nosource"]))
    groups.append(("mechanism", "all_wrong_nosource_offceiling",
                   [r for r in mech if r["condition"] == "all_wrong_nosource" and r["init_conf"] < 0.9]))
    groups.append(("mechanism_random_target", "all_wrong_nosource",
                   [r for r in rows if r["grid"] == "mechanism" and r["target_mode"] == "random"
                    and r["condition"] == "all_wrong_nosource"]))
    for grid in ("controls", "paraphrase", "prefix", "preamble", "userattr"):
        sub = [r for r in top if r["grid"] == grid]
        for cond in sorted({r["condition"] for r in sub}):
            groups.append((grid, cond, [r for r in sub if r["condition"] == cond]))
    out = []
    for grid, cond, sub in groups:
        if not sub:
            continue
        p, lo, hi, n = hrr(sub)
        avg, ncell = cell_average(sub)
        out.append({"grid": grid, "condition": cond, "HRR_pooled": r4(p), "HRR_lo": r4(lo),
                    "HRR_hi": r4(hi), "HRR_cell_average": r4(avg), "N": n, "cells": ncell})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="out")
    ap.add_argument("--out", default="tables")
    args = ap.parse_args()

    rows = load(args.runs)
    main_rows = [r for r in rows if r["target_mode"] == "top"]
    by_grid = {g: [r for r in main_rows if r["grid"] == g] for g in ("mechanism", "role", "dose")}
    os.makedirs(args.out, exist_ok=True)

    if by_grid["mechanism"]:
        write(os.path.join(args.out, "table1_mechanism_contrasts.csv"),
              ["structure", "framing", "condition", "N_items", "HRR", "HRR_lo", "HRR_hi", "BenR",
               "delta_target_prob"], table1(by_grid["mechanism"]))
        write(os.path.join(args.out, "table2_crossmodel.csv"),
              ["model"] + TABLE2_CONDITIONS, table2(by_grid["mechanism"]))
    if by_grid["role"]:
        write(os.path.join(args.out, "table3_role_control.csv"),
              ["structure", "role", "HRR", "HRR_lo", "HRR_hi", "N"], table3(by_grid["role"]))
    if by_grid["dose"]:
        write(os.path.join(args.out, "table4_dose_curve.csv"),
              ["condition", "HRR", "HRR_lo", "HRR_hi", "N"], table4(by_grid["dose"]))
    controls = table6(rows)
    if controls:
        write(os.path.join(args.out, "table6_controls.csv"),
              ["grid", "condition", "HRR_pooled", "HRR_lo", "HRR_hi", "HRR_cell_average", "N", "cells"],
              controls)
    print(f"wrote tables for grids: {', '.join(g for g, v in by_grid.items() if v)} -> {args.out}/")


if __name__ == "__main__":
    main()
