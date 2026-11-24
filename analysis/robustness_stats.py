#!/usr/bin/env python
"""Mixed-effects model, random-slope and covariate variants, cell-level contrast,
and sign test for experts vs. no-source.

Usage:
    python analysis/robustness_stats.py --runs out/mechanism
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEEDED = ["model", "dataset", "variant", "id", "correct0", "init_conf",
          "structure", "framing", "harmful"]


def load_grid(runs_dir, variant=0):
    paths = sorted(p for p in glob.glob(os.path.join(runs_dir, "**", "*_long.csv"), recursive=True)
                   if ".partial." not in os.path.basename(p))
    if not paths:
        sys.exit(f"no *_long.csv files under {runs_dir}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        missing = [c for c in NEEDED if c not in df.columns]
        if missing:
            sys.exit(f"{p} lacks columns {missing}")
        if "grid" in df.columns:
            df = df[df["grid"] == "mechanism"]
        if "target_mode" in df.columns:          # exclude random-target runs if present
            df = df[df["target_mode"].fillna("top") == "top"]
        frames.append(df)
    d = pd.concat(frames, ignore_index=True)
    d = d[d["variant"].astype(str) == str(variant)].copy()
    dup = d.duplicated(["model", "dataset", "id", "condition"]).sum()
    if dup:
        sys.exit(f"{dup} duplicate (model, dataset, id, condition) rows: pass one seed per cell")
    return d, len(paths)


def option_counts(items_dir):
    """(dataset, id) -> number of options, from the item files."""
    out = {}
    for p in glob.glob(os.path.join(items_dir, "*.jsonl")):
        ds = os.path.splitext(os.path.basename(p))[0]
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    out[(ds, str(r["original_instance_ID"]))] = len(r["options"])
    return out


def cluster_ci(cells):
    cells = np.asarray(cells, float)
    n = len(cells)
    se = cells.std(ddof=1) / np.sqrt(n)
    tc = stats.t.ppf(.975, n - 1)
    return cells.mean(), cells.mean() - tc * se, cells.mean() + tc * se, n


def cell_deltas(df, a="experts", b="nosource", min_n=1):
    out = []
    for _, s in df.groupby(["model", "dataset"]):
        sa, sb = s[s.framing == a], s[s.framing == b]
        if len(sa) >= min_n and len(sb) >= min_n:
            out.append((sa.harmful.mean() - sb.harmful.mean()) * 100)
    return out


def fit(formula, vc, data, term):
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    res = BinomialBayesMixedGLM.from_formula(formula, vc, data).fit_vb()
    names = list(res.model.exog_names)
    terms = [term] if isinstance(term, str) else term
    out = {}
    for t in terms:
        i = names.index(t)
        b, s = res.fe_mean[i], res.fe_sd[i]
        out[t] = (b, s, math.exp(b), math.exp(b - 1.96 * s), math.exp(b + 1.96 * s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="Directory with mechanism-grid *_long.csv files.")
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--items-dir", default=os.path.join(PKG, "data"),
                    help="Item files, used only for the option-count covariate.")
    ap.add_argument("--item-effect", default="id", choices=["id", "dataset_id"],
                    help="Item random intercept grouping. 'id' (paper) groups by item id alone, "
                         "so equal ids in different datasets share an intercept; "
                         "'dataset_id' keeps them separate.")
    ap.add_argument("--skip-slopes", action="store_true")
    ap.add_argument("--skip-covariates", action="store_true")
    a = ap.parse_args()

    d, nfiles = load_grid(a.runs, a.variant)
    d["id"] = d["id"].astype(str)
    if a.item_effect == "dataset_id":
        d["id"] = d["dataset"] + ":" + d["id"]
    aw = d[(d.structure == "all_wrong") & (d.correct0 == 1)].copy()
    print(f"files={nfiles}  models={d.model.nunique()}  datasets={d.dataset.nunique()}  "
          f"cells={d.groupby(['model', 'dataset']).ngroups}  all-wrong initially-correct trials={len(aw)}")
    ns_ex = aw[aw.framing.isin(["nosource", "experts"])].copy()
    ns_ex["fr"] = (ns_ex.framing == "experts").astype(int)
    p_ns = ns_ex[ns_ex.fr == 0].harmful.mean()
    p_ex = ns_ex[ns_ex.fr == 1].harmful.mean()
    print(f"pooled HRR: no-source {p_ns * 100:.1f}%  experts {p_ex * 100:.1f}%  "
          f"difference {(p_ex - p_ns) * 100:+.1f} pp")

    # 1. sign test across models
    print("\n[1] sign test across models (experts - no-source)")
    deltas = {m: (s[s.fr == 1].harmful.mean() - s[s.fr == 0].harmful.mean()) * 100
              for m, s in ns_ex.groupby("model")}
    for m, v in deltas.items():
        print(f"    {m:28s} {v:+.1f} pp")
    k, n = sum(v > 0 for v in deltas.values()), len(deltas)
    p2 = stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue
    p1 = stats.binomtest(k, n, 0.5, alternative="greater").pvalue
    print(f"    {k}/{n} positive; exact p two-sided={p2:.4f}, one-sided={p1:.4f}")

    # 2. cell-level contrast
    cells = cell_deltas(ns_ex)
    m, lo, hi, nc = cluster_ci(cells)
    print(f"\n[2] cell-level (model x dataset) experts - no-source: {m:+.1f} pp "
          f"95% CI [{lo:+.1f},{hi:+.1f}], {sum(x > 0 for x in cells)}/{nc} cells positive, "
          f"t-test p={stats.ttest_1samp(cells, 0).pvalue:.2e}")

    # 3. off-ceiling
    off = ns_ex[ns_ex.init_conf < 0.9]
    gap = cell_deltas(off, min_n=5)
    m, lo, hi, nc = cluster_ci(gap)
    print(f"\n[3] off-ceiling (init_conf < 0.9) cell-level: {m:+.1f} pp 95% CI [{lo:+.1f},{hi:+.1f}], "
          f"{sum(x > 0 for x in gap)}/{nc} cells positive, p={stats.ttest_1samp(gap, 0).pvalue:.2e}")

    # 4. GLMM, random intercepts
    vc = {"model": "0+C(model)", "dataset": "0+C(dataset)", "item": "0+C(id)"}
    b, s, orr, olo, ohi = fit("harmful ~ fr", vc, ns_ex, "fr")["fr"]
    l0 = math.log(p_ns / (1 - p_ns))
    inv = lambda x: 1 / (1 + math.exp(-x))  # noqa: E731
    print(f"\n[4] GLMM (random intercepts: model, dataset, item): experts log-odds {b:.3f} "
          f"(sd {s:.3f}), OR {orr:.2f} [{olo:.2f},{ohi:.2f}]")
    print(f"    implied marginal {p_ns * 100:.1f}% -> {inv(l0 + b) * 100:.1f}% "
          f"({(inv(l0 + b) - p_ns) * 100:+.1f} pp, CI [{(inv(l0 + b - 1.96 * s) - p_ns) * 100:+.1f},"
          f"{(inv(l0 + b + 1.96 * s) - p_ns) * 100:+.1f}])")

    # 5. random slopes
    if not a.skip_slopes:
        vc1 = {"model": "0+C(model)", "model_fr": "0+C(model):fr",
               "dataset": "0+C(dataset)", "item": "0+C(id)"}
        b, s, orr, olo, ohi = fit("harmful ~ fr", vc1, ns_ex, "fr")["fr"]
        print(f"\n[5] GLMM + framing x model random slope: experts log-odds {b:.3f} (sd {s:.3f}), "
              f"OR {orr:.2f} [{olo:.2f},{ohi:.2f}]")

    # 6. covariate-adjusted, all four framings
    if not a.skip_covariates:
        nopt = option_counts(a.items_dir)
        aw["n_options"] = [nopt.get((ds, i)) for ds, i in
                           zip(aw["dataset"], aw["id"].str.split(":").str[-1])]
        miss = aw["n_options"].isna().sum()
        if miss:
            sys.exit(f"{miss} trials have no option count in {a.items_dir}")
        print("\n[6] GLMM + covariates (initial top-option probability, option count); "
              "one model per framing vs no-source:")
        for fr in ("experts", "people", "richpeers"):
            sub = aw[aw.framing.isin(["nosource", fr])].copy()
            sub["fr"] = (sub.framing == fr).astype(int)
            b, s, orr, olo, ohi = fit("harmful ~ fr + init_conf + n_options", vc, sub, "fr")["fr"]
            print(f"    {fr:10s} log-odds {b:+.3f} (sd {s:.3f})  OR {orr:.2f} [{olo:.2f},{ohi:.2f}]")


if __name__ == "__main__":
    main()
