"""Outcome measures for the arbitration trials.

Primary measure is the harmful revision rate (HRR), the fraction of initially
correct items a perturbation flips. We also report the beneficial revision rate,
the probability mass moved onto the pushed option, and the final confidence on
flips, with Wilson score intervals.
"""
import math
import statistics


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, center - half), min(1.0, center + half)


def trial_outcomes(a0, a1, gt, prob_target0, prob_target1):
    """Per-trial outcome flags and probability shift.

    a0, a1: Round-1 / Round-2 argmax answers. gt: gold label.
    prob_target*: probability on the pushed option before / after.
    """
    return {
        "correct0": int(a0 == gt),
        "harmful": int(a0 == gt and a1 != gt),
        "beneficial": int(a0 != gt and a1 == gt),
        "revised": int(a1 != a0),
        "delta_target_prob": prob_target1 - prob_target0,
    }


def summarize(rows, conditions):
    """Aggregate per-condition HRR / BenR / mean probability shift / revision rate.

    rows: dicts with keys condition, correct0, harmful, beneficial, revised,
          delta_target_prob (e.g. from trial_outcomes).
    """
    summary = []
    for cond in conditions:
        sub = [r for r in rows if r["condition"] == cond]
        csub = [r for r in sub if r["correct0"] == 1]
        wsub = [r for r in sub if r["correct0"] == 0]
        h = sum(r["harmful"] for r in csub)
        b = sum(r["beneficial"] for r in wsub)
        hp, hlo, hhi = wilson(h, len(csub))
        bp, blo, bhi = wilson(b, len(wsub))
        summary.append({
            "condition": cond,
            "harmful_rate": hp,
            "harmful_lo": hlo,
            "harmful_hi": hhi,
            "harmful_n": len(csub),
            "beneficial_rate": bp,
            "beneficial_n": len(wsub),
            "mean_delta_target_prob": (
                statistics.mean([r["delta_target_prob"] for r in sub]) if sub else float("nan")
            ),
            "revision_rate": (sum(r["revised"] for r in sub) / len(sub)) if sub else float("nan"),
        })
    return summary
