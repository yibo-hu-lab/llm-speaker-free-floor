#!/usr/bin/env python
"""Run one condition grid for one model on one item file.

For each item: read Round-1, pick the pushed wrong target (the model's own top
non-gold option), then for every condition in the grid insert the perturbation and
read Round-2. Writes a per-trial long CSV and a per-condition summary CSV.

Grids:
    mechanism   framing x pressure shape + baselines     (main results; --target random for the random-target check)
    role        role ladder x pressure shape + baselines (role labels)
    dose        repeated vs distinct, N in {1,2,3,6}      (statement count)
    controls    invalid-label placebo and three containers
    paraphrase  six differently worded assertions vs. no-source
    prefix      fixed-answer source-prefix probe (no preamble)
    preamble    expert tag without the panel preamble
    userattr    "I think the answer is X." with and without the preamble

Usage:
    python run_arbitration.py --model Qwen/Qwen2.5-7B-Instruct \
        --items data/arc_challenge.jsonl --grid mechanism \
        --out-prefix out/qwen7b_arc_challenge --allow-download
    (drop --allow-download when the weights are already cached locally)

Input schema (one JSON object per line):
    {"original_instance_ID": "...", "question": "...",
     "options": [{"label": "(A)", "text": "..."}, ...],
     "answer": {"label": "(A)", "text": "..."}}
"""
import argparse
import csv
import json
import os
import random

from speaker_free.arbitration import (
    letter_ids, load_model, messages_round1, messages_round2, read_dist, top_wrong,
)
from speaker_free.metrics import summarize, trial_outcomes
from speaker_free.perturbations import (
    BASELINES, FRAMINGS, ROLE_FRAMINGS, condition_text, controls_conditions, controls_text,
    dose_conditions, dose_text, grid_conditions, paraphrase_conditions, paraphrase_text,
    parse_condition, preamble_conditions, preamble_text, prefix_conditions, prefix_text,
    role_condition_text, role_conditions, userattr_conditions, userattr_text,
)

# grid name -> (condition list, text builder, framings used to split condition names)
# Text builders take (condition, gt, wrong, variant, letters).
GRIDS = {
    "mechanism": (grid_conditions, lambda c, g, w, v, l: condition_text(c, g, w, v), FRAMINGS),
    "role": (role_conditions, lambda c, g, w, v, l: role_condition_text(c, g, w, v), ROLE_FRAMINGS),
    "dose": (dose_conditions, lambda c, g, w, v, l: dose_text(c, w), None),
    "controls": (controls_conditions, controls_text, "all_wrong"),
    "paraphrase": (paraphrase_conditions, paraphrase_text, "all_wrong"),
    "prefix": (prefix_conditions, prefix_text, "all_wrong"),
    "preamble": (preamble_conditions, preamble_text, "all_wrong"),
    "userattr": (userattr_conditions, userattr_text, "all_wrong"),
}

LONG_FIELDS = [
    "model", "dataset", "grid", "target_mode", "variant", "id", "gt", "wrong_target", "a0", "correct0",
    "init_conf", "condition", "structure", "framing", "a1", "conf",
    "target_prob0", "target_prob1", "delta_target_prob",
    "gt_prob0", "gt_prob1", "delta_gt_prob",
    "harmful", "beneficial", "revised",
]


def load_items(path):
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def option_letter(label):
    """Extract the single option character from a label like '(A)', 'A' or '(1)'."""
    s = label.strip().strip("()").strip()
    if len(s) != 1 or not s.isalnum():
        raise ValueError(f"option label must be a single letter or digit, got {label!r}")
    return s.upper()


def split_condition(condition, framings):
    if framings == "all_wrong":  # single-structure control grids
        if condition in BASELINES:
            return "baseline", condition
        return "all_wrong", condition.replace("all_wrong_", "", 1)
    if framings is None:
        return ("baseline", condition) if condition in ("neutral_reask", "length_control") else ("dose", condition)
    return parse_condition(condition, framings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path (instruction-tuned).")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--items", required=True, help="Normalized multiple-choice JSONL.")
    ap.add_argument("--grid", default="mechanism", choices=sorted(GRIDS))
    ap.add_argument("--variant", type=int, default=0, choices=[0, 1, 2],
                    help="Order of the six asserted lines: 0 identity, 1 reversed, 2 interleaved.")
    ap.add_argument("--target", default="top", choices=["top", "random"],
                    help="Wrong target: the model's top non-gold option (paper default) or a random non-gold option.")
    ap.add_argument("--seed", type=int, default=0, help="Seed for --target random.")
    ap.add_argument("--limit", type=int, default=0, help="Use only the first N items (0 = all).")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--allow-download", action="store_true",
                    help="Allow downloading weights (default: local files only).")
    args = ap.parse_args()

    conditions_fn, text_fn, framings = GRIDS[args.grid]
    conditions = conditions_fn()
    items = load_items(args.items)
    if args.limit:
        items = items[:args.limit]
    dataset = os.path.splitext(os.path.basename(args.items))[0]
    model, tok, device = load_model(args.model, args.tokenizer, local_files_only=not args.allow_download)
    model_label = os.path.basename(args.model.rstrip("/"))

    rows = []
    for idx, item in enumerate(items, 1):
        letters = [option_letter(opt["label"]) for opt in item["options"]]
        lid = letter_ids(tok, letters)
        options_text = "\n".join(f"({letter}) {opt['text']}" for letter, opt in zip(letters, item["options"]))
        m1 = messages_round1(item["question"], options_text, args.model)
        lp0 = read_dist(model, tok, m1, letters, lid, device)
        gt = f"({option_letter(item['answer']['label'])})"
        assert gt in lp0["dist"], (
            f"gold label {gt} not among option letters {letters} for item {item['original_instance_ID']}"
        )
        a0 = lp0["answer"]
        if args.target == "random":
            candidates = [label for label in lp0["dist"] if label != gt]
            wrong = random.Random(f"{args.seed}|{item['original_instance_ID']}").choice(candidates)
        else:
            wrong = top_wrong(lp0["dist"], gt)
        for condition in conditions:
            structure, framing = split_condition(condition, framings)
            pressure_text = text_fn(condition, gt, wrong, args.variant, letters)
            m2 = messages_round2(m1, a0, pressure_text, item["question"], options_text)
            lp1 = read_dist(model, tok, m2, letters, lid, device)
            oc = trial_outcomes(a0, lp1["answer"], gt, lp0["dist"][wrong], lp1["dist"][wrong])
            rows.append({
                "model": model_label, "dataset": dataset, "grid": args.grid, "target_mode": args.target,
                "variant": args.variant,
                "id": item["original_instance_ID"], "gt": gt,
                "wrong_target": wrong, "a0": a0, "correct0": oc["correct0"],
                "init_conf": round(lp0["conf"], 6), "condition": condition,
                "structure": structure, "framing": framing, "a1": lp1["answer"],
                "conf": round(lp1["conf"], 6),
                "target_prob0": round(lp0["dist"][wrong], 6),
                "target_prob1": round(lp1["dist"][wrong], 6),
                "delta_target_prob": round(oc["delta_target_prob"], 6),
                "gt_prob0": round(lp0["dist"][gt], 6),
                "gt_prob1": round(lp1["dist"][gt], 6),
                "delta_gt_prob": round(lp1["dist"][gt] - lp0["dist"][gt], 6),
                "harmful": oc["harmful"], "beneficial": oc["beneficial"], "revised": oc["revised"],
            })
        if idx % 25 == 0 or idx == len(items):
            print(f"[{model_label} {dataset} {args.grid}] {idx}/{len(items)} items", flush=True)

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    with open(args.out_prefix + "_long.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LONG_FIELDS)
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows, conditions)
    with open(args.out_prefix + "_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"done -> {args.out_prefix}_long.csv / _summary.csv ({len(rows)} trials)")


if __name__ == "__main__":
    main()
