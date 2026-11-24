#!/usr/bin/env python
"""Message-role check: the same six no-source statements inline, in a separate
user turn, or in a separate system turn.

Usage:
    python run_message_role.py --model Qwen/Qwen2.5-7B-Instruct \
        --items data/arc_challenge.jsonl --out-prefix out/message_role/qwen7b_arc --allow-download
"""
import argparse
import csv
import json
import os
import re

from run_arbitration import LONG_FIELDS, load_items
from speaker_free.arbitration import (
    letter_ids, load_model, messages_round1, messages_round2, read_dist, top_wrong,
)
from speaker_free.metrics import summarize, trial_outcomes
from speaker_free.perturbations import BASELINES, baseline_text, grid_text

BLOCK_ARMS = ["user_inline", "user_split", "system_split"]
ARMS = BASELINES + BLOCK_ARMS

# Role of the turn that carries the six statements.
BLOCK_ROLE = {"user_inline": "user", "user_split": "user", "system_split": "system"}

# Turn headers of the chat templates we check (Qwen/ChatML, Llama-3, gemma).
TURN_RE = re.compile(r"<\|im_start\|>(\w+)"
                     r"|<\|start_header_id\|>(\w+)<\|end_header_id\|>"
                     r"|<start_of_turn>(\w+)")

ROLE_FIELDS = LONG_FIELDS[:LONG_FIELDS.index("condition") + 1] + ["block_role"] + \
    LONG_FIELDS[LONG_FIELDS.index("condition") + 1:]


def option_key(label):
    """Option key from a label like '(A)' or '(1)'."""
    s = label.strip().strip("()").strip()
    if len(s) != 1 or not s.isalnum():
        raise ValueError(f"option label must be a single letter or digit, got {label!r}")
    return s.upper()


def block_text(gt, wrong, variant=0):
    """The main grid's no-source block (all_wrong_nosource)."""
    return grid_text("all_wrong", "nosource", gt, wrong, variant)


def messages_for_arm(arm, round1_messages, a0, gt, wrong, question, options_text, variant=0):
    """Round-2 messages for one arm. Only the turn that carries the block varies."""
    if arm in BASELINES:
        return messages_round2(round1_messages, a0, baseline_text(arm), question, options_text)
    block = block_text(gt, wrong, variant)
    inline = messages_round2(round1_messages, a0, block, question, options_text)
    if arm == "user_inline":
        return inline
    # Split the last user turn at the block boundary: the block becomes its own turn
    # and the rest (re-ask + question + options) stays as the next user turn.
    last = inline[-1]["content"]
    assert last.startswith(block + "\n\n")
    reask = last[len(block) + 2:]
    return inline[:-1] + [
        {"role": BLOCK_ROLE[arm], "content": block},
        {"role": "user", "content": reask},
    ]


def check_template(tok, model_name, question="Q?", options_text="(A) x\n(B) y"):
    """Render each block arm with the model's own template. Return (eligible, report)."""
    eligible, report = [], {}
    r1 = messages_round1(question, options_text, model_name)
    head_line = block_text("(A)", "(B)").split("\n")[0]
    for arm in BLOCK_ARMS:
        msgs = messages_for_arm(arm, r1, "(A)", "(A)", "(B)", question, options_text)
        try:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            report[arm] = f"REJECTED by template: {type(e).__name__}: {str(e)[:120]}"
            continue
        if head_line not in s:
            report[arm] = "DROPPED: block text absent from rendered prompt"
            continue
        last = None
        for m in TURN_RE.finditer(s[:s.find(head_line)]):
            found = [x for x in m.groups() if x]
            if found:
                last = found[0]
        if last != BLOCK_ROLE[arm]:
            report[arm] = f"INELIGIBLE: block rendered under role `{last}`, wanted `{BLOCK_ROLE[arm]}`"
            continue
        report[arm] = f"OK: block under role `{last}`"
        eligible.append(arm)
    return eligible, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path (instruction-tuned).")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--items", help="Normalized multiple-choice JSONL.")
    ap.add_argument("--variant", type=int, default=0, choices=[0, 1, 2],
                    help="Order of the six asserted lines (all identical here, so 0 suffices).")
    ap.add_argument("--limit", type=int, default=0, help="Use only the first N items (0 = all).")
    ap.add_argument("--out-prefix")
    ap.add_argument("--allow-download", action="store_true",
                    help="Allow downloading weights (default: local files only).")
    ap.add_argument("--check-template", action="store_true",
                    help="Load only the tokenizer, print the template check, and exit.")
    args = ap.parse_args()

    if args.check_template:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model,
                                            local_files_only=not args.allow_download)
        eligible, report = check_template(tok, args.model)
        print(json.dumps({"model": args.model, "eligible": eligible, "report": report}, indent=2))
        return
    if not args.items or not args.out_prefix:
        ap.error("--items and --out-prefix are required unless --check-template is given")

    items = load_items(args.items)
    if args.limit:
        items = items[:args.limit]
    dataset = os.path.splitext(os.path.basename(args.items))[0]
    model, tok, device = load_model(args.model, args.tokenizer, local_files_only=not args.allow_download)
    model_label = os.path.basename(args.model.rstrip("/"))

    eligible, report = check_template(tok, args.model)
    print(f"[{model_label}] template check: {json.dumps(report, indent=2)}", flush=True)
    if "user_inline" not in eligible or "user_split" not in eligible:
        raise SystemExit("both user-turn arms must render as user turns; the contrast "
                         "would not be interpretable")
    if "system_split" not in eligible:
        print(f"[{model_label}] system arm not eligible for this template; skipped", flush=True)
    arms = BASELINES + eligible

    rows = []
    for idx, item in enumerate(items, 1):
        letters = [option_key(opt["label"]) for opt in item["options"]]
        lid = letter_ids(tok, letters)
        options_text = "\n".join(f"({letter}) {opt['text']}" for letter, opt in zip(letters, item["options"]))
        m1 = messages_round1(item["question"], options_text, args.model)
        lp0 = read_dist(model, tok, m1, letters, lid, device)
        gt = f"({option_key(item['answer']['label'])})"
        assert gt in lp0["dist"], (
            f"gold label {gt} not among option letters {letters} for item {item['original_instance_ID']}"
        )
        a0 = lp0["answer"]
        wrong = top_wrong(lp0["dist"], gt)
        for arm in arms:
            m2 = messages_for_arm(arm, m1, a0, gt, wrong, item["question"], options_text, args.variant)
            lp1 = read_dist(model, tok, m2, letters, lid, device)
            oc = trial_outcomes(a0, lp1["answer"], gt, lp0["dist"][wrong], lp1["dist"][wrong])
            is_base = arm in BASELINES
            rows.append({
                "model": model_label, "dataset": dataset, "grid": "message_role",
                "variant": args.variant,
                "id": item["original_instance_ID"], "gt": gt,
                "wrong_target": wrong, "a0": a0, "correct0": oc["correct0"],
                "init_conf": round(lp0["conf"], 6), "condition": arm,
                "block_role": "" if is_base else BLOCK_ROLE[arm],
                "structure": arm if is_base else "all_wrong",
                "framing": "baseline" if is_base else "nosource",
                "a1": lp1["answer"], "conf": round(lp1["conf"], 6),
                "target_prob0": round(lp0["dist"][wrong], 6),
                "target_prob1": round(lp1["dist"][wrong], 6),
                "delta_target_prob": round(oc["delta_target_prob"], 6),
                "gt_prob0": round(lp0["dist"][gt], 6),
                "gt_prob1": round(lp1["dist"][gt], 6),
                "delta_gt_prob": round(lp1["dist"][gt] - lp0["dist"][gt], 6),
                "harmful": oc["harmful"], "beneficial": oc["beneficial"], "revised": oc["revised"],
            })
        if idx % 25 == 0 or idx == len(items):
            print(f"[{model_label} {dataset} message_role] {idx}/{len(items)} items", flush=True)

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    with open(args.out_prefix + "_long.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ROLE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows, arms)
    with open(args.out_prefix + "_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"done -> {args.out_prefix}_long.csv / _summary.csv ({len(rows)} trials)")


if __name__ == "__main__":
    main()
