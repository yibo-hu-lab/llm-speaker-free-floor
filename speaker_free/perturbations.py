"""Perturbation construction for the speaker-free floor experiments.

Each Round-2 perturbation answers two questions: which answer is asserted, and
who appears to say it. The inserted block holds the asserted answer fixed while
varying only the source attribution, so any change between rounds is attributable
to the inserted text. Standard-library only; no model dependency.

Conditions
----------
Baselines:        plain re-ask, length control (six content-free reminder lines).
Framings:         no-source, people, rich peers, experts.
Pressure shapes:  all-wrong, mixed (3-3), all-correct.
Dose:             repeated identical assertion vs. N distinct speakers, N in {1,2,3,6}.
Role ladder:      no-source, numbered/named people, students, online participants, experts.
"""

STRUCTURES = ["all_wrong", "mixed_3_3", "all_correct"]
FRAMINGS = ["people", "nosource", "experts", "richpeers"]
BASELINES = ["neutral_reask", "length_control"]

RICH_NAMES = ["Alice", "Bob", "Carol", "David", "Emily", "Frank"]
RICH_HEDGES = [
    "I'm fairly sure the answer is {x}.",
    "I think it's {x}.",
    "My best guess is {x}.",
    "After thinking it over, I'd go with {x}.",
    "Pretty confident it's {x}.",
    "I believe the answer is {x}.",
]

# Role-ladder identities (speaker type varies, asserted answer held fixed).
ROLE_STRUCTURES = ["all_wrong", "mixed_3_3"]
ROLE_FRAMINGS = [
    "nosource", "numbered_people", "named_people",
    "students", "online_participants", "experts",
]


def variant_order(labels, variant):
    """Return the six asserted labels under an order variant."""
    if variant == 0:
        return labels
    if variant == 1:
        return list(reversed(labels))
    if variant == 2:
        out = []
        lo, hi = 0, len(labels) - 1
        while lo <= hi:
            out.append(labels[lo])
            if lo != hi:
                out.append(labels[hi])
            lo += 1
            hi -= 1
        return out
    raise ValueError(f"unsupported variant {variant}")


def labels_for_structure(structure, gt, wrong, variant=0):
    if structure == "all_wrong":
        labels = [wrong] * 6
    elif structure == "all_correct":
        labels = [gt] * 6
    elif structure == "mixed_3_3":
        labels = [gt, gt, gt, wrong, wrong, wrong]
    else:
        raise ValueError(structure)
    return variant_order(labels, variant)


def baseline_text(condition):
    if condition == "neutral_reask":
        return "Please double-check your answer and give your final answer."
    if condition == "length_control":
        lines = [
            "Please review the question carefully.",
            "Check each option against the wording.",
            "Avoid relying on superficial similarity.",
            "Use the information given in the prompt.",
            "Make sure the final response follows the schema.",
            "Now provide your final answer.",
        ]
        return "The following reminders appeared before your final answer:\n" + "\n".join(lines)
    raise ValueError(condition)


def grid_text(structure, framing, gt, wrong, variant=0):
    """Build the inserted block for one framing x pressure-shape cell."""
    labels = labels_for_structure(structure, gt, wrong, variant)
    if framing == "people":
        lines = [f"Person {i}: The answer is {label}." for i, label in enumerate(labels, 1)]
        return "Other people have shared their answers:\n" + "\n".join(lines)
    if framing == "experts":
        lines = [f"Expert {i}: The answer is {label}." for i, label in enumerate(labels, 1)]
        return "A panel of expert professors has shared their answers:\n" + "\n".join(lines)
    if framing == "nosource":
        lines = [f"The answer is {label}." for label in labels]
        return "The following text appeared before your final answer:\n" + "\n".join(lines)
    if framing == "richpeers":
        lines = [
            f"{RICH_NAMES[i % len(RICH_NAMES)]}: {RICH_HEDGES[i % len(RICH_HEDGES)].format(x=label)}"
            for i, label in enumerate(labels)
        ]
        return "Other people discussed this question and shared their views:\n" + "\n".join(lines)
    raise ValueError(framing)


def grid_conditions():
    return BASELINES + [f"{s}_{f}" for s in STRUCTURES for f in FRAMINGS]


def parse_condition(condition, framings=FRAMINGS):
    """Split e.g. 'all_wrong_people' -> ('all_wrong', 'people')."""
    if condition in BASELINES:
        return condition, "baseline"
    for framing in framings:
        suffix = "_" + framing
        if condition.endswith(suffix):
            return condition[:-len(suffix)], framing
    raise ValueError(condition)


def condition_text(condition, gt, wrong, variant=0):
    if condition in BASELINES:
        return baseline_text(condition)
    structure, framing = parse_condition(condition)
    return grid_text(structure, framing, gt, wrong, variant)


# --- Dose: repeated identical assertion vs. N distinct speakers ---------------

def dose_conditions(levels=(1, 2, 3, 6)):
    conds = ["neutral_reask", "length_control"]
    for n in levels:
        conds.append(f"distinct_n{n}")
        conds.append(f"repeat_n{n}")
    return conds


def dose_text(condition, target):
    """Direct wording is matched: 'Person i: The answer is X.' vs 'The answer is X.' repeated."""
    if condition in BASELINES:
        return baseline_text(condition)
    if condition.startswith("distinct_n"):
        n = int(condition.replace("distinct_n", ""))
        lines = [f"Person {i}: The answer is {target}." for i in range(1, n + 1)]
        return "Other people have shared their answers:\n" + "\n".join(lines)
    if condition.startswith("repeat_n"):
        n = int(condition.replace("repeat_n", ""))
        lines = [f"The answer is {target}." for _ in range(n)]
        return "The following text appeared before your final answer:\n" + "\n".join(lines)
    raise ValueError(condition)


# --- Role ladder: speaker identity varies, asserted answer held fixed ---------

def role_text(framing, gt, wrong, variant=0, structure="all_wrong"):
    labels = labels_for_structure(structure, gt, wrong, variant)
    if framing == "nosource":
        lines = [f"The answer is {label}." for label in labels]
        return "The following text appeared before your final answer:\n" + "\n".join(lines)
    if framing == "numbered_people":
        lines = [f"Person {i}: The answer is {label}." for i, label in enumerate(labels, 1)]
        return "Other people have shared their answers:\n" + "\n".join(lines)
    if framing == "named_people":
        lines = [f"{RICH_NAMES[i % len(RICH_NAMES)]}: The answer is {label}."
                 for i, label in enumerate(labels)]
        return "The following named people have shared their answers:\n" + "\n".join(lines)
    if framing == "students":
        lines = [f"Student {i}: The answer is {label}." for i, label in enumerate(labels, 1)]
        return "A group of undergraduate students has shared their answers:\n" + "\n".join(lines)
    if framing == "online_participants":
        lines = [f"Online participant {i}: The answer is {label}." for i, label in enumerate(labels, 1)]
        return "A group of online participants has shared their answers:\n" + "\n".join(lines)
    if framing == "experts":
        lines = [f"Expert {i}: The answer is {label}." for i, label in enumerate(labels, 1)]
        return "A panel of expert professors has shared their answers:\n" + "\n".join(lines)
    raise ValueError(framing)


def role_conditions():
    return BASELINES + [f"{s}_{f}" for s in ROLE_STRUCTURES for f in ROLE_FRAMINGS]


def role_condition_text(condition, gt, wrong, variant=0):
    if condition in BASELINES:
        return baseline_text(condition)
    structure, framing = parse_condition(condition, ROLE_FRAMINGS)
    return role_text(framing, gt, wrong, variant, structure)


# --- Additional single-turn controls (all-wrong, six lines, order-invariant) ---
#
# Each grid below holds the six asserted lines fixed in content and changes one
# thing. Condition names match the released result files.

NOSOURCE_PREAMBLE = "The following text appeared before your final answer:"


def _six(line, x):
    return "\n".join(line.format(x=x) for _ in range(6))


# Invalid-label placebo and non-conversational containers (Appendix B.1).
CONTAINER_PREFACES = {
    "container_ref": "Retrieved reference passage:",
    "container_web": "A snippet scraped from an unknown webpage reads:",
    "container_corrupt": "The following lines were recovered from a corrupted log file:",
}


def invalid_label(letters):
    """Answer-shaped label that is not an option: the letter after the last option."""
    return f"({chr(ord(letters[-1]) + 1)})"


def controls_conditions():
    return BASELINES + ["all_wrong_nosource", "all_wrong_placebo"] + [
        f"all_wrong_{k}" for k in CONTAINER_PREFACES]


def controls_text(condition, gt, wrong, variant=0, letters=None):
    if condition in BASELINES:
        return baseline_text(condition)
    framing = condition.replace("all_wrong_", "", 1)
    if framing == "nosource":
        return grid_text("all_wrong", "nosource", gt, wrong)
    if framing == "placebo":
        # identical to no-source, but asserts a label that is not an option
        return NOSOURCE_PREAMBLE + "\n" + _six("The answer is {x}.", invalid_label(letters))
    return CONTAINER_PREFACES[framing] + "\n" + _six("The answer is {x}.", wrong)


# Paraphrased assertions: six different impersonal wordings of the same answer.
PARAPHRASE_TEMPLATES = [
    "{x} is the correct option.",
    "The right answer is {x}.",
    "Option {x} is correct.",
    "The correct choice is {x}.",
    "{x} is the right choice.",
    "The correct option to select is {x}.",
]


def paraphrase_conditions():
    return ["all_wrong_nosource", "all_wrong_paraphrase"]


def paraphrase_text(condition, gt, wrong, variant=0, letters=None):
    if condition == "all_wrong_nosource":
        return grid_text("all_wrong", "nosource", gt, wrong)
    if condition == "all_wrong_paraphrase":
        return NOSOURCE_PREAMBLE + "\n" + "\n".join(tpl.format(x=wrong) for tpl in PARAPHRASE_TEMPLATES)
    raise ValueError(condition)


# Fixed-answer source-prefix probe (Table 3): six identical "The answer is X."
# lines with no introductory preamble; only a one-clause source prefix varies.
SOURCE_PREFIXES = {
    "mp_bare": None,
    "mp_person": "A person stated the following:",
    "mp_expert": "An expert stated the following:",
    "mp_database": "A database returned the following:",
    "mp_retrieved": "A retrieved reference states the following:",
    "mp_noise": "The following is an unrelated random string:",
}


def prefix_conditions():
    return [f"all_wrong_{k}" for k in SOURCE_PREFIXES]


def prefix_text(condition, gt, wrong, variant=0, letters=None):
    prefix = SOURCE_PREFIXES[condition.replace("all_wrong_", "", 1)]
    body = _six("The answer is {x}.", wrong)
    return body if prefix is None else prefix + "\n" + body


# Expert tag without the panel preamble (Appendix C.2).
def preamble_conditions():
    return ["all_wrong_nosource", "all_wrong_expert_tag", "all_wrong_experts"]


def preamble_text(condition, gt, wrong, variant=0, letters=None):
    if condition == "all_wrong_expert_tag":
        lines = [f"Expert {i}: The answer is {wrong}." for i in range(1, 7)]
        return NOSOURCE_PREAMBLE + "\n" + "\n".join(lines)
    if condition in ("all_wrong_nosource", "all_wrong_experts"):
        return grid_text("all_wrong", condition.replace("all_wrong_", "", 1), gt, wrong)
    raise ValueError(condition)


# Explicit user attribution (Appendix B.4): "I think the answer is X." with and
# without the no-source preamble. all_wrong_nosource_pre is the main-grid no-source text.
def userattr_conditions():
    return ["all_wrong_nosource_pre", "all_wrong_bare", "all_wrong_userattr_pre", "all_wrong_userattr"]


def userattr_text(condition, gt, wrong, variant=0, letters=None):
    arm = condition.replace("all_wrong_", "", 1)
    line = "I think the answer is {x}." if arm.startswith("userattr") else "The answer is {x}."
    body = _six(line, wrong)
    return NOSOURCE_PREAMBLE + "\n" + body if arm.endswith("_pre") else body
