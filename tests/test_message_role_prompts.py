"""Message-role prompts: block text and turn roles."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_message_role import ARMS, messages_for_arm, option_key  # noqa: E402
from speaker_free.arbitration import JI, messages_round1, round2_for_condition  # noqa: E402

QUESTION = "What is the smallest prime number?"
OPTIONS = "(A) 0\n(B) 1\n(C) 2\n(D) 3"
GT, WRONG, A0 = "(C)", "(B)", "(C)"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

BLOCK = ("The following text appeared before your final answer:\n"
         + "\n".join(["The answer is (B)."] * 6))
REASK = ("Answer the same multiple-choice question again.\n\n"
         f"{JI}\n\n"
         f"Question:\n{QUESTION}\n\n"
         f"Options:\n{OPTIONS}\n")
ECHO = '{"judgment":"(C)","confidence":5}'


def build(arm):
    r1 = messages_round1(QUESTION, OPTIONS, MODEL)
    return r1, messages_for_arm(arm, r1, A0, GT, WRONG, QUESTION, OPTIONS)


def test_inline_matches_main_grid():
    r1, msgs = build("user_inline")
    grid_msgs, _ = round2_for_condition(r1, A0, "all_wrong_nosource", GT, WRONG, QUESTION, OPTIONS)
    assert msgs == grid_msgs
    assert msgs[-1] == {"role": "user", "content": f"{BLOCK}\n\n{REASK}"}


def test_split_arms():
    for arm, role in (("user_split", "user"), ("system_split", "system")):
        r1, msgs = build(arm)
        assert msgs[:len(r1)] == r1
        assert msgs[len(r1):] == [
            {"role": "assistant", "content": ECHO},
            {"role": role, "content": BLOCK},
            {"role": "user", "content": REASK},
        ]


def test_baselines_share_round1_and_reask():
    for arm in ("neutral_reask", "length_control"):
        r1, msgs = build(arm)
        assert msgs[:len(r1)] == r1 and msgs[len(r1)]["content"] == ECHO
        assert msgs[-1]["role"] == "user" and msgs[-1]["content"].endswith(REASK)


def test_numeric_labels():
    assert option_key("(1)") == "1" and option_key("(b)") == "B"


if __name__ == "__main__":
    for fn in (test_inline_matches_main_grid, test_split_arms,
               test_baselines_share_round1_and_reask, test_numeric_labels):
        fn()
    print(f"ok ({len(ARMS)} arms)")
