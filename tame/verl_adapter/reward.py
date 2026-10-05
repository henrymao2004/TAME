"""verl custom reward function for TAME.

Pass it to verl with
    reward.custom_reward_function.path=<repo>/tame/verl_adapter/reward.py
    reward.custom_reward_function.name=compute_score
VIRL is scored by answer matching, SPA-VL by an LLM judge against the reference response. The optional CoT-monitor
bonus reproduces the reward baseline of tame.train (--cot_mon_weight).
"""

from functools import lru_cache

from ..common import virl_reward
from ..monitor import LLM, predict, spavl_reward
from .config import from_env


@lru_cache(maxsize=None)
def _llm(model):
    return LLM(model)


@lru_cache(maxsize=1)
def _cfg():
    return from_env()


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    cfg = _cfg()
    dataset = str(data_source).removeprefix("tame_")
    if dataset == "virl":
        score = virl_reward(solution_str, ground_truth)
    elif dataset == "spavl":
        score = spavl_reward(_llm(cfg["reward_judge"]), solution_str, ground_truth)
    else:
        raise ValueError(f"unknown data_source {data_source!r}")
    if cfg["cot_mon_weight"] > 0:
        sample = {"question": (extra_info or {}).get("question", ""), "response": solution_str,
                  "chosen": ground_truth if dataset == "spavl" else ""}
        score += cfg["cot_mon_weight"] * float(predict(_llm(cfg["predictor"]), sample, dataset) == "A")
    return {"score": float(score), "acc": float(score)}
