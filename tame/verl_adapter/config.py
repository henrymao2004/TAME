"""TAME settings shared by the driver, the trainer actor and the FSDP workers.

The settings travel as one JSON string in the TAME_CONFIG environment variable, which main.py injects into the Ray
runtime environment so that every worker process sees the same values.
"""

import json
import os

from ..common import PRESETS

ENV_VAR = "TAME_CONFIG"
METHODS = ("grpo", "dapo", "tame")

DEFAULTS = {
    "method": "tame",
    "dataset": "virl",
    "sae": None,
    "features": None,
    "gamma": None,
    "tau_r": None,
    "no_feedback": False,
    "random_features": False,
    "update_every": 10,
    "updater": "openai/gpt-4o",
    "reward_judge": "deepseek/deepseek-v3.2",
    "predictor": "openai/gpt-4o",
    "cot_mon_weight": 0.0,
    "dynamic_sampling_rounds": 3,
    "seed": 0,
}


def resolve(raw=None):
    """Merge user settings over the defaults and apply the same per-method rules as tame.train."""
    cfg = {**DEFAULTS, **{k: v for k, v in (raw or {}).items() if v is not None}}
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown tame settings: {sorted(unknown)}")
    if cfg["method"] not in METHODS:
        raise ValueError(f"tame.method must be one of {METHODS}, got {cfg['method']!r}")
    if cfg["dataset"] not in PRESETS:
        raise ValueError(f"tame.dataset must be one of {list(PRESETS)}, got {cfg['dataset']!r}")
    preset = PRESETS[cfg["dataset"]]
    if cfg["method"] != "tame":
        cfg["gamma"], cfg["no_feedback"] = 0.0, True
    elif cfg["gamma"] is None:
        cfg["gamma"] = preset["gamma"]
    if cfg["tau_r"] is None:
        cfg["tau_r"] = preset["tau_r"]
    if cfg["gamma"] > 0 and not (cfg["sae"] and cfg["features"]):
        raise ValueError("TAME with gamma > 0 needs tame.sae and tame.features")
    return cfg


def dumps(cfg):
    return json.dumps(cfg, sort_keys=True)


def from_env():
    raw = os.environ.get(ENV_VAR)
    return resolve(json.loads(raw)) if raw else resolve({})
