"""A tiny random Qwen3-VL with LoRA, random SAEs and a features file, for CPU tests of the verl adapter."""

import json
import os

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, Qwen3VLConfig

from tame.common import module_keys
from tame.sae import TopKSAE, save_saes

LAYERS = [1, 2]
RESP = 4
PROMPT = 5


def build_model(seed=0, randomize_adapter=True):
    torch.manual_seed(seed)
    cfg = Qwen3VLConfig(
        text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, vocab_size=128, max_position_embeddings=256,
                         rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True}),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=32,
                           patch_size=4, spatial_merge_size=1, temporal_patch_size=1, deepstack_visual_indexes=[0],
                           num_position_embeddings=16),
    )
    base = AutoModelForImageTextToText.from_config(cfg)
    lora = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                      target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(base, lora)
    if randomize_adapter:  # LoRA B starts at zero; make the policy differ from the pre-RL model
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.data.normal_(0, 0.5)
    return model


def write_sae_files(model, tmp, n_latents=16, k=4, feature_ids=(0, 3, 5)):
    keys = module_keys(LAYERS)
    saes = {}
    for key in keys:
        mod = next(m for n, m in model.named_modules() if n.replace("base_model.model.", "").endswith("language_model." + key))
        saes[key] = TopKSAE(mod.base_layer.out_features, n_latents, k)
    save_saes(saes, f"{tmp}/sae", {"keys": keys, "layers": LAYERS, "modules": [], "k": k, "n_latents": n_latents,
                                   "models": ["tiny"]})
    json.dump({"targeted": {key: list(feature_ids) for key in keys}}, open(f"{tmp}/features.json", "w"))
    return f"{tmp}/sae", f"{tmp}/features.json"


def tame_env(sae, feats, **overrides):
    cfg = {"method": "tame", "dataset": "virl", "sae": sae, "features": feats, "gamma": 0.1}
    cfg.update(overrides)
    os.environ["TAME_CONFIG"] = json.dumps(cfg)


def init_dist(tmp):
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp}/pg")


def make_batch(batch=4, seed=1, rows_left_padded=(1,), rows_right_padded=(2,)):
    from verl import DataProto
    g = torch.Generator().manual_seed(seed)
    seqlen = PROMPT + RESP
    input_ids = torch.randint(1, 128, (batch, seqlen), generator=g)
    attention_mask = torch.ones(batch, seqlen, dtype=torch.long)
    for r in (r for r in rows_left_padded if r < batch):
        attention_mask[r, :2] = 0
    for r in (r for r in rows_right_padded if r < batch):
        attention_mask[r, -1] = 0
    response_mask = attention_mask[:, -RESP:].clone()
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    adv = torch.randn(batch, 1, generator=g).expand(batch, RESP).contiguous()
    return DataProto.from_dict(tensors={
        "responses": input_ids[:, -RESP:], "response_mask": response_mask, "input_ids": input_ids,
        "attention_mask": attention_mask, "position_ids": position_ids,
        "old_log_probs": torch.zeros(batch, RESP), "advantages": adv,
    }, meta_info={"temperature": 1.0, "pad_token_id": 0})


def make_setup(tmp, **model_kw):
    model = build_model(**model_kw)
    sae, feats = write_sae_files(model, str(tmp))
    init_dist(str(tmp))
    return model, sae, feats


def actor_config(**kw):
    from verl.workers.config.actor import FSDPActorConfig
    base = dict(strategy="fsdp2", ppo_mini_batch_size=4, ppo_micro_batch_size_per_gpu=2, use_torch_compile=False,
                rollout_n=1, ppo_epochs=1)
    base.update(kw)
    return FSDPActorConfig(**base)
