"""CPU tests of the verl adapter. They need verl v0.7.1 on PYTHONPATH and run on a tiny random Qwen3-VL."""

import copy
import json
import os
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("verl")
sys.path.insert(0, str(Path(__file__).parent))

import _tiny  # noqa: E402
from tame.verl_adapter import actor as A  # noqa: E402
from tame.verl_adapter import config as C  # noqa: E402
from tame.verl_adapter import data as D  # noqa: E402
from tame.verl_adapter import reward as R  # noqa: E402


# --------------------------------------------------------------------------------------------------------------------
# config, data and reward
# --------------------------------------------------------------------------------------------------------------------
def test_config_rules_follow_tame_train():
    cfg = C.resolve({"method": "tame", "dataset": "spavl", "sae": "s", "features": "f"})
    assert cfg["gamma"] == 0.05 and cfg["tau_r"] == 0.8 and not cfg["no_feedback"]
    cfg = C.resolve({"method": "grpo", "dataset": "virl"})
    assert cfg["gamma"] == 0.0 and cfg["no_feedback"] and cfg["tau_r"] == 1.0
    with pytest.raises(ValueError):
        C.resolve({"method": "tame", "dataset": "virl"})  # gamma > 0 needs the SAE files
    with pytest.raises(ValueError):
        C.resolve({"method": "nope"})
    with pytest.raises(ValueError):
        C.resolve({"bogus": 1})


def test_data_conversion(tmp_path):
    from PIL import Image
    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(img)
    rows = [{"id": "q1", "question": "What is shown?", "images": [str(img)], "answer": "B"},
            {"id": "q2", "question": "No image here", "images": [], "answer": "3"}]
    out = D.to_verl_rows(rows, "virl", "train")
    assert out[0]["data_source"] == "tame_virl"
    assert out[0]["prompt"][0] == {"role": "system", "content": "You are a helpful vision-language assistant."}
    assert out[0]["prompt"][1]["content"].startswith("<image>What is shown?")
    assert out[0]["prompt"][1]["content"].endswith("Final:")
    assert out[0]["images"] == [{"image": "file://" + str(img)}]
    assert out[1]["images"] == [] and "<image>" not in out[1]["prompt"][1]["content"]
    assert out[0]["reward_model"]["ground_truth"] == "B"
    spavl = D.to_verl_rows([{"id": "s", "question": "q", "images": [], "chosen": "I cannot help."}], "spavl", "train")
    assert spavl[0]["reward_model"]["ground_truth"] == "I cannot help."

    src = tmp_path / "jsonl"
    src.mkdir()
    (src / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    written = D.convert(str(src), str(tmp_path / "pq"), "virl")
    assert written == [("train", 2)]
    import datasets
    ds = datasets.load_dataset("parquet", data_files=str(tmp_path / "pq" / "train.parquet"))["train"]
    assert ds[0]["images"] == [{"image": "file://" + str(img)}] and ds[1]["images"] == []


def test_reward_virl_and_spavl(monkeypatch):
    monkeypatch.setenv(C.ENV_VAR, json.dumps({"method": "grpo", "dataset": "virl"}))
    R._cfg.cache_clear()
    good = R.compute_score("tame_virl", "reasoning...\nFinal: \\boxed{B}", "B", {"question": "q"})
    bad = R.compute_score("tame_virl", "reasoning...\nFinal: \\boxed{C}", "B", {"question": "q"})
    assert good == {"score": 1.0, "acc": 1.0} and bad["score"] == 0.0

    class Judge:
        def __call__(self, prompt, **kw):
            return "0.75"
    monkeypatch.setattr(R, "_llm", lambda model: Judge())
    out = R.compute_score("tame_spavl", "cot\nFinal: I will not help with that.", "I cannot help.", {"question": "q"})
    assert out["score"] == 0.75

    monkeypatch.setenv(C.ENV_VAR, json.dumps({"method": "grpo", "dataset": "virl", "cot_mon_weight": 0.5}))
    R._cfg.cache_clear()
    monkeypatch.setattr(R, "predict", lambda llm, t, dataset: "A")
    assert R.compute_score("tame_virl", "x\nFinal: B", "B", {"question": "q"})["score"] == 1.5
    R._cfg.cache_clear()
    with pytest.raises(ValueError):
        R.compute_score("other", "x", "y")


# --------------------------------------------------------------------------------------------------------------------
# activation selection
# --------------------------------------------------------------------------------------------------------------------
def test_response_hidden_remove_padding_matches_padded_layout():
    torch.manual_seed(0)
    b, s, r, d = 3, 9, 4, 6
    attention_mask = torch.ones(b, s, dtype=torch.long)
    attention_mask[0, :3] = 0   # left padded prompt
    attention_mask[1, -2:] = 0  # right padded response
    response_mask = attention_mask[:, -r:].clone()
    padded = torch.randn(b, s, d)
    packed = padded[attention_mask.bool()].unsqueeze(0)  # (1, total_nnz, d) as flash-attn varlen produces
    want = A.response_hidden(padded, attention_mask, response_mask, remove_padding=False)
    got = A.response_hidden(packed, attention_mask, response_mask, remove_padding=True)
    assert want.shape == (int(response_mask.sum()), d)
    assert torch.equal(want, got)


def test_find_module_strips_fsdp_and_checkpoint_wrappers(tmp_path):
    model, _, _ = _tiny.make_setup(tmp_path)
    key = "layers.1.self_attn.q_proj"
    found = A.find_module(model, key)
    assert found.__class__.__name__ == "Linear" and hasattr(found, "base_layer")  # the LoRA wrapper, not the base layer
    inner = model.base_model.model.model.language_model.layers[1]
    wrapper = torch.nn.Module()
    wrapper._fsdp_wrapped_module = inner
    holder = torch.nn.Module()
    holder.language_model = torch.nn.Module()
    holder.language_model.layers = torch.nn.ModuleList([torch.nn.Module(), wrapper])
    assert A.find_module(holder, "layers.1.self_attn.q_proj") is found


# --------------------------------------------------------------------------------------------------------------------
# the actor
# --------------------------------------------------------------------------------------------------------------------
@pytest.fixture()
def setup(tmp_path):
    model, sae, feats = _tiny.make_setup(tmp_path)
    _tiny.tame_env(sae, feats)
    return model, sae, feats


def reference_token_penalty(actor, model, data):
    """The penalty exactly as tame.train computes it: encode the policy and the adapter-disabled model."""
    rec = {}
    handles = [A.find_module(model, k).register_forward_hook(
        lambda m, i, o, k=k: rec.__setitem__(k, o[0] if isinstance(o, tuple) else o)) for k in actor.feats]
    inputs = dict(input_ids=data.batch["input_ids"], attention_mask=data.batch["attention_mask"],
                  position_ids=data.batch["position_ids"], use_cache=False)
    mask = data.batch["response_mask"].bool()
    with torch.no_grad():
        model(**inputs)
        student = {k: actor.saes[k].encode(rec[k][:, -_tiny.RESP:][mask])[:, f] for k, f in actor.feats.items()}
        with model.disable_adapter():
            model(**inputs)
            base = {k: actor.saes[k].encode(rec[k][:, -_tiny.RESP:][mask])[:, f] for k, f in actor.feats.items()}
    for h in handles:
        h.remove()
    return sum((student[k] - base[k]).relu().pow(2).sum(-1) for k in student)


def test_sae_penalty_matches_original_formula_per_micro_batch(setup):
    model, _, _ = setup
    actor = A.TameActor(_tiny.actor_config(), model, torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0))
    actor.param_dtype = torch.float32  # compare in float32; bf16 autocast on CPU only adds rounding noise
    data = _tiny.make_batch(batch=4)
    metrics = actor.update_policy(data)
    logged = metrics["actor/tame_sae_loss"]
    assert len(logged) == 2  # two micro-batches of two sequences
    for i, got in enumerate(logged):
        sub = data.select_idxs([2 * i, 2 * i + 1])
        want = reference_token_penalty(actor, model, sub).mean().item()  # token-mean over the micro-batch
        assert got == pytest.approx(want, rel=1e-5), (i, got, want)
        assert want > 0


def test_penalty_is_zero_when_policy_equals_pre_rl_model(tmp_path):
    model, sae, feats = _tiny.make_setup(tmp_path, randomize_adapter=False)  # lora_B = 0 -> identical to the base model
    _tiny.tame_env(sae, feats)
    actor = A.TameActor(_tiny.actor_config(), model, torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0))
    metrics = actor.update_policy(_tiny.make_batch())
    assert all(v == pytest.approx(0.0, abs=1e-6) for v in metrics["actor/tame_sae_loss"])


def test_base_activations_use_the_adapter_free_model_without_grad(setup):
    model, _, _ = setup
    actor = A.TameActor(_tiny.actor_config(), model, torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0))
    data = _tiny.make_batch()
    inputs = {**data.batch, "pad_token_id": 0}
    base = actor._base_activations(inputs, 1.0)
    assert all(not v.requires_grad for v in base.values())
    with model.disable_adapter(), torch.no_grad():
        again = actor._forward_micro_batch(inputs, 1.0, record=True)["tame_acts"]
    for k in base:
        assert torch.allclose(base[k], again[k])
    assert not actor._rec.active and not actor._rec.out  # hooks are idle afterwards


def test_sae_gradient_reaches_lora_parameters(tmp_path):
    grads = {}
    for gamma in (0.0, 10.0):
        model, sae, feats = _tiny.make_setup(tmp_path, seed=0)
        _tiny.tame_env(sae, feats, gamma=gamma, method="tame")
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0)
        actor = A.TameActor(_tiny.actor_config(ppo_mini_batch_size=2), model, opt)
        captured = {}
        orig = actor._optimizer_step

        def spy(captured=captured, actor=actor, orig=orig):
            captured.update({n: p.grad.clone() for n, p in actor.actor_module.named_parameters() if p.grad is not None})
            return orig()
        actor._optimizer_step = spy
        actor.update_policy(_tiny.make_batch(batch=2))
        grads[gamma] = captured
    layer1 = [n for n in grads[10.0] if "layers.1." in n and "lora_B" in n]
    assert layer1
    assert any(not torch.allclose(grads[0.0][n], grads[10.0][n]) for n in layer1)


def test_gamma_zero_is_identical_to_verl_actor(tmp_path):
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    out = []
    for cls in (DataParallelPPOActor, A.TameActor):
        model, sae, feats = _tiny.make_setup(tmp_path, seed=0)
        _tiny.tame_env(sae, feats, method="grpo", gamma=0.0)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        actor = cls(_tiny.actor_config(), model, opt)
        metrics = actor.update_policy(_tiny.make_batch(batch=4))
        out.append((metrics["actor/pg_loss"], {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}))
    assert out[0][0] == pytest.approx(out[1][0], rel=1e-6)
    for n in out[0][1]:
        assert torch.allclose(out[0][1][n], out[1][1][n], atol=1e-6), n


def test_feedback_mode_updates_on_the_whole_batch(tmp_path):
    steps = {}
    for method, no_feedback in (("tame", False), ("tame", True)):
        model, sae, feats = _tiny.make_setup(tmp_path, seed=0)
        _tiny.tame_env(sae, feats, method=method, no_feedback=no_feedback)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0)
        actor = A.TameActor(_tiny.actor_config(ppo_mini_batch_size=2, ppo_micro_batch_size_per_gpu=2), model, opt)
        metrics = actor.update_policy(_tiny.make_batch(batch=6))
        steps[no_feedback] = len(metrics["actor/grad_norm"])
    assert steps[False] == 1   # feedback adds rows, so everything is one mini-batch as in tame.train
    assert steps[True] == 3    # fixed-size batch keeps verl's own mini-batching


def test_reference_policy_instance_never_applies_the_penalty(setup):
    model, _, _ = setup
    ref = A.TameActor(_tiny.actor_config(), model, None)
    assert ref.gamma == 0.0 and not ref._use_sae and ref._rec is None
