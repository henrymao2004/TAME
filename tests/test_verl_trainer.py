"""CPU tests of TameTrainer's batch logic with a fake rollout manager."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("verl")
sys.path.insert(0, str(Path(__file__).parent))

from verl import DataProto  # noqa: E402
from tame.verl_adapter import trainer as T  # noqa: E402

P, R, SEQ = 6, 5, 11


def make_rows(n, groups=2, seed=0, base_reward=0.0):
    """A batch shaped like verl's after generation: prompts left padded to P, responses right padded to R."""
    g = torch.Generator().manual_seed(seed)
    prompts = torch.randint(1, 100, (n, P), generator=g)
    responses = torch.randint(1, 100, (n, R), generator=g)
    attention = torch.ones(n, SEQ, dtype=torch.long)
    attention[:, :2] = 0                                       # two left pads in every prompt
    resp_len = torch.tensor([R, R - 1, R - 2, R, R - 3, R, R - 1, R][:n])
    for i in range(n):
        attention[i, P + int(resp_len[i]):] = 0
    pos = (attention.cumsum(-1) - 1).clamp(min=0)
    position_ids = torch.stack([pos, pos + 7, pos + 7, pos + 7], dim=1)   # (n, 4, SEQ) like Qwen-VL mrope
    rm = torch.zeros(n, R)
    for i in range(n):
        rm[i, int(resp_len[i]) - 1] = base_reward
    uid = np.array([f"u{i // (n // groups)}" for i in range(n)], dtype=object)
    return DataProto.from_dict(
        tensors={"prompts": prompts, "responses": responses, "input_ids": torch.cat([prompts, responses], 1),
                 "attention_mask": attention, "position_ids": position_ids, "rm_scores": rm,
                 "response_mask": attention[:, -R:].clone()},
        non_tensors={"uid": uid,
                     "extra_info": np.array([{"question": f"question {i}"} for i in range(n)], dtype=object),
                     "data_source": np.array(["tame_virl"] * n, dtype=object),
                     "multi_modal_inputs": np.array([{"image_grid_thw": torch.ones(1, 3)} for _ in range(n)], dtype=object),
                     "__num_turns__": np.array([2] * n, dtype=object)},
        meta_info={"temperature": 1.0},
    )


def set_rewards(batch, rewards):
    rm = torch.zeros_like(batch.batch["rm_scores"])
    last = batch.batch["response_mask"].sum(-1) - 1
    rm[torch.arange(len(batch)), last] = torch.tensor(rewards, dtype=torch.float32)
    batch.batch["rm_scores"] = rm


def test_resplice_position_ids_text_and_mrope():
    text = torch.tensor([[0, 0, 0, 1, 2, 3]])                  # last prompt token (index 3) has position 1... use index P-1
    out = T.resplice_position_ids(text, 4, 3)
    assert out.tolist() == [[0, 0, 0, 1, 2, 3, 4]]
    mrope = torch.stack([torch.tensor([[0, 1, 2, 3, 0, 0]]), torch.tensor([[5, 6, 7, 8, 0, 0]])], dim=1)  # (1, 2, 6)
    out = T.resplice_position_ids(mrope, 4, 2)
    assert out.shape == (1, 2, 6)
    assert out[0, 0].tolist() == [0, 1, 2, 3, 4, 5]
    assert out[0, 1].tolist() == [5, 6, 7, 8, 9, 10]


def test_splice_distilled_pairs_original_prompt_with_second_response():
    orig = make_rows(4, seed=1)
    second = make_rows(4, seed=2, base_reward=0.5)
    out = T.splice_distilled(orig, second)
    assert torch.equal(out.batch["prompts"], orig.batch["prompts"])
    assert torch.equal(out.batch["input_ids"][:, :P], orig.batch["prompts"])
    assert torch.equal(out.batch["input_ids"][:, P:], second.batch["responses"])
    assert torch.equal(out.batch["responses"], second.batch["responses"])
    assert torch.equal(out.batch["attention_mask"][:, :P], orig.batch["attention_mask"][:, :P])
    assert torch.equal(out.batch["attention_mask"][:, P:], second.batch["attention_mask"][:, P:])
    assert torch.equal(out.batch["response_mask"], second.batch["attention_mask"][:, P:])
    assert torch.equal(out.batch["rm_scores"], second.batch["rm_scores"])
    last = orig.batch["position_ids"][:, :, P - 1]
    assert torch.equal(out.batch["position_ids"][:, :, P], last + 1)
    assert torch.equal(out.batch["position_ids"][:, :, P + R - 1], last + R)
    assert torch.equal(out.batch["position_ids"][:, :, :P], orig.batch["position_ids"][:, :, :P])
    assert out.non_tensor_batch["uid"].tolist() == orig.non_tensor_batch["uid"].tolist()
    assert orig.batch["input_ids"][:, P:].tolist() != out.batch["input_ids"][:, P:].tolist()  # the original is untouched


def test_informative_indices_keep_groups_in_order():
    uids = ["a", "a", "b", "b", "c", "c"]
    rewards = [0.0, 1.0, 0.5, 0.5, 1.0, 0.0]
    assert T.informative_indices(uids, rewards) == [0, 1, 4, 5]
    assert T.informative_indices(["a", "a"], [1.0, 1.0]) == []


def test_fix_advantages_matches_tame_train_rules():
    kind = np.array(["first", "second", "second", "second", "distilled"], dtype=object)
    uids = np.array(["u", "u#second", "u#second", "v#second", "u#distilled#0"], dtype=object)
    rewards = [0.0, 1.0, 0.0, 1.0, 0.75]
    adv = torch.full((5, 3), 9.0)
    mask = torch.tensor([[1, 1, 1]] * 4 + [[1, 1, 0]])
    out = T.fix_advantages(kind, uids, rewards, adv, mask)
    assert out[0].tolist() == [9.0, 9.0, 9.0]            # first attempts keep verl's group advantage
    assert out[1].tolist() == [9.0, 9.0, 9.0]            # second attempts in a group of two keep it
    assert out[3].tolist() == [0.0, 0.0, 0.0]            # a singleton second group has zero advantage
    assert out[4].tolist() == [0.75, 0.75, 0.0]          # distilled: max(reward, 0) on response tokens only
    assert adv[3].tolist() == [9.0, 9.0, 9.0]            # input is not modified


class FakeEvolver:
    def __init__(self):
        self.history, self.updates = [], 0

    def update(self):
        self.updates += 1
        self.history = []

    def system_for(self, question, response, reward):
        return f"REFINE[{question}|{response}|{reward:.2f}]"


class FakeRollout:
    """Returns responses for the regenerated prompts; reward of second attempt k is rewards2[k]."""

    def __init__(self, rewards2):
        self.rewards2, self.calls = rewards2, []

    def generate_sequences(self, gen2):
        self.calls.append(gen2)
        out = make_rows(len(gen2), groups=1, seed=99)
        set_rewards(out, self.rewards2)
        out.meta_info["timing"] = {"x": 1.0}
        return out


def make_trainer(rewards2, tau=1.0, n=8):
    t = T.TameTrainer.__new__(T.TameTrainer)
    t.tame_cfg = {"method": "tame", "tau_r": tau, "update_every": 10, "dataset": "virl"}
    t.evolver = FakeEvolver()
    t.global_steps = 3
    t.tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens=True: "tok" + "-".join(str(int(x)) for x in ids[:2]))
    t.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda: None)
    t.async_rollout_manager = FakeRollout(rewards2)
    return t


def test_tame_augment_builds_second_and_distilled_rows():
    n = 8
    batch = make_rows(n, groups=2, seed=3)
    rewards = [1.0, 0.0, 1.0, 0.0, 0.2, 1.0, 0.0, 0.5]       # gated (< tau_r = 1): rows 1, 3, 4, 6, 7
    set_rewards(batch, rewards)
    gen_input = DataProto.from_dict(
        tensors={"dummy": torch.zeros(n, 1)},
        non_tensors={"raw_prompt": np.array([[{"role": "system", "content": "BASE"},
                                              {"role": "user", "content": f"u{i}"}] for i in range(n)], dtype=object)})
    t = make_trainer(rewards2=[0.0, 1.0, 0.1, 0.0, 0.9])      # second rewards for rows 1, 3, 4, 6, 7
    metrics = {}
    out = t._tame_augment(batch, gen_input, {}, False, metrics)

    gated = [1, 3, 4, 6, 7]
    better = [k for k, i in enumerate(gated) if [0.0, 1.0, 0.1, 0.0, 0.9][k] > rewards[i]]
    assert better == [1, 4]   # 1.0 > 0.0 and 0.9 > 0.5; 0.1 < 0.2 and 0.0 == 0.0 do not count
    assert len(out) == n + len(gated) + len(better)
    kinds = out.non_tensor_batch["tame_kind"].tolist()
    assert kinds == ["first"] * n + ["second"] * len(gated) + ["distilled"] * len(better)

    # the second attempts were generated from the same prompts with the evolved system prompt
    gen2 = t.async_rollout_manager.calls[0]
    assert len(gen2) == len(gated)
    for slot, i in enumerate(gated):
        msgs = gen2.non_tensor_batch["raw_prompt"][slot]
        assert msgs[0]["role"] == "system" and msgs[0]["content"].startswith(f"REFINE[question {i}|")
        assert msgs[1]["content"] == f"u{i}"
    assert gen_input.non_tensor_batch["raw_prompt"][1][0]["content"] == "BASE"   # inputs are not mutated

    uids = out.non_tensor_batch["uid"].tolist()
    assert all(u.endswith("#second") for u in uids[n:n + len(gated)])
    assert len(set(uids[n:n + len(gated)])) == 2        # second attempts of one prompt share a uid: u0#second, u1#second
    assert len(set(uids[n + len(gated):])) == len(better)   # every distilled row is its own group
    assert all("#distilled#" in u for u in uids[n + len(gated):])

    # distilled rows: original prompt of the gated row, response of its better second attempt
    d0 = out.select_idxs([n + len(gated)])
    orig = batch.select_idxs([gated[better[0]]])
    assert torch.equal(d0.batch["input_ids"][:, :P], orig.batch["prompts"])
    sec = out.select_idxs([n + better[0]])
    assert torch.equal(d0.batch["responses"], sec.batch["responses"])
    assert d0.batch["rm_scores"].sum().item() == pytest.approx(1.0)
    assert out.meta_info["temperature"] == 1.0

    assert metrics["tame/gated_frac"] == pytest.approx(5 / 8)
    assert metrics["tame/refined_kept"] == pytest.approx(2 / 8)
    assert metrics["tame/first_reward"] == pytest.approx(sum(rewards) / n)
    assert t.evolver.updates == 0 and len(t.evolver.history) == n

    t.global_steps = 10
    t.async_rollout_manager = FakeRollout([0.0] * 5)
    batch2 = make_rows(n, groups=2, seed=3)
    set_rewards(batch2, rewards)
    t._tame_augment(batch2, gen_input, {}, False, {})
    assert t.evolver.updates == 1   # the prompt updater runs every update_every steps


def test_tame_augment_without_gated_rows_returns_the_batch_unchanged():
    batch = make_rows(8, seed=4)
    set_rewards(batch, [1.0] * 8)
    t = make_trainer([])
    out = t._tame_augment(batch, DataProto.from_dict(tensors={"dummy": torch.zeros(8, 1)}), {}, False, {})
    assert len(out) == 8 and t.async_rollout_manager.calls == []
    assert out.non_tensor_batch["tame_kind"].tolist() == ["first"] * 8


def test_trim_to_dp_multiple_drops_trailing_rows():
    t = T.TameTrainer.__new__(T.TameTrainer)
    t.actor_rollout_wg = SimpleNamespace(_dispatch_info={"actor": [0, 1, 2, 3]}, _query_dispatch_info=lambda r: None)
    batch = make_rows(8, seed=5)
    assert len(t._trim_to_dp_multiple(batch)) == 8
    batch = DataProto.concat([batch, make_rows(8, seed=6).select_idxs([0, 1, 2])])   # 11 rows, dp size 4
    out = t._trim_to_dp_multiple(batch)
    assert len(out) == 8


def test_dynamic_sampling_fills_the_batch_with_informative_groups():
    t = T.TameTrainer.__new__(T.TameTrainer)
    t.config = SimpleNamespace(data=SimpleNamespace(train_batch_size=2),
                               actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=4)))
    t.tame_cfg = {"dynamic_sampling_rounds": 3}
    first = make_rows(8, groups=2, seed=7)
    set_rewards(first, [1, 1, 1, 1, 0, 1, 0, 1])               # group u0 is uniform, group u1 is informative
    extras = []

    def rollout(batch_dict, timing_raw, profile):
        b = make_rows(8, groups=2, seed=20 + len(extras))
        set_rewards(b, [0, 1, 0, 1, 1, 1, 1, 1])               # u0 informative, u1 uniform
        b.non_tensor_batch["uid"] = np.array([f"x{len(extras)}{u}" for u in b.non_tensor_batch["uid"]], dtype=object)
        extras.append(b)
        return b, None
    t._rollout_batch = rollout
    t._next_extra_batch = lambda: {}
    out = t._dynamic_sampling(first, {})
    assert len(out) == 8                                         # 2 groups x 4 rollouts
    assert len(extras) == 1                                      # one extra round was enough
    uids = out.non_tensor_batch["uid"].tolist()
    assert uids[:4] == ["u1"] * 4 and all(u.startswith("x0") for u in uids[4:])

    # nothing informative anywhere: fall back to the plain batch
    allsame = make_rows(8, groups=2, seed=8)
    set_rewards(allsame, [1.0] * 8)
    t._rollout_batch = lambda *a, **k: (allsame, None)
    assert t._dynamic_sampling(allsame, {}) is allsame


def test_fit_is_verl_fit_plus_the_marked_changes():
    """Drift guard: TameTrainer.fit must differ from verl v0.7.1's RayPPOTrainer.fit only where we meant it to."""
    import difflib
    import inspect
    import textwrap

    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    a = textwrap.dedent(inspect.getsource(RayPPOTrainer.fit)).split("\n")
    b = textwrap.dedent(inspect.getsource(T.TameTrainer.fit)).split("\n")
    ops = [op for op in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op[0] != "equal"]
    removed = ["\n".join(a[i1:i2]) for tag, i1, i2, j1, j2 in ops if tag in ("delete", "replace")]
    added = ["\n".join(b[j1:j2]) for tag, i1, i2, j1, j2 in ops if tag in ("insert", "replace")]
    assert len(removed) == 2, "exactly two verl blocks are replaced: batch construction and generation"
    assert "DataProto.from_single_dict(batch_dict)" in removed[0] and "gen_batch.repeat(" in removed[0]
    assert "# generate a batch" in removed[1] and "compute_response_mask(batch)" in removed[1]
    assert "REMAX" in removed[1]
    assert len(added) == 3 and all("TAME" in block for block in added)
    assert "_extra_iter" in added[0] and "_rollout_batch" in added[1] and "_tame_fix_advantages" in added[2]
