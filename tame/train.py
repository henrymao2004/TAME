"""GRPO, DAPO and TAME training with LoRA.

torchrun --nproc_per_node 8 -m tame.train --dataset virl --method tame --model Qwen/Qwen3-VL-8B-Instruct \
    --train data/virl/train.jsonl --sae sae/virl --features analysis/virl_grpo_step200/features.json --out runs/virl_tame
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText

from . import prompts as P
from .common import (DEVICE, PRESETS, Recorder, append_response, collate, dist_info, encode_prompt, generate,
                     load_processor, read_jsonl, to_device, virl_reward)
from .generate import random_features
from .monitor import LLM, pmap, predict, spavl_reward
from .sae import load_saes


@dataclass
class Rollout:
    sample: dict
    enc: dict
    ids: list
    text: str
    group: int
    reward: float = 0.0
    adv: float = 0.0
    refined: bool = False


class PromptEvolver:
    """Behavioral feedback loop: refinement prompt p_e evolved by the prompt updater U."""

    def __init__(self, dataset, updater):
        self.prompt = P.SECOND_ATTEMPT_VISUAL if dataset == "virl" else P.SECOND_ATTEMPT_SAFETY
        self.llm = LLM(updater)
        self.history = []

    def summary(self, k=16):
        h = sorted(self.history, key=lambda r: r[2])
        picked = h[:k // 2] + h[-(k // 2):] if len(h) > k else h
        return "\n\n".join(f"[Question]: {q[:400]}\n[Attempt]: {a[:800]}\n[Reward]: {r:.2f}" for q, a, r in picked)

    def update(self):
        out = self.llm(P.UPDATER_PROMPT.format(current_prompt=self.prompt, summary=self.summary()),
                       temperature=0.7, max_tokens=2048)
        m = re.search(r"<prompt>(.*?)</prompt>", out, re.S)
        if m and m.group(1).strip():
            self.prompt = m.group(1).strip()
        self.history = []

    def system(self, r):
        traj = P.TRAJECTORY_TEMPLATE.format(question=r.sample["question"], response=r.text, reward=r.reward)
        return self.prompt + P.ATTEMPT_CONTEXT.format(trajectory=traj)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--dataset", choices=list(PRESETS), required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["grpo", "dapo", "tame"], default="grpo")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64, help="prompts per step")
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--kl_coef", type=float, default=0.01)
    ap.add_argument("--clip_low", type=float, default=0.2)
    ap.add_argument("--clip_high", type=float, default=None)
    ap.add_argument("--n_minibatches", type=int, default=1)
    ap.add_argument("--micro_batch_size", type=int, default=2)
    ap.add_argument("--gen_batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lora_rank", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reward_judge", default="deepseek/deepseek-v3.2")
    ap.add_argument("--sae")
    ap.add_argument("--features")
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--random_features", action="store_true")
    ap.add_argument("--no_feedback", action="store_true")
    ap.add_argument("--update_every", type=int, default=10)
    ap.add_argument("--updater", default="openai/gpt-4o")
    ap.add_argument("--cot_mon_weight", type=float, default=0.0)
    ap.add_argument("--predictor", default="openai/gpt-4o")
    args = ap.parse_args()
    dapo = args.method == "dapo"
    if args.clip_high is None:
        args.clip_high = 0.28 if dapo else args.clip_low
    if dapo:
        args.kl_coef = 0.0
    if args.method != "tame":
        args.gamma, args.no_feedback = 0.0, True
    elif args.gamma is None:
        args.gamma = PRESETS[args.dataset]["gamma"]
    return args


def all_sum(x):
    t = torch.tensor(float(x), device=DEVICE)
    if dist.is_initialized():
        dist.all_reduce(t)
    return t.item()


class Trainer:
    def __init__(self, args):
        self.args = args
        self.rank, self.world, local = dist_info()
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        if self.world > 1:
            dist.init_process_group("nccl")
        torch.manual_seed(args.seed + self.rank)
        self.rng = random.Random(args.seed + 1000 + self.rank)

        base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).to(DEVICE)
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        lora = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                          target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
        self.policy = get_peft_model(base, lora)
        self.params = [p for p in self.policy.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=args.lr, weight_decay=0.0)
        self.processor = load_processor(args.model)
        tok = self.processor.tokenizer
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.data = read_jsonl(args.train)

        self.judge = LLM(args.reward_judge) if args.dataset == "spavl" else None
        self.monitor = LLM(args.predictor) if args.cot_mon_weight > 0 else None
        self.evolver = None if args.no_feedback else PromptEvolver(args.dataset, args.updater)

        self.saes, self.feats = {}, {}
        if args.gamma > 0:
            feats = json.load(open(args.features))["targeted"]
            self.saes = load_saes(args.sae, keys=set(feats))
            if args.random_features:
                feats = random_features(feats, next(iter(self.saes.values())).W_enc.shape[0], args.seed)
            self.feats = {k: torch.tensor(v, device=DEVICE) for k, v in feats.items()}
            self.recorder = Recorder(self.policy, list(self.feats))
        if self.rank == 0:
            os.makedirs(args.out, exist_ok=True)
            json.dump(vars(args), open(f"{args.out}/args.json", "w"), indent=2)

    # Rollouts and rewards
    def sample(self, samples, systems, n):
        gen_encs = [encode_prompt(self.processor, s, sys) for s, sys in zip(samples, systems)]
        gen_encs = [e for e in gen_encs for _ in range(n)]
        self.policy.eval()
        ids, texts = [], []
        for i in range(0, len(gen_encs), self.args.gen_batch_size):
            a, b = generate(self.policy, self.processor, gen_encs[i:i + self.args.gen_batch_size],
                            self.args.max_new_tokens, self.args.temperature, self.args.gen_batch_size)
            ids += a
            texts += b
        self.policy.train()
        return ids, texts

    def reward(self, rollouts):
        def score(r):
            s = r.sample
            if self.args.dataset == "virl":
                v = virl_reward(r.text, s["answer"])
            else:
                v = spavl_reward(self.judge, r.text, s.get("chosen", ""))
            if self.monitor is not None:
                t = dict(s, response=r.text)
                v += self.args.cot_mon_weight * float(predict(self.monitor, t, self.args.dataset) == "A")
            return v
        for r, v in zip(rollouts, pmap(score, rollouts)):
            r.reward = v

    def rollout_groups(self, samples, first_group):
        encs = [encode_prompt(self.processor, s) for s in samples]
        G = self.args.group_size
        ids, texts = self.sample(samples, [P.BASE_SYSTEM_PROMPT] * len(samples), G)
        rollouts = [Rollout(samples[i // G], encs[i // G], ids[i], texts[i], first_group + i // G)
                    for i in range(len(ids))]
        self.reward(rollouts)
        return rollouts

    def collect(self, step):
        a = self.args
        batch = random.Random(a.seed + step).sample(self.data, a.batch_size)[self.rank::self.world]
        rollouts = self.rollout_groups(batch, 0)
        if a.method == "dapo":
            rollouts = self.dynamic_sampling(rollouts, len(batch))
        originals = list(rollouts)

        if self.evolver is not None:
            self.evolver.history += [(r.sample["question"], r.text, r.reward) for r in originals]
            if step % a.update_every == 0:
                if self.rank == 0:
                    self.evolver.update()
                obj = [self.evolver.prompt]
                if self.world > 1:
                    dist.broadcast_object_list(obj, src=0)
                self.evolver.prompt = obj[0]
            samples = [r.sample for r in originals]
            ids, texts = self.sample(samples, [self.evolver.system(r) for r in originals], 1)
            refined = [Rollout(r.sample, r.enc, i, t, r.group, refined=True) for r, i, t in zip(originals, ids, texts)]
            self.reward(refined)
            rollouts += [f for f, r in zip(refined, originals) if f.reward >= r.reward]

        groups = defaultdict(list)
        for r in rollouts:
            groups[r.group].append(r)
        for g in groups.values():
            rs = torch.tensor([r.reward for r in g])
            std = rs.std(unbiased=False)
            for r in g:
                r.adv = ((r.reward - rs.mean()) / (std + 1e-6)).item()
        stats = {"reward": sum(r.reward for r in originals) / max(len(originals), 1),
                 "response_len": sum(len(r.ids) for r in originals) / max(len(originals), 1),
                 "refined_kept": (len(rollouts) - len(originals)) / max(len(originals), 1)}
        return rollouts, stats

    def dynamic_sampling(self, rollouts, n_groups, max_rounds=3):
        def informative(rs):
            groups = defaultdict(list)
            for r in rs:
                groups[r.group].append(r)
            return [g for g in groups.values() if len({r.reward for r in g}) > 1]
        kept = informative(rollouts)
        for i in range(max_rounds):
            if len(kept) >= n_groups:
                break
            extra = self.rng.sample(self.data, n_groups - len(kept))
            kept += informative(self.rollout_groups(extra, (i + 1) * 10 ** 6))
        return [r for g in kept[:n_groups] for r in g]

    # Policy update
    def make_batch(self, micro):
        encs = [append_response(r.enc, r.ids) for r in micro]
        batch = to_device(collate(encs, self.pad_id), DEVICE)
        L = batch["input_ids"].shape[1]
        starts = [r.enc["input_ids"].shape[1] for r in micro]
        keep = L - min(starts) + 1
        pos = torch.arange(L - keep + 1, L, device=DEVICE)
        mask = torch.stack([(pos >= s) & (pos < s + len(r.ids)) for s, r in zip(starts, micro)])
        return batch, mask, keep

    def forward(self, batch, mask, keep, record):
        if record:
            self.recorder.out.clear()
            self.recorder.active = True
        logits = self.policy(**batch, logits_to_keep=keep, use_cache=False).logits[:, :-1].float()
        tgt = batch["input_ids"][:, -(keep - 1):]
        lp = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)
        acts = None
        if record:
            self.recorder.active = False
            acts = {k: self.saes[k].encode(self.recorder.out[k][:, -(keep - 1):][mask])[:, f]
                    for k, f in self.feats.items()}
        return lp, acts

    def update(self, rollouts):
        a = self.args
        self.rng.shuffle(rollouts)
        minibatches = [rollouts[i::a.n_minibatches] for i in range(a.n_minibatches)]
        plan = [[mb[i:i + a.micro_batch_size] for i in range(0, len(mb), a.micro_batch_size)] for mb in minibatches]
        use_sae = a.gamma > 0

        pre = []
        with torch.no_grad():
            for micro in chain(*plan):
                batch, mask, keep = self.make_batch(micro)
                p = {}
                if a.n_minibatches > 1:
                    p["old"] = self.forward(batch, mask, keep, False)[0]
                if a.kl_coef > 0 or use_sae:
                    with self.policy.disable_adapter():
                        p["ref"], p["base"] = self.forward(batch, mask, keep, use_sae)
                pre.append(p)

        stats, idx = defaultdict(float), 0
        for mb, micros in zip(minibatches, plan):
            n_seq = all_sum(len(mb))
            n_tok = all_sum(sum(len(r.ids) for r in mb))
            for micro in micros:
                p = pre[idx]
                idx += 1
                batch, mask, keep = self.make_batch(micro)
                lp, acts = self.forward(batch, mask, keep, use_sae)
                old = p.get("old", lp.detach())
                adv = torch.tensor([r.adv for r in micro], device=DEVICE).unsqueeze(1)
                ratio = (lp - old).exp()
                tok = -torch.min(ratio * adv, ratio.clamp(1 - a.clip_low, 1 + a.clip_high) * adv)
                if a.kl_coef > 0:
                    d = p["ref"] - lp
                    kl = d.exp() - d - 1
                    tok = tok + a.kl_coef * kl
                    stats["kl"] += (kl * mask).sum().item() / n_tok
                if a.method == "dapo":
                    w = mask.float() / n_tok
                else:
                    w = mask.float() / (mask.sum(1, keepdim=True).clamp_min(1) * n_seq)
                loss = (tok * w).sum()
                stats["pg_loss"] += loss.item()
                if use_sae:
                    sae_tok = sum((acts[k] - p["base"][k]).relu().pow(2).sum(-1) for k in acts)
                    sae_loss = (sae_tok * w[mask]).sum()
                    loss = loss + a.gamma * sae_loss
                    stats["sae_loss"] += sae_loss.item()
                loss.backward()
            for prm in self.params:
                if prm.grad is None:
                    prm.grad = torch.zeros_like(prm)
                if self.world > 1:
                    dist.all_reduce(prm.grad)
            stats["grad_norm"] += torch.nn.utils.clip_grad_norm_(self.params, 1.0).item() / a.n_minibatches
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        return {k: v if k == "grad_norm" else all_sum(v) for k, v in stats.items()}

    def save(self, step):
        if self.rank == 0:
            path = f"{self.args.out}/step_{step}"
            self.policy.save_pretrained(path)
            self.processor.save_pretrained(path)
            if self.evolver is not None:
                open(f"{path}/refinement_prompt.txt", "w").write(self.evolver.prompt)

    def train(self):
        self.policy.train()
        for step in range(1, self.args.steps + 1):
            rollouts, stats = self.collect(step)
            stats = {k: all_sum(v) / self.world for k, v in stats.items()}
            stats.update(self.update(rollouts))
            if self.rank == 0:
                stats["step"] = step
                print(json.dumps(stats), flush=True)
                with open(f"{self.args.out}/log.jsonl", "a") as f:
                    f.write(json.dumps(stats) + "\n")
            if step % self.args.save_every == 0 or step == self.args.steps:
                self.save(step)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    Trainer(parse_args()).train()
