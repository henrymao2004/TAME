"""Generate CoT traces, optionally under SAE feature ablation or injection.

python -m tame.generate --model ckpt --data data/virl/test.jsonl --out traces.jsonl
python -m tame.generate --model ckpt --data ... --out ... --sae sae/virl --features features.json --intervene ablate [--random]
"""

import argparse
import json
import random

import torch

from .common import encode_prompt, generate, get_module, load_model, read_jsonl, virl_reward, write_jsonl
from .sae import load_saes


def random_features(targeted, n_latents, seed=0):
    """Same count per module as the targeted set, drawn from latents outside it."""
    rng = random.Random(seed)
    return {k: rng.sample(sorted(set(range(n_latents)) - set(v)), len(v)) for k, v in targeted.items()}


def add_intervention(model, saes, features, mode, beta=0.8, lam=1.0):
    """y_new = y + lam * (Dec(z') - Dec(z)), with z' = (1 -/+ beta) z on the selected latents."""
    sign = -1.0 if mode == "ablate" else 1.0
    handles = []
    for key, ids in features.items():
        sae = saes[key]
        ids = torch.tensor(ids, device=sae.W_enc.device)

        def hook(module, inputs, output, sae=sae, ids=ids):
            z = sae.encode(output.float())
            dz = torch.zeros_like(z)
            dz[..., ids] = sign * beta * z[..., ids]
            return (output.float() + lam * (dz @ sae.W_dec)).to(output.dtype)

        handles.append(get_module(model, key).register_forward_hook(hook))
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=None, help="number of prompts")
    ap.add_argument("--samples", type=int, default=1, help="responses per prompt")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--sae")
    ap.add_argument("--features")
    ap.add_argument("--intervene", choices=["ablate", "inject"])
    ap.add_argument("--random", action="store_true", help="matched random features")
    ap.add_argument("--beta", type=float, default=0.8)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    model, processor = load_model(args.model)
    data = read_jsonl(args.data)[:args.n]
    if args.intervene:
        feats = json.load(open(args.features))["targeted"]
        saes = load_saes(args.sae, keys=set(feats))
        if args.random:
            feats = random_features(feats, next(iter(saes.values())).W_enc.shape[0], args.seed)
        add_intervention(model, saes, feats, args.intervene, args.beta, args.lam)

    samples = [s for s in data for _ in range(args.samples)]
    texts = []
    for i in range(0, len(samples), args.batch_size):
        encs = [encode_prompt(processor, s) for s in samples[i:i + args.batch_size]]
        texts += generate(model, processor, encs, args.max_new_tokens, args.temperature, args.batch_size)[1]
    rows = []
    for s, text in zip(samples, texts):
        row = dict(s, response=text)
        if "answer" in s:
            row["reward"] = virl_reward(text, s["answer"])
        rows.append(row)
    write_jsonl(args.out, rows)
    if rows and "answer" in rows[0]:
        print("task_acc", sum(r["reward"] for r in rows) / len(rows))


if __name__ == "__main__":
    main()
