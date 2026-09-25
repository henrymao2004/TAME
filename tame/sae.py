"""TopK sparse autoencoders on the outputs of projection modules.

python -m tame.sae --model Qwen/Qwen3-VL-8B-Instruct --traces traces_step0.jsonl --dataset virl --out sae/virl
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from .common import (DEVICE, PRESETS, MODULES, Recorder, encode_trace, image_token_id, load_model, module_keys,
                     read_jsonl, to_device)


class TopKSAE(nn.Module):
    def __init__(self, d_in, n_latents=4096, k=32):
        super().__init__()
        self.k = k
        W = torch.randn(n_latents, d_in)
        W = W / W.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W)
        self.W_enc = nn.Parameter(W.clone())
        self.b_enc = nn.Parameter(torch.zeros(n_latents))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def encode(self, x):
        pre = (x.to(self.W_enc.dtype) - self.b_dec) @ self.W_enc.T + self.b_enc
        vals, idx = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, idx, vals.relu())

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)


def save_saes(saes, out, meta):
    Path(out).mkdir(parents=True, exist_ok=True)
    tensors = {f"{key}/{n}": p.detach().cpu().contiguous() for key, sae in saes.items() for n, p in sae.named_parameters()}
    save_file(tensors, f"{out}/sae.safetensors")
    json.dump(meta, open(f"{out}/config.json", "w"), indent=2)


def load_saes(path, device=DEVICE, keys=None):
    meta = json.load(open(f"{path}/config.json"))
    tensors = load_file(f"{path}/sae.safetensors")
    saes = {}
    for key in meta["keys"]:
        if keys is not None and key not in keys:
            continue
        W = tensors[f"{key}/W_dec"]
        sae = TopKSAE(W.shape[1], W.shape[0], meta["k"])
        sae.load_state_dict({n: tensors[f"{key}/{n}"] for n in ("W_dec", "W_enc", "b_enc", "b_dec")})
        saes[key] = sae.to(device).eval().requires_grad_(False)
    return saes


@torch.no_grad()
def collect(model, processor, traces, keys, n_tokens):
    """Collect module outputs on non-image tokens until n_tokens are gathered."""
    rec = Recorder(model, keys, detach=True)
    img_id, device = image_token_id(model), next(model.parameters()).device
    buf, count = {k: [] for k in keys}, 0
    rec.active = True
    while count < n_tokens:
        t = random.choice(traces)
        enc, _, _ = encode_trace(processor, t, t["response"])
        enc = to_device(enc, device)
        model(**enc)
        keep = enc["input_ids"][0] != img_id
        for k in keys:
            buf[k].append(rec.out[k][0][keep].to("cpu", torch.bfloat16))
        count += int(keep.sum())
    rec.remove()
    return {k: torch.cat(v)[:n_tokens] for k, v in buf.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True, help="one model, or several to mix activations equally")
    ap.add_argument("--traces", nargs="+", required=True, help="traces jsonl per model")
    ap.add_argument("--dataset", choices=list(PRESETS))
    ap.add_argument("--layers", type=int, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_latents", type=int, default=4096)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--steps", type=int, default=19000)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--buffer_tokens", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    layers = args.layers or PRESETS[args.dataset]["layers"]
    keys = module_keys(layers)
    models = [load_model(m) for m in args.model]
    traces = [read_jsonl(t) for t in args.traces]
    per_model = args.buffer_tokens // len(models)

    saes, opts, seen = {}, {}, {}
    step = 0
    while step < args.steps:
        parts = [collect(m, p, tr, keys, per_model) for (m, p), tr in zip(models, traces)]
        buf = {k: torch.cat([part[k] for part in parts]) for k in keys}
        if not saes:
            for k in keys:
                saes[k] = TopKSAE(buf[k].shape[1], args.n_latents, args.k).to(DEVICE)
                saes[k].b_dec.data = buf[k][:args.batch_size].float().to(DEVICE).median(0).values
                opts[k] = torch.optim.Adam(saes[k].parameters(), lr=args.lr)
                seen[k] = torch.zeros(args.n_latents, dtype=torch.bool, device=DEVICE)
        perm = torch.randperm(len(buf[keys[0]]))
        for i in range(0, len(perm) - args.batch_size + 1, args.batch_size):
            logs = {}
            for k in keys:
                x = buf[k][perm[i:i + args.batch_size]].to(DEVICE).float()
                x_hat, z = saes[k](x)
                fvu = (x_hat - x).pow(2).sum() / (x - x.mean(0)).pow(2).sum()
                opts[k].zero_grad()
                fvu.backward()
                opts[k].step()
                saes[k].normalize_decoder()
                seen[k] |= (z > 0).any(0)
                logs[k] = fvu.item()
            step += 1
            if step % 100 == 0:
                dead = {k: 1 - seen[k].float().mean().item() for k in keys}
                print(f"step {step} fvu {sum(logs.values()) / len(logs):.4f} dead {sum(dead.values()) / len(dead):.3f}")
                for k in keys:
                    seen[k].zero_()
            if step >= args.steps:
                break

    save_saes(saes, args.out, {"keys": keys, "layers": layers, "modules": MODULES, "k": args.k,
                               "n_latents": args.n_latents, "models": args.model})


if __name__ == "__main__":
    main()
