"""Decision-point attention mass, normalized Cheating Index, head screening and head patching.

python -m tame.attention screen --model ckpt --traces annotated.jsonl --out ci.json
python -m tame.attention rank --early ci_step0.json --late ci_step200.json --out heads.json
python -m tame.attention patch --early_model step0 --late_model step200 --traces annotated.jsonl --heads heads.json --out patch.json
"""

import argparse
import json

import torch

from .common import (CATEGORIES, decision_index, decoder_layers, encode_trace, load_model, read_jsonl, text_config,
                     to_device, write_json)
from .features import token_labels

EPS = 1e-6


def prepare(processor, t):
    """Inputs truncated at the decision point, category masks over positions, token labels and the next token."""
    enc, start, offsets = encode_trace(processor, t, t["response"])
    dpos = start + decision_index(t["response"], offsets)
    next_id = enc["input_ids"][0, min(dpos + 1, enc["input_ids"].shape[1] - 1)].item()
    for k in ("input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids"):
        if k in enc:
            enc[k] = enc[k][:, :dpos + 1]
    labels = token_labels(t["response"], t.get("spans", []), offsets)
    masks = {}
    for c in CATEGORIES:
        m = torch.zeros(dpos + 1, dtype=torch.bool)
        for i, lab in enumerate(labels[:dpos + 1 - start]):
            m[start + i] = lab == c
        masks[c] = m
    return enc, masks, labels, next_id


def ci_uniform(label_lists):
    labels = [l for ls in label_lists for l in ls]
    p = {c: labels.count(c) / max(len(labels), 1) for c in CATEGORIES}
    return p["TEMPLATE"] / max(p["GROUND"] + p["ENTITY"], EPS)


@torch.no_grad()
def attention_rows(model, enc, patch=None):
    """Attention of the last position to all positions, per layer and head: [L, H, S].

    patch maps layer -> (head, tensor [S, d]) substituted into that head's o_proj input."""
    layers, rows, handles = decoder_layers(model), {}, []
    d = text_config(model).head_dim
    for l, layer in enumerate(layers):
        def grab(module, inputs, output, l=l):
            rows[l] = output[1][0, :, -1, :].float()
        handles.append(layer.self_attn.register_forward_hook(grab))
    for l, (h, x) in (patch or {}).items():
        def sub(module, args, h=h, x=x):
            y = args[0].clone()
            y[0, :, h * d:(h + 1) * d] = x.to(y.dtype)
            return (y,) + args[1:]
        handles.append(layers[l].self_attn.o_proj.register_forward_pre_hook(sub))
    model(**to_device(enc, next(model.parameters()).device))
    for hd in handles:
        hd.remove()
    return torch.stack([rows[l] for l in range(len(layers))])


def category_mass(rows, masks):
    return {c: rows[..., masks[c].to(rows.device)].sum(-1) for c in CATEGORIES}


def ci_raw(mass):
    return mass["TEMPLATE"] / (mass["GROUND"] + mass["ENTITY"] + EPS)


@torch.no_grad()
def head_outputs(model, enc, layer_ids):
    """Per-layer o_proj inputs [S, H*d] for the given layers."""
    layers, out, handles = decoder_layers(model), {}, []
    for l in layer_ids:
        def grab(module, args, l=l):
            out[l] = args[0][0].detach().clone()
        handles.append(layers[l].self_attn.o_proj.register_forward_pre_hook(grab))
    model(**to_device(enc, next(model.parameters()).device))
    for hd in handles:
        hd.remove()
    return out


def screen(model, processor, traces):
    ci_sum, mass_sum, labels_all = 0, {c: 0 for c in CATEGORIES}, []
    for t in traces:
        enc, masks, labels, _ = prepare(processor, t)
        mass = category_mass(attention_rows(model, enc), masks)
        ci_sum = ci_sum + ci_raw(mass)
        for c in CATEGORIES:
            mass_sum[c] = mass_sum[c] + mass[c].mean(1)
        labels_all.append(labels)
    n, u = len(traces), ci_uniform(labels_all)
    ci = ci_sum / n
    return {"ci": (ci / u).tolist(), "ci_raw": ci.tolist(), "ci_uniform": u,
            "mass": {c: (mass_sum[c] / n).tolist() for c in CATEGORIES}, "n": n}


def rank(early, late, top_layers=5, top_heads=15):
    e, l = torch.tensor(early["ci"]), torch.tensor(late["ci"])
    layer_delta = l.mean(1) - e.mean(1)
    layers = layer_delta.topk(top_layers).indices.tolist()
    head_delta = (l - e)[layers]
    flat = head_delta.flatten().topk(min(top_heads, head_delta.numel())).indices.tolist()
    H = e.shape[1]
    heads = [[layers[i // H], i % H, head_delta.flatten()[i].item()] for i in flat]
    return {"layer_delta": layer_delta.tolist(), "layers": layers, "heads": heads}


def patch(early, late, processor, traces, heads):
    """Delta CI_patch = CI_baseline - CI_patched at layers downstream of each patched head."""
    per_trace, labels_all = [], []
    for t in traces:
        enc, masks, labels, _ = prepare(processor, t)
        labels_all.append(labels)
        cached = head_outputs(early, enc, sorted({l for l, _ in heads}))
        base = ci_raw(category_mass(attention_rows(late, enc), masks))
        deltas = []
        for l, h in heads:
            x = cached[l][:, h * text_config(late).head_dim:(h + 1) * text_config(late).head_dim]
            patched = ci_raw(category_mass(attention_rows(late, enc, {l: (h, x)}), masks))
            deltas.append((base[l + 1:].mean() - patched[l + 1:].mean()).item())
        per_trace.append(deltas)
    u = ci_uniform(labels_all)
    mean = torch.tensor(per_trace).mean(0) / u
    return {f"L{l}H{h}": v for (l, h), v in zip(heads, mean.tolist())}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screen")
    s.add_argument("--model", required=True)
    s.add_argument("--traces", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int)
    r = sub.add_parser("rank")
    r.add_argument("--early", required=True)
    r.add_argument("--late", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--top_layers", type=int, default=5)
    r.add_argument("--top_heads", type=int, default=15)
    p = sub.add_parser("patch")
    p.add_argument("--early_model", required=True)
    p.add_argument("--late_model", required=True)
    p.add_argument("--traces", required=True)
    p.add_argument("--heads", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int)
    args = ap.parse_args()

    if args.cmd == "screen":
        model, processor = load_model(args.model, attn="eager")
        write_json(args.out, screen(model, processor, read_jsonl(args.traces)[:args.n]))
    elif args.cmd == "rank":
        write_json(args.out, rank(json.load(open(args.early)), json.load(open(args.late)),
                                  args.top_layers, args.top_heads))
    else:
        early, _ = load_model(args.early_model, attn="eager")
        late, processor = load_model(args.late_model, attn="eager")
        heads = [(l, h) for l, h, _ in json.load(open(args.heads))["heads"]]
        write_json(args.out, patch(early, late, processor, read_jsonl(args.traces)[:args.n], heads))


if __name__ == "__main__":
    main()
