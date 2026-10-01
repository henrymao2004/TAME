"""Layer-wise Integrated Gradients at the decision point, averaged per token type.

python -m tame.attribution --model ckpt --traces annotated.jsonl --out ig.json
python -m tame.attribution --model step200 --traces ... --out ig_patched.json --patch_from step0 --heads heads.json
"""

import argparse
import json

import torch

from .attention import head_outputs, prepare
from .common import CATEGORIES, decoder_layers, image_token_id, load_model, read_jsonl, text_config, to_device, write_json

TYPES = ["IMAGE"] + CATEGORIES


def expand(enc, b):
    return {k: v.repeat(b, *[1] * (v.dim() - 1)) for k, v in enc.items()}


def integrated_gradients(model, enc, target, layer_ids, steps=50, chunk=10):
    """IG_j^(l) with a zero baseline on the input hidden states of each layer: [n_layers, S]."""
    layers = decoder_layers(model)
    device = next(model.parameters()).device
    enc = to_device(enc, device)
    clean, handles = {}, []
    for l in layer_ids:
        def grab(module, args, kwargs, l=l):
            clean[l] = (args[0] if args else kwargs["hidden_states"]).detach()
        handles.append(layers[l].register_forward_pre_hook(grab, with_kwargs=True))
    with torch.no_grad():
        model(**enc)
    for h in handles:
        h.remove()

    out = []
    alphas = (torch.arange(steps, device=device, dtype=torch.float32) + 0.5) / steps
    for l in layer_ids:
        h0 = clean[l][:1]
        grad_sum = torch.zeros_like(h0, dtype=torch.float32)
        for a in alphas.split(chunk):
            x = (a.view(-1, 1, 1) * h0.float()).to(h0.dtype).requires_grad_(True)

            def scale(module, args, kwargs, x=x):
                if args:
                    return (x,) + args[1:], kwargs
                kwargs["hidden_states"] = x
                return args, kwargs

            hd = layers[l].register_forward_pre_hook(scale, with_kwargs=True)
            logits = model(**expand(enc, len(a)), logits_to_keep=1).logits[:, -1].float()
            hd.remove()
            logp = logits.log_softmax(-1)[:, target].sum()
            grad_sum += torch.autograd.grad(logp, x)[0].float().sum(0, keepdim=True)
        out.append((h0.float() * grad_sum / steps).sum(-1)[0].cpu())
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="+")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--n", type=int)
    ap.add_argument("--patch_from", help="earlier checkpoint whose head outputs replace the flagged heads")
    ap.add_argument("--heads", help="heads.json from tame.attention rank")
    args = ap.parse_args()

    model, processor = load_model(args.model)
    model.requires_grad_(False)
    n_layers = len(decoder_layers(model))
    layer_ids = args.layers or list(range(n_layers))
    early = load_model(args.patch_from)[0] if args.patch_from else None
    heads = [(l, h) for l, h, _ in json.load(open(args.heads))["heads"]] if args.heads else []
    d = text_config(model).head_dim
    img_id = image_token_id(model)

    sums = {c: torch.zeros(len(layer_ids)) for c in TYPES}
    counts = {c: 0 for c in TYPES}
    for t in read_jsonl(args.traces)[:args.n]:
        enc, masks, _, target = prepare(processor, t)
        masks["IMAGE"] = enc["input_ids"][0] == img_id
        handles = []
        if early is not None:
            cached = head_outputs(early, enc, sorted({l for l, _ in heads}))
            for l, h in heads:
                def sub(module, a, l=l, h=h):
                    y = a[0].clone()
                    y[:, :, h * d:(h + 1) * d] = cached[l][:, h * d:(h + 1) * d].to(y.dtype)
                    return (y,) + a[1:]
                handles.append(decoder_layers(model)[l].self_attn.o_proj.register_forward_pre_hook(sub))
        ig = integrated_gradients(model, enc, target, layer_ids, args.steps, args.chunk)
        for hd in handles:
            hd.remove()
        for c in TYPES:
            m = masks[c]
            if m.any():
                sums[c] += ig[:, m].mean(1)
                counts[c] += 1
    write_json(args.out, {"layers": layer_ids,
                          **{c: (sums[c] / max(counts[c], 1)).tolist() for c in TYPES}})


if __name__ == "__main__":
    main()
