"""Token-type annotation, template-feature selection by Cohen's d, and SAE diagnostics.

python -m tame.features annotate --traces traces.jsonl --out annotated.jsonl
python -m tame.features analyze --model ckpt --sae sae/virl --traces annotated.jsonl --out analysis/step200
"""

import argparse
import json
import re

import torch

from . import prompts as P
from .common import (CATEGORIES, DEVICE, Recorder, encode_trace, layer_of, load_images, load_model, read_jsonl, split_cot,
                     to_device, write_json, write_jsonl)
from .sae import load_saes


# Annotation
def annotate(llm, t):
    cot = split_cot(t["response"])[0]
    out = llm(P.TOKEN_ANNOTATION_PROMPT.format(question=t["question"], cot=cot), images=load_images(t), max_tokens=4096)
    m = re.search(r"\[.*\]", out, re.S)
    try:
        spans = json.loads(m.group()) if m else []
    except json.JSONDecodeError:
        spans = []
    return [{"span": s["span"], "label": s["label"].upper()} for s in spans
            if isinstance(s, dict) and s.get("label", "").upper() in CATEGORIES and s.get("span")]


def char_labels(response, spans):
    labels, cursor = [None] * len(response), 0
    for s in spans:
        i = response.find(s["span"], cursor)
        if i < 0:
            i = response.find(s["span"])
        if i < 0:
            continue
        for j in range(i, i + len(s["span"])):
            labels[j] = s["label"]
        cursor = i + len(s["span"])
    return labels


def token_labels(response, spans, offsets):
    chars = char_labels(response, spans)
    out = []
    for s, e in offsets:
        lab = next((chars[j] for j in range(s, e) if chars[j]), None)
        out.append(lab or "OTHER")
    return out


# Activations
@torch.no_grad()
def collect_latents(model, processor, saes, traces):
    """Sparse SAE latents (indices, values) on response tokens plus token labels."""
    rec = Recorder(model, list(saes), detach=True)
    device = next(model.parameters()).device
    idx, val, labels = {k: [] for k in saes}, {k: [] for k in saes}, []
    rec.active = True
    for t in traces:
        enc, start, offsets = encode_trace(processor, t, t["response"])
        model(**to_device(enc, device))
        for k, sae in saes.items():
            z = sae.encode(rec.out[k][0, start:].float())
            v, i = z.topk(sae.k, dim=-1)
            idx[k].append(i.cpu())
            val[k].append(v.cpu())
        labels += token_labels(t["response"], t.get("spans", []), offsets)
    rec.remove()
    return {k: (torch.cat(idx[k]), torch.cat(val[k])) for k in saes}, labels


def densify(idx, val, n_latents, device=DEVICE):
    z = torch.zeros(idx.shape[0], n_latents, device=device)
    return z.scatter_(1, idx.to(device), val.to(device))


def cohens_d(z, mask):
    a, b = z[mask], z[~mask]
    n1, n0 = a.shape[0], b.shape[0]
    pooled = (((n1 - 1) * a.var(0) + (n0 - 1) * b.var(0)) / max(n1 + n0 - 2, 1)).sqrt()
    return (a.mean(0) - b.mean(0)) / (pooled + 1e-8)


def jsd(p, q):
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    kl = lambda x, y: (x * (x / y).log2()).sum()
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def diagnostics(z, lab, feats, tau_quantile=0.5, bins=20, eps=1e-6):
    """CES, TEMPLATE precision/recall and GROUND/TEMPLATE JSD for selected latents."""
    ground, template = lab == 0, lab == 2
    zf = z[:, feats]
    tau = torch.stack([c[c > 0].quantile(tau_quantile) if (c > 0).any() else c.new_tensor(float("inf"))
                       for c in zf.T])
    high = zf > tau
    ces = ((high[ground].float().mean(0) + eps) / (high[~ground].float().mean(0) + eps)).log()
    precision = high[template].sum() / high.sum().clamp_min(1)
    recall = high[template].any(1).float().mean()
    js = []
    for c in zf.T:
        hi = c.max().item() + eps
        hg = torch.histc(c[ground], bins, 0, hi) + eps
        ht = torch.histc(c[template], bins, 0, hi) + eps
        js.append(jsd(hg, ht))
    return {"ces": ces.mean().item(), "precision": precision.item(), "recall": recall.item(),
            "jsd": torch.stack(js).mean().item()}


def analyze(model, processor, saes, traces, k_target=20, k_diag=100, tau_quantile=0.5):
    latents, labels = collect_latents(model, processor, saes, traces)
    lab = torch.tensor([{"GROUND": 0, "ENTITY": 1, "TEMPLATE": 2}.get(l, 3) for l in labels], device=DEVICE)
    n = len(labels)
    props = {c: (lab == i).sum().item() / n for i, c in enumerate(CATEGORIES)}
    ci_uniform = props["TEMPLATE"] / max(props["GROUND"] + props["ENTITY"], 1e-8)

    d_all, diag, per_module = {}, {}, {}
    for key, sae in saes.items():
        z = densify(*latents[key], sae.W_enc.shape[0])
        d = cohens_d(z, lab == 2)
        d_all[key] = d.cpu()
        if key.endswith(("q_proj", "k_proj")):
            feats = d.topk(k_diag).indices
            diag[key] = feats.tolist()
            per_module[key] = diagnostics(z, lab, feats, tau_quantile)
        del z

    targeted = {}
    for layer in sorted({layer_of(k) for k in saes}):
        keys = [k for k in saes if layer_of(k) == layer]
        scores = torch.cat([d_all[k] for k in keys])
        n_lat = d_all[keys[0]].shape[0]
        for flat in scores.topk(k_target).indices.tolist():
            targeted.setdefault(keys[flat // n_lat], []).append(flat % n_lat)

    summary = {m: sum(v[m] for v in per_module.values()) / max(len(per_module), 1)
               for m in ("ces", "precision", "recall", "jsd")}
    return {"targeted": targeted, "diagnostic": diag}, {
        "summary": summary, "per_module": per_module, "token_proportions": props, "ci_uniform": ci_uniform}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("annotate")
    a.add_argument("--traces", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--annotator", default="openai/gpt-4o")
    a.add_argument("--workers", type=int, default=32)
    b = sub.add_parser("analyze")
    b.add_argument("--model", required=True)
    b.add_argument("--sae", required=True)
    b.add_argument("--traces", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--k_target", type=int, default=20)
    b.add_argument("--k_diag", type=int, default=100)
    b.add_argument("--tau_quantile", type=float, default=0.5)
    args = ap.parse_args()

    if args.cmd == "annotate":
        from .monitor import LLM, pmap
        traces, llm = read_jsonl(args.traces), LLM(args.annotator)
        for t, spans in zip(traces, pmap(lambda t: annotate(llm, t), traces, args.workers)):
            t["spans"] = spans
        write_jsonl(args.out, traces)
    else:
        model, processor = load_model(args.model)
        saes = load_saes(args.sae)
        feats, diag = analyze(model, processor, saes, read_jsonl(args.traces), args.k_target, args.k_diag,
                              args.tau_quantile)
        write_json(f"{args.out}/features.json", feats)
        write_json(f"{args.out}/diagnostics.json", diag)
        print(json.dumps(diag["summary"], indent=2))


if __name__ == "__main__":
    main()
