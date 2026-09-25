"""Shared utilities: dataset presets, I/O, model loading, prompt encoding, generation, answer parsing."""

import json
import os
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .prompts import ANSWER_INSTRUCTION, BASE_SYSTEM_PROMPT

PRESETS = {
    "virl": {"layers": [12, 13, 20], "gamma": 0.1},
    "spavl": {"layers": [8, 10, 18], "gamma": 0.05},
}
MODULES = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]
CATEGORIES = ["GROUND", "ENTITY", "TEMPLATE"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_KEYS = ("input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids")


def module_keys(layers, modules=MODULES):
    return [f"layers.{l}.{m}" for l in layers for m in modules]


def layer_of(key):
    return int(key.split(".")[1])


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_images(sample):
    return [Image.open(p).convert("RGB") for p in sample.get("images", [])]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model(path, device=DEVICE, attn="sdpa", max_pixels=1003520):
    """Load a VLM checkpoint; LoRA checkpoints are merged into their base model."""
    adapter_cfg = Path(path) / "adapter_config.json"
    base = json.load(open(adapter_cfg))["base_model_name_or_path"] if adapter_cfg.exists() else path
    model = AutoModelForImageTextToText.from_pretrained(
        base, dtype=torch.bfloat16, attn_implementation=attn
    ).to(device)
    if adapter_cfg.exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, path).merge_and_unload()
    model.eval()
    return model, load_processor(base, max_pixels)


def load_processor(path, max_pixels=1003520):
    processor = AutoProcessor.from_pretrained(path)
    ip = getattr(processor, "image_processor", None)
    if ip is not None and max_pixels:
        if hasattr(ip, "max_pixels"):
            ip.max_pixels = max_pixels
        if isinstance(getattr(ip, "size", None), dict) and "longest_edge" in ip.size:
            ip.size["longest_edge"] = max_pixels
    processor.tokenizer.padding_side = "left"
    return processor


def text_config(model):
    cfg = model.config
    return getattr(cfg, "text_config", cfg)


def image_token_id(model):
    cfg = model.config
    for name in ("image_token_id", "image_token_index"):
        if getattr(cfg, name, None) is not None:
            return getattr(cfg, name)
    return None


def decoder_layers(model):
    """The ModuleList of language-model decoder layers."""
    for name, mod in model.named_modules():
        if name.endswith("language_model.layers") and isinstance(mod, torch.nn.ModuleList):
            return mod
    raise ValueError("language_model.layers not found")


def get_module(model, key):
    """Resolve a key such as 'layers.12.self_attn.q_proj' inside the language model (works under PEFT)."""
    suffix = "language_model." + key
    for name, mod in model.named_modules():
        if name.endswith(suffix):
            return mod
    raise KeyError(key)


class Recorder:
    """Records the outputs of selected modules while active."""

    def __init__(self, model, keys, detach=False):
        self.out, self.active, self.detach = {}, False, detach
        self.handles = [get_module(model, k).register_forward_hook(self._hook(k)) for k in keys]

    def _hook(self, key):
        def fn(module, inputs, output):
            if self.active:
                y = output[0] if isinstance(output, tuple) else output
                self.out[key] = y.detach() if self.detach else y
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


# ---------------------------------------------------------------------------
# Prompts and encoding
# ---------------------------------------------------------------------------
def build_messages(sample, system=BASE_SYSTEM_PROMPT):
    content = [{"type": "image"} for _ in sample.get("images", [])]
    content.append({"type": "text", "text": sample["question"] + ANSWER_INSTRUCTION})
    messages = [{"role": "system", "content": system}] if system else []
    return messages + [{"role": "user", "content": content}]


def encode_prompt(processor, sample, system=BASE_SYSTEM_PROMPT, images=None):
    """Processor outputs (batch of one) for the chat prompt with the generation header."""
    images = load_images(sample) if images is None else images
    text = processor.apply_chat_template(build_messages(sample, system), tokenize=False, add_generation_prompt=True)
    return dict(processor(text=[text], images=images or None, return_tensors="pt"))


def append_response(enc, response_ids):
    """Concatenate response token ids to an encoded prompt."""
    out = dict(enc)
    resp = torch.tensor([response_ids], dtype=enc["input_ids"].dtype)
    out["input_ids"] = torch.cat([enc["input_ids"], resp], 1)
    out["attention_mask"] = torch.ones_like(out["input_ids"])
    for k in ("token_type_ids", "mm_token_type_ids"):
        if k in enc:
            out[k] = torch.cat([enc[k], torch.zeros_like(resp)], 1)
    return out


def encode_trace(processor, sample, response, system=BASE_SYSTEM_PROMPT):
    """Encode prompt + response text. Returns inputs, response start, and response-token char offsets."""
    enc = encode_prompt(processor, sample, system)
    tok = processor.tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    start = enc["input_ids"].shape[1]
    return append_response(enc, tok["input_ids"]), start, tok["offset_mapping"]


def collate(encs, pad_id, left=False):
    """Pad sequence tensors and concatenate vision tensors of single-sample encodings."""
    L = max(e["input_ids"].shape[1] for e in encs)
    out = {}
    for k in encs[0]:
        if k in SEQ_KEYS:
            pad = pad_id if k == "input_ids" else 0
            rows = []
            for e in encs:
                x = e[k]
                p = torch.full((1, L - x.shape[1]), pad, dtype=x.dtype)
                rows.append(torch.cat([p, x] if left else [x, p], 1))
            out[k] = torch.cat(rows, 0)
        else:
            out[k] = torch.cat([e[k] for e in encs], 0)
    return out


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def decision_index(response, offsets):
    """Index of the last response token before 'Final:' (the decision point)."""
    cut = response.rfind("Final:")
    if cut < 0:
        return len(offsets) - 1
    idx = [i for i, (s, e) in enumerate(offsets) if e <= cut]
    return idx[-1] if idx else 0


@torch.no_grad()
def generate(model, processor, encs, max_new_tokens=1024, temperature=1.0, batch_size=16):
    """Sample one response per encoding. Returns lists of response token ids (ending at EOS) and texts."""
    tok = processor.tokenizer
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else next(iter(eos))
    device = next(model.parameters()).device
    ids_out, texts = [], []
    for i in range(0, len(encs), batch_size):
        chunk = encs[i:i + batch_size]
        batch = to_device(collate(chunk, pad_id, left=True), device)
        kwargs = dict(do_sample=temperature > 0, max_new_tokens=max_new_tokens, pad_token_id=pad_id)
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=1.0, top_k=0)
        out = model.generate(**batch, **kwargs)[:, batch["input_ids"].shape[1]:]
        for row in out.tolist():
            ids = []
            for t in row:
                ids.append(t)
                if t in eos:
                    break
            ids_out.append(ids)
            texts.append(tok.decode(ids, skip_special_tokens=True))
    return ids_out, texts


# ---------------------------------------------------------------------------
# Answer parsing and the VIRL reward
# ---------------------------------------------------------------------------
def last_boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    i += len("\\boxed{")
    depth, j = 1, i
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i:j - 1] if depth == 0 else None


def split_cot(response):
    """Split a response into (CoT, final answer)."""
    i = response.rfind("Final:")
    if i >= 0:
        cot, final = response[:i], response[i + len("Final:"):]
    else:
        cot, final = response, last_boxed(response) or ""
    boxed = last_boxed(final)
    return cot.strip(), (boxed if boxed is not None else final).strip()


def _norm(s):
    s = str(s).strip()
    s = last_boxed(s) or s
    s = re.sub(r"^(answer|final answer)\s*[:：]\s*", "", s, flags=re.I)
    return s.replace("$", "").replace(" ", "").rstrip(".").lower()


def answer_match(pred, gt):
    """Binary answer matching for VIRL."""
    p, g = _norm(pred), _norm(gt)
    if not p:
        return False
    if re.fullmatch(r"[a-h]", g):
        m = re.match(r"^\(?([a-h])\)?(?![a-z])", p)
        return bool(m) and m.group(1) == g
    if p == g:
        return True
    try:
        from math_verify import parse, verify
        return bool(verify(parse(f"${gt}$"), parse(f"${pred}$")))
    except Exception:
        return False


def virl_reward(response, answer):
    return float(answer_match(split_cot(response)[1], answer))


def dist_info():
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
