"""Convert the jsonl files from tame.data into verl parquet files.

python -m tame.verl_adapter.data --dataset virl --src data/virl --out data/verl/virl
"""

import argparse
import os
from pathlib import Path

import datasets

from ..common import read_jsonl
from ..prompts import ANSWER_INSTRUCTION, BASE_SYSTEM_PROMPT

SPLITS = ("train", "val", "test", "test_harm", "test_help")


def to_verl_rows(rows, dataset, split):
    """One verl row per sample. Images stay on disk and are referenced by file:// path."""
    out = []
    for i, r in enumerate(rows):
        images = [{"image": "file://" + os.path.abspath(p)} for p in r.get("images", [])]
        user = "<image>" * len(images) + r["question"] + ANSWER_INSTRUCTION
        truth = r["answer"] if dataset == "virl" else r.get("chosen", "")
        out.append({
            "data_source": f"tame_{dataset}",
            "prompt": [{"role": "system", "content": BASE_SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "images": images,
            "ability": "vqa" if dataset == "virl" else "safety",
            "reward_model": {"style": "rule", "ground_truth": truth},
            "extra_info": {"split": split, "index": i, "id": r["id"], "question": r["question"]},
        })
    return out


def convert(src, out, dataset):
    Path(out).mkdir(parents=True, exist_ok=True)
    written = []
    for split in SPLITS:
        path = Path(src) / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = to_verl_rows(read_jsonl(path), dataset, split)
        datasets.Dataset.from_list(rows).to_parquet(str(Path(out) / f"{split}.parquet"))
        written.append((split, len(rows)))
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["virl", "spavl"], required=True)
    ap.add_argument("--src", required=True, help="directory with the jsonl files written by tame.data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    for split, n in convert(args.src, args.out, args.dataset):
        print(f"{split}: {n} rows -> {args.out}/{split}.parquet")


if __name__ == "__main__":
    main()
