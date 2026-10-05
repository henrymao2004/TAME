"""Build train/val/test jsonl files with fields id, question, images, answer (VIRL) or chosen (SPA-VL).

python -m tame.data virl --src ViRL39K --out data/virl
python -m tame.data spavl --out data/spavl [--test_refs refs.jsonl]
"""

import argparse
import os
import random
import re

import pandas as pd

from .common import last_boxed, read_jsonl, write_jsonl


def prepare_virl(src, out, n_train, n_val, n_test, seed):
    df = pd.read_parquet(os.path.join(src, "39Krelease.parquet"))
    rows = []
    for _, r in df.iterrows():
        images = [os.path.abspath(os.path.join(src, p)) for p in (r["image"] if r["image"] is not None else [])]
        question = re.sub(r"<image>(/n|\n)?", "", r["question"]).strip()
        answer = str(r["answer"])
        answer = (last_boxed(answer) or answer).strip().strip("$")
        rows.append({"id": str(r["qid"]), "question": question, "images": images, "answer": answer})
    random.Random(seed).shuffle(rows)
    write_jsonl(f"{out}/train.jsonl", rows[:n_train])
    write_jsonl(f"{out}/val.jsonl", rows[n_train:n_train + n_val])
    write_jsonl(f"{out}/test.jsonl", rows[n_train + n_val:n_train + n_val + n_test])


def prepare_spavl(out, n_train, n_val, seed, test_refs):
    from datasets import load_dataset

    img_dir = os.path.abspath(f"{out}/images")
    os.makedirs(img_dir, exist_ok=True)

    def dump(ds, prefix, idx):
        rows = []
        for i in idx:
            r = ds[i]
            rid = f"{prefix}_{i}"
            path = f"{img_dir}/{rid}.png"
            r["image"].convert("RGB").save(path)
            rows.append({"id": rid, "question": r["question"], "images": [path], "chosen": r.get("chosen") or ""})
        return rows

    rng = random.Random(seed)
    train = load_dataset("sqrti/SPA-VL", "default", split="train")
    val = load_dataset("sqrti/SPA-VL", "validation", split="validation")
    write_jsonl(f"{out}/train.jsonl", dump(train, "train", rng.sample(range(len(train)), n_train)))
    write_jsonl(f"{out}/val.jsonl", dump(val, "val", rng.sample(range(len(val)), n_val)))

    refs = {r["id"]: r["chosen"] for r in read_jsonl(test_refs)} if test_refs else {}
    for split in ("harm", "help"):
        ds = load_dataset("sqrti/SPA-VL", "test", split=split)
        rows = dump(ds, f"test_{split}", range(len(ds)))
        for r in rows:
            r["chosen"] = refs.get(r["id"], "")
        write_jsonl(f"{out}/test_{split}.jsonl", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["virl", "spavl"])
    ap.add_argument("--src", help="ViRL39K directory with 39Krelease.parquet and images/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=1934)
    ap.add_argument("--test_refs", help="jsonl with id and chosen for the SPA-VL test splits")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.dataset == "virl":
        prepare_virl(args.src, args.out, args.n_train, args.n_val, args.n_test, args.seed)
    else:
        prepare_spavl(args.out, args.n_train, args.n_val, args.seed, args.test_refs)


if __name__ == "__main__":
    main()
