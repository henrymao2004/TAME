<h2 align="center">TAME</h2>
<p align="center"><b>Taming CoT Obfuscation in VLMs: From Mechanistic Evidence to Activation-Level Enforcement</b></p>

<p align="center">
  <a href="assets/tame-overview.pdf">
    <img src="assets/tame-overview.png" width="100%" alt="TAME overview">
  </a>
</p>
<p align="center"><sub>Overview of the TAME training pipeline. Click for the PDF.</sub></p>

<p align="center">
  <img src="https://img.shields.io/badge/NeurIPS-2026-4B2E83?style=flat" alt="NeurIPS 2026"> ·
  <a href="https://arxiv.org/abs/2609.24243"><img src="https://img.shields.io/badge/arXiv-2609.24243-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="arXiv 2609.24243"></a> ·
  <a href="https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Qwen3--VL--8B-yellow?style=flat" alt="Qwen3-VL-8B"></a> ·
  <a href="https://huggingface.co/datasets/TIGER-Lab/ViRL39K"><img src="https://img.shields.io/badge/%F0%9F%A4%97-ViRL39K-yellow?style=flat" alt="ViRL39K"></a> ·
  <a href="https://huggingface.co/datasets/sqrti/SPA-VL"><img src="https://img.shields.io/badge/%F0%9F%A4%97-SPA--VL-yellow?style=flat" alt="SPA-VL"></a> ·
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python 3.10+">
</p>

Reinforcement learning improves reasoning in vision-language models but can induce **CoT obfuscation**:
task reward rises while reasoning traces become less grounded and less monitorable. This repository
contains the code to measure that decline, localize its representational correlates, test their
causal contribution, and train with **TAME**, **T**argeted **A**nti-obfuscation with **M**echanistic
**E**nforcement.

## 📰 News

- **[2026-09]** TAME is accepted to **NeurIPS 2026**.
- **[2026-09]** The paper is on [arXiv](https://arxiv.org/abs/2609.24243).

## 🔬 TAME at a glance

TAME adds two complementary objectives to GRPO.

| Objective | Level | What it does |
|---|---|---|
| $\mathcal{L}_{\mathrm{behav}}$ | Behavior | A prompt updater evolves a refinement prompt every 10 steps from batch attempts and rewards. Refined traces whose reward matches or exceeds the original attempt are re-paired with the original prompt and trained with GRPO and a KL penalty to the initial policy. |
| $\mathcal{L}_{\mathrm{SAE}}$ | Representation | A frozen TopK SAE encodes target-layer activations of the policy and of the frozen pre-RL model. Template-associated latents are penalized only above their pre-RL baseline. |

$$
\mathcal{L}_{\mathrm{SAE}}=\sum_{i\in\mathcal{F}_{\mathrm{template}}}\mathrm{ReLU}\left(a_i^{\mathrm{student}}-a_i^{\mathrm{base}}\right)^2,
\qquad
\mathcal{L}_{\mathrm{TAME}}=\mathcal{L}_{\mathrm{behav}}+\gamma\,\mathcal{L}_{\mathrm{SAE}}
$$

with $\gamma=0.1$ on VIRL and $\gamma=0.05$ on SPA-VL.

## 🏗️ How it works

| Stage | Question | Module | Output |
|---|---|---|---|
| **Phenomenon** | Does RL raise reward while lowering monitorability? | `tame.monitor` | G-mean², TPR, TNR, monitor accuracy |
| **Localization** | Which layers and heads shift toward template routing? | `tame.attention`, `tame.attribution` | Normalized Cheating Index, head patching, Integrated Gradients |
| **Representation** | Do template and ground activations become less separable? | `tame.sae`, `tame.features` | CES, template precision and recall, JSD |
| **Causal test** | Do the selected features contribute beyond random ones? | `tame.generate --intervene` | Ablation and injection with matched random controls |
| **Mitigation** | Can training restore monitorability? | `tame.train` | GRPO, DAPO, CoT-monitor reward, TAME and its ablations |

Monitorability is the balanced recoverability of a **two-gate reference label**. On VIRL a trace is
labeled **A** when a text-only DeepSeek-V3.2 answers correctly from the CoT alone and an image-aware
Gemini-2.5-Flash confirms its visual evidence. On SPA-VL it needs a safety reward of at least 0.8 and
a Gemini-2.5-Flash auditor judging the trace policy-auditable. A GPT-4o predictor sees the question,
CoT and answer without the image and predicts the label, and
$\mathrm{G\text{-}mean}^2=\mathrm{TPR}\times\mathrm{TNR}$.

## 🚀 Quick start

Requirements: Python 3.10+, CUDA GPUs, and an OpenAI-compatible API key.

```bash
git clone https://github.com/henrymao2004/TAME.git
cd TAME
pip install -r requirements.txt
export OPENROUTER_API_KEY=<api-key>
```

All judges, annotators and monitors go through OpenRouter by default. Set `LLM_BASE_URL` and
`LLM_API_KEY` to use another OpenAI-compatible endpoint.

Prepare the data:

```bash
# VIRL-39k: download TIGER-Lab/ViRL39K and unzip images.zip
python -m tame.data virl --src ViRL39K --out data/virl

# SPA-VL: test references are passed as a jsonl with id and chosen
python -m tame.data spavl --out data/spavl --test_refs spavl_test_refs.jsonl
```

Each example is one json line with `id`, `question`, `images`, and `answer` for VIRL or `chosen` for SPA-VL.

## 🏋️ Training

```bash
M=Qwen/Qwen3-VL-8B-Instruct
COMMON="--dataset virl --model $M --train data/virl/train.jsonl"

# GRPO, checkpoints every 10 steps
torchrun --nproc_per_node 8 -m tame.train --method grpo $COMMON --out runs/virl_grpo

# DAPO and the CoT-monitor reward baseline
torchrun --nproc_per_node 8 -m tame.train --method dapo $COMMON --out runs/virl_dapo
torchrun --nproc_per_node 8 -m tame.train --method grpo --cot_mon_weight 0.5 $COMMON --out runs/virl_cotmon

# TAME
torchrun --nproc_per_node 8 -m tame.train --method tame $COMMON \
    --sae sae/virl --features analysis/virl_step200/features.json --out runs/virl_tame
```

| Variant | Flags |
|---|---|
| **TAME** | `--method tame` |
| SAE-only | `--method tame --no_feedback` |
| Behav-only | `--method tame --gamma 0` |
| Behav + Rand-SAE | `--method tame --random_features` |

Use `--dataset spavl` with `data/spavl/*` for SPA-VL. The SAE and feature files come from the
mechanistic analysis below.

## 📏 Monitorability evaluation

```bash
python -m tame.generate --model runs/virl_tame/step_200 --data data/virl/test.jsonl --out traces/virl_tame.jsonl
python -m tame.monitor --traces traces/virl_tame.jsonl --dataset virl --out traces/virl_tame_labeled.jsonl

# held-out monitors on the same frozen traces and reference labels
python -m tame.monitor --traces traces/virl_tame_labeled.jsonl --dataset virl \
    --predictor minimax/minimax-m3 --out traces/virl_tame_labeled.jsonl
python -m tame.monitor --traces traces/virl_tame_labeled.jsonl --dataset virl \
    --predictor moonshotai/kimi-k2.6 --out traces/virl_tame_labeled.jsonl
```

Each run writes a metrics file with G-mean², TPR, TNR, monitor accuracy, and task accuracy or reward.

## 🔍 Mechanistic analysis

Traces, token-type annotation, and step-0 SAEs:

```bash
python -m tame.generate --model $M --data data/virl/val.jsonl --n 500 --out traces/step0.jsonl
python -m tame.features annotate --traces traces/step0.jsonl --out traces/step0_annot.jsonl
python -m tame.sae --model $M --traces traces/step0.jsonl --dataset virl --out sae/virl
```

Repeat generation and annotation for `runs/virl_grpo/step_200` to get `traces/step200_annot.jsonl`.

Attention screening, head ranking, head patching, and Integrated Gradients:

```bash
python -m tame.attention screen --model $M --traces traces/step0_annot.jsonl --out analysis/ci_step0.json
python -m tame.attention screen --model runs/virl_grpo/step_200 --traces traces/step200_annot.jsonl --out analysis/ci_step200.json
python -m tame.attention rank --early analysis/ci_step0.json --late analysis/ci_step200.json --out analysis/heads.json
python -m tame.attention patch --early_model $M --late_model runs/virl_grpo/step_200 \
    --traces traces/step200_annot.jsonl --heads analysis/heads.json --out analysis/patch.json

python -m tame.attribution --model runs/virl_grpo/step_200 --traces traces/step200_annot.jsonl --out analysis/ig_step200.json
python -m tame.attribution --model runs/virl_grpo/step_200 --traces traces/step200_annot.jsonl --out analysis/ig_patched.json \
    --patch_from $M --heads analysis/heads.json
```

Template-feature selection, SAE diagnostics, and controlled interventions:

```bash
python -m tame.features analyze --model runs/virl_grpo/step_200 --sae sae/virl \
    --traces traces/step200_annot.jsonl --out analysis/virl_step200

FEATS="--sae sae/virl --features analysis/virl_step200/features.json"
python -m tame.generate --model runs/virl_grpo/step_200 --data data/virl/test.jsonl --out traces/ablate.jsonl $FEATS --intervene ablate
python -m tame.generate --model runs/virl_grpo/step_200 --data data/virl/test.jsonl --out traces/ablate_rand.jsonl $FEATS --intervene ablate --random
python -m tame.generate --model $M --data data/virl/test.jsonl --out traces/inject.jsonl $FEATS --intervene inject
python -m tame.generate --model $M --data data/virl/test.jsonl --out traces/inject_rand.jsonl $FEATS --intervene inject --random
```

For SAE-source robustness, pass several models with their traces to `tame.sae`, for example
`--model $M runs/virl_grpo/step_200 --traces traces/step0.jsonl traces/step200.jsonl` for the 50/50 mixture.

## 📁 Repository layout

```text
.
├── assets/             # overview figure (PNG + vector PDF)
├── tame/
│   ├── data.py         # VIRL-39k and SPA-VL preparation
│   ├── prompts.py      # monitor, reward, gate, annotation, and refinement prompts
│   ├── common.py       # model loading, encoding, generation, answer parsing
│   ├── monitor.py      # rewards, two-gate reference labels, monitor predictions, G-mean²
│   ├── generate.py     # trace generation and SAE ablation / injection
│   ├── sae.py          # TopK SAE training
│   ├── features.py     # token-type annotation, Cohen's d selection, CES / precision / recall / JSD
│   ├── attention.py    # decision-point attention, normalized CI, head screening and patching
│   ├── attribution.py  # layer-wise Integrated Gradients
│   └── train.py        # GRPO, DAPO, TAME, and ablations
└── requirements.txt
```

## Citation

```bibtex
@article{mao2026tame,
  title   = {Taming CoT Obfuscation in VLMs: From Mechanistic Evidence to Activation-Level Enforcement},
  author  = {Xutao Mao and Jianing Zhu and Jinman Zhao and Tongliang Liu and Xiaowen Chu and Cong Wang and Bo Han},
  journal = {arXiv preprint arXiv:2609.24243},
  year    = {2026},
  note    = {Accepted to NeurIPS 2026}
}
```

## 🙏 Acknowledgements

- [Transformers](https://github.com/huggingface/transformers) and [PEFT](https://github.com/huggingface/peft)
  provide the model and LoRA implementations.
- [ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K) and [SPA-VL](https://huggingface.co/datasets/sqrti/SPA-VL)
  provide the training and evaluation data.
- The behavioral feedback loop builds on experiential reinforcement learning.
