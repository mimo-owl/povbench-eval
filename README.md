# POVBench
  
[[Project page](https://mimo-owl.github.io/POVBench/)] &nbsp;|&nbsp; [[Dataset](https://huggingface.co/datasets/owl-owl/POVBench)]

---

## Overview

This repository contains the evaluation code for POVBench, the benchmark introduced in
*Contextual Observer Grounding: Evaluating Situated Spatial Reasoning in
Vision-Language Models* (Findings of EMNLP 2026). The dataset itself lives on the
[Hugging Face Hub](https://huggingface.co/datasets/owl-owl/POVBench).

### Models evaluated

Backends are selected with `--vlm`. API model identifiers are fixed in
`evaluation/run_vlm_eval.py`; the results in the paper were produced in May-July 2026.
API providers retire models over time, so a given identifier may no longer be
available. Open-weight models are unaffected.

| Paper | `--vlm` | Model |
|---|---|---|
| MolmoPoint-8B | `molmopoint-8b` | `allenai/MolmoPoint-8B` |
| GPT-5.4 | `gpt` | `gpt-5.4` |
| Qwen3-VL-8B (Transformer) | `qwen` | `Qwen/Qwen3-VL-8B-Instruct` |
| Qwen3-VL-8B (vLLM) | `qwen3vl-8b-vllm` | `Qwen/Qwen3-VL-8B-Instruct` |
| Qwen3-VL-32B (vLLM) | `qwen3vl-32b` | `Qwen/Qwen3-VL-32B-Instruct` |
| InternVL3-38B | `internvl3-38b-8bit` | `OpenGVLab/InternVL3-38B` (8-bit) |
| Gemma-4 | `gemma` | `gemma-4-26b-a4b-it` |
| Gemini-2.5-Flash | `gemini` | `gemini-2.5-flash` |
| Gemini-Robotics-ER | `gemini-robotics` | `gemini-robotics-er-1.6-preview` |
| RoboPoint | `robopoint` | `wentao-yuan/robopoint-v1-vicuna-v1.5-13b` |
| Llama-3.2-Vision | `llama32-vision-11b` | `meta-llama/Llama-3.2-11B-Vision-Instruct` |

RoboPoint and Llama-3.2-Vision are evaluated under Type C only. `run_vlm_eval.py
--help` lists further backends (`qwen36`, `internvl3-78b-8bit`, `roborefer-8b`, ...)
that are available but not part of the main table.

---

## Repository structure

```
evaluation/
├── run_vlm_eval.py          # Main runner
├── run_vlm_eval_ab0.py      # no annotation
├── run_vlm_eval_ab1.py      # O-Plot
├── run_vlm_eval_ab2.py      # O&A-Plot
├── run_vlm_eval_ab3.py      # CoT
├── run_vlm_eval_ab4.py      # Spatial-CoT
├── eval_results.py          # Compute metrics (clamped L2 + verified-surface GT + response rate)
├── aggregate_runs.py        # Aggregate multiple runs into a long CSV
└── run_all_eval.py          # Batch-evaluate every run + write summary.csv
```

(The `dataset/` directory is downloaded separately — see below.)

---

## Setup

```bash
git clone https://github.com/mimo-owl/povbench-eval.git
cd povbench-eval

# (optional but recommended)
conda create -n povbench python=3.12 -y   # 3.12 or newer
conda activate povbench

pip install -r requirements.txt
cp .env.example .env
# Fill in your API keys in .env
```

This is enough for the API models. See [API keys](#api-keys) and
[Local Qwen inference](#local-qwen-inference).

---

## Dataset

The dataset is hosted on the Hugging Face Hub. Download it into `dataset/POVBench`:

```bash
pip install -U huggingface_hub
hf download owl-owl/POVBench --repo-type dataset --local-dir dataset/POVBench
```

Or from Python:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="owl-owl/POVBench", repo_type="dataset",
                  local_dir="dataset/POVBench")
```

Every command below takes the download location as `--dataset-dir`, so any path
works; `dataset/POVBench` is only the convention used in these examples.

```
dataset/POVBench/
├── meta.json
└── artifacts/
    ├── train_house_00000/
    │   ├── train_house_00000.json          # evaluation pairs for this house
    │   ├── exploration_images/             # exploration images (Type A/B input)
    │   ├── img_from_observer/              # observer images (Type C input)
    │   ├── exploration_log.json
    │   └── per_image_visible_objects.json
    └── ...                                 # 47 houses
```

---

## Running evaluation

Two stages: `run_vlm_eval.py` writes predictions, then `eval_results.py` computes
metrics from them. Use the single-worker flow (simplest) or the multi-worker flow
(faster); both feed the same metrics step. See [Options](#options) for every flag.

### Single worker

**1. Run predictions** → writes `eval_results/gemini_run1/predictions.json`:

```bash
python evaluation/run_vlm_eval.py \
    --dataset-dir dataset/POVBench \
    --output-dir  eval_results/gemini_run1 \
    --vlm         gemini
```

**2. Compute metrics** → writes `eval_results/gemini_run1/eval_results.json`
(per-item clamped L2, per-group response rates, and summary statistics):

```bash
python evaluation/eval_results.py \
    --predictions eval_results/gemini_run1 \
    --dataset-dir dataset/POVBench \
    --output-dir  eval_results/gemini_run1
```

Predictions are written incrementally; re-running with the same `--output-dir`
resumes and skips pairs already done.

### Multiple workers (parallel)

Split one run across `N` processes that share a single `--output-dir`. Give every
process the same `--num-workers N` and a distinct `--worker-id` in `0..N-1`; worker
`W` handles every `N`-th pair and writes its own `predictions_<W>.json`.

**1. Run the workers** (example with `N = 3`; run each in its own terminal / GPU):

```bash
python evaluation/run_vlm_eval.py --dataset-dir dataset/POVBench \
    --output-dir eval_results/qwen_run1 --vlm qwen --worker-id 0 --num-workers 3
python evaluation/run_vlm_eval.py --dataset-dir dataset/POVBench \
    --output-dir eval_results/qwen_run1 --vlm qwen --worker-id 1 --num-workers 3
python evaluation/run_vlm_eval.py --dataset-dir dataset/POVBench \
    --output-dir eval_results/qwen_run1 --vlm qwen --worker-id 2 --num-workers 3
```

This leaves `predictions_0.json`, `predictions_1.json`, `predictions_2.json` in
`eval_results/qwen_run1/`.

**2. Merge the shards** → regroups all predictions into one `<house_id>.json` per
house (mirroring the dataset layout; the `predictions_<W>.json` shards are kept):

```bash
python evaluation/run_vlm_eval.py \
    --merge --num-workers 3 \
    --dataset-dir dataset/POVBench \
    --output-dir  eval_results/qwen_run1
```

**3. Compute metrics** → identical to the single-worker step; just point
`--predictions` at the directory:

```bash
python evaluation/eval_results.py \
    --predictions eval_results/qwen_run1 \
    --dataset-dir dataset/POVBench \
    --output-dir  eval_results/qwen_run1
```

### Options

`run_vlm_eval.py`:

| Option | Default | Description |
|---|---|---|
| `--dataset-dir` | *(required)* | Dataset directory (the one containing `artifacts/`). |
| `--output-dir` | *(required)* | Where predictions are written. |
| `--vlm` | `gemini` | Backend; see [Models evaluated](#models-evaluated) or `--help` for the full list. |
| `--types` | `a b c` | Sentence types to evaluate; any subset of `a b c`. |
| `--qwen-model-dir` | Hugging Face id | Local path to Qwen3-VL weights (`qwen` backend only); see [Local Qwen inference](#local-qwen-inference). |
| `--max-pairs` | *(all)* | Stop after this many predictions — handy for a quick smoke test. |
| `--worker-id` / `--num-workers` | `0` / `1` | Parallel sharding (see [Multiple workers](#multiple-workers-parallel)). |
| `--merge` | *(off)* | Merge the `predictions_<0..N-1>.json` shards into per-house `<house_id>.json` files and exit. |

`eval_results.py`:

| Option | Default | Description |
|---|---|---|
| `--predictions` | *(required)* | A prediction directory (single- or multi-worker output), or a single `predictions.json` file. |
| `--dataset-dir` | *(required)* | Dataset directory (for ground-truth positions). |
| `--output-dir` | *(required)* | Where to write `eval_results.json` and plots. |
| `--no-verified-surface` | *(off)* | Use the raw stored GT placements instead of the stage-3 surface-verified GT (the default, matching the paper). |
| `--no-plot` | *(off)* | Skip plot and overlay-image generation (metrics only). |

### Large local models (`qwen36`)

`--vlm qwen36` loads Qwen3.6-35B-A3B locally and shards it across all visible GPUs
with `device_map="auto"` (no worker loop needed). Weights (~72 GB) download on first
use, so point the cache at a disk with room:

```bash
export HF_HOME=/path/with/space/hf_cache      # avoid filling your home directory
python evaluation/run_vlm_eval.py \
    --vlm         qwen36 \
    --dataset-dir dataset/POVBench \
    --output-dir  eval_results/qwen36_run1
```

Notes:
- Needs a recent `transformers` (≥ 4.57, which knows the `qwen3_5_moe` architecture):
  `pip install -U transformers`.
- Thinking mode is disabled automatically (`enable_thinking=False`) so the reply is the
  answer only, and a leading `<think>…</think>` block, if any, is stripped.
- `--qwen-model-dir <path>` overrides the default Hugging Face id with a local path.

### Chain-of-thought variant (S-CoT)

For Spatial Chain-of-Thought prompting, use `run_vlm_eval_ab4.py` in place of
`run_vlm_eval.py` in either flow above. It takes the **same options**, except
`--types` accepts only `a b` (S-CoT is Type A/B only). Evaluate its output with
`eval_results.py` exactly as before.

```bash
python evaluation/run_vlm_eval_ab4.py \
    --dataset-dir dataset/POVBench \
    --output-dir  eval_results/qwen_cot_run1 \
    --vlm         qwen \
    --types       a b
```

### Aggregating multiple runs (optional)

Evaluate every prediction sub-directory under a root at once and write a
consolidated `summary.csv` / `summary.md` (under `--pred-root`):

```bash
python evaluation/run_all_eval.py \
    --pred-root   eval_results \
    --dataset-dir dataset/POVBench
```

Or collect per-item rows from several runs into one long CSV for your own analysis
(`--run` is `MODEL:RUN_ID:EVAL_DIR`, repeated per run; output `results/all_runs_long.csv`):

```bash
python evaluation/aggregate_runs.py \
    --run gemini:1:eval_results/gemini_run1 \
    --run gemini:2:eval_results/gemini_run2 \
    --output-dir results/
```

---

## Local Qwen inference

To run Qwen locally instead of via API, install the model weights separately
(see [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)) and pass
`--qwen-model-dir <path>`.

---

## API keys

Copy `.env.example` to `.env` and fill in the keys you need:

```
GEMINI_API_KEY=...
OPENAI_API_KEY=...
```

Alternatively, pass them as environment variables inline:

```bash
GEMINI_API_KEY=<key> python evaluation/run_vlm_eval.py ...
```

When running Gemini across several workers you may give each one its own key as
`GEMINI_API_KEY_<worker_id>`; a worker falls back to `GEMINI_API_KEY`, then
`GOOGLE_API_KEY`.

### Backend-specific setup

Most `--vlm` choices need nothing beyond the two API keys. These do:

| Backend | Environment variables |
|---|---|
| `qwen3vl-8b-vllm`, `qwen3vl-32b` | `VLLM_BASE_URL` (default `http://localhost:8000/v1`), `VLLM_MODEL` |
| `molmopoint-8b` | `MOLMO_MODEL` — needs `transformers==4.57.1` in a separate environment |
| `robopoint` | `ROBOPOINT_MODEL` — install the external `robopoint` package |
| `roborefer-8b`, `roborefer-8b-depth` | `ROBOREFER_URL` (default `http://localhost:25547`) |
| `internvl3-*` | `INTERNVL_MODEL` |
| `llama32-vision-*` | `LLAMA_MODEL` — requires accepting the model licence on the Hub |

---

## Citation

```bibtex
@inproceedings{shirasaka2026povbench,
  title     = {Contextual Observer Grounding: Evaluating Situated Spatial
               Reasoning in Vision-Language Models},
  author    = {Shirasaka, Mimo and Zhang, Haochen and Bisk, Yonatan},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```
