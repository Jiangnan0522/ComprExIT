# ComprExIT: Context Compression via Explicit Information Transmission

Official implementation of **ComprExIT**, a lightweight framework for *soft* context
compression in Large Language Models. Instead of re-purposing the LLM as a trainable
compressor that aggregates context through layer-by-layer self-attention, ComprExIT
formulates compression as **explicit information transmission over frozen LLM hidden
states**, performing coordinated **depth-wise** (inter-layer) and **width-wise**
(token-to-slot) transmission. This decouples compression from the model's internal
self-attention dynamics, mitigating progressive representation overwriting and enabling
globally-coordinated, controllable aggregation.

📄 **Paper:** [Context Compression via Explicit Information Transmission](https://arxiv.org/abs/2602.03784)

> **Method-name map.** In the paper the method is called **ComprExIT**. In the code it is
> selected with `--model_structure hier --pooling_method ot-dy-src` (the Optimal-Transport
> dynamic-source pooling that realises the width-wise transmission plan). The repository
> package is historically named `CompressIn`; the two names refer to the same project.

---

## ✨ Reproduce it in a few commands

We worked hard to make ComprExIT **painless to reproduce** — no scavenger hunts for weights or
data, no half-documented glue scripts. If you can copy-paste, you can reproduce the paper. 🎯

- 🤗 **Weights & data live on HuggingFace.** Every released checkpoint and the preprocessed
  datasets are public — so you never have to retrain or rebuild anything just to *evaluate*.
- 🧪 **Evaluation, end-to-end.** Three little steps, each a ready-to-run command: **download a
  checkpoint → download the datasets → run eval** (see [Pretrained weights](#pretrained-weights),
  [Data](#data), and [Evaluation](#evaluation)). Point it at a checkpoint and go.
- 🏋️ **Training, fully scripted.** Both phases (NTP → SFT) ship as one-line wrappers
  pre-configured for the paper's setup — just pass your base model / checkpoint (see
  [Training](#training)).
- 🧩 **Baselines, batteries included.** ICAE (`--model_structure icae`), 500x (`500x`), and
  Activation Beacon all run through the same machinery, so apples-to-apples comparison is one
  flag away.
- 🧬 **Bonus — SAC.** We also threw in an implementation of **SAC** (`--model_structure sac`) for
  easy side-by-side comparison. Fair warning: it's only been validated on the **1B** setting,
  isn't fully tested, and isn't part of the paper — treat it as experimental. 🚧
- 🔭 **More on the way.** We plan to keep adding implementations of other compression methods —
  contributions and requests welcome! 💛

---

## 🗓️ Milestones

Where we are, and where we're headed — we'd love your help ticking more of these off! 🎉

- [x] 🚀 Release **ComprExIT** — method + two-phase training (NTP → SFT) & evaluation code
- [x] 🧩 Add the **ICAE** baseline (`--model_structure icae`)
- [x] 🧩 Add the **500x** baseline (`--model_structure 500x`)
- [x] 🧩 Add the **Activation Beacon** baseline
- [x] 🤗 Upload **datasets** to HuggingFace
- [x] 🤗 Upload **checkpoints / weights** to HuggingFace
- [x] 🧬 Add the **SAC** implementation (`--model_structure sac`; experimental, 1B-only)
- [ ] 🔭 Add more compression-method implementations
- [ ] 💬 *Your idea here* — open an issue or PR!

---

## Method overview

| Paper concept | Where in the code |
|---|---|
| ComprExIT (depth-wise + width-wise transmission) | `--model_structure hier --pooling_method ot-dy-src` |
| Depth-wise gating over frozen layers (token anchors) | `src/model/pooling.py`, `--top_k_layers`, `--layerwise_pooling_*` |
| Width-wise transmission plan (Sinkhorn / OT) | `src/model/pooling.py`, `--ot_window_size`, `--ot_n_iter`, `--ot_metric_dim` |
| Two-phase training (NTP → SFT) | `train.py --mode {ntp,sft}` |

### Baselines (reported in the paper)

| Baseline | `--model_structure` |
|---|---|
| ICAE | `icae` |
| 500x | `500x` |
| Activation Beacon | [`src/baselines/activation_beacon/`](src/baselines/activation_beacon/README.md) (separate codebase, env & train/eval) |
| Prompt-tuning / Zero-shot | eval modes in `src/evaluation/eval_datasets.py` |

---

## Repository layout

```
train.py                         # main training entry point (NTP & SFT), launched via torchrun
src/
  model/        model.py         # all model classes + get_model_factory()
                pooling.py       # Pooling subclasses (ot-dy-src = ComprExIT)
                inference.py     # compress-and-generate helpers
  data_processing/               # NTP + SFT preprocessing, data loading
  training/     trainer.py       # CompressInTrainer (DDP loss averaging, SFT eval)
  evaluation/   eval_datasets.py # QA evaluator (in-domain + OOD partitions), metrics
  baselines/activation_beacon/   # Activation Beacon baseline
scripts/
  downloading/                   # dataset / model download helpers
  llama-3.2/
    ntp/ ntp_3b/                 # NTP pretraining (1B / 3B) for each method
    sft/ sft_3b/                 # supervised fine-tuning (1B / 3B) for each method
configs/                         # config templates
```

---

## Installation

We use [`uv`](https://docs.astral.sh/uv/) for fast, reproducible environments. If you
don't have it: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

```bash
uv sync                 # create the venv and install pinned deps (Python 3.9)
uv sync --extra viz     # also install plotting deps (paper-figure reproduction)
```

The training scripts under `scripts/` already wrap their launch in `uv run`, so you can
run them directly — no manual activation needed:

```bash
bash scripts/llama-3.2/ntp/512_ot_2mlp_1b.sh
```

For ad-hoc commands, prefix them with `uv run` (e.g. `uv run python train.py ...`), or
activate the venv once with `source .venv/bin/activate`. All `python ...` commands in this
README assume one of those.

The code uses [`liger_kernel`](https://github.com/linkedin/Liger-Kernel) for efficient GPU
kernels and runs on CUDA GPUs (DDP via `nccl`). Training is launched with `torchrun`.

> **torch builds.** The default PyPI `torch==2.8.0` wheel on Linux x86_64 bundles a CUDA
> 12.8 build, which works on most NVIDIA GPUs. For CPU-only or a different CUDA version,
> uncomment the matching `[[tool.uv.index]]` / `[tool.uv.sources]` block in
> `pyproject.toml` and re-run `uv lock && uv sync` (or see [pytorch.org](https://pytorch.org)).

> **Want a plain `requirements.txt`?** `pyproject.toml` + `uv.lock` are the source of truth.
> If you need a pip-style requirements file (e.g. for a different toolchain), generate one
> with `uv export --format requirements-txt > requirements.txt`.

> **Activation Beacon baseline.** This `uv` environment covers ComprExIT and the ICAE / 500x /
> prompt-tuning baselines. The **Activation Beacon** baseline is a separate, self-contained
> codebase with its own heavier dependencies (`deepspeed`, `flash-attn`, `vllm`, `fuzzywuzzy`,
> `jieba`, …) that are **not** installed by `uv sync`. Anything Beacon-related — the
> `beacon.sh` / `mrqa_beacon*.sh` scripts and evaluating a Beacon checkpoint (loaded lazily by
> `src/evaluation/utils_eval.py`) — needs that environment. See
> [`src/baselines/activation_beacon/README.md`](src/baselines/activation_beacon/README.md) for
> its setup and usage.

---

## Configuration (paths)

Two kinds of paths, handled differently:

- **Stable locations → environment variables.** They rarely change once set, so put them in
  your shell (copy `.env.example` to `.env` and edit, or `export` them):

  | Variable | Meaning |
  |---|---|
  | `DATA_DIR`   | Root directory holding the datasets |
  | `OUTPUT_DIR` | Where runs write checkpoints/logs (default `./training_outputs`) |
  | `MASTER_ADDR` | *(multi-node only)* `torchrun` rendezvous host; unset ⇒ current hostname |

- **The model / checkpoint you operate on → a script argument.** This changes every run, so
  you pass it explicitly rather than via the environment:

  | What | How it's passed |
  |---|---|
  | Base model (NTP & prompt-tuning scripts) | **first argument** to the script — local path or HF repo id |
  | NTP checkpoint to fine-tune (SFT scripts) | **first argument** to the SFT script |
  | Checkpoint to evaluate | `--model_folders <path>` on `eval_datasets.py` |

  Each script aborts with a usage message if the required argument is missing.

---

## Data

We use **SlimPajama-DC** (1B tokens) for NTP pretraining and **MRQA** for SFT and evaluation
(6 in-domain: SQuAD, NewsQA, TriviaQA, SearchQA, HotpotQA, NaturalQuestions; 6 out-of-domain:
BioASQ, DROP, DuoRC, RACE, RelationExtraction, TextbookQA).

**MRQA (SFT + evaluation).** We host the preprocessed MRQA splits as HuggingFace datasets.
Download them into `$DATA_DIR` with the directory layout the loader expects — each repo is a
tree of `<subset>/<split>_<name>.jsonl.gz` files, so fetch the files verbatim with
`hf download` (not `load_dataset`):

```bash
hf download Jiang-nan/comprexit-mrqa     --repo-type dataset --local-dir "$DATA_DIR/mrqa"
hf download Jiang-nan/comprexit-mrqa-ood --repo-type dataset --local-dir "$DATA_DIR/mrqa_ood"
```

**SlimPajama-6B (NTP pretraining).** Downloaded directly from its HuggingFace repo:

```bash
uv run python scripts/downloading/download_dataset.py \
    --repo_id DKYoon/SlimPajama-6B --target_dir "$DATA_DIR/slim_pajama_6b"
```

---

## Pretrained weights

All released checkpoints are gathered in the
**🤗 [ComprExIT collection](https://huggingface.co/collections/Jiang-nan/comprexit-6a340cd07b4f45c96fc6da65)**.

We release ComprExIT and ICAE-baseline checkpoints at **×4 compression** for both training
phases (NTP pretraining and MRQA SFT), spanning Llama-3.2 **1B**/**3B** and Llama-3.1 **8B**,
plus **1B long-context (8192-token)** variants. The `{ntp,sft}` suffix selects the phase.

| Variant | ComprExIT | ICAE baseline |
|---|---|---|
| 1B · 512-ctx · 4× | `Jiang-nan/comprexit-1b-4x-{ntp,sft}` | `Jiang-nan/icae-1b-4x-{ntp,sft}` |
| 3B · 512-ctx · 4× | `Jiang-nan/comprexit-3b-4x-{ntp,sft}` | `Jiang-nan/icae-3b-4x-{ntp,sft}` |
| 8B · 512-ctx · 4× | `Jiang-nan/comprexit-8b-4x-{ntp,sft}` | `Jiang-nan/icae-8b-4x-{ntp,sft}` |
| 1B · 8192-ctx · 4× | `Jiang-nan/comprexit-1b-4x-long8192-{ntp,sft}` | `Jiang-nan/icae-1b-4x-long8192-{ntp,sft}` |

```bash
uv run python scripts/downloading/download_model.py \
    --repo_id Jiang-nan/comprexit-1b-4x-sft --target_dir ./checkpoints/comprexit-1b-4x-sft
```

> **Weights license.** The released checkpoints are derived from Meta's Llama-3.2 models and
> are therefore subject to the [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE).
> The **code** in this repository is licensed under Apache-2.0 (see `LICENSE`).

---

## Training

ComprExIT uses a two-phase procedure: NTP pretraining, then SFT on QA. Set `DATA_DIR` /
`OUTPUT_DIR` first (see above). Each script takes the model/checkpoint path as its **first
argument**; scripts are pre-configured for the paper's setup (×4 compression, 512-token
context, BF16).

```bash
# 1) NTP pretraining — ComprExIT 1B. Arg = base model (local path or HF repo id).
bash scripts/llama-3.2/ntp/512_ot_2mlp_1b.sh meta-llama/Llama-3.2-1B

# 2) SFT on MRQA — arg = the NTP checkpoint produced by step 1.
bash scripts/llama-3.2/sft/mrqa_ot_2mlp_1b.sh \
    "$OUTPUT_DIR/512-128-1b-ot-learnable-a-ntp1.0-gpu/checkpoint-763"
```

> **Treat the per-experiment scripts as examples.** `512_icae.sh`, `512_500x.sh`,
> `beacon.sh`, the prompt-tuning scripts, and the `ntp_3b/` + `sft_3b/` variants all follow
> the same pattern — they differ only in a few hyperparameters (model size, batch size /
> grad-accum to fit GPU memory, method-specific knobs). Each `.sh` is a thin wrapper that
> sets those values and calls `train.py` via `torchrun`. Copy the closest one and adjust;
> see `train.py --help` for the full argument list.

---

## Evaluation

`src/evaluation/eval_datasets.py` reproduces the QA tables. Choose the dataset partition:

```bash
# In-domain (Table 1 / Table 2) — pass the checkpoint dir(s) to --model_folders
uv run python -m src.evaluation.eval_datasets \
    --model_folders ./checkpoints/comprexit-1b-4x-sft --partition mrqa --compress_ratio 4

# Out-of-domain (Table 5)
uv run python -m src.evaluation.eval_datasets \
    --model_folders ./checkpoints/comprexit-1b-4x-sft --partition mrqa_ood --compress_ratio 4
```

Metrics (Exact Match, F1) are computed by `src/evaluation/metrics.py`.

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{ye2026fixstructuralbottleneckcontext,
      title={Fix the Structural Bottleneck: Context Compression via Explicit Information Transmission}, 
      author={Jiangnan Ye and Hanqi Yan and Zhenyi Shen and Heng Chang and Ye Mao and Yulan He},
      year={2026},
      eprint={2602.03784},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.03784}, 
}
```
<!-- TODO(release): fill in the author list. -->

## License

Code: Apache-2.0 (see [`LICENSE`](LICENSE)). Released model weights are additionally subject
to the Llama 3.2 Community License (see above).
