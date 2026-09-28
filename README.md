# FactorJEPA: World Models for Crowded, Chaotic Global South Urban Scenes

**Factorizing monolithic future prediction into layout, agent, and interaction channels, trained and evaluated on DenseWorld.**

> This fork also contains independent future-latent RGB readout experiments. For their
> setup, training, inference, and checkpoint instructions, see
> [Future prediction readout experiments](#future-prediction-readout-experiments)
> below. The original FactorJEPA pipeline is documented separately in the sections
> immediately following this note.

[![arXiv](https://img.shields.io/badge/arXiv-2608.01049-b31b1b?logo=arxiv)](https://arxiv.org/abs/2608.01049)
[![Project Page](https://img.shields.io/badge/Project-Page-8B3A2A)](https://kapilw25.github.io/factorjepa/)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-ffd21e?logo=huggingface)](https://huggingface.co/datasets/anonymousML123/denseworld-115k)

> **115,687 clips** | 714 videos | 22 cities | 276 hours | 121 GB

**[See 192 video clips from all cities and taxonomy categories on the project page](https://kapilw25.github.io/factorjepa/)**

---

## Key Finding

Frozen video encoders are **nearly motion-blind** on crowded Global South scenes; factor-view predictor surgery (**FactorJEPA**) restores predictive structure. With an *identical probe head* over either backbone, only the FactorJEPA-adapted encoder answers motion questions reliably:

| Motion probe (1,825 held-out clips) | Frozen V-JEPA 2.1 | FactorJEPA (ours) |
|-------------------------------------|-------------------|-------------------|
| Motion-speed quartile (chance 25.6%) | 60.9% | **69.8%** |

At the full 115k-clip scale, FactorJEPA separates from the strongest fine-tuning rival on **all four predictive diagnostics** (in 95%-CI units): mask-ratio slope **43.3×**, future-frame L1 **33.2×**, motion-cosine **20.0×**, causal L1 **13.9×**. Method rankings replicate across the 2B and 1B V-JEPA 2.1 backbones (Spearman ρ = 0.895–0.978).

📄 **[Read the paper on arXiv (2608.01049)](https://arxiv.org/abs/2608.01049)**

---

## Setup

```bash
git clone https://github.com/kapilw25/factorjepa.git && cd factorjepa
./setup_env_uv.sh --gpu          # Nvidia GPU server (installs PyTorch, FAISS-GPU, cuML, FA2)
# or: ./setup_env_uv.sh --mac    # M1 Mac (CPU-only, for development/testing)
source venv_walkindia/bin/activate
```

---

## Pipeline

Five scripts, single responsibility each. All use checkpoint/resume — safe to interrupt and restart.

```
scripts/
├── train_frozen.sh     → Ch9:  VLM tags + motion features
├── train_pretrain.sh   → Ch10: Continual pretraining (V-JEPA loss + EMA)
├── train_surgery.sh    → Ch11: Surgical fine-tuning (TODO)
├── run_embed.sh        → ALL:  Embedding extraction (auto-detects encoders)
└── run_eval.sh         → ALL:  Evaluation (auto-detects encoders, radar plot)
```

### Quick start

```bash
# Fast iteration (~7h): train 115K + embed 10K + eval 10K
./scripts/train_pretrain.sh --FULL
./scripts/run_embed.sh --FULL --subset data/subset_10k.json \
    --local-data data/subset_10k_local --encoders vjepa_lambda0_001
./scripts/legacy2/run_eval.sh --POC

# Paper result (~22h): full embed + eval
./scripts/run_embed.sh --FULL --local-data data/full_local
./scripts/legacy2/run_eval.sh --FULL
```

### Ch9: Frozen encoder data (tags + motion)

```bash
./scripts/train_frozen.sh --FULL   # m04 (VLM tagging) + m04d (RAFT motion)
```

### Ch10: Continual pretraining

Self-supervised JEPA loss on Indian clips. Student-teacher with EMA, ImageNet normalization, 16f training / 64f eval (Meta recipe).

```bash
./scripts/train_pretrain.sh --FULL  # m09 (training only)
```

### Ch11: Representation surgery (TODO)

Progressive prefix unfreezing with factor datasets (Layout &#8594; Agent &#8594; Interaction) from SAM3 segmentation.

```bash
./scripts/train_surgery.sh --FULL   # m10 → m10b → m10c → m09 (surgical)
```

### Embedding + Evaluation (reusable across all chapters)

```bash
./scripts/run_embed.sh --FULL --local-data data/full_local   # all encoders
./scripts/legacy2/run_eval.sh --FULL                                  # m06→m08b radar
```

---

## Dataset

| Tier | Cities | Clips | Hours | GB |
|------|--------|-------|-------|----|
| Tier 1 | 6 metros | 68,614 | 161h | 74 |
| Goa | 1 | 5,835 | 14h | 6 |
| Tier 2 | 15 cities | 40,743 | 99h | 41 |
| Monuments | 3 | 495 | 1h | 1 |
| **Total** | **22** | **115,687** | **276h** | **121** |

---

## Code Structure

```
src/
├── m00-m03          # Data pipeline (YouTube → clips → WebDataset → HF)
├── m04              # VLM tagging (Qwen3-VL-8B, 16-field taxonomy)
├── m04d             # GPU-RAFT optical flow (13D motion features)
├── m05/m05b/m05c    # Embeddings (V-JEPA + 4 baselines + True Overlap)
├── m06/m06b         # Spatial metrics (FAISS) + temporal correlation
├── m07              # UMAP (cuML GPU)
├── m08/m08b         # Plots + multi-encoder comparison
└── utils/           # Config, bootstrap CI, gpu_batch, wandb
```

## Future prediction readout experiments

This branch adds **separate** adapters and caches to visualize frozen world-model
predictions. It does not change the FactorJEPA training pipeline above. Readouts
are diagnostic images, not samples from a newly trained video generator. Ground
truth and true-future JEPA features are never fed into the causal prediction
path; they are used only for training supervision or evaluation diagnostics.

### Start on a GPU machine

Run these commands from a new checkout of this branch, with enough persistent disk
for the chosen dataset/cache. The repository setup script creates a Python **venv**
(not a Conda environment); a compatible pre-existing Conda environment can also
be used.

```bash
git clone -b cosmos-latent-adapter https://github.com/aayush-2021003/jepa-latent-to-rgb.git
cd jepa-latent-to-rgb
bash setup_env_uv.sh --gpu --from-wheels
source venv_walkindia/bin/activate
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
export WANDB_API_KEY='your_new_wandb_key'
export HF_TOKEN='your_new_huggingface_token'
```

Keep credentials in your shell or secret manager, **not** in YAML, this README,
Git, or logs. Use fresh keys if any older keys were exposed. On this fork,
`setup_env_uv.sh --gpu --from-wheels` assumes the checkout's `origin` is GitHub;
that is true for the clone command above. If CUDA is unavailable, fix the GPU
PyTorch install before downloading large assets. The preflight `validate` stage
also checks expected packages, paths, splits, and (with `--online`) accounts.

All four launchers share the same ordered workflow. Set `RUNNER` and `CONFIG`
from the tables below, then run:

```bash
bash "$RUNNER" validate "$CONFIG"
bash "$RUNNER" assets "$CONFIG"
bash "$RUNNER" data "$CONFIG"
bash "$RUNNER" cache "$CONFIG"
bash "$RUNNER" validate "$CONFIG" --require-assets --require-cache --online
bash "$RUNNER" train "$CONFIG"       # use train-image for the image-loss/MSE variants below
```

Run `data` before `cache`; `cache` writes frozen predicted features, Cosmos
targets, and (where configured) frame-15 RGB targets. The downloader records
the selected DenseWorld clip keys and source-video-disjoint splits in the
dataset directory. For a fixed archive and seed, rerunning on the *same*
dataset directory reuses that selection. Do not silently regenerate a new
split when comparing checkpoints. If data and cache already exist, skip their
stages and use `validate --require-assets --require-cache` before training.
The cache stage accepts `--split train|val|test|all` in the single-frame family.

### Experiment families

| Family | Set `RUNNER` to | Main `CONFIG` | Predicted target | Training entry point |
| --- | --- | --- | --- | --- |
| FactorJEPA → Cosmos video | `scripts/run_jepa_cosmos.sh` | `configs/experiments/jepa_cosmos_adapter.yaml` | frames 9–16 after frames 1–8 | `train` (true-JEPA readout) |
| FactorJEPA predicted → Cosmos video | same | `configs/experiments/jepa_cosmos_predicted.yaml` | frames 9–16 | `train` (predicted latents) |
| V-JEPA 2.1 → JEPA-WMs decoder | `scripts/run_vjepa21_jepawms.sh` | `configs/experiments/vjepa21_jepawms_two_frame.yaml` | frames 15–16 | `train` |
| V-JEPA 2.1 → Cosmos video | `scripts/run_vjepa21_cosmos.sh` | `configs/experiments/vjepa21_cosmos_predicted_4frames.yaml` | frames 13–16 after frames 1–12 | `train` |
| V-JEPA 2.1 → Cosmos-CI image | `scripts/run_vjepa21_cosmos_single_frame.sh` | see next table | frame 15 after frames 1–14 | `train` or `train-image` |

The single-frame experiment masks the V-JEPA tubelet for frames 15–16 but
supervises and decodes **only frame 15**. V-JEPA 2.1, Cosmos encoder/decoder,
and the JEPA-WMs head (in its family) stay frozen; only the adapter trains.
The true-JEPA and Cosmos self-reconstruction panels are evaluation controls,
not causal inputs. Detailed family runbooks: [FactorJEPA/Cosmos](experiments/jepa_cosmos/README.md),
[V-JEPA/JEPA-WMs](experiments/vjepa21_jepawms/README.md),
[V-JEPA/Cosmos four-frame](experiments/vjepa21_cosmos/README.md), and
[V-JEPA/Cosmos single-frame](experiments/vjepa21_cosmos_single_frame/README.md).

For example, to run the 75K experiment:

```bash
RUNNER=scripts/run_vjepa21_cosmos_single_frame.sh
CONFIG=configs/experiments/vjepa21_cosmos_predicted_1frame_75k_drive_walk_mse.yaml
bash "$RUNNER" validate "$CONFIG"
bash "$RUNNER" assets "$CONFIG"
bash "$RUNNER" data "$CONFIG"
bash "$RUNNER" cache "$CONFIG"
bash "$RUNNER" validate "$CONFIG" --require-assets --require-cache --online
mkdir -p logs
nohup bash "$RUNNER" train-image "$CONFIG" > logs/75k_mse.log 2>&1 &
echo $! > logs/75k_mse.pid
tail -f logs/75k_mse.log
```

### Single-frame runs and ablations

Every path in this table is relative to `configs/experiments/`; prepend that
directory when setting `CONFIG`. `B×acc` is per-GPU physical batch × gradient
accumulation. Validation cadence is counted in **optimizer steps** (also at
epoch end unless already due for the image-loss trainer).

| Config YAML | Data (train/val/test) | Loss or purpose | Epochs; B×acc; LR; validation | Stage |
| --- | --- | --- | --- | --- |
| `vjepa21_cosmos_predicted_1frame.yaml` | 4,500/500/– | latent L1 + 0.1 cosine | 20; 2×8; 2e-4; 150 | `train` |
| `vjepa21_cosmos_predicted_1frame_10k.yaml` | 9,500/500/– | latent L1 + 0.1 cosine | 20; 2×8; 2e-4; 150 | `train` |
| `vjepa21_cosmos_predicted_1frame_10k_image_loss.yaml` | 9,000/500/500 | latent L1 + cosine + RGB MSE + Alex-LPIPS | 20; 2×8; 2e-4; 300 | `train-image` |
| `vjepa21_cosmos_predicted_1frame_10k_latent_mse.yaml` | 9,000/500/500 | latent MSE only | 20; 2×8; 2e-4; 300 | `train-image` |
| `vjepa21_cosmos_predicted_1frame_60k_drive_walk_mse.yaml` | 48,000/6,000/6,000 | walk/drive, latent MSE | 5; 2×8; 2e-4; 1,500 | `train-image` |
| `vjepa21_cosmos_predicted_1frame_75k_drive_walk_mse.yaml` | 60,000/7,500/7,500 | walk/drive, latent MSE | 5; 2×8; 2e-4; 1,000 | `train-image` |
| `vjepa21_cosmos_predicted_1frame_75k_linear_mse.yaml` | same 75K split/cache | affine 1×1 linear readout, latent MSE | 5; 2×8; 2e-4; 1,000 | `train-image` |
| `vjepa21_cosmos_predicted_1frame_75k_drive_walk_mse_high_lr_continue.yaml` | same 75K split/cache | fresh 5-epoch adapter-weight warm start, latent MSE | 5; 2×8; 2e-4; 1,000 | `train-image` |
| `vjepa21_cosmos_predicted_1frame_75k_drive_walk_mse_lpips_finetune.yaml` | same 75K split/cache | 1 latent MSE + 5 RGB MSE + 2 Alex-LPIPS | 3; 2×8; 5e-5; 1,000 | `train-image` |
| `vjepa21_cosmos_predicted_1frame_75k_jepawms_rgb_lpips.yaml` | same 75K split/cache | 10 RGB MSE + VGG-LPIPS | 3; 2×8; 5e-5; 1,000 | `train-image` |
| `vjepa21_cosmos_predicted_1frame_75k_jepawms_rgb_lpips_latent_mse.yaml` | same 75K split/cache | 0.1 latent MSE + 10 RGB MSE + VGG-LPIPS | 3; 2×8; 5e-5; 1,000 | `train-image` |

The 60K/75K runs filter the dataset to `walking` and `drive` categories and
use an 80/10/10 source-video-disjoint split; the 75K selection uses seed 42.
The 10K image-loss and 10K latent-MSE configs deliberately share the same
9K/500/500 data/cache but use separate output directories, W&B run names, and
HF repos. The 75K follow-up configs reuse the **original 75K cache** and each
start from the original latent-MSE **best** adapter, not from one another.
See their exact commands and caveats in
[10K image loss](experiments/vjepa21_cosmos_single_frame/IMAGE_LOSS_10K.md),
[10K MSE](experiments/vjepa21_cosmos_single_frame/LATENT_MSE_10K.md),
[60K MSE](experiments/vjepa21_cosmos_single_frame/DRIVE_WALK_60K_MSE.md),
[75K MSE](experiments/vjepa21_cosmos_single_frame/DRIVE_WALK_75K_MSE.md),
[75K linear baseline](experiments/vjepa21_cosmos_single_frame/LINEAR_BASELINE_75K.md),
[75K MSE continuation](experiments/vjepa21_cosmos_single_frame/LATENT_MSE_75K_HIGH_LR_CONTINUE.md),
[75K perceptual fine-tune](experiments/vjepa21_cosmos_single_frame/MSE_LPIPS_75K_FINETUNE.md), and
[JEPA-WMs-style loss ablations](experiments/vjepa21_cosmos_single_frame/JEPAWMS_STYLE_75K_FINETUNES.md).

For any 75K fine-tune, download the **source** model first; its destination
matches `training.init_adapter_checkpoint` in those configs:

```bash
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='Aaypom/vjepa21-cosmos-ci-predicted-1frame-75k-drive-walk-mse-adapter', filename='checkpoints/best.pt', local_dir='checkpoints/vjepa21_75k_mse_source'))"
CONFIG=configs/experiments/vjepa21_cosmos_predicted_1frame_75k_jepawms_rgb_lpips_latent_mse.yaml
RUNNER=scripts/run_vjepa21_cosmos_single_frame.sh
bash "$RUNNER" validate "$CONFIG" --require-assets --require-cache --online
bash "$RUNNER" train-image "$CONFIG"
```

This is **weight initialization**, not an exact optimizer-state resume. To
resume an interrupted run, retain that run's own output directory and config;
its `adapter_latest.pt` includes optimizer/scheduler state. On a fresh instance,
transfer the original 75K dataset/cache to preserve identical splits or rebuild
and verify the recorded clip keys before claiming a matched comparison.

### Inference, evaluation, and artifacts

Use the checkpoint matching the selected config. `infer` downloads that
config's HF `checkpoints/best.pt` if a local adapter path is not supplied
where supported; pass an explicit path to eliminate ambiguity. Examples:

```bash
# One held-out frame (input clip must contain at least the 14 observed frames).
bash scripts/run_vjepa21_cosmos_single_frame.sh infer \
  configs/experiments/vjepa21_cosmos_predicted_1frame_75k_drive_walk_mse.yaml \
  --input-video /absolute/path/to/clip.mp4 \
  --adapter-checkpoint outputs/vjepa21_cosmos_predicted_1frame_75k_drive_walk_mse/adapter_best.pt \
  --output-dir outputs/manual_frame15_inference

# Four-frame V-JEPA/Cosmos experiment.
bash scripts/run_vjepa21_cosmos.sh infer \
  configs/experiments/vjepa21_cosmos_predicted_4frames.yaml \
  --input-video /absolute/path/to/clip.mp4

# Two-frame V-JEPA/JEPA-WMs experiment (different argument names).
bash scripts/run_vjepa21_jepawms.sh infer \
  configs/experiments/vjepa21_jepawms_two_frame.yaml \
  --video /absolute/path/to/clip.mp4 \
  --checkpoint outputs/vjepa21_jepawms_two_frame/adapter_best.pt

# FactorJEPA/Cosmos experiment.
bash scripts/run_jepa_cosmos.sh infer \
  configs/experiments/jepa_cosmos_predicted.yaml \
  --input-video /absolute/path/to/clip.mp4
```

For the pretrained JEPA-WMs head **without adapter training**, evaluate the
existing single-frame 10K validation/test cache with:

```bash
python -m experiments.vjepa21_jepawms.evaluate_pretrained_head \
  --config configs/experiments/vjepa21_cosmos_predicted_1frame_10k_image_loss.yaml \
  --head-config configs/experiments/vjepa21_jepawms_two_frame.yaml \
  --split val
```

The image-loss trainer computes full-split validation and final test metrics
on frame 15 (latent MSE, RGB MSE/PSNR, LPIPS, plus controls where available)
and logs step/epoch metrics, comparison videos, and loss curves to W&B. Local
checkpoints/reports are under each config's `outputs/` path; model checkpoints
are also uploaded to its `huggingface.repo_id` as `checkpoints/best.pt` and
`checkpoints/latest.pt`, and numbered epoch checkpoints when
`push_every_epoch: true`. This is **model-artifact** publishing, not a GitHub
upload of videos, latent caches, or large checkpoints. The repository ignores
those large/generated files; inspect the W&B run and HF repo to verify remote
artifacts for a particular training run. The chosen validation metric is in
each YAML; do not compare runs solely by a differently weighted training loss.

---

## Authors

Kapil Wanaskar¹, Gaytri Jena², Aman Chadha³, Vinija Jain⁴, Vasu Sharma⁵, Amitava Das⁶

¹San Jose State University, USA · ²UC Berkeley, USA · ³Apple, USA · ⁴Meta, USA · ⁵PocketFM, USA · ⁶Pragya Lab, BITS Pilani Goa, India

Part of the **DenseWorld** research program — *World Models for Populous, Crowded, and Chaotic Global South*

## Citation

```bibtex
@article{wanaskar2026factorjepa,
  title={FactorJEPA: Factorizing Monolithic Futures into Layout-Agent-Interaction Channels for Crowded and Chaotic Global South Urban Worlds},
  author={Wanaskar, Kapil and Jena, Gaytri and Chadha, Aman and Jain, Vinija and Sharma, Vasu and Das, Amitava},
  journal={arXiv preprint arXiv:2608.01049},
  year={2026}
}
```
