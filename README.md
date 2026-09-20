# FactorJEPA: World Models for Crowded, Chaotic Global South Urban Scenes

**Factorizing monolithic future prediction into layout, agent, and interaction channels, trained and evaluated on DenseWorld.**

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

### Future-latent RGB visualization experiment

The frozen FactorJEPA future-token to Cosmos continuous-latent adapter lives in
[`experiments/jepa_cosmos`](experiments/jepa_cosmos/README.md). Its runbook covers
efficient 5K/10K DenseWorld preparation, frozen-latent caching, adapter training,
W&B validation logging, Hugging Face checkpoint publishing, and RGB inference.

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
