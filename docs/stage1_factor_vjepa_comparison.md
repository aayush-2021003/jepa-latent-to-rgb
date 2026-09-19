# Stage-1 FactorJEPA vs V-JEPA Comparison

## What this evaluates

`src/stage1_compare_factor_vjepa.py` applies the same trained Stage-1 decoder to:

1. FactorJEPA ViT-g visible plus predicted tokens.
2. Vanilla V-JEPA 2.1 ViT-g visible plus predicted tokens.

Both branches receive the same 16 sampled frames and the same deterministic mask for each clip.
Both must produce a combined token grid of `[1, 4608, 5632]`. The shared decoder maps each grid
to a normalized Cosmos latent of `[1, 16, 4, 48, 48]`, and the same frozen Cosmos VAE decodes it.

This is a decoder-transfer test. It measures how compatible vanilla V-JEPA features are with a
decoder trained on FactorJEPA features. It is not fully equivalent to training one matched decoder
for each backbone.

## Files to transfer after cloning the original repository

The safest small transfer is the complete modified source and configuration trees. Together they
are only about 5.5 MB and avoid missing a transitive helper imported by the Stage-1 training file:

```text
src/
configs/
requirements.txt
requirements_gpu.txt
```

For this comparison specifically, the central custom files are:

```text
src/stage1_compare_factor_vjepa.py
src/stage1_jepa_decoder_train.py
src/utils/vjepa2_imports.py
configs/model/vjepa2_1_vitg.yaml
configs/train/surgery_3stage_DI_diheavy_encoder.yaml
configs/train/surgery_3stage_DI_encoder.yaml
configs/train/surgery_base.yaml
configs/train/base_optimization.yaml
configs/pipeline.yaml
```

Do not copy `outputs/`, downloaded datasets, Hugging Face caches, or old Python environments.
Clone Meta's V-JEPA repository separately under `deps/vjepa2`.

## Vast.ai setup

Use a current NVIDIA PyTorch/CUDA image and at least 48 GB VRAM for comfortable execution. Replace
`VAST_HOST` and `VAST_PORT` with the values shown by Vast.ai.

On the Vast instance:

```bash
mkdir -p /workspace
cd /workspace
git clone https://github.com/kapilw25/factorjepa.git
cd factorjepa

mkdir -p deps
git clone --depth 1 https://github.com/facebookresearch/vjepa2.git deps/vjepa2
```

From the local Mac, while the instance is running:

```bash
cd "/Users/aayush/Documents/Pragya AI Research (YONDER)/Experiments"

rsync -az -e "ssh -p VAST_PORT" factorjepa/src/ \
  root@VAST_HOST:/workspace/factorjepa/src/
rsync -az -e "ssh -p VAST_PORT" factorjepa/configs/ \
  root@VAST_HOST:/workspace/factorjepa/configs/
rsync -az -e "ssh -p VAST_PORT" \
  factorjepa/requirements.txt factorjepa/requirements_gpu.txt \
  root@VAST_HOST:/workspace/factorjepa/
rsync -az -e "ssh -p VAST_PORT" \
  "/Users/aayush/Documents/Pragya AI Research (YONDER)/30clips.zip" \
  root@VAST_HOST:/workspace/factorjepa/
```

Back on Vast, create an isolated environment. CUDA 12.8 is appropriate for Blackwell instances;
use the CUDA wheel selected for the rented GPU/image when it differs.

```bash
cd /workspace/factorjepa

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda create -n factorjepa-compare python=3.12 -y
  conda activate factorjepa-compare
else
  python3 -m venv .venv-factorjepa-compare
  source .venv-factorjepa-compare/bin/activate
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install xformers --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
  PyYAML huggingface_hub safetensors datasets \
  diffusers==0.39.0 transformers==5.5.4 accelerate \
  timm einops av Pillow tqdm imageio imageio-ffmpeg \
  wandb lpips
python -m pip install -e deps/vjepa2
```

Verify the important runtime pieces:

```bash
python - <<'PY'
import torch
from diffusers import AutoencoderKLWan

print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("BF16 supported:", torch.cuda.is_bf16_supported())
print("AutoencoderKLWan:", AutoencoderKLWan.__name__)
PY

python -m xformers.info | head -40
python -m py_compile src/stage1_compare_factor_vjepa.py
```

## Credentials and inputs

Enter secrets interactively so they are not written into shell history:

```bash
read -s -p "HF token: " HF_TOKEN; echo
export HF_TOKEN
wandb login
```

Extract the clips:

```bash
mkdir -p data/eval_30clips
unzip -q 30clips.zip -d data/eval_30clips
find data/eval_30clips -type f -iname '*.mp4' | sort
```

The script automatically downloads these two source checkpoints when they are absent:

- FactorJEPA `m09c_ckpt_best.pt` from `anonymousML123/factorjepa-outputs`.
- Vanilla V-JEPA 2.1 **ViT-g** `vjepa2_1_vitg_384.pt` from Meta.

The lowercase `vitg` checkpoint is required. The uppercase `vitG` checkpoint is a 1664-dimensional
2B model and is rejected by the comparison script.

## Download the Stage-1 decoder from W&B

List the artifacts logged by the Stage-1 run:

```bash
python - <<'PY'
import wandb

run_path = (
    "aayush21003-indraprastha-institute-of-information-techno/"
    "factorjepa-stage1/8ahcb0na"
)
run = wandb.Api().run(run_path)
for artifact in run.logged_artifacts():
    print(artifact.name, "type=", artifact.type, "aliases=", list(artifact.aliases))
PY
```

Choose the numbered checkpoint artifact corresponding to approximately step 12000, then download
it by replacing `ARTIFACT_NAME_OR_VERSION` with the exact printed value:

```bash
mkdir -p checkpoints/stage1_12k
python - <<'PY'
import wandb

artifact_path = (
    "aayush21003-indraprastha-institute-of-information-techno/"
    "factorjepa-stage1/ARTIFACT_NAME_OR_VERSION"
)
artifact = wandb.Api().artifact(artifact_path)
print("Downloaded to:", artifact.download(root="checkpoints/stage1_12k"))
PY

find checkpoints/stage1_12k -type f -name '*.pt' -ls

export DECODER_CKPT="$(find checkpoints/stage1_12k -type f \
  -name 'stage1_jepa_decoder_step_*.pt' | sort | tail -1)"
test -n "$DECODER_CKPT" && test -f "$DECODER_CKPT"
echo "Using decoder: $DECODER_CKPT"
```

Use the resulting numbered `stage1_jepa_decoder_step_*.pt` file as `--decoder-ckpt`.

## Smoke test

Always validate one clip before launching all 30:

```bash
python -u src/stage1_compare_factor_vjepa.py \
  --clips-dir data/eval_30clips/FactorJEPA \
  --decoder-ckpt "$DECODER_CKPT" \
  --model-config configs/model/vjepa2_1_vitg.yaml \
  --train-config configs/train/surgery_3stage_DI_diheavy_encoder.yaml \
  --output-dir outputs/eval/stage1_factor_vs_vjepa_smoke \
  --mask-seed 1 \
  --fps 16 \
  --limit 1 \
  --lpips
```

Inspect the four-column video under `videos/comparison/`. Also inspect
`checkpoint_load_reports.json`; every encoder and predictor should clear the default 90 percent
key and parameter threshold.

## Full 30-clip comparison

```bash
python -u src/stage1_compare_factor_vjepa.py \
  --clips-dir data/eval_30clips/FactorJEPA \
  --decoder-ckpt "$DECODER_CKPT" \
  --model-config configs/model/vjepa2_1_vitg.yaml \
  --train-config configs/train/surgery_3stage_DI_diheavy_encoder.yaml \
  --output-dir outputs/eval/stage1_factor_vs_vjepa_30clips \
  --mask-seed 1 \
  --mask-ratio 0.9 \
  --fps 16 \
  --lpips \
  --wandb \
  --wandb-entity aayush21003-indraprastha-institute-of-information-techno \
  --wandb-project factorjepa-stage1-comparison \
  --wandb-run-name stage1-step11934-factorjepa-vs-vjepa-mask90-30clips \
  --wandb-max-videos 30
```

The script writes:

```text
videos/reference/              sampled and center-cropped 16-frame input
videos/cosmos_vae_roundtrip/   upper-bound VAE reconstruction
videos/factorjepa/             shared decoder applied to FactorJEPA tokens
videos/vjepa/                  shared decoder applied to vanilla V-JEPA tokens
videos/comparison/             labeled four-column videos
videos/masks/                  exact per-clip JEPA mask visualization
metrics_per_clip.csv
summary.json
checkpoint_load_reports.json
inference_manifest.json
latent_cache/                  small resumable predictions, never source checkpoints
```

The source videos are sampled uniformly across each complete MP4. The output FPS changes playback
speed only; it does not change the selected source frames.

Each mask video contains two synchronized panels. `Encoder-visible tokens` preserves the pixels
whose tubelet tokens are sent to the JEPA encoder and darkens predicted regions. The second panel
overlays visible tokens in green and masked/predictor-target tokens in red. These are expanded from
the exact deterministic `m_enc` and `m_pred` indices used for that clip, with a temporal tubelet of
2 frames and 16x16 spatial patches. W&B logs them under `comparison/masks/`; set
`--wandb-max-videos 30` to display all 30 clips.

`--mask-ratio 0.9` replaces the standard block mask with an exact deterministic random-token
partition for each clip. On the 4608-token grid it assigns 4147 tokens (89.996%) to prediction and
461 tokens (10.004%) to visible context. FactorJEPA and V-JEPA receive the same per-clip partition.
The selected ratio is part of the latent-cache fingerprint, so caches from another mask regime are
not reused.

## Causal future prediction

Use `--future-visible-frames N` to expose a temporal prefix instead of random tokens. Every spatial
patch in the first `N` sampled frames is visible to the JEPA encoder, while every spatial patch in
the remaining sampled frames is supplied by the predictor. This option is mutually exclusive with
`--mask-ratio`.

The ViT-g setup samples 16 frames and uses two-frame tubelets, producing an `8 x 24 x 24` token
grid. For example, `--future-visible-frames 4` exposes the first two temporal token planes (1152
tokens) and predicts the last six planes (3456 tokens). The current Stage-1 checkpoint outputs four
Cosmos temporal latents, which natively decode to 13 frames. The comparison therefore exports 13
aligned frames without repetition: frames 0-3 are visible-token context and frames 4-12 are
predicted future. JEPA still predicts frames 13-15 internally to satisfy the fixed 16-frame token
grid, but those three predictions are not renderable by this checkpoint. The mask video marks the
displayed boundary in green and red.

```bash
python -u src/stage1_compare_factor_vjepa.py \
  --clips-dir data/eval_30clips/FactorJEPA \
  --decoder-ckpt checkpoints/stage1_step_11934/stage1_jepa_decoder_step_0011934.pt \
  --model-config configs/model/vjepa2_1_vitg.yaml \
  --train-config configs/train/surgery_3stage_DI_diheavy_encoder.yaml \
  --output-dir outputs/eval/stage1_factor_vs_vjepa_future4f_native13_30clips \
  --future-visible-frames 4 \
  --fps 16 \
  --lpips \
  --wandb \
  --wandb-entity aayush21003-indraprastha-institute-of-information-techno \
  --wandb-project factorjepa-stage1-comparison \
  --wandb-run-name stage1-step11934-factorjepa-vs-vjepa-future4f-native13-30clips \
  --wandb-max-videos 30
```

The 16 frames are sampled across each source MP4 before masking. Therefore, "first 4 frames" means
the first four frames in that sampled sequence, not necessarily four adjacent frames from the
original file.
