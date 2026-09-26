# V-JEPA 2.1 predicted features to two RGB future frames

This directory is an independent experiment. It does not change or reuse the
checkpoints/caches/outputs of `experiments/jepa_cosmos`.

## Task

The official frozen V-JEPA 2.1 ViT-g/16 1B encoder and predictor receive frames
1-14. Because the pretrained model uses a temporal tubelet of two, it predicts
one held-out latent tubelet representing frames 15-16. A trainable adapter uses
all four 1,408-dimensional deep-supervision levels (5,632 channels total),
expands that tubelet into two decoder feature grids, and passes them through
Meta's frozen JEPA-WMs `vjepa2_vitg_256_INet` decoder to render both RGB frames.

Neither target frame is given to the context encoder. The adapter is trained
on predicted latents, not oracle target latents.

## Components

- Frozen V-JEPA 2.1 checkpoint: `vjepa2_1_vitg_384.pt`
- Frozen upstream JEPA-WMs decoder implementation and checkpoint
- Trainable 5,632 -> 1,408 fusion, six ConvNeXt-style spatial residual blocks,
  and two frame-specific 1,408-channel output heads
- RGB L1 + LPIPS + decoder-feature distribution regularization
- Last-frame-copy, oracle-latent, and predicted-latent validation panels
- Separate cache, outputs, W&B run/project, and Hugging Face model repository

## Setup on a fresh Vast.ai instance

Use a 48 GB GPU and at least 200 GB of container storage. From the repository:

```bash
bash setup_env_uv.sh --gpu --from-wheels
source venv_walkindia/bin/activate

export WANDB_API_KEY="your_rotated_wandb_key"
export HF_TOKEN="your_huggingface_token"
```

Do not put secrets in YAML files or commit them.

Validate the configuration, download public model assets, and reuse/download
the source-disjoint DenseWorld 5K subset:

```bash
bash scripts/run_vjepa21_jepawms.sh validate \
  configs/experiments/vjepa21_jepawms_two_frame.yaml

bash scripts/run_vjepa21_jepawms.sh assets \
  configs/experiments/vjepa21_jepawms_two_frame.yaml

bash scripts/run_vjepa21_jepawms.sh data \
  configs/experiments/vjepa21_jepawms_two_frame.yaml
```

The data command exits without downloading when the existing
`data/jepa_cosmos_5k` subset has the requested 4,500/500 split.

## Cache and train

Caching is resumable. It stores predicted V-JEPA 2.1 features for all training
and validation clips and true target features only for validation:

```bash
bash scripts/run_vjepa21_jepawms.sh cache \
  configs/experiments/vjepa21_jepawms_two_frame.yaml
```

Then validate the completed cache and train:

```bash
bash scripts/run_vjepa21_jepawms.sh validate \
  configs/experiments/vjepa21_jepawms_two_frame.yaml \
  --require-assets --require-cache

bash scripts/run_vjepa21_jepawms.sh train \
  configs/experiments/vjepa21_jepawms_two_frame.yaml
```

The default run uses 20 epochs, batch size 2, gradient accumulation 8
(effective batch 16), learning rate `1e-4`, validation every 150 optimizer
steps, checkpoint backup every 300 steps, and eight labeled validation images
every 150 steps. Frequent validation uses 128 clips; final validation uses all
500 clips.

To cache and train unattended:

```bash
mkdir -p logs
nohup bash -c '
set -e
bash scripts/run_vjepa21_jepawms.sh cache configs/experiments/vjepa21_jepawms_two_frame.yaml
bash scripts/run_vjepa21_jepawms.sh train configs/experiments/vjepa21_jepawms_two_frame.yaml
' > logs/vjepa21_jepawms_two_frame.log 2>&1 &
echo $! > logs/vjepa21_jepawms_two_frame.pid
```

For a genuinely fresh run without overwriting an earlier one:

```bash
bash scripts/run_vjepa21_jepawms.sh train \
  configs/experiments/vjepa21_jepawms_two_frame.yaml \
  --fresh --output-dir outputs/vjepa21_jepawms_two_frame_run2
```

## Inference

```bash
bash scripts/run_vjepa21_jepawms.sh infer \
  configs/experiments/vjepa21_jepawms_two_frame.yaml \
  --video /path/to/video.mp4 \
  --checkpoint outputs/vjepa21_jepawms_two_frame/adapter_best.pt \
  --output-dir outputs/vjepa21_jepawms_inference \
  --wandb
```

Inference saves the last context frame, both held-out ground truths, both oracle
and predicted decodings, a labeled comparison, and per-frame JSON metrics.

## Interpretation

This is a short-horizon diagnostic, not an eight-frame video prediction test.
The predictor's smallest native unit covers two frames. The shared adapter trunk
therefore feeds two learned frame-specific heads; both outputs are supervised,
while frame-16 LPIPS selects the best checkpoint. Always compare frame 16 against
the last-context-frame copy baseline: otherwise a mostly static dataset can make
immediate-future quality look deceptively strong.

## Pretrained JEPA-WMs head on the current 10K frame-15 split (no training)

The two-frame adapter above is **not** a matched baseline for the current
Cosmos-CI frame-15 experiment: it uses a separate 5K split, supervises both
frames, and trains another adapter. To inspect the *pretrained decoder head*
on exactly the same cached V-JEPA 2.1 predictions and RGB targets as the
10K Cosmos image-loss run, use the separate evaluator:

```bash
COSMOS_CONFIG=configs/experiments/vjepa21_cosmos_predicted_1frame_10k_image_loss.yaml
HEAD_CONFIG=configs/experiments/vjepa21_jepawms_two_frame.yaml

# Reuses the V-JEPA checkpoint; obtains the JEPA-WMs source and decoder weight.
bash scripts/run_vjepa21_jepawms.sh assets "$HEAD_CONFIG"

# Check that the 10K Cosmos cache exists (9K train / 500 val / 500 test).
bash scripts/run_vjepa21_cosmos_single_frame.sh validate "$COSMOS_CONFIG" \
  --require-cache

# Small smoke test; separate output directory. Omit --no-wandb to log it.
python -m experiments.vjepa21_jepawms.evaluate_pretrained_head \
  --config "$COSMOS_CONFIG" --head-config "$HEAD_CONFIG" \
  --split val --max-samples 8 --no-wandb \
  --output-dir outputs/jepawms_pretrained_head_smoke

# Complete validation first; use test only once the protocol is settled.
python -m experiments.vjepa21_jepawms.evaluate_pretrained_head \
  --config "$COSMOS_CONFIG" --head-config "$HEAD_CONFIG" --split val
python -m experiments.vjepa21_jepawms.evaluate_pretrained_head \
  --config "$COSMOS_CONFIG" --head-config "$HEAD_CONFIG" --split test
```

The evaluator consumes frame-15 `jepa_predicted`, `jepa_target`,
`target_rgb`, and frame-14 `previous_rgb` from the Cosmos cache. It does **not**
re-download videos, re-run V-JEPA, require a Cosmos adapter checkpoint, train
an adapter, or create a Hugging Face checkpoint. It resizes the 24×24 feature
grid to 16×16, the V-JEPA-2/256 geometry of the pretrained JEPA-WMs head,
then scales the decoded image to the cached 384×384 target for matched RGB L1,
MSE, PSNR, and LPIPS-Alex metrics. It reports oracle (true JEPA), predicted
JEPA, and frame-14 persistence readouts for held-out frame 15.

Outputs are under `outputs/vjepa21_jepawms_pretrained_head_frame15/{val,test}`:
`summary.json`, `per_sample.csv`, and eight four-panel PNGs. Metrics and
panels also go to a separate run in the Cosmos W&B project. This is a
**zero-shot, cross-version diagnostic**, because the JEPA-WMs head was trained
for V-JEPA-2, not V-JEPA 2.1. Poor quality alone would not establish that a
capacity-matched JEPA-WMs head trained on the 9K DenseWorld examples is worse
than the trained Cosmos adapter.
