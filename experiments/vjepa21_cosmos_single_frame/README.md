# V-JEPA 2.1 predicted tubelet to one Cosmos-CI frame

This experiment is isolated from `experiments/vjepa21_cosmos` and
`experiments/vjepa21_jepawms`; it has separate code, cache, outputs, W&B run,
and Hugging Face checkpoint repository.

## Exact objective

- Frames 1-14 are visible context.
- Frozen V-JEPA 2.1 predicts its shortest native unit: one tubelet jointly
  representing held-out frames 15-16.
- The adapter receives only that predicted tubelet, shaped
  `(B, 1408, 1, 24, 24)`.
- Frozen Cosmos-CI8x8 encodes frame 15 alone into `(B, 16, 48, 48)`.
- Only the adapter is trained, using
  `latent_L1 + 0.1 * latent_cosine_distance`.
- Frozen Cosmos-CI decodes one output image: frame 15.

This is a genuine single-frame reconstruction target and loss. It remains a
single-frame **readout** from a two-frame V-JEPA prediction; the pretrained
V-JEPA tubelet size is not changed.

## Fresh Vast.ai setup

```bash
git clone -b cosmos-latent-adapter \
  https://github.com/aayush-2021003/jepa-latent-to-rgb.git
cd jepa-latent-to-rgb

bash setup_env_uv.sh --gpu --from-wheels
source venv_walkindia/bin/activate

export WANDB_API_KEY="your_rotated_wandb_key"
export HF_TOKEN="your_huggingface_token"
```

Keep tokens in environment variables; do not place them in YAML or shell files.

## Prepare and train

```bash
CONFIG=configs/experiments/vjepa21_cosmos_predicted_1frame.yaml

bash scripts/run_vjepa21_cosmos_single_frame.sh validate "$CONFIG"
bash scripts/run_vjepa21_cosmos_single_frame.sh assets "$CONFIG"
bash scripts/run_vjepa21_cosmos_single_frame.sh data "$CONFIG"
bash scripts/run_vjepa21_cosmos_single_frame.sh cache "$CONFIG"
bash scripts/run_vjepa21_cosmos_single_frame.sh validate "$CONFIG" \
  --require-assets --require-cache --online
bash scripts/run_vjepa21_cosmos_single_frame.sh train "$CONFIG"
```

The `data` stage reuses `data/jepa_cosmos_5k` if the existing 4,500/500 split
is complete. The single-frame latent cache is new and must be built at
`data/vjepa21_cosmos_5k_predicted_1frame`.

Unattended cache plus training:

```bash
mkdir -p logs
nohup bash -c '
set -e
CONFIG=configs/experiments/vjepa21_cosmos_predicted_1frame.yaml
bash scripts/run_vjepa21_cosmos_single_frame.sh cache "$CONFIG"
bash scripts/run_vjepa21_cosmos_single_frame.sh validate "$CONFIG" --require-assets --require-cache
bash scripts/run_vjepa21_cosmos_single_frame.sh train "$CONFIG"
' > logs/vjepa21_cosmos_predicted_1frame.log 2>&1 &
echo $! > logs/vjepa21_cosmos_predicted_1frame.pid
```

Defaults are 20 epochs, physical batch size 2, gradient accumulation 8
(effective batch 16), learning rate `2e-4` with cosine decay to `2e-6`, and
validation every 150 optimizer steps over 500 clips. Eight labeled comparison
images are logged at each step-based validation. The best checkpoint is selected
by predicted latent loss. Latest and best adapters are uploaded to
`Aaypom/vjepa21-cosmos-ci-predicted-1frame-adapter`.

## One-sample overfit diagnostic

Run this before the 5K training if desired:

```bash
bash scripts/run_vjepa21_cosmos_single_frame.sh train "$CONFIG" \
  --overfit --no-hf-push
```

The overfit output is isolated under
`outputs/vjepa21_cosmos_predicted_1frame/overfit`.

## Inference

```bash
bash scripts/run_vjepa21_cosmos_single_frame.sh infer "$CONFIG" \
  --input-video /path/to/video.mp4 \
  --adapter-checkpoint outputs/vjepa21_cosmos_predicted_1frame/adapter_best.pt
```

Inference writes the 14-frame context video, ground-truth frame 15, predicted
frame 15, a labeled four-panel comparison image, and metrics JSON.
