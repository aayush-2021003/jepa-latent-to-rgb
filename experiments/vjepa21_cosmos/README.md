# V-JEPA 2.1 predicted latents to Cosmos CV4

This is an isolated experiment. It does not change the existing
`experiments/jepa_cosmos` or `experiments/vjepa21_jepawms` runs, caches, or outputs.

## Experiment

- Frames 1-12 are visible context.
- Frozen V-JEPA 2.1 predicts two tubelets representing frames 13-16.
- Only its final 1,408-channel hierarchical level enters the adapter, matching
  the previous FactorJEPA-to-Cosmos experiment.
- The known frame 12 is the causal Cosmos anchor.
- Cosmos encodes `[frame 12, frames 13-16]`, a valid five-frame `4k+1` sequence.
- The anchor latent is removed, leaving one 16-channel Cosmos future slot.
- Only the adapter is trained.

The exact objective retained from the earlier predicted-latent experiment is:

```text
loss = latent_L1 + 0.1 * latent_cosine_distance
```

RGB, LPIPS, and temporal training losses are disabled. During validation, the
frozen Cosmos decoder renders all four predicted frames. RGB L1 and PSNR are
reported separately for frames 13, 14, 15, and 16 and as four-frame averages.
Eight four-frame comparison videos are also logged. Checkpoint selection remains
based on predicted latent loss, so decoded metrics do not alter training.

## Setup and run

```bash
bash setup_env_uv.sh --gpu --from-wheels
source venv_walkindia/bin/activate

export WANDB_API_KEY="your_rotated_wandb_key"
export HF_TOKEN="your_huggingface_token"

bash scripts/run_vjepa21_cosmos.sh validate \
  configs/experiments/vjepa21_cosmos_predicted_4frames.yaml

bash scripts/run_vjepa21_cosmos.sh assets \
  configs/experiments/vjepa21_cosmos_predicted_4frames.yaml

bash scripts/run_vjepa21_cosmos.sh data \
  configs/experiments/vjepa21_cosmos_predicted_4frames.yaml

bash scripts/run_vjepa21_cosmos.sh cache \
  configs/experiments/vjepa21_cosmos_predicted_4frames.yaml

bash scripts/run_vjepa21_cosmos.sh validate \
  configs/experiments/vjepa21_cosmos_predicted_4frames.yaml \
  --require-assets --require-cache

bash scripts/run_vjepa21_cosmos.sh train \
  configs/experiments/vjepa21_cosmos_predicted_4frames.yaml
```

The data stage reuses `data/jepa_cosmos_5k` when its 4,500/500 source-disjoint
split is already present. The new V-JEPA/Cosmos latent cache must still be built.

For unattended caching and training:

```bash
mkdir -p logs
nohup bash -c '
set -e
bash scripts/run_vjepa21_cosmos.sh cache configs/experiments/vjepa21_cosmos_predicted_4frames.yaml
bash scripts/run_vjepa21_cosmos.sh train configs/experiments/vjepa21_cosmos_predicted_4frames.yaml
' > logs/vjepa21_cosmos_predicted_4frames.log 2>&1 &
echo $! > logs/vjepa21_cosmos_predicted_4frames.pid
```

## Defaults

- 20 epochs
- batch size 2
- gradient accumulation 8 (effective batch 16)
- learning rate `2e-4` with cosine decay to `2e-6`
- validation every 150 optimizer steps
- 500 validation clips
- eight four-frame validation videos at validation events
- best checkpoint selected by predicted latent loss
- latest and best checkpoints published to the separate configured HF repository

## Inference

```bash
bash scripts/run_vjepa21_cosmos.sh infer \
  configs/experiments/vjepa21_cosmos_predicted_4frames.yaml \
  --input-video /path/to/video.mp4 \
  --adapter-checkpoint outputs/vjepa21_cosmos_predicted_4frames/adapter_best.pt \
  --output-dir outputs/vjepa21_cosmos_inference
```

The four decoded future frames correspond to frames 13-16. Frame 13 can be
treated as the primary shortest-horizon output, but the underlying Cosmos latent
still represents the complete four-frame block.
