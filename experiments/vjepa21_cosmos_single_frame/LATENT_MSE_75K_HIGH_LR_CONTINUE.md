# Five-epoch high-LR continuation of the 75K latent-MSE adapter

Config: `configs/experiments/vjepa21_cosmos_predicted_1frame_75k_drive_walk_mse_high_lr_continue.yaml`.

This **separate run** starts from the original 75K latent-MSE run's **best**
adapter weights, not its latest weights. It retains the exact latent-MSE
objective, predicted V-JEPA input, adapter architecture, seed, 60K/7.5K/7.5K
source-disjoint split, and cache. Only adapter weights are loaded from the
source checkpoint; AdamW, the cosine scheduler, counters, and best-metric
state start fresh. The original run and the two JEPA-WMs-style runs are not
modified.

The restart uses `2e-4` initially, decaying to `2e-6` over **five additional
epochs**. This is the original run's *peak* LR and 4x the later image-loss
fine-tune's `5e-5`, not a higher peak than the original run. Restarting this
high from an already-trained checkpoint can temporarily worsen validation.
The step-0 checkpoint preserves the source weights in the new run, and the
best checkpoint is selected solely by full-validation predicted latent MSE.
Check RGB MSE, PSNR, LPIPS, and held-out previews as diagnostics rather than
assuming lower latent MSE means better images.

Batch size is 2, accumulation 8 (effective 16), with validation at step 0,
every 1,000 optimizer steps, and otherwise at epoch end. W&B logs loss,
validation metrics, 20 preview videos per validation, final test metrics,
and curves. The separate HF model repo stores best, latest, and every-epoch
checkpoints.

## Push from this checkout

Here, `github` points to `aayush-2021003/jepa-latent-to-rgb`; `origin` is a
different local checkout. Push to **github**, not `origin`:

```bash
git add \
  configs/experiments/vjepa21_cosmos_predicted_1frame_75k_drive_walk_mse_high_lr_continue.yaml \
  experiments/vjepa21_cosmos_single_frame/test_image_loss.py \
  experiments/vjepa21_cosmos_single_frame/LATENT_MSE_75K_HIGH_LR_CONTINUE.md
git diff --cached --check
git commit -m "Add 75K high-LR latent-MSE continuation"
git push github cosmos-latent-adapter
```

This stages no unrelated untracked files. Do not force-push if the remote
has changed.

## Run on Vast.ai with the existing 75K cache

Use the same known-good CUDA environment as the previous run. Activate it,
and export fresh credentials in your shell (never commit their values):

```bash
cd /aayush/jepa-latent-to-rgb
git pull --ff-only origin cosmos-latent-adapter
export HF_TOKEN='your_huggingface_token'
export WANDB_API_KEY='your_wandb_key'
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

On a standard Vast GitHub clone, `origin` is the GitHub remote; confirm with
`git remote -v` if uncertain. Download the source best checkpoint once into
the path expected by the config:

```bash
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='Aaypom/vjepa21-cosmos-ci-predicted-1frame-75k-drive-walk-mse-adapter', filename='checkpoints/best.pt', local_dir='checkpoints/vjepa21_75k_mse_source'))"
```

Do **not** rerun `data` or `cache` when the original 75K dataset/cache is
already present. Check the setup, then start the new run:

```bash
CONFIG=configs/experiments/vjepa21_cosmos_predicted_1frame_75k_drive_walk_mse_high_lr_continue.yaml
bash scripts/run_vjepa21_cosmos_single_frame.sh validate "$CONFIG" --require-assets --require-cache --online
mkdir -p logs
nohup bash scripts/run_vjepa21_cosmos_single_frame.sh train-image "$CONFIG" > logs/vjepa21_75k_mse_high_lr_continue.log 2>&1 &
echo $! > logs/vjepa21_75k_mse_high_lr_continue.pid
tail -f logs/vjepa21_75k_mse_high_lr_continue.log
```

The step-0 validation metrics should closely match the original checkpoint's
validation metrics. If not, stop and check source checkpoint/cache/model
versions. The new local outputs are under
`outputs/vjepa21_cosmos_predicted_1frame_75k_drive_walk_mse_high_lr_continue`;
the HF destination is
`Aaypom/vjepa21-cosmos-ci-predicted-1frame-75k-drive-walk-mse-high-lr-continue-adapter`.

On a fresh instance, transfer the **original**
`data/denseworld_75k_drive_walk` and
`data/vjepa21_cosmos_75k_drive_walk_frame15` directories to preserve exact
splits. If you rebuild them from the archive, verify that the clip-key
manifests match the originals before claiming the same split. Rebuilding
requires substantially more disk than merely reusing the existing cache.
