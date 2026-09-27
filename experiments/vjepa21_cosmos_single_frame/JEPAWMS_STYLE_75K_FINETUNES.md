# Two 75K JEPA-WMs-style image-loss continuations

Both runs warm-start **adapter weights only** from the best 75K latent-MSE
checkpoint. They use the same frozen V-JEPA 2.1 prediction, frozen Cosmos-CI
decoder, frame-15 target, adapter architecture, seed, 60K/7.5K/7.5K
source-disjoint splits, and existing cache. They do **not** initialize from
the previous AlexNet-LPIPS fine-tune. Optimizer, scheduler, and epoch state
start fresh in each run. Neither run trains the V-JEPA predictor or Cosmos.

| Run | Config | Training loss |
| --- | --- | --- |
| 2: JEPA-WMs-style | `configs/experiments/vjepa21_cosmos_predicted_1frame_75k_jepawms_rgb_lpips.yaml` | `10 * RGB_MSE + 1 * VGG_LPIPS` |
| 3: plus latent anchor | `configs/experiments/vjepa21_cosmos_predicted_1frame_75k_jepawms_rgb_lpips_latent_mse.yaml` | `0.1 * latent_MSE + 10 * RGB_MSE + 1 * VGG_LPIPS` |

The 0.1 latent weight is a starting ablation value, not a calibrated
gradient balance. Run 2's zero latent weight means its target Cosmos latent
is **not** used as training supervision; it remains available for reporting
latent-error diagnostics. This adapts the JEPA-WMs image-head loss to a
**frozen Cosmos decoder** and **predicted V-JEPA features**; it is not the
same architecture or training-input distribution as the JEPA-WMs head.

Training LPIPS uses VGG. Validation/test LPIPS uses **AlexNet**, matching the
previous 75K runs for apples-to-apples comparison. The best checkpoint in
each new run is selected by full-validation predicted-frame AlexNet-LPIPS.
W&B config and summary identify both backbones.

Defaults for each run: 3 epochs, batch size 2, accumulation 8 (effective
batch 16), learning rate `5e-5` cosine-decayed to `5e-6`, validation at step
0 and every 1,000 optimizer steps (and at epoch end if not already due),
20 validation preview videos per pass. Separate W&B run names, output folders,
and Hugging Face model repos keep the experiments independent. Best, latest,
and every-epoch checkpoints are published to each run's HF repo. Final test
metrics are computed only after checkpoint selection and logged to W&B.

## Run on an existing Vast.ai instance with the original 75K cache

Activate the already-working environment; export fresh credentials in your
shell. Never put actual token values in this document or a Git commit:

```bash
cd /aayush/jepa-latent-to-rgb
git pull --ff-only origin cosmos-latent-adapter
export HF_TOKEN='your_huggingface_token'
export WANDB_API_KEY='your_wandb_key'
python -c 'import torch, lpips, cosmos_tokenizer; assert torch.cuda.is_available()'
python -c "import lpips; lpips.LPIPS(net='vgg'); lpips.LPIPS(net='alex'); print('LPIPS backbones ready')"
```

Fetch the **source** checkpoint once. The destination exactly matches both
configs' `training.init_adapter_checkpoint`:

```bash
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='Aaypom/vjepa21-cosmos-ci-predicted-1frame-75k-drive-walk-mse-adapter', filename='checkpoints/best.pt', local_dir='checkpoints/vjepa21_75k_mse_source'))"
```

Check that `data/denseworld_75k_drive_walk` and
`data/vjepa21_cosmos_75k_drive_walk_frame15` still exist. Do **not** rerun
`data` or `cache` when they are already complete. Run 2 first, then run 3:

```bash
CONFIG=configs/experiments/vjepa21_cosmos_predicted_1frame_75k_jepawms_rgb_lpips.yaml
bash scripts/run_vjepa21_cosmos_single_frame.sh validate "$CONFIG" --require-assets --require-cache --online
mkdir -p logs
nohup bash scripts/run_vjepa21_cosmos_single_frame.sh train-image "$CONFIG" > logs/vjepa21_75k_jepawms_rgb_lpips.log 2>&1 &
echo $! > logs/vjepa21_75k_jepawms_rgb_lpips.pid
tail -f logs/vjepa21_75k_jepawms_rgb_lpips.log
```

Wait for run 2 to finish before starting run 3 on a single GPU:

```bash
CONFIG=configs/experiments/vjepa21_cosmos_predicted_1frame_75k_jepawms_rgb_lpips_latent_mse.yaml
bash scripts/run_vjepa21_cosmos_single_frame.sh validate "$CONFIG" --require-assets --require-cache --online
nohup bash scripts/run_vjepa21_cosmos_single_frame.sh train-image "$CONFIG" > logs/vjepa21_75k_jepawms_rgb_lpips_latent_mse.log 2>&1 &
echo $! > logs/vjepa21_75k_jepawms_rgb_lpips_latent_mse.pid
tail -f logs/vjepa21_75k_jepawms_rgb_lpips_latent_mse.log
```

Check step-0 validation against the original source checkpoint before letting
either fine-tune continue: AlexNet-LPIPS, RGB MSE, and latent MSE should be
close to the original 75K checkpoint's validation numbers. If not, stop and
inspect the source checkpoint, cache, and model versions. The two new runs'
step-0 values should also be close to each other. A GPU with 48 GB VRAM is
preferable; if batch 2 OOMs, set batch size 1 and accumulation 16 in **both**
configs before starting either run to preserve effective batch 16.

## Fresh instance

Clone the same branch and install/activate a working CUDA environment using
the repository's GPU setup instructions:

```bash
git clone -b cosmos-latent-adapter https://github.com/aayush-2021003/jepa-latent-to-rgb.git
cd jepa-latent-to-rgb
bash setup_env_uv.sh --gpu --from-wheels
source venv_walkindia/bin/activate
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

The setup script makes a **venv**, not a conda environment. An existing
known-good conda environment is equally usable; do not reinstall over it.
Export credentials as above and download assets with either new config:

```bash
CONFIG=configs/experiments/vjepa21_cosmos_predicted_1frame_75k_jepawms_rgb_lpips.yaml
bash scripts/run_vjepa21_cosmos_single_frame.sh assets "$CONFIG"
```

For an **identical** split, transfer the original
`data/denseworld_75k_drive_walk` and
`data/vjepa21_cosmos_75k_drive_walk_frame15` directories intact. If transferring
is impossible, rebuild once with `data` and `cache` using the same config;
compare clip-key manifests with the original instance before calling the
split identical. A fixed seed alone cannot protect against changed upstream
archives. Then fetch the source checkpoint above and run the preflight checks.
Budget at least 350 GB free disk (preferably 400-450 GB) if rebuilding the
dataset and cache from scratch.

## Push these changes from your development checkout

Stage only the files for this experiment; unrelated untracked files can stay
untouched. Check `git status -sb` first: pushing this branch also publishes
any earlier local commits that have not yet reached `origin`.

```bash
git status -sb
git log --oneline origin/cosmos-latent-adapter..HEAD
```

Then commit and push:

```bash
git add configs/experiments/vjepa21_cosmos_predicted_1frame_75k_jepawms_rgb_lpips.yaml \
  configs/experiments/vjepa21_cosmos_predicted_1frame_75k_jepawms_rgb_lpips_latent_mse.yaml \
  experiments/vjepa21_cosmos_single_frame/train_image_adapter.py \
  experiments/vjepa21_cosmos_single_frame/tracking.py \
  experiments/vjepa21_cosmos_single_frame/test_image_loss.py \
  experiments/vjepa21_cosmos_single_frame/JEPAWMS_STYLE_75K_FINETUNES.md
git diff --cached --check
git commit -m "Add two JEPA-WMs-style 75K adapter fine-tunes"
git push origin cosmos-latent-adapter
```
