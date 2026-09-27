# 75K predicted-latent perceptual fine-tune

Use `configs/experiments/vjepa21_cosmos_predicted_1frame_75k_drive_walk_mse_lpips_finetune.yaml`.
This is a **new run**, initialized from the **best** checkpoint of the completed
75K drive/walk latent-MSE run. It loads adapter weights only. It does not load
the old optimizer, scheduler, epoch, or best-metric state. The frozen V-JEPA
features, Cosmos-CI target latents, frame-15 RGB targets, and 60K/7.5K/7.5K
source-disjoint split are unchanged.

The training objective, with RGB normalized to `[0,1]`, is

`1.0 * latent_MSE + 5.0 * RGB_MSE + 2.0 * LPIPS_Alex`.

The coefficients are initial hyperparameters, not calibrated gradient-norm
weights. W&B logs raw and weighted loss terms separately. The best checkpoint
is selected on the full validation split's **predicted frame-15 RGB LPIPS**;
latent MSE, RGB MSE, PSNR, persistence, and Cosmos self-reconstruction remain
diagnostics. Do not select or tune on the test split.

The default fine-tune is 2 epochs, batch 1, accumulation 16 (effective 16),
learning rate `3e-5` decaying to `3e-6`, and full validation at step 0,
every 1,000 optimizer steps, and epoch end if needed. The separate W&B run
receives stepwise raw/weighted losses, validation metrics, 20 preview videos
per validation, curves, and final test results. The separate HF model repo
receives best/latest and numbered epoch checkpoints. A continuation of this
*new* fine-tune resumes its own latest optimizer/scheduler state normally.

## Run on the same Vast.ai instance

Use the existing configured Python/conda environment and export credentials
in the shell (`export HF_TOKEN='...'` and `export WANDB_API_KEY='...'`); do not
write tokens into config files or commit them. From the repository root:

```bash
cd /aayush/jepa-latent-to-rgb
CONFIG=configs/experiments/vjepa21_cosmos_predicted_1frame_75k_drive_walk_mse_lpips_finetune.yaml
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='Aaypom/vjepa21-cosmos-ci-predicted-1frame-75k-drive-walk-mse-adapter', filename='checkpoints/best.pt', local_dir='checkpoints/vjepa21_75k_mse_source'))"
bash scripts/run_vjepa21_cosmos_single_frame.sh validate "$CONFIG" --require-assets --require-cache --online
mkdir -p logs
nohup bash scripts/run_vjepa21_cosmos_single_frame.sh train-image "$CONFIG" > logs/vjepa21_75k_mse_lpips_finetune.log 2>&1 &
echo $! > logs/vjepa21_75k_mse_lpips_finetune.pid
tail -f logs/vjepa21_75k_mse_lpips_finetune.log
```

If the original `outputs/vjepa21_cosmos_predicted_1frame_75k_drive_walk_mse/adapter_best.pt`
is still present, edit `training.init_adapter_checkpoint` to that path instead
of downloading. The initial step-0 validation should closely reproduce that
checkpoint's prediction metrics on the same validation cache. If it does not,
stop and check cache/model versions before spending GPU time.

The existing `data/vjepa21_cosmos_75k_drive_walk_frame15` cache has frame-15
RGB targets and can be reused. No new DenseWorld download or cache build is
needed on the same instance. On a fresh instance, transfer or rebuild the
original data/cache first using the 75K MSE experiment's instructions; the
new config deliberately names the same paths and split.

The new local checkpoints and reports are under
`outputs/vjepa21_cosmos_predicted_1frame_75k_drive_walk_mse_lpips_finetune`.
The HF destination is
`Aaypom/vjepa21-cosmos-ci-predicted-1frame-75k-drive-walk-mse-lpips-finetune-adapter`.
If CUDA memory is tight, keep batch 1 and lower preview or validation
frequency before changing the effective batch size. Backpropagating through
the frozen Cosmos decoder is more memory-intensive than latent-MSE training.
