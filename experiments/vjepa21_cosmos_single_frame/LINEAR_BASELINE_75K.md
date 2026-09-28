# 75K linear readout baseline

This is a **new training run from random initialization**, not an inference
shortcut or a warm-start from the six-block adapter. It reuses exactly the
same 60,000/7,500/7,500 source-video-disjoint walking/driving split and
cached predicted V-JEPA 2.1 tubelets/Cosmos-CI frame-15 targets as
`vjepa21_cosmos_predicted_1frame_75k_drive_walk_mse.yaml`.

The readout is an affine 1×1 projection from 1,408 JEPA channels to 16
Cosmos channels on the 24×24 grid, followed by fixed bilinear upsampling to
48×48. It has 22,544 trainable parameters, no hidden layers, normalization,
activation, or learned spatial filtering. It receives **predicted**, not
true-future, JEPA features. Like the main latent-MSE run it trains for five
epochs with batch size 2, accumulation 8, AdamW, a cosine learning-rate
schedule starting at 2e-4, and latent MSE alone. The best checkpoint is
selected on full validation latent MSE, never on the test split.

On the existing Vast.ai instance, from the repository root with the working
Python environment active:

```bash
git pull --ff-only origin cosmos-latent-adapter
CONFIG=configs/experiments/vjepa21_cosmos_predicted_1frame_75k_linear_mse.yaml
bash scripts/run_vjepa21_cosmos_single_frame.sh validate "$CONFIG" \
  --require-assets --require-cache
mkdir -p logs
nohup bash scripts/run_vjepa21_cosmos_single_frame.sh train-image "$CONFIG" \
  > logs/vjepa21_75k_linear_mse.log 2>&1 &
echo $! > logs/vjepa21_75k_linear_mse.pid
tail -f logs/vjepa21_75k_linear_mse.log
```

**Do not run `data` or `cache` again** if the original 75K cache is complete.
The config uses its existing `cache_root`; it cannot overwrite the original
adapter because it has a separate output directory and W&B run name. W&B
receives train/validation/test scalar metrics, but validation media is disabled
to save time. The config disables HF publishing by default, including
per-epoch uploads, and sets any future HF repo to private. The local best and
latest checkpoints are still saved.

The final selected checkpoint's held-out test means are in:

`outputs/vjepa21_cosmos_predicted_1frame_75k_linear_mse/test_metrics.json`

For the paper table, use `predicted_rgb_lpips`, `predicted_rgb_psnr`,
`predicted_rgb_mse`, and `predicted_latent_mse` from that file, in that order.
This test pass runs only after selecting the best validation checkpoint.
The `persistence_*` and `cosmos_*` keys are controls, not linear-readout
scores. No placeholder or intermediate validation value belongs in the
results table.

This is matched to the **latent-MSE** six-block adapter. It is not a
matched-capacity/objective comparison against an adapter additionally
fine-tuned on RGB/LPIPS. To compare architecture after image-loss fine-tuning,
continue this linear model under the same image-loss schedule separately.
