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
schedule starting at 2e-4, and latent MSE alone. This variant **does not cache
validation latents or run validation**. It evaluates the **final epoch-5
checkpoint exactly once on test**, with no test-based checkpoint selection.
This differs from the six-block run's validation-selected checkpoint and
should be disclosed in the paper.

On the existing Vast.ai instance, from the repository root with the working
Python environment active:

```bash
git pull --ff-only origin cosmos-latent-adapter
CONFIG=configs/experiments/vjepa21_cosmos_predicted_1frame_75k_linear_mse.yaml
bash scripts/run_vjepa21_cosmos_single_frame.sh assets "$CONFIG"
bash scripts/run_vjepa21_cosmos_single_frame.sh data "$CONFIG"
bash scripts/run_vjepa21_cosmos_single_frame.sh cache "$CONFIG"
bash scripts/run_vjepa21_cosmos_single_frame.sh validate "$CONFIG" \
  --require-assets --require-cache
mkdir -p logs
nohup bash scripts/run_vjepa21_cosmos_single_frame.sh train-image "$CONFIG" \
  > logs/vjepa21_75k_linear_mse.log 2>&1 &
echo $! > logs/vjepa21_75k_linear_mse.pid
tail -f logs/vjepa21_75k_linear_mse.log
```

The `cache` command's default `all` mode reads `data.cache_splits: [train, test]`
for this config; it **does not build the validation cache**. It resumes any
completed train/test shards and skips already-cached clips. The original
75K cache root is reused, so skip `data` and `cache` if those train/test
caches already exist and match this config. The dataset download still
prepares a 60K/7.5K/7.5K disjoint split; it is only the val *latent encoding*
that is skipped.

The linear run has a separate output directory and W&B run name. W&B receives
training and final test scalars, with no validation scalars or previews. The
config disables HF publishing by default and sets any future HF repo to private.
`adapter_latest.pt` contains the epoch-5 weights used for the final test;
no validation-selected `adapter_best.pt` is created.

The final epoch-5 checkpoint's held-out test means are in:

`outputs/vjepa21_cosmos_predicted_1frame_75k_linear_mse/test_metrics.json`

For the paper table, use `predicted_rgb_lpips`, `predicted_rgb_psnr`,
`predicted_rgb_mse`, and `predicted_latent_mse` from that file, in that order.
This test pass runs only after the fifth epoch completes. Its JSON records
`"selected_on": "final_epoch"`.
The `persistence_*` and `cosmos_*` keys are controls, not linear-readout
scores. No placeholder or intermediate validation value belongs in the
results table.

This is matched to the **latent-MSE objective and five-epoch optimization
budget** of the six-block adapter, but not its validation-based model
selection. It is not a matched-capacity/objective comparison against an
adapter additionally fine-tuned on RGB/LPIPS. To compare architecture after
image-loss fine-tuning, continue this linear model under the same image-loss
schedule separately.
