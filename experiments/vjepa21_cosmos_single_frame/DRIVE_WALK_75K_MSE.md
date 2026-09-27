# 75K walking/driving latent-MSE experiment

Use `configs/experiments/vjepa21_cosmos_predicted_1frame_75k_drive_walk_mse.yaml`.
The 60K config remains available but is not used for this run.

- Eligible clips: DenseWorld `tour_type` is `walking` or `drive` only.
- Exact sample counts: 60,000 train / 7,500 validation / 7,500 test (80/10/10).
- Split isolation: no source-video overlap; at least 60 source videos per
  evaluation split. The assignment is seed-42 deterministic for a fixed archive.
- Training: predicted V-JEPA 2.1 tubelet from frames 1-14 to the frame-15
  Cosmos-CI latent, using latent MSE only. The validation latent MSE chooses the
  best checkpoint; RGB MSE, PSNR, and LPIPS are diagnostics, not loss terms.
- Optimization: 5 epochs, batch 2, accumulation 8, effective batch 16, about
  18,750 optimizer updates; full validation at step 0 and every 1,000 updates.
- Tracking: separate W&B dataset, cache, and training runs. The training run
  logs train MSE/LR, every validation metric and 20 preview videos at each
  validation, per-epoch metrics, curves, and final test metrics/videos. HF stores
  `checkpoints/best.pt`, `checkpoints/latest.pt`, and permanent
  `checkpoints/epoch_001.pt` through `epoch_005.pt`, each with a matching metrics
  JSON. The training W&B summary records links to these HF checkpoints.

Validation also runs at an epoch end unless that optimizer step was already
validated. Five epochs therefore produce 23 passes in total: one at step 0,
18 at 1,000-step intervals, and four additional epoch-end passes. Logging
20 videos each time means 460 validation videos plus 20 final test videos
on W&B. Build the cache after setting
`preview_samples: 20`; the preflight check rejects an older eight-preview cache.

This matches the FactorJEPA paper's 80/10/10 ratio and source-video isolation,
but does not reproduce the paper's original source assignments or its explicit
city/capture-mode stratification. The downloader records selected clip keys and
per-split walking/driving counts in
`data/denseworld_75k_drive_walk/dataset_summary.json`.

On a configured Vast.ai instance with `HF_TOKEN` and `WANDB_API_KEY` exported:

```bash
cd /aayush/jepa-latent-to-rgb
CONFIG=configs/experiments/vjepa21_cosmos_predicted_1frame_75k_drive_walk_mse.yaml
bash scripts/run_vjepa21_cosmos_single_frame.sh assets "$CONFIG"
bash scripts/run_vjepa21_cosmos_single_frame.sh data "$CONFIG"
bash scripts/run_vjepa21_cosmos_single_frame.sh cache "$CONFIG"
bash scripts/run_vjepa21_cosmos_single_frame.sh validate "$CONFIG" \
  --require-assets --require-cache --online
mkdir -p logs
nohup bash scripts/run_vjepa21_cosmos_single_frame.sh train-image "$CONFIG" \
  > logs/vjepa21_75k_drive_walk_mse.log 2>&1 &
echo $! > logs/vjepa21_75k_drive_walk_mse.pid
tail -f logs/vjepa21_75k_drive_walk_mse.log
```

Estimated cache size is about 178 GiB before serialization overhead, with
training RGB retained for later image-loss fine-tuning. The selected video
subset is roughly 82 GB from the public manifest's mean walking/driving clip
size. About 103 remote archive shards are downloaded initially and normally
removed after repacking; additional shards are fetched if source-disjoint
selection needs them. Provision at least 350 GB free disk and preferably
400-450 GB to cover cache, video subset, temporary shards, assets, and retries.
