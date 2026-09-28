# 60K walking/driving single-frame latent-MSE run

Config: `configs/experiments/vjepa21_cosmos_predicted_1frame_60k_drive_walk_mse.yaml`.
This is a new run with distinct dataset, cache, output, W&B name, and Hugging Face
model repository; it does not overwrite the 10K experiments.

The gated DenseWorld archive is sampled with seed 42. Only clips whose JSON
`tour_type` is `walking` or `drive` are eligible. The downloader selects exactly
48,000 training, 6,000 validation, and 6,000 test clips, with no source-video
overlap. It reserves at least 60 source videos for each evaluation split. The
archive may need more than the initial 88 remote shards; the downloader fetches
additional shards automatically if the filtered source-disjoint pools are too
small. The exact chosen clip keys and type counts are saved in
`data/denseworld_60k_drive_walk/dataset_summary.json` and split manifests.
This matches the paper's 80/10/10 ratio and source isolation, but is a newly
sampled subset, not the paper authors' original split assignment.

The frozen V-JEPA 2.1 model predicts the tubelet for frames 15-16 from frames
1-14. The adapter predicts a Cosmos-CI latent for frame 15. The **only training
loss** is latent MSE. The best checkpoint is selected on validation latent MSE;
RGB MSE, PSNR, and LPIPS are evaluation metrics only. Five epochs with batch 2
and accumulation 8 give an effective batch of 16 and approximately 15,000
optimizer steps. Full validation runs every 1,500 optimizer steps; the disjoint
test set is evaluated after training. The cache retains training RGB for later
image-loss fine-tuning without recaching.

On a prepared Vast.ai instance with `HF_TOKEN` and `WANDB_API_KEY` exported:

```bash
cd /aayush/jepa-latent-to-rgb
CONFIG=configs/experiments/vjepa21_cosmos_predicted_1frame_60k_drive_walk_mse.yaml
bash scripts/run_vjepa21_cosmos_single_frame.sh assets "$CONFIG"
bash scripts/run_vjepa21_cosmos_single_frame.sh data "$CONFIG"
bash scripts/run_vjepa21_cosmos_single_frame.sh cache "$CONFIG"
bash scripts/run_vjepa21_cosmos_single_frame.sh validate "$CONFIG" \
  --require-assets --require-cache --online
mkdir -p logs
nohup bash scripts/run_vjepa21_cosmos_single_frame.sh train-image "$CONFIG" \
  > logs/vjepa21_60k_drive_walk_mse.log 2>&1 &
echo $! > logs/vjepa21_60k_drive_walk_mse.pid
tail -f logs/vjepa21_60k_drive_walk_mse.log
```

The cache payload is approximately 143 GiB before serialization overhead:
about 2.0 MiB per training clip and 4.0 MiB per validation/test clip. The
repacked 60K video subset is approximately 66 GB at the public manifest's
mean size for walking/driving clips. During dataset preparation the initial 88
remote archive shards can require roughly 88-100 GB temporarily in addition
to the repacked subset; by default they are removed after repacking. Reserve
at least 300 GB free disk, preferably 350-400 GB for headroom, checkpoints,
download retries, and other installed assets. These are estimates, not measured
usage on the Vast.ai instance.
