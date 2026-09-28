# Linear readout and paired 500-clip test comparison

Use the existing, source-disjoint 75K walking/driving split. The linear
readout is fitted on the 60K **train** cache for five epochs; its config
requests only `train` and `test` caches and does not load or score the 7.5K
validation cache. Full 7.5K **test** metrics are recorded after each epoch
and at completion. These per-epoch test scores are diagnostic only: the
checkpoint is fixed to epoch five, not selected using test results.

The test comparison uses a deterministic uniform sample of 500 distinct
keys from the same 7.5K test split. Four fixed adapters share each cached
predicted V-JEPA feature, Cosmos target latent, frame-15 RGB target, Cosmos
decoder, and AlexNet LPIPS evaluator. The four adapters are: the final
five-epoch linear readout, the validation-selected pure latent-MSE adapter,
the validation-selected AlexNet-LPIPS/RGB-MSE image-loss fine-tune, and the
validation-selected VGG-LPIPS/RGB-MSE fine-tune with latent-MSE anchor.
Results are paired per clip, not drawn from separate random batches.

The seven-panel MP4 for each selected clip contains moving context frames
1–14, true frame 15, Cosmos reconstruction of true frame 15, and the four
predicted-JEPA adapter readouts, in that order. The three non-linear model
checkpoints are read from their local `adapter_best.pt` paths or downloaded
from their configured Hugging Face repos as `checkpoints/best.pt`. The linear
checkpoint must be local and is read from `adapter_latest.pt` after epoch 5.
Checkpoint lineage and the test manifest are checked before evaluation.

## Safe restart after interrupting an in-progress cache

Press Ctrl-C once and wait for the process to exit. The cache writer flushes
its partial shard in `finally`. Existing completed keys are read from
`data/vjepa21_cosmos_75k_drive_walk_frame15/train/manifest.json`; do not
remove or edit this manifest or its shard files. Do not start a second cache
process until the first has exited. After pulling the new code:

```bash
CONFIG=configs/experiments/vjepa21_cosmos_predicted_1frame_75k_linear_mse.yaml
bash scripts/run_vjepa21_cosmos_single_frame.sh cache "$CONFIG"
bash scripts/run_vjepa21_cosmos_single_frame.sh validate "$CONFIG" \
  --require-assets --require-cache --online
bash scripts/run_vjepa21_cosmos_single_frame.sh train-image "$CONFIG"
bash scripts/run_vjepa21_cosmos_single_frame.sh compare-test \
  configs/experiments/vjepa21_cosmos_75k_test_compare_500.yaml
```

The cache command honors `data.cache_splits: [train, test]`: it will skip
previously completed train keys, finish train, then cache test. The setup
`validate` command checks assets and cache readiness; it does **not** run
model validation. The comparison runs after training completes.

For unattended execution, put the commands after `CONFIG=...` in one
`nohup bash -c 'set -euo pipefail; ...'` shell. Stop any existing cache
process first. The shell inherits the currently activated Python environment
and exported `HF_TOKEN`/`WANDB_API_KEY`.

Local outputs:

- `outputs/vjepa21_cosmos_predicted_1frame_75k_linear_mse/adapter_epoch_*.pt`
- `outputs/vjepa21_cosmos_predicted_1frame_75k_linear_mse/adapter_latest.pt`
- `outputs/vjepa21_cosmos_predicted_1frame_75k_linear_mse/test_history.json`
- `outputs/vjepa21_cosmos_predicted_1frame_75k_linear_mse/test_metrics.json`
- `outputs/vjepa21_cosmos_75k_test_compare_500/comparison_manifest.json`
- `outputs/vjepa21_cosmos_75k_test_compare_500/per_sample_metrics.json`
- `outputs/vjepa21_cosmos_75k_test_compare_500/summary_metrics.json`
- `outputs/vjepa21_cosmos_75k_test_compare_500/videos/sample_*.mp4`

W&B receives the epoch test metrics from training and a separate comparison
run with all 500 seven-panel MP4s and paired metrics. The comparison summary
reports RGB MSE, PSNR, AlexNet LPIPS, and latent MSE (for adapters); the
Cosmos reconstruction is an encoding/decoding ceiling and has no predicted
latent MSE. The 500 clips are a subset of test, so their summary is **not**
an independent 500-clip evaluation in addition to the 7.5K test results.
Avoid tuning model selection or hyperparameters to these monitored test
scores; doing so would invalidate an untouched-test claim.
