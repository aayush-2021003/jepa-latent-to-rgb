# 10K pure latent-MSE comparison

This is a **fresh adapter training run**, not a continuation of the image-loss
checkpoint. The input remains the frozen V-JEPA 2.1 prediction for frames
15–16 after observing frames 1–14. The target remains the Cosmos-CI latent of
frame 15. The only training loss is the mean squared error between the adapter
output and that target latent, calculated in float32. The Cosmos decoder and
LPIPS model are used only for validation, test metrics, and preview media.

The config deliberately reuses the exact 9,000/500/500 source-disjoint dataset
and latent/RGB cache of the image-loss experiment. It retains the same adapter,
seed, batch size 2, accumulation 8, 20 epochs, 2e-4 cosine learning rate, and
validation every 300 optimizer steps. The output directory, W&B run name, and
HF model repository are new. Best-checkpoint selection remains validation
predicted-frame LPIPS, so visual quality is compared using the same criterion;
validation also logs latent MSE, latent L1, RGB MSE, PSNR, and persistence.
The disjoint test set is evaluated once using the selected best checkpoint.

On the existing Vast.ai instance with the image-loss cache already complete:

```bash
cd /aayush/jepa-latent-to-rgb
git pull --ff-only origin cosmos-latent-adapter
CONFIG=configs/experiments/vjepa21_cosmos_predicted_1frame_10k_latent_mse.yaml
bash scripts/run_vjepa21_cosmos_single_frame.sh validate "$CONFIG" \
  --require-assets --require-cache --online
mkdir -p logs
nohup bash scripts/run_vjepa21_cosmos_single_frame.sh train-image "$CONFIG" \
  > logs/vjepa21_10k_latent_mse.log 2>&1 &
echo $! > logs/vjepa21_10k_latent_mse.pid
tail -f logs/vjepa21_10k_latent_mse.log
```

Do not rerun `data` or `cache` on that instance if the validation command
confirms they are complete. On a fresh instance, run `assets`, `data`, and
`cache` with this config before the required validation command. Set
`HF_TOKEN` and `WANDB_API_KEY` as environment variables; never put them in Git.
The best/latest checkpoints go to
`Aaypom/vjepa21-cosmos-ci-predicted-1frame-10k-latent-mse-adapter`.
