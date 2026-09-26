# 10K predicted-JEPA image-loss experiment

This is a separate run from the earlier latent-only 10K experiment. It keeps
the frozen V-JEPA 2.1 encoder/predictor and Cosmos-CI image encoder/decoder,
with the same adapter architecture. Frames 1–14 are context; V-JEPA predicts
the frame-15/16 tubelet; the adapter reads that prediction and learns a
Cosmos-CI latent for frame 15 alone. No true-future JEPA feature enters the
training input.

The objective is `latent L1 + 0.1 latent cosine + RGB MSE + 0.1 LPIPS`.
RGB MSE is measured in `[0,1]`; LPIPS (AlexNet) receives `[-1,1]` images.
Cosmos is frozen but its decoder stays in the autograd path. The best adapter
is chosen by validation predicted-frame LPIPS, **not** latent loss. The test
set is evaluated once, after loading that selected best checkpoint.

## Data protocol

The new 10K draw has 9,000 train, 500 validation, and 500 test clips. The
downloader samples about 22 widely spaced DenseWorld TARs by default, reserves
at least 20 different source videos for each evaluation split, and prevents
any source video from appearing in more than one split. If the selected TARs
do not have enough source-disjoint clips, increase `download_margin_shards`
in the new config and retry with its same empty `local_root`. The selected
10K clips are repacked locally; remote source TARs are discarded afterward.

The cache is new because training now needs frame-15 RGB. Evaluation caches
also store frame 14 to score the persistence baseline. Validation and test
report predicted RGB MSE, PSNR, LPIPS; Cosmos reconstruction ceiling; and
frame-14 persistence. Validation also logs eight five-panel videos, including
the actual observed 14-frame context, at every scheduled validation. The
best/latest adapter checkpoints go to the separate HF model repository in
the config. The final test uses the selected best checkpoint and is logged to
W&B, with eight additional test-preview videos.

## Vast.ai run

Choose an NVIDIA CUDA GPU with **48 GB VRAM preferred**, 32 GB only after a
batch-size-1 smoke test, and **150 GB persistent disk**. Roughly 22 remote
TARs occupy 20–25 GB during preparation, the repacked 10K subset around
10–12 GB, the new latent/RGB cache around 22–25 GB, and the V-JEPA checkpoint
around 14 GB; allow additional space for the environment, Cosmos weights,
checkpoints, and download overhead. These are planning estimates, not a measured peak; the frozen
decoder is backpropagated through for image losses. `batch_size: 2` and
`gradient_accumulation_steps: 8` make effective batch 16. Training is 20
epochs, roughly 563 optimizer steps/epoch, with validation every 150 steps.

```bash
git clone -b cosmos-latent-adapter https://github.com/aayush-2021003/jepa-latent-to-rgb.git
cd jepa-latent-to-rgb
bash setup_env_uv.sh --gpu --from-wheels
source venv_walkindia/bin/activate
export HF_TOKEN="your_current_HF_token"
export WANDB_API_KEY="your_current_WandB_key"
CONFIG=configs/experiments/vjepa21_cosmos_predicted_1frame_10k_image_loss.yaml
bash scripts/run_vjepa21_cosmos_single_frame.sh validate "$CONFIG"
bash scripts/run_vjepa21_cosmos_single_frame.sh assets "$CONFIG"
bash scripts/run_vjepa21_cosmos_single_frame.sh data "$CONFIG"
bash scripts/run_vjepa21_cosmos_single_frame.sh cache "$CONFIG"
bash scripts/run_vjepa21_cosmos_single_frame.sh validate "$CONFIG" --require-assets --require-cache --online
mkdir -p logs
nohup bash scripts/run_vjepa21_cosmos_single_frame.sh train-image "$CONFIG" > logs/vjepa21_10k_image_loss.log 2>&1 &
echo $! > logs/vjepa21_10k_image_loss.pid
tail -f logs/vjepa21_10k_image_loss.log
```

The setup script creates its own Python virtual environment; a separate Conda
environment is not needed. Do not paste tokens into the config or Git. The
dataset/cache/output roots are isolated from previous runs. The data, assets,
and cache stages are restartable. `train-image` resumes from the latest
completed epoch in its new output directory. Final test metrics are at
`outputs/vjepa21_cosmos_predicted_1frame_10k_image_loss/test_metrics.json`.

Before paying for a long run, verify one differentiable batch on the selected
GPU. An unsupported Cosmos-JIT autograd path fails loudly with a decoder
detachment error; lower `training.batch_size` to 1 and raise accumulation to
16 if the first batch runs out of VRAM. The two sizes keep effective batch 16.
