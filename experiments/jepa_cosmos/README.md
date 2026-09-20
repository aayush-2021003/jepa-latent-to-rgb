# FactorJEPA future latents to Cosmos RGB

This experiment trains a small adapter while keeping FactorJEPA and the Cosmos
continuous video tokenizer frozen:

```text
ground-truth future RGB -> frozen FactorJEPA encoder -> true JEPA future tokens
last context frame + future RGB -> frozen Cosmos encoder -> future Cosmos targets

true JEPA future tokens -> trainable adapter -> two predicted future Cosmos latents
```

At inference time, true future tokens are replaced by the frozen FactorJEPA
predictor's causal output:

```text
8 context frames -> FactorJEPA predictor -> adapter -> two future Cosmos latents
last observed context frame -> frozen Cosmos encoder -> one known anchor latent
known anchor + predicted future latents -> Cosmos decoder -> discard anchor -> 8 future frames
```

Cosmos-CV4x8x8 follows a `4k+1` temporal cadence. The decoder therefore receives
the nine-frame-equivalent sequence `[frame 8 | frames 9..16]`: one known latent for
the last observed context frame and two adapter-predicted future latent slots. It
decodes nine frames and discards the reconstructed context frame. Future RGB is
never supplied to the FactorJEPA predictor or adapter at inference.

## Security and account setup

The code never accepts credentials as command-line arguments and never stores them in
configuration or checkpoints. Copy `.env.example` to `.env`, place `HF_TOKEN` and
`WANDB_API_KEY` there, and keep `.env` untracked. Because credentials were pasted into
a chat, rotate them after setup if that chat is not considered an approved secret
store.

The Hugging Face account associated with `HF_TOKEN` must accept the access conditions
for [DenseWorld archive](https://huggingface.co/datasets/anonymousML123/denseworld-115k_archive).

## Environment

Use a Linux NVIDIA GPU machine; a 48 GB card is suitable for validation and latent
precomputation at batch size 1, while 96 GB gives more headroom.

```bash
cd /Users/aayush/Documents/World-Models-JEPA/factorjepa
cp .env.example .env
# edit .env locally
./setup_env_uv.sh --gpu
source venv_walkindia/bin/activate
```

The GPU requirements install the final official Cosmos Tokenizer commit, LPIPS, and
the existing FactorJEPA stack. `setup_env_uv.sh` also checks out V-JEPA 2 under
`deps/vjepa2`.

## Preflight

```bash
bash scripts/run_jepa_cosmos.sh validate configs/experiments/jepa_cosmos_adapter.yaml
bash scripts/run_jepa_cosmos.sh validate configs/experiments/jepa_cosmos_adapter.yaml --online
```

The first command is local. The second also verifies the Hugging Face identity and
that both credentials are present, without printing either credential.

## Prepare a 5K dataset efficiently

The supplied config creates 4,500 train and 500 validation clips. It downloads six
evenly spaced remote TAR files, rather than scanning all 116, and then discards
unselected clips while repacking local 250-clip shards. Train and validation source
videos are disjoint.

```bash
bash scripts/run_jepa_cosmos.sh data configs/experiments/jepa_cosmos_adapter.yaml
```

For 10K, copy the YAML and change:

```yaml
data:
  train_samples: 9000
  val_samples: 1000
  local_root: data/jepa_cosmos_10k
  cache_root: data/jepa_cosmos_10k_latents
```

That downloads eleven remote shards. If a particular spread of shards does not leave
enough whole source videos after the leakage-safe split, increase
`download_margin_shards` by one; the script fails explicitly instead of mixing source
videos across splits.

## Download model assets

```bash
bash scripts/run_jepa_cosmos.sh assets configs/experiments/jepa_cosmos_adapter.yaml
```

This fetches the published FactorJEPA diheavy checkpoint containing both encoder and
predictor and the official `Cosmos-0.1-Tokenizer-CV4x8x8` encoder/decoder JIT files.

## Precompute frozen latents

```bash
bash scripts/run_jepa_cosmos.sh cache configs/experiments/jepa_cosmos_adapter.yaml
```

The cache is resumable. Train shards contain true JEPA future volumes, a one-slot
Cosmos context anchor, and two future Cosmos target slots. Validation shards
additionally contain causal FactorJEPA predictions. For 5K, budget roughly 40 GB for
fp16 caches; 10K needs roughly 75–80 GB. A small separate preview cache retains RGB
frames for W&B videos. Caches made by the earlier repeated-frame formulation are
rejected by a schema check and must not be reused.

## Train

```bash
bash scripts/run_jepa_cosmos.sh train configs/experiments/jepa_cosmos_adapter.yaml
```

The default objective is:

```text
L = 1.0 * L1(adapter(z_jepa), z_cosmos)
  + 0.1 * (1 - cosine(adapter(z_jepa), z_cosmos))
```

Set `rgb_l1`, `temporal`, or `perceptual` to non-zero and set
`decoded_loss_every_steps` to a positive interval to add losses through the frozen
Cosmos decoder. The known context anchor is prepended before decoding and its RGB
frame is removed before computing loss. These losses compare against the anchored
Cosmos reconstruction ceiling, which avoids repeatedly storing all RGB training
frames.

Every epoch logs training and validation loss, oracle adapter metrics, causal
world-model metrics, learning rate, and validation videos to the configured W&B
project. In addition, the default configuration runs the complete 500-sample
validation set every 200 **optimizer updates** (after gradient accumulation) and logs
all oracle and causal metrics against `optimizer_step` in W&B. End-of-epoch
validation always runs even when it does not coincide with the interval. The full
step-level record is also saved as `validation_history.json`.

`adapter_latest.pt` and improvements to `adapter_best.pt` are uploaded to
`Aaypom/factorjepa-cosmos-future-adapter` with metrics, the exact config, and a model
card. `--no-wandb` and `--no-hf-push` exist only for local debugging.

For a 48 GB GPU, retain `batch_size: 2` and
`gradient_accumulation_steps: 8` (effective batch size 16). Change
`validate_every_optimizer_steps` to tune validation frequency; smaller values add
validation overhead but do not change optimization.

### Overfit smoke test

Before the full run, memorize a tiny subset with:

```bash
bash scripts/run_jepa_cosmos.sh train configs/experiments/jepa_cosmos_adapter.yaml --overfit
```

This mode uses the same 16 cached validation examples for training and validation,
disables data-loader parallelism and validation-video uploads, and writes to
`outputs/jepa_cosmos/overfit` so it cannot resume from or overwrite the full run.
With batch size 2, no gradient accumulation, and 125 epochs, it performs exactly
1,000 optimizer updates. Validation runs every 50 updates and at every epoch end.
Metrics and checkpoints are still logged to a clearly named W&B run; Hugging Face
uploading is disabled for this diagnostic mode by default. The latent cache has no
training-time augmentation, so augmentation is effectively disabled.

## Inference

Pass a DenseWorld-style clip containing both context and held-out future frames:

```bash
bash scripts/run_jepa_cosmos.sh infer configs/experiments/jepa_cosmos_adapter.yaml \
  --input-video /absolute/path/to/clip.mp4
```

The output directory contains `context.mp4`, `ground_truth_future.mp4`,
`predicted_future.mp4`, `metrics.json`, and `comparison.mp4`. The comparison columns
are, from left to right: ground truth, Cosmos reconstruction ceiling, adapter from
true JEPA future tokens, and adapter from the actual causal FactorJEPA prediction.
