# FactorJEPA latent-to-video experiment

This is an isolated three-phase experiment. It does not import or modify the
existing `stage_1_*` or `stage_2_*` decoder code.

The experiment uses only the Cosmos-Predict2.5 video VAE. It does **not** load the
Cosmos text encoder or diffusion transformer. FactorJEPA features are mapped into
the Cosmos VAE's normalized latent space, then the frozen VAE converts those
latents into MP4 pixels.

## Fixed checkpoint

- Hugging Face repo: `anonymousML123/factorjepa-outputs`
- Repo type: `dataset`
- File:
  `outputs/full/vjepa_2_1_vitg_1B/train/m09c_surgery_3stage_DI_diheavy_encoder/m09c_ckpt_best.pt`
- Cosmos model: `nvidia/Cosmos-Predict2.5-2B`
- Cosmos revision: `diffusers/base/post-trained`

The loader checks the FactorJEPA student and predictor load percentages and stores
the source checkpoint's SHA-256 in the oracle checkpoint. Predicted evaluation
refuses to run against a different source checkpoint.

## Recommended progression

| Mode | Oracle train | Validation | Held-out test | Maximum epochs |
|---|---:|---:|---:|---:|
| Sanity | 32 | 16 | 16 | 1 |
| Proof of concept | 10,000 | 1,000 | 1,000 | 5 |
| Full | all clips in the source-safe 80% partition | all clips in 10% | all clips in 10% | 5 |

Start with sanity, then run the 10K proof of concept. Do not move to the full
roughly 115K corpus unless the proof of concept shows all of the following:

1. The Cosmos VAE roundtrip is visually adequate on Indian street footage.
2. Oracle decoding improves during validation and produces recognizable motion.
3. Predicted decoding is measurably worse than oracle decoding but remains
   semantically recognizable. If oracle decoding itself fails, more predictor
   training cannot fix the decoder bottleneck.

Early stopping uses the full validation manifest and patience of two epochs. The
test manifest is never used for fitting or early stopping.

## Phase A: source-video-safe manifests

The split utility assigns complete source videos before selecting clips. Clips
from the same original YouTube video cannot leak across train, validation, and
test even when they occur in different TAR shards.

```bash
python -u src/utils/jepa_decode_splits.py \
  --manifest data/full_local/full_local.json \
  --decode-config configs/jepa_pixel_decode.yaml \
  --mode poc \
  --output-dir data/jepa_decode_splits/poc \
  --cache-policy 2
```

For the 32-clip smoke test, replace both occurrences of `poc` with `sanity`. For
the complete partition, replace both with `full`.

## Phase 0: measure the Cosmos VAE ceiling

This encodes real video with the frozen Cosmos VAE and decodes it immediately.
The result is the best pixel quality this latent target can represent. If these
videos are poor, stop: the chosen Cosmos VAE/crop/frame preprocessing is the
bottleneck.

```bash
python -u src/phase0_cosmos_vae_ceiling.py \
  --model-config configs/model/vjepa2_1_vitg.yaml \
  --train-config configs/train/surgery_3stage_DI_diheavy_encoder.yaml \
  --decode-config configs/jepa_pixel_decode.yaml \
  --local-data data/full_local \
  --manifest data/jepa_decode_splits/poc/validation.json \
  --output-dir outputs/jepa_pixel_decode/phase0_poc \
  --wandb-group factorjepa-poc \
  --cache-policy 2
```

Outputs include input-versus-roundtrip MP4s, per-clip JSONL/CSV, and bootstrap
95% confidence intervals for pixel error, PSNR, LPIPS, and temporal-difference
error. With `WANDB_ENABLED=1`, metrics, MP4s, and JSON/CSV/JSONL artifacts are
uploaded to a Phase-0 W&B run.

## Phase 1: train the oracle decoder

The FactorJEPA encoder and Cosmos VAE are frozen. The trainable decoder sees only
the full, unmasked FactorJEPA encoder grid and learns:

`full FactorJEPA feature grid -> normalized Cosmos VAE latent`

It never sees FactorJEPA predictor tokens. This prevents it from learning to hide
predictor mistakes.

```bash
python -u src/oracle_jepa_decoder_train.py \
  --mode poc \
  --experiment-mode generalization \
  --model-config configs/model/vjepa2_1_vitg.yaml \
  --train-config configs/train/surgery_3stage_DI_diheavy_encoder.yaml \
  --decode-config configs/jepa_pixel_decode.yaml \
  --factorjepa-repo anonymousML123/factorjepa-outputs \
  --factorjepa-repo-type dataset \
  --factorjepa-filename outputs/full/vjepa_2_1_vitg_1B/train/m09c_surgery_3stage_DI_diheavy_encoder/m09c_ckpt_best.pt \
  --local-data data/full_local \
  --train-manifest data/jepa_decode_splits/poc/train.json \
  --validation-manifest data/jepa_decode_splits/poc/validation.json \
  --test-manifest data/jepa_decode_splits/poc/test.json \
  --output-dir outputs/jepa_pixel_decode/oracle_poc \
  --wandb-group factorjepa-poc \
  --cache-policy 2
```

Use `--cache-policy 1` to resume from `oracle_decoder_latest.pt`. A resume is
rejected if the model, data, decoder, Cosmos, or FactorJEPA checkpoint identity
changed. `oracle_decoder_best.pt` is selected by validation latent mean absolute
error.

## One-sample overfit diagnostic

This is a memorization test, not a generalization experiment. It uses one fixed
center-cropped clip for the VAE ceiling, initial decoder measurement, training,
periodic evaluation, and final reconstruction. Frozen FactorJEPA features and
the Cosmos target latent are computed once per process.

First create the one-clip manifest. Passing it for all three split arguments is
deliberate and the overfit contract rejects any other arrangement:

```bash
python -u src/utils/jepa_one_clip_manifest.py \
  --source-manifest data/full_local/manifest.json \
  --clip-index 0 \
  --output-manifest data/jepa_decode_splits/overfit_one/one.json \
  --cache-policy 2
```

Enable W&B explicitly and run the complete oracle memorization diagnostic:

```bash
wandb login
export WANDB_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u src/oracle_jepa_decoder_train.py \
  --mode sanity \
  --experiment-mode overfit_one \
  --model-config configs/model/vjepa2_1_vitg.yaml \
  --train-config configs/train/surgery_3stage_DI_diheavy_encoder.yaml \
  --decode-config configs/jepa_pixel_decode.yaml \
  --factorjepa-repo anonymousML123/factorjepa-outputs \
  --factorjepa-repo-type dataset \
  --factorjepa-filename outputs/full/vjepa_2_1_vitg_1B/train/m09c_surgery_3stage_DI_diheavy_encoder/m09c_ckpt_best.pt \
  --local-data data/full_local \
  --train-manifest data/jepa_decode_splits/overfit_one/one.json \
  --validation-manifest data/jepa_decode_splits/overfit_one/one.json \
  --test-manifest data/jepa_decode_splits/overfit_one/one.json \
  --output-dir outputs/jepa_pixel_decode/overfit_one \
  --wandb-group factorjepa-one-sample \
  --cache-policy 2 \
  2>&1 | tee logs/overfit_one_$(date +%Y%m%d_%H%M%S).log
```

W&B receives every training update, periodic latent/pixel measurements,
four-panel reconstruction MP4s, best/final checkpoint artifacts, and durable
training/evaluation JSON artifacts. Use
`--cache-policy 1` with the otherwise identical command to resume.

Then evaluate both official FactorJEPA masks on the same clip:

```bash
python -u src/predicted_jepa_decoder_eval.py \
  --model-config configs/model/vjepa2_1_vitg.yaml \
  --train-config configs/train/surgery_3stage_DI_diheavy_encoder.yaml \
  --decode-config configs/jepa_pixel_decode.yaml \
  --factorjepa-repo anonymousML123/factorjepa-outputs \
  --factorjepa-repo-type dataset \
  --factorjepa-filename outputs/full/vjepa_2_1_vitg_1B/train/m09c_surgery_3stage_DI_diheavy_encoder/m09c_ckpt_best.pt \
  --oracle-decoder-checkpoint outputs/jepa_pixel_decode/overfit_one/overfit_one_best.pt \
  --local-data data/full_local \
  --test-manifest data/jepa_decode_splits/overfit_one/one.json \
  --output-dir outputs/jepa_pixel_decode/overfit_one_predicted \
  --wandb-group factorjepa-one-sample \
  --cache-policy 2 \
  2>&1 | tee logs/overfit_one_predicted_$(date +%Y%m%d_%H%M%S).log
```

The one-record 95% confidence intervals necessarily collapse to the point
estimate. These results must not be reported as held-out performance.
Phase 0, Phase 1, and Phase 2 create separate W&B runs under the same
`factorjepa-one-sample` group.

## Phase 2: predicted-mode evaluation

This loads the best oracle decoder frozen. For each held-out clip and each
FactorJEPA mask generator, it:

1. obtains the full oracle encoder grid;
2. obtains FactorJEPA predictions for the target-token indices;
3. replaces only those target locations in the oracle grid;
4. runs the same frozen decoder and Cosmos VAE;
5. compares input, Cosmos VAE ceiling, oracle decode, and predicted decode.

There is no token-type embedding, so the decoder is not told which tokens were
replaced. Any quality drop from oracle to predicted mode is therefore evidence of
predictor error under a decoder held constant.

```bash
python -u src/predicted_jepa_decoder_eval.py \
  --model-config configs/model/vjepa2_1_vitg.yaml \
  --train-config configs/train/surgery_3stage_DI_diheavy_encoder.yaml \
  --decode-config configs/jepa_pixel_decode.yaml \
  --factorjepa-repo anonymousML123/factorjepa-outputs \
  --factorjepa-repo-type dataset \
  --factorjepa-filename outputs/full/vjepa_2_1_vitg_1B/train/m09c_surgery_3stage_DI_diheavy_encoder/m09c_ckpt_best.pt \
  --oracle-decoder-checkpoint outputs/jepa_pixel_decode/oracle_poc/oracle_decoder_best.pt \
  --local-data data/full_local \
  --test-manifest data/jepa_decode_splits/poc/test.json \
  --output-dir outputs/jepa_pixel_decode/predicted_poc \
  --wandb-group factorjepa-poc \
  --cache-policy 2
```

The two mask regimes are summarized separately with per-clip bootstrap 95%
confidence intervals. Important fields are:

- `oracle_vs_target_*`: decoder error with perfect full-video JEPA features;
- `predicted_vs_target_*`: total error after predictor substitution;
- `predicted_vs_oracle_*`: direct effect of replacing target features;
- `prediction_penalty_*`: paired per-clip increase over the oracle condition;
- `masked_predicted_vs_oracle_*`: pixel change inside target-token regions;
- `predicted_target_features_vs_oracle_*`: predictor feature error before decoding.

This predicted condition is a controlled intervention, not a standalone video
generator: any grid positions outside both context and target remain oracle. The
per-record `oracle_preserved_noncontext_fraction` makes that amount explicit. If
it is non-zero, do not describe the output as fully generated from context alone.

## Installation and hardware

Use `setup_env_uv.sh --gpu`; do not install packages individually. The GPU
requirements pin `diffusers==0.39.0`, which includes the Cosmos-Predict2.5 VAE
class used here. A CUDA GPU is mandatory. Start on the 48 GB sanity machine; use
the 96 GB machine if decoded RGB/LPIPS backpropagation exhausts memory.

All experiment constants are in `configs/jepa_pixel_decode.yaml`. Change that
file rather than embedding new values in the Python entry points.
