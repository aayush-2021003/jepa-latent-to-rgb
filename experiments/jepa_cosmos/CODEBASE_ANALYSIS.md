# FactorJEPA codebase analysis for the latent-visualization experiment

## Active architecture

The active implementation is under `src/`, with experiment configuration under
`configs/` and thin orchestration in `scripts/`. The large `iter/`, `Literature/`,
`overleaf/`, and copied supplement trees are research history or duplicated release
material; they are not runtime dependencies of this experiment.

The current model is native V-JEPA 2.1 rather than the Hugging Face V-JEPA wrapper.
`src/utils/vjepa2_imports.py` isolates the upstream Meta checkout under `deps/vjepa2`
and exposes the encoder, 2.1 predictor, masking utility, and mask generator. The
primary 2B configuration is `configs/model/vjepa2_1.yaml`: ViT-G with
1664-dimensional final-layer tokens. This experiment instead selects
`configs/model/vjepa2_1_vitg.yaml`, the matching 1B ViT-g architecture for the
published checkpoint: 1408-dimensional final-layer tokens, four-layer hierarchical
output, 16-pixel patches, two-frame tubelets, and a 24-layer predictor.

`src/utils/predictor_eval.py` is the important reuse boundary for this experiment.
It loads a combined FactorJEPA checkpoint, validates encoder and predictor coverage,
constructs temporal token masks, and runs the same causal predictor used by the
FactorJEPA evaluation suite. `m12d_future_mse.py`, `m12e_predictor_temporal.py`, and
`m14_metric_demo.py` demonstrate the expected checkpoint schema and establish that
the final `embed_dim` columns of the hierarchical prediction are the final encoder
layer.

The numbered active modules form the following pipeline:

- `m00a`–`m03`: source parsing, duration lookup, subset selection, video download,
  scene splitting, and WebDataset packing.
- `m04*`: VLM scene taxonomy, optical-flow motion features, action labels, and
  taxonomy labels.
- `m05*` and `m07`: frozen/model embeddings, baseline encoders, overlap controls,
  and UMAP analysis.
- `m09a*`: continual V-JEPA pretraining split into encoder and head variants.
- `m09b`, `m09d`, `m09e`, and `m09f`: PEFT, continual-learning, adaptive-gradient,
  and naive/full fine-tuning baselines.
- `m09c*`, `m10`, and `m11`: FactorJEPA surgery, SAM-derived masks, and streamed
  layout/agent/interaction factor views.
- `m12a`–`m12f`: action, taxonomy, motion, future-prediction, predictor-temporal,
  and encoder-temporal evaluation families with per-clip metrics.
- `m13` and `m14`: aggregate reports and qualitative predictor demonstrations.
- `m15`: the earlier lightweight pixel-decoder experiment; useful as evidence that
  final-layer JEPA predictions are the correct feature slice, but not used here
  because Cosmos supplies a substantially stronger frozen generative decoder.
- `m16`–`m18`: retrieval, VQA demonstrations, and VLM evaluation extensions.

Shared utilities are organized around configuration/path contracts, WebDataset and
video I/O, checkpoint safety, V-JEPA import isolation, frozen feature extraction,
training primitives, factor streaming, probe metrics, GPU batch sizing, W&B, and
Hugging Face publication. The new experiment reuses only stable boundaries from
those utilities and does not import one numbered `m*.py` module from another.

## Data and preprocessing

DenseWorld clips use paired `.mp4` and `.json` members in WebDataset TAR files. The
canonical clip key is `section/video_id/source_file`. `src/utils/data_download.py`
provides the parallel local TAR reader; `src/utils/video_io.py` performs deterministic
uniform frame sampling; and `src/utils/frozen_features.py` defines the evaluation
resize, center crop, and ImageNet normalization used by FactorJEPA.

The public `anonymousML123/denseworld-115k` repository now contains metadata and a
deterministic reconstruction program, not videos. The original 116 TAR shards are in
the gated `anonymousML123/denseworld-115k_archive` repository. The new downloader
therefore selects evenly distributed archive shards, downloads only the minimum plus
one margin shard, enforces source-video-disjoint train/validation splits, and repacks
exact subsets into the local layout already understood by FactorJEPA.

## Training and checkpoints

FactorJEPA training modules `m09a*` and `m09c*` produce two checkpoint forms. The
encoder-only `student_encoder.pt` is sufficient for embedding evaluation, while the
combined `m09*_ckpt_best.pt` contains both `student` and `predictor` and is required
for causal future prediction. This experiment uses the combined diheavy surgery
checkpoint published in the FactorJEPA outputs dataset.

The adapter experiment freezes the FactorJEPA encoder/predictor and both Cosmos
tokenizer halves. For a 16-frame clip, frames 1–8 are context and frames 9–16 are the
future. The true future FactorJEPA volume is taken from a full frozen-encoder pass;
the causal validation volume is produced by masking the second half and calling the
frozen predictor. Both use only the final 1408-dimensional layer. To respect
Cosmos-CV4x8x8's `4k+1` cadence without repeating a future frame, Cosmos encodes the
nine-frame sequence `[last context frame | eight future frames]`. The known context
frame is cached as one anchor latent; the adapter targets only the remaining two
future latent slots. Decoding prepends the anchor and removes its reconstructed RGB
frame, so the evaluated output remains exactly the eight held-out future frames.

Only `JEPAToCosmosAdapter` is optimized. It layer-normalizes JEPA tokens, projects
them to a smaller channel width, applies residual 3D convolutions, resamples onto the
Cosmos temporal/spatial grid, and predicts the two-slot, 16-channel continuous future
latent. The adapter never predicts or receives the known anchor latent.

## Experiment stages

1. `download_denseworld.py`: minimal gated archive download and exact local splits.
2. `download_assets.py`: FactorJEPA combined checkpoint plus official Cosmos JITs.
3. `prepare_latents.py`: resumable frozen-target caches; validation also caches causal
   JEPA predictions and a small RGB preview set.
4. `train_adapter.py`: latent L1 plus cosine alignment, with optional frozen-decoder
   RGB, temporal, and LPIPS losses; configurable optimizer-step and end-of-epoch
   validation measure both oracle adapter quality and actual causal-world-model
   visualization quality.
5. `infer.py`: saves context, ground-truth future, predicted future, and the four-way
   comparison used to separate tokenizer ceiling, adapter error, and JEPA prediction
   error.

W&B receives configurations, dataset/cache manifests, step and epoch metrics, loss
curves, validation comparisons, and inference videos. Hugging Face receives only the
adapter checkpoints, their metrics, the exact YAML configuration, and a model card.
