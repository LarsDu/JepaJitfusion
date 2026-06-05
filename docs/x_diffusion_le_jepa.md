# X-Diffusion over Sequential LeJEPA Embeddings

*Status: proposal / planning. Author-assisted draft, 2026-06-02.*

## TL;DR

Today this repo conditions an **image-space** X-diffusion model (JiT) on a frozen
LeJEPA embedding to generate pixels. This proposal flips the target: keep the
LeJEPA encoder **frozen**, and train a diffusion model that **generates LeJEPA
embeddings conditioned on other LeJEPA embeddings** — e.g. predicting the
embedding(s) of future video frames from the embeddings of past frames.

In other words: a **generative latent dynamics / world model that operates
entirely in LeJEPA representation space**. Pixels are optional — they only re-enter
at visualization time, where we can reuse the *existing* fusion JiT decoder to turn
a predicted embedding back into an image.

## Why this is a natural fit for LeJEPA specifically

The key insight: **SIGReg trains the LeJEPA embedding marginal to be an isotropic
standard Gaussian, N(0, I).** That is *exactly* the prior that flow-matching /
diffusion models sample from and integrate toward.

Consequences:

1. **The diffusion source and the data marginal are the same family.** The
   `z_0 ~ N(0, I)` noise prior and the target embedding distribution are both
   (approximately) isotropic unit Gaussians. The flow only has to learn the
   *conditional structure* (how the future depends on the past), not also reshape
   a wildly anisotropic, heavy-tailed latent — which is the usual pain of latent
   diffusion (cf. the variance-normalization hacks in Stable Diffusion's VAE
   latents). LeJEPA gives us a well-conditioned latent *for free*.
2. **`noise_scale` ≈ 1.0**, not `img_size/256`. The pixel pipeline scales noise
   to match pixel statistics; here the target is already ~unit variance, so the
   standard flow-matching schedule applies directly. (We should *measure* the
   empirical per-dim std of the frozen encoder's embeddings and confirm it's ≈1
   before fixing this.)
3. **Whitening is essentially done by the encoder.** No per-channel latent
   normalization step is needed (and — see the SIGReg fixes — we should be careful
   not to *re-introduce* erroneous normalization here either).

This is the conceptual payoff of the SIGReg correctness work: an isotropic-Gaussian
latent is not just "nice for probing," it's the ideal substrate for a generative
dynamics model.

## What we are modeling

Given a sequence of frames `x_1 … x_T` (e.g. a video clip), the frozen encoder
maps each to an embedding `e_t = Encoder(x_t) ∈ R^D` (the CLS token; `D = embed_dim`).
We model the conditional distribution of future embeddings given context:

```
p(e_{c+1}, …, e_{c+H} | e_1, …, e_c)
```

- `c` = number of context frames, `H` = prediction horizon.
- The diffusion model denoises a block of `H` future embedding vectors,
  conditioned on the `c` context embeddings.
- Crucially this is a **distribution**, not a point: the future of a video is
  genuinely uncertain (aleatoric), so a generative model is the right tool even
  though the encoder is deterministic per frame.

This is "predict in representation space" (à la JEPA) but **generative and across
time** — closer to latent world models (DINO-WM, navigation/world-model diffusion,
diffusion forcing) than to single-image I-JEPA.

## Relationship to the existing codebase

| Concern | Existing (image X-diffusion) | New (embedding-sequence X-diffusion) |
|---|---|---|
| Diffusion target | image pixels `(B, C, H, W)` | embedding sequence `(B, H, D)` |
| Backbone | JiT ViT over 2D patches | 1D temporal transformer over embedding tokens |
| Positional enc. | 2D VisionRoPE | 1D temporal RoPE / learned temporal pos |
| Conditioning | timestep adaLN (+ optional CLS via `JepaConditioner`) | timestep adaLN + **context embeddings** (prefix tokens and/or cross-attn) |
| Objective | x-prediction + flow-matching v-loss | **same** (reuse `compute_z_t`, `compute_v_loss`, logit-normal time) |
| noise_scale | `img_size/256` | **≈ 1.0** (measure & confirm) |
| Encoder | frozen LeJEPA (for conditioning) | frozen LeJEPA (for *both* context and target) |

We deliberately **reuse**:
- `decoder/diffusion.py` (`sample_logit_normal_time`, `compute_z_t`, `compute_v_loss`)
  — these are shape-agnostic and apply to `(B, H, D)` with minor view changes.
- `decoder/conditioning.py::TimestepEmbedder` verbatim.
- The adaLN-Zero / RMSNorm / SwiGLU building blocks from `decoder/jit_model.py`
  (factor them out or import) for the temporal transformer.
- `models/ema.py`, `trainers/base_trainer.py`, the Hydra/dataclass config pattern,
  and the app entrypoint pattern.
- The **fusion JiT decoder** as an optional *visualizer*: predicted embedding → image.

## Architecture: the Embedding-Sequence Diffuser (ESD)

A plain transformer over a token sequence of embedding vectors. One token = one
frame's embedding (`R^D`).

```
context embeddings        noised future embeddings (z_t)
[e_1 … e_c]               [ẽ_{c+1} … ẽ_{c+H}]
   │                          │
   ▼                          ▼
 (context proj)            (noisy-token proj)         t ──▶ TimestepEmbedder ─┐
   │                          │                                               │
   └───────────► token sequence (c + H tokens) ◄──────────┘                  │
                              │  + 1D temporal RoPE                            │
                       N × ESDBlock  ◄── adaLN-Zero(c = time emb) ────────────┘
                              │
                    predict clean future embeddings  ê_{c+1..c+H}  (x-prediction)
```

Design choices to start (simplest that can work), with alternatives noted:

- **Token = CLS embedding per frame.** Start here (matches `JepaConditioner` and
  the fusion path). Later option: use the full patch-token grid per frame for a
  richer target (much larger sequence; defer).
- **Context injection: prefix tokens** (concatenate context tokens before the
  noised future tokens, attend over the whole sequence). Simple, lets the model
  use full temporal attention. *Only the future tokens carry the diffusion
  loss; context tokens are clean and act as conditioning.* Alternative:
  cross-attention to a frozen context memory (cleaner separation, more params).
- **Timestep conditioning via adaLN-Zero** exactly as JiT, fed by `TimestepEmbedder`.
- **x-prediction + flow-matching v-loss**, identical scheme to the image pipeline,
  for consistency and because it's known-good here.
- **Diffusion-forcing option (later):** allow *per-token* independent noise levels
  so the same model does both teacher-forced training and autoregressive rollout
  with variable horizons. Powerful but adds complexity — phase 3+.

### Why a transformer (not the 2D JiT)
The target is a short 1D sequence of vectors, not a 2D image. We drop
`BottleneckPatchEmbed`/`unpatchify` and 2D RoPE, replace patch embedding with a
linear `R^D → dim` token embedder, and use 1D temporal RoPE. Everything else
(RMSNorm, SwiGLU, adaLN-Zero, zero-init final layer) carries over.

## Data pipeline

1. **Source:** a video dataset. MVP candidates (small, fast):
   - A toy/synthetic moving-sprites or bouncing-ball video set (deterministic
     dynamics → easy to sanity-check rollouts).
   - Then a real small video dataset (e.g. UCF101 subset, Moving-MNIST,
     something on HuggingFace). Pick in Phase 0.
2. **Frozen-encoder feature extraction (offline, cached):** run the corrected
   LeJEPA encoder over every frame at the encoder's input resolution and **cache
   the embeddings to disk** (`.npy`/`.pt` per clip, shape `(T, D)`). Training then
   reads embeddings, never images — fast and decoupled from augmentation.
   - Use the encoder in `eval()` with `inference_mode`. **No augmentation** for the
     target/context frames (we want the true dynamics, not augmentation noise).
   - Record encoder checkpoint hash + config alongside the cache for provenance.
3. **Sampling:** draw clips of length `c + H` with a configurable frame stride
   (stride controls how much motion is between tokens).
4. **Sanity stats:** compute per-dim mean/std and a slice-Gaussianity check on the
   cached embeddings to (a) confirm the encoder really is ≈N(0,I) and (b) set
   `noise_scale`.

## Training objective

Reuse the flow-matching machinery, applied to the future-embedding block
`E_fut ∈ R^{B×H×D}`:

```
t   ~ logit-normal(P_mean, P_std)          # per-sample (or per-token in DF mode)
z0  ~ N(0, I)  · noise_scale (≈1.0)
z_t =  t · E_fut + (1 - t) · z0            # compute_z_t, view as (B,H,D)
Ê   =  ESD(z_t, t, context=E_ctx)          # x-prediction of clean future embeddings
loss = v-loss(Ê, E_fut, z_t, t)            # compute_v_loss, MSE in velocity space
```

Notes:
- `compute_z_t`/`compute_v_loss` need a tensor-shape generalization (currently
  `view(-1,1,1,1)` for images → `view(-1,1,1)` for `(B,H,D)`). Small refactor:
  broadcast `t` to `x.ndim` instead of hardcoding 4D.
- **Do not L2-normalize or per-dim standardize** the embeddings before the loss
  (same mistake class as the SIGReg bugs just fixed). Operate on raw embeddings.

## Conditioning & guidance

- **Context = the `c` clean context embeddings**, injected as prefix tokens.
- **Classifier-free guidance over context:** randomly drop the context (replace
  with a learned null token / zeros) with prob `p_drop` during training; at
  inference interpolate between context-conditioned and unconditional predictions
  to trade fidelity vs. diversity of the rollout.

## Inference / rollout

1. Encode the available context frames → `E_ctx`.
2. Sample `z0 ~ N(0,I)`, integrate the ODE/flow (Heun, reuse the sampler pattern in
   `decoder/sampler.py`) to produce `Ê_{c+1..c+H}`.
3. **Autoregressive rollout:** append predictions to the context window and repeat
   for long-horizon generation. Watch for drift (errors compounding); mitigations:
   train with longer `H`, scheduled sampling, or diffusion-forcing.
4. **Visualization (closing the loop):** feed each predicted embedding through the
   existing fusion JiT decoder (`conditioning_mode="jepa"`) to render predicted
   frames as images. This makes rollouts inspectable and ties the new pipeline
   back into the repo's image-generation stack.

## Evaluation

- **Embedding-space metrics:** MSE / cosine similarity between predicted and true
  future embeddings (teacher-forced, 1-step and H-step).
- **Distributional:** does the *generated* embedding marginal stay ≈N(0,I)
  (slice-Gaussianity) over a rollout? Drift away from the prior signals collapse.
- **Probe transfer:** if a linear probe (e.g. action/class) exists on embeddings,
  measure probe accuracy on *predicted* vs *true* embeddings.
- **Perceptual (via decoder):** decode rollouts and eyeball / FID against held-out
  frames. Secondary, since the decoder adds its own error.
- **Calibration:** sample multiple rollouts per context; check the spread covers
  plausible futures (the generative payoff vs. a deterministic regressor baseline).

**Baseline to beat:** a deterministic MSE regressor `E_ctx → E_fut` (a plain
transformer, no diffusion). The diffusion model should match it on 1-step MSE
*and* produce diverse, plausible multi-step rollouts where the regressor blurs to
the mean.

## Proposed file / module layout (matches repo conventions)

```
src/jepajitfusion/
  data/
    video_datasets.py        # clip sampling + cached-embedding dataset
    feature_cache.py         # offline frozen-encoder feature extraction
  sequence/                  # new "embedding dynamics" package
    esd_model.py             # Embedding-Sequence Diffuser (temporal transformer)
    temporal_rope.py         # 1D RoPE (or reuse a generalized RoPE)
  config/
    sequence_config.py       # SeqDiffTrainConfig dataclass
  conf/
    train_seqdiff.yaml       # hydra config
    sequence/esd_small.yaml  # model size presets
  trainers/
    seqdiff_trainer.py       # frozen encoder -> cache -> ESD training loop
  train_seqdiff_app.py       # entrypoint
tests/
  test_esd_model.py
  test_feature_cache.py
  test_seqdiff_diffusion.py  # shape-generalized z_t / v-loss
```

Refactors to support reuse (small, low-risk):
- Generalize `compute_z_t` / `compute_v_loss` to arbitrary tensor rank (broadcast
  `t` to `x.ndim`) so both pixel and embedding pipelines share them.
- Factor `RMSNorm`, `SwiGLU`, adaLN-Zero `Block`, `FinalLayer` out of
  `jit_model.py` into a shared `decoder/blocks.py` (or `nn/`), imported by both
  the 2D JiT and the 1D ESD.

## Config additions (sketch)

```python
@dataclass
class SeqDiffTrainConfig:
    dataset: VideoDataConfig          # source video + frame stride + resolution
    encoder: EncoderConfig            # must match the frozen checkpoint
    encoder_checkpoint: str = "checkpoints/lejepa_last.pth"
    # sequence
    context_len: int = 4
    horizon: int = 4
    frame_stride: int = 1
    # ESD model
    dim: int = 384
    depth: int = 8
    num_heads: int = 6
    # diffusion
    P_mean: float = -0.8
    P_std: float = 0.8
    noise_scale: float = 1.0          # measure encoder std first
    context_dropout: float = 0.1      # CFG over context
    # standard training fields (lr, wd, epochs, ema_decays, ...) as elsewhere
```

## Phased milestones

- **Phase 0 — Data & sanity (small).** Pick a video dataset; build the offline
  feature cache; verify cached embeddings are ≈N(0,I) and fix `noise_scale`.
  Deliverable: `feature_cache.py`, stats notebook.
- **Phase 1 — Deterministic baseline.** Plain transformer regressor `E_ctx→E_fut`,
  report teacher-forced MSE. Establishes the bar and the data/eval harness.
- **Phase 2 — ESD v1 (single-step, fixed H).** Implement the temporal diffusion
  transformer, reuse the flow-matching loss, train, beat the baseline on 1-step
  MSE and show rollout diversity. Decode a few rollouts via the fusion decoder.
- **Phase 3 — Long-horizon & guidance.** Autoregressive rollout, CFG over context,
  mitigate drift; evaluate H-step metrics and marginal-Gaussianity over rollouts.
- **Phase 4 (optional) — Diffusion forcing / per-token noise** for flexible
  horizons and stabler long rollouts; richer targets (patch-token grids).

## Open questions / risks

- **Information content of CLS embeddings over time.** A single 256-d CLS vector
  may be too coarse to capture fine motion. Mitigation: full patch-token targets
  (Phase 4) or a temporally-aware encoder. Validate in Phase 0/1 via baseline MSE.
- **Encoder must be exactly reproduced** at feature-extraction time (resolution,
  normalization to `[-1,1]`, eval mode). Cache provenance guards this.
- **Rollout drift** in autoregressive mode — the central hard problem of latent
  world models. Phases 3–4 target it directly.
- **Is the embedding marginal *really* ≈N(0,I)?** Depends on the quality of the
  corrected LeJEPA training run. If not, either (a) train LeJEPA longer/better, or
  (b) fit a fixed whitening transform from the cache (and apply its inverse at
  decode time) — but prefer (a), since clean isotropy is the whole advantage.
- **Determinism vs. distribution.** If the chosen dataset's dynamics are nearly
  deterministic, the generative model may collapse to the regressor; pick data
  with genuine future uncertainty to make the diffusion formulation pay off.

## References / prior art (conceptual)

- LeJEPA: Balestriero & LeCun, 2025 (arXiv:2511.08544) — isotropic-Gaussian SSL.
- JiT "Just Image Transformers": Li & He, 2025 — x-prediction flow-matching ViT.
- Latent world models in self-supervised feature space (DINO-WM and related),
  diffusion forcing, and latent video diffusion — for context-conditioned
  sequence generation in representation space.
```
