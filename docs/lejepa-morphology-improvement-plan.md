# Strengthening LeJEPA Embeddings for Morphology

*Status: brainstorm / planning. Author-assisted draft, 2026-06-06.*

## The observation

Manual review of `lejepa_embeddings.ipynb` nearest-neighbor results shows our
LeJEPA space is **color-dominated**: a query's nearest neighbors tend to be
*different* Pokémon that share a color palette, rather than the *same* Pokémon
under color variation (e.g. shiny variants land far apart). We want embeddings
that encode **morphology** — shape, silhouette, body plan — so that "same creature,
different palette" is *near* and "different creature, same palette" is *far*.

This document (a) diagnoses *why* color wins, and (b) lays out a ranked set of
interventions, of which Sub-JEPA (see
[`subjepa-implementation.md`](./subjepa-implementation.md)) is only one — and not
the most direct.

## Root-cause analysis

### 1. Augmentation–invariance coupling (the primary cause)

This is the central mechanism of all joint-embedding SSL: **the encoder is trained
to be invariant to exactly what the augmentations destroy, and stays discriminative
about what they preserve.** LeJEPA's invariance term pulls together the embeddings
of two crops of the same image; whatever differs between those crops is signal the
model learns to *ignore*.

Look at what our `MultiCropAugmentation` actually varies
(`src/jepajitfusion/encoder/multicrop.py`):

```python
transforms.RandomHorizontalFlip(),
transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),   # brightness, contrast, saturation, hue
transforms.ToTensor(),
# + RandomResizedCrop(scale=...)
```

- **Color is only *mildly* perturbed** (hue jitter `0.1`, saturation `0.2`) and is
  **never removed** — no `RandomGrayscale`, no channel drop, no solarize. So a
  Pokémon's palette survives augmentation almost intact and remains a *highly
  reliable, low-effort* cue for the invariance objective. The model has every
  incentive to encode it.
- **Shape is partly *destroyed*** by `RandomResizedCrop(scale=(0.3,1.0))` for
  globals and `scale=(0.05,0.3)` for locals. Aggressive cropping (especially the
  tiny local crops) means two views often show different *parts* of the creature,
  so a stable global silhouette is a *harder* invariant to learn than color.

Net effect: color is the path of least resistance. **This is the highest-leverage
lever and the cheapest to change.**

### 2. CLS-token global summary at low resolution

Downstream code uses the **encoder CLS token** (`embed_dim = 256`) at `img_size = 96`,
`patch_size = 8` → an 12×12 patch grid. A single pooled vector at modest resolution
favors coarse global statistics (dominant color, overall brightness) over fine
contour/topology. The patch tokens themselves carry more spatial/structural
information (the notebook's patch-PCA heatmaps already hint at this) but are not used
for retrieval.

### 3. Full-space isotropy doesn't prioritize morphology

SIGReg shapes the *distribution* (`N(0,I)`) but is **agnostic to which semantic
factor occupies which direction**. It will happily make color PC1 and shape PC7. So
the regularizer is neither the cause nor, by itself, the cure. Sub-JEPA changes the
*rank/geometry* of the space but likewise does not *target* color-vs-shape — keep
expectations calibrated (see caveat in the companion doc).

## Interventions, ranked by leverage / effort

### A. Destroy color in augmentation (highest leverage, lowest effort)

Make color something the model *must* be invariant to, so morphology becomes the
only consistent signal across views. Add to the common augmentation stack:

```python
transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
transforms.RandomGrayscale(p=0.2),                      # SimCLR/DINO-style
# optional, stronger: random channel permutation / solarization
transforms.RandomSolarize(threshold=0.5, p=0.2),
```

This is the standard SimCLR/DINO color-augmentation recipe and is precisely what's
*missing* here. The aggressive variant — to push *hard* toward palette-invariance —
is **random grayscale at high `p` (0.5+)** or a per-view random channel shuffle,
which makes the absolute palette nearly useless and forces reliance on shape/texture
relationships.

> **Design tension to decide deliberately:** for shiny-variant retrieval we *want*
> color-invariance (shiny ≈ normal). But color is not *pure* nuisance for Pokémon —
> type is correlated with palette. Full grayscale invariance throws that away. Two
> coherent stances: **(i)** go fully color-invariant (best for the stated morphology
> goal), or **(ii)** keep a *small* color-sensitive capacity (see D). Recommend
> starting with (i) — it directly tests the hypothesis — and measuring.

**Geometry side:** consider making global crops *less* destructive of silhouette
(raise `global_crop_scale_min` toward `0.5`) so a stable whole-creature outline is a
learnable invariant rather than a coincidence.

### B. Shape-biased input / texture randomization

CNN/ViT encoders are known to be texture-biased (Geirhos et al.). Two cheap ways to
tilt toward shape:

1. **Edge/Sobel augmentation:** with some probability, replace a view with its edge
   map (Sobel/Canny) or stack an edge channel. Edges encode silhouette and internal
   contour while discarding flat color regions. Invariance between an RGB view and an
   edge view forces the shared representation to be shape-centric.
2. **Texture/style randomization:** randomize low-level appearance while preserving
   geometry (e.g. random recoloring of segmented regions, or lightweight style
   transfer). Heavier to implement; revisit only if A+1 underperform.

### C. Sub-JEPA subspace regularization (secondary, compactness)

Implement per [`subjepa-implementation.md`](./subjepa-implementation.md). Expected
effect: lower **effective rank**, a more compact latent matched to intrinsic
dimensionality. *Hypothesis:* a tighter, less color-saturated space may let
morphological structure dominate the leading directions — **to be measured, not
assumed.** Best run *in combination with* A, since it cannot remove a color axis
that the augmentations keep feeding in.

### D. Explicit color/shape factorization (more speculative, higher payoff)

Use structure to *separate* the factors instead of just suppressing color:

- **Two-view-family invariance.** Define a "shape" invariance over *color-destroyed*
  view pairs (grayscale/edge) and a separate "appearance" head over color-jittered
  pairs, with the shape head's subspace **orthogonal** to the appearance head's.
  Sub-JEPA's frozen orthogonal subspaces are a natural substrate: dedicate some
  subspaces to color-invariant views and others to color-sensitive views. Retrieval
  then queries the shape subspace.
- This keeps color information *available* (stance ii above) while making morphology
  independently addressable — the cleanest answer to the stated goal, at the cost of
  a more involved objective.

### E. Predictive / dense structure (orthogonal direction)

- **I-JEPA-style masked latent prediction:** add a term that predicts the embeddings
  of masked patches from visible context. Masked prediction forces reasoning about
  spatial layout and part structure — inherently morphological — and composes with
  the existing invariance loss. This also dovetails with the repo's planned
  x-diffusion-over-embeddings work (`docs/x_diffusion_le_jepa.md`).
- **Use patch tokens for retrieval:** mean/attention-pool patch tokens (or concat
  with CLS) instead of CLS alone. Nearly free; test in the notebook first — it may
  shift retrieval toward spatial structure with zero retraining.

## Evaluation: make "morphology" measurable first

We cannot tune toward morphology without a metric. **Build this before running the
sweeps** — the dataset already supports it: `pokemon_11k` was recently split *by
Pokédex entry* (commit `807376a`), i.e. multiple images per species (normal/shiny/
forms) carry a shared identity label.

Proposed probes (add a `notebooks/morphology_eval.ipynb` or extend the existing one):

1. **Same-species retrieval recall@k.** For each query, fraction of top-k neighbors
   sharing its Pokédex identity (across palette variants). This is the headline
   number we want to go *up*.
2. **Color-confound control.** Compute a cheap palette descriptor per image (mean
   Lab / color histogram). Report (a) same-species recall@k vs (b) same-dominant-color
   recall@k. The current failure mode is (b) ≫ (a); success is (a) rising and the gap
   closing.
3. **Shiny-pair distance.** For species with a shiny variant, the rank/distance of
   the shiny as a neighbor of the normal form. Should *decrease* under color-invariant
   training.
4. **Effective rank** (`r_eff`, see companion doc) as a descriptive covariate.

Report all probes on the **held-out split** to avoid the identity-leakage the recent
split fix was about.

## Recommended sequence

1. **Build the evaluation probes** (esp. same-species vs same-color recall@k).
   Baseline the current checkpoint — quantify the color-dominance we observe by eye.
2. **Intervention A** (stronger color destruction + RandomGrayscale; optionally raise
   `global_crop_scale_min`). Retrain, re-probe. Expectation: same-species recall up,
   shiny pairs closer. This alone may largely fix the reported behavior.
3. **A + C** (Sub-JEPA on top). Check whether lower `r_eff` adds retrieval gains.
4. **Cheap retrieval-only tweak E** (patch-token pooling) — test on existing
   checkpoints, no retraining needed.
5. If color must be *preserved* (stance ii) or A over-suppresses type signal, escalate
   to **D** (factorized subspaces) and/or **B/E** (edge views, masked prediction).

## Honest expectations

- The **augmentation change (A) is the most likely single fix** for "same color,
  different Pokémon" neighbors, because it attacks the actual mechanism producing them.
- **Sub-JEPA is complementary, not a substitute** — it reshapes rank/geometry but does
  not target color; bundling it with A is sensible, leading with it is not.
- Everything here is **measurable on data we already have**; the evaluation harness is
  the prerequisite that makes the rest falsifiable.

## References

- Chen et al., *SimCLR* (color-augmentation ablations show color jitter + grayscale
  are decisive for SSL), ICML 2020.
- Caron et al., *DINO*, ICCV 2021 (multi-crop + color augmentation recipe).
- Assran et al., *I-JEPA: Self-supervised learning from images with a JEPA*, CVPR 2023
  (masked latent prediction).
- Geirhos et al., *ImageNet-trained CNNs are biased towards texture; increasing shape
  bias improves accuracy and robustness*, ICLR 2019.
- Zhao et al., *Sub-JEPA*, arXiv:2605.09241 (2026); Balestriero & LeCun, *LeJEPA*, 2025.
