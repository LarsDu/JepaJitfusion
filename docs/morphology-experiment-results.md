# Morphology color-augmentation experiments — results

*Status: results log. 2026-06-07. Branch: `feat/morphology-color-aug`.*

First experimental pass on Intervention A from
[`lejepa-morphology-improvement-plan.md`](./lejepa-morphology-improvement-plan.md):
randomize the *absolute* palette while preserving *relative* color geometry, to
move LeJEPA embeddings off the color-dominated regime (different Pokemon with
similar palettes as neighbors; same Pokemon's shiny variants far apart).

## Setup

- Dataset: `pokemon_11k`, vit_tiny, img 96, evaluated on the held-out **test** split.
- All experiment runs: 200 epochs, seed 1999, `global_crop_scale_min=0.5`.
- Metrics (`scripts/morph_eval.py`, `eval/morphology.py`), EMA weights:
  - **shiny median rank** — rank of a sprite's shiny twin among 1739 (lower better).
  - **shiny r@10** — fraction of shiny twins in the top 10 (higher better).
  - **color corr** — Pearson(embedding distance, palette Lab distance); the
    color-dominance signature (lower better).
  - **eff rank** — effective dimensionality (descriptive).
  - **morph r@k** — same-creature retrieval recall@k (saturated by near-identical
    cross-game renders; least sensitive).

## Results

| run | color aug | hue range | shiny med rank | shiny r@10 | color corr | eff rank | morph r@5 |
|---|---|---|---|---|---|---|---|
| `lejepa_5eb15573` (old) | default | ±~18° | 151 | 0.157 | 0.694 | 8.4 | 0.962 |
| `lejepa_morph_baseline` | default | ±~18° | 129 | 0.153 | 0.426 | 28.0 | 0.961 |
| `lejepa_morph_colorjitter` | colorjitter | ±45° | 111 | 0.181 | 0.370 | 28.9 | 0.962 |
| `lejepa_morph_cj_hue90` | colorjitter | ±90° | 45 | 0.214 | 0.286 | 29.5 | 0.964 |
| `lejepa_morph_cj_hue180` | colorjitter | ±180° | **30** | 0.245 | **0.281** | 23.3 | 0.971 |
| `lejepa_morph_lab` | lab | ±45° | 112 | 0.144 | 0.415 | 29.4 | 0.964 |
| `lejepa_morph_lab_hue180` | lab | ±180° | 38 | **0.270** | 0.350 | 22.0 | **0.974** |

(The old `5eb15573` baseline is 99 epochs / crop 0.3; `lejepa_morph_baseline` is the
clean same-recipe control — use it, not the old one, for attributions.)

## Findings

1. **The clean baseline was essential — much of the initial "win" was a confound.**
   The old 99-epoch / crop-0.3 baseline → 199-epoch / crop-0.5 default-aug baseline
   already moves color_corr 0.69→0.43 and eff_rank 8.4→28 with **no** color-aug
   change. Most of that shift is training length + crop floor, not the color
   augmentation. Always compare experiments to `lejepa_morph_baseline`.

2. **Hue range is the decisive lever, monotonically.** Holding all else fixed,
   ±18°→±45°→±90°→±180° drives ColorJitter shiny median rank 129→111→45→**30**
   (~4.3× vs. clean baseline) and color_corr 0.426→0.370→0.286→0.281. Color
   *decoupling* mostly saturates by ±90°; shiny *retrieval* keeps improving to ±180°.

3. **Lab vs. ColorJitter:** at matched ±45°, Lab ≈ baseline while ColorJitter is
   clearly better — empirically the *magnitude* of palette swing matters more than
   the metric-preservation purity of the Lab isometry. But at ±180° Lab catches up
   and posts the **best shiny r@10 (0.270)** and **morph r@5 (0.974)** while keeping
   more color info (corr 0.350): Lab rotates only hue, so it is fully hue-invariant
   yet preserves saturation/value/chroma structure — a distinct, useful operating
   point from ColorJitter (which also crushes saturation/brightness).

4. **Effective-rank reversal at ±180°:** very aggressive color invariance *lowers*
   eff rank (29→22–23), unlike milder settings that raise it. Removing color as a
   variation axis recompacts the representation.

## Recommended operating points

- **Balanced (keep partial hue/type signal):** `cj_hue90` — color_corr 0.286, shiny
  rank 45, retains some absolute hue (fire≈warm, water≈cool).
- **Max morphology:** `cj_hue180` — best shiny rank (30) + lowest color_corr, at the
  cost of hue-based type signal.
- **Hue-invariant but chroma-aware:** `lab_hue180` — best shiny r@10 and morph r@5,
  retains saturation/value structure; good if you want hue-invariance without
  discarding all color.

## Honest ceiling

Even the best run leaves shiny twins poorly retrieved in absolute terms (best
shiny r@10 = 0.270; best median rank 30 / 1739). Global color augmentation cannot
fully collapse shiny pairs because many shinies are **non-global recolors** (regions
recolored independently), not a single global hue rotation. Closing the remaining
gap likely needs the deferred interventions (masked latent prediction, per-region /
shape-biased augmentation) rather than more hue range.

## Reproduce

```bash
# sweep points (configs in conf/)
python -m jepajitfusion.train_lejepa_app --config-name train_lejepa_morph_colorjitter cj_hue=0.25 run_id=lejepa_morph_cj_hue90
python -m jepajitfusion.train_lejepa_app --config-name train_lejepa_morph_lab lab_hue_deg=180 run_id=lejepa_morph_lab_hue180
python -m jepajitfusion.train_lejepa_app color_aug=default global_crop_scale_min=0.5 num_epochs=200 +run_id=lejepa_morph_baseline
# eval
python scripts/morph_eval.py checkpoints/<run>/lejepa_last.pth ...
```
