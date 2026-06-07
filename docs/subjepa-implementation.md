# Sub-JEPA: Subspace Gaussian Regularization for this repo's LeJEPA

*Status: proposal / planning. Author-assisted draft, 2026-06-06.*

## TL;DR

Our `SIGReg` enforces an isotropic Gaussian `N(0, I)` on the **full** projector
output by testing random 1D slices drawn over the *entire* `D`-sphere. **Sub-JEPA**
([Zhao et al., 2026](https://arxiv.org/pdf/2605.09241),
[code](https://github.com/intcomp/sub-jepa)) keeps the exact same Epps–Pulley test
but moves it into `K` *frozen random orthogonal subspaces* of dimension
`d_s = ⌊D/K⌋`. This relaxes the global isotropy constraint, letting the latent
**contract toward its intrinsic dimensionality** (lower effective rank) while
preserving leJEPA's anti-collapse guarantee.

The change is small and self-contained: a new `SubspaceSIGReg` regularizer that
reuses our existing `SlicingUnivariateTest`, plus a handful of config flags. It is
opt-in and reduces *exactly* to the current behavior at `K = 1, d_s = D`.

> **Scope note.** Sub-JEPA was published for *latent world models* (action-conditioned,
> temporal `L_pred` between consecutive latents on continuous-control envs). The
> **regularizer** transfers cleanly to image SSL; the paper's "prediction loss"
> simply maps to our **multi-crop invariance loss**. The reported planning-success
> numbers (Two-Room, PushT, OGB-Cube) are *not* image-retrieval results, so we
> should treat the morphology benefit as a hypothesis to measure, not a given —
> see [`lejepa-morphology-improvement-plan.md`](./lejepa-morphology-improvement-plan.md).

## Background: what our SIGReg does today

`src/jepajitfusion/encoder/sigreg.py`:

- `SIGReg.forward(*views)` computes `total = λ·reg + (1−λ)·inv` with `λ = 0.02`.
- `inv` (invariance) = variance of the view embeddings about their per-sample mean
  (raw, *not* L2-normalized).
- `reg` (regularization) = `SlicingUnivariateTest`, averaged over views. For each
  view of shape `(B, D)` it draws **fresh** random unit directions
  `directions ∈ R^{D×n_slices}` over the full sphere, projects `z @ directions →
  (B, n_slices)`, and runs the Epps–Pulley `UnivariateGaussianityTest` on each
  slice.

By the Cramér–Wold theorem, forcing *every* 1D marginal to `N(0,1)` forces the
joint `D`-dim distribution to `N(0, I_D)`. The regularizer acts on the **projector
output** (`proj_output_dim = 128`), not the encoder CLS token (`embed_dim = 256`)
used downstream. So in what follows, `D = proj_output_dim = 128`.

## The Sub-JEPA modification (math)

Sub-JEPA introduces `K` projection matrices `P_k ∈ R^{d_s × D}`, each
**row-orthonormal** and **frozen**, built by QR-decomposing a random Gaussian
matrix. Default `d_s = ⌊D/K⌋`.

For an embedding batch `Z ∈ R^{B×D}`:

1. **Project into each subspace:** `Z^(k) = Z P_kᵀ ∈ R^{B×d_s}`, for `k = 1…K`.
2. **Slice within the subspace:** draw `M` random unit vectors `u^(m) ∈ S^{d_s−1}`
   and project `z^{(k,m)}_b = ⟨Z^(k)_{b,:}, u^(m)⟩`.
3. **Epps–Pulley per (subspace, slice):** `T^{(k,m)} = T({z^{(k,m)}_b}_b)`.
4. **Average:** `L_reg = (1/KM) Σ_k Σ_m T^{(k,m)}`.

Total objective (our image-SSL mapping):

```
L_total = (1−λ)·L_invariance + λ·L_reg_subspace
```

### Why this differs from full-space SIGReg

The inner steps 2–3 are *literally* our current `SlicingUnivariateTest`, but applied
to the `d_s`-dim vector `Z^(k)` instead of the full `D`-dim `Z`. The slice
directions therefore only ever live **inside the fixed `K` subspaces** — the test
never probes arbitrary global directions. With small `d_s`, each subspace estimates
a low-dimensional Gaussianity that is a *weaker, more flexible* constraint than
full-`D` isotropy. The paper frames this as a **bias–variance trade-off**: full-space
isotropy over-regularizes when intrinsic dimension `≪ D`, inflating effective rank;
subspace regularization lets the representation contract. Empirically, the
**effective-rank reduction from LeWM→Sub-JEPA correlated with downstream gains**
(their Eq. 9, Fig. 2).

Two design facts from their ablations worth importing:

- **Frozen + orthogonal is load-bearing**, not an implementation detail. Random
  (non-orthogonal) frozen projections and *trainable* projections both did
  markedly worse (their Table 3) — trainable projections co-adapt with the encoder
  and weaken the anti-collapse signal.
- **`d_s` too small breaks it.** When subspaces become very narrow (e.g. `d_s = 6`),
  the per-slice normality estimate gets noisy and performance can collapse (PushT at
  `K=32`). There is a usable mid-range; don't push `K` to the extreme.

## Implementation in this repo

### 1. New regularizer in `sigreg.py`

Add a `SubspaceSIGReg` that owns the frozen projections and reuses
`SlicingUnivariateTest`:

```python
class SubspaceSIGReg(nn.Module):
    """leJEPA SIGReg restricted to K frozen random orthogonal subspaces.

    Reduces to full-space SIGReg at K=1, d_s=D.
    """

    def __init__(
        self,
        dim: int,                 # D = projector output dim
        n_subspaces: int = 4,     # K
        subspace_dim: int | None = None,  # d_s; default floor(D / K)
        n_slices: int = 256,      # M, slices *per subspace*
        t_max: float = 3.0,
        n_quad: int = 17,
        seed: int = 0,
    ):
        super().__init__()
        d_s = subspace_dim or (dim // n_subspaces)
        assert d_s >= 1 and d_s <= dim
        self.n_subspaces = n_subspaces
        self.subspace_dim = d_s
        self.test = SlicingUnivariateTest(n_slices, t_max, n_quad)

        # K independent row-orthonormal projections, built deterministically and
        # FROZEN (registered as a buffer, not a Parameter): moves with .to(device),
        # is saved/restored in checkpoints, and never receives gradients.
        gen = torch.Generator().manual_seed(seed)
        mats = []
        for _ in range(n_subspaces):
            g = torch.randn(dim, dim, generator=gen)
            q, _ = torch.linalg.qr(g)        # (D, D) orthonormal columns
            mats.append(q[:, :d_s].T)        # (d_s, D) row-orthonormal
        self.register_buffer("projections", torch.stack(mats))  # (K, d_s, D)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, D). Project into each subspace and run the sliced test there.
        # (K, d_s, D) x (B, D) -> (K, B, d_s)
        zk = torch.einsum("ksd,bd->kbs", self.projections.to(z.dtype), z)
        return sum(self.test(zk[k]) for k in range(self.n_subspaces)) / self.n_subspaces
```

Then make `SIGReg` delegate its regularization term to either the existing
`SlicingUnivariateTest` (full space) or `SubspaceSIGReg`, selected by a flag. The
invariance term and the `λ` convex combination are **unchanged**:

```python
class SIGReg(nn.Module):
    def __init__(self, ..., subspace: bool = False, n_subspaces: int = 4,
                 subspace_dim: int | None = None, proj_dim: int | None = None):
        ...
        if subspace:
            assert proj_dim is not None, "subspace SIGReg needs the projector out dim"
            self.slicing_test = SubspaceSIGReg(
                dim=proj_dim, n_subspaces=n_subspaces, subspace_dim=subspace_dim,
                n_slices=n_slices, t_max=t_max, n_quad=n_quad,
            )
        else:
            self.slicing_test = SlicingUnivariateTest(n_slices, t_max, n_quad)
```

Because `SubspaceSIGReg.forward` has the same `(B, D) → scalar` signature as
`SlicingUnivariateTest.forward`, the rest of `SIGReg.forward` (the
`sum(self.slicing_test(v) for v in views) / len(views)` line) needs no change.

### 2. Config flags

`src/jepajitfusion/conf/train_lejepa.yaml` (and the matching dataclass field on
`LeJEPATrainConfig`):

```yaml
# Sub-JEPA subspace regularization (opt-in). K=1 == current full-space SIGReg.
subjepa_enabled: false
subjepa_n_subspaces: 4        # K; with proj_output_dim=128 -> d_s = 32
subjepa_subspace_dim: null    # d_s; null => floor(proj_output_dim / K)
# Reuse sigreg_n_slices as M (slices per subspace).
```

Wire these into `LeJEPATrainer.__init__` where `SIGReg(...)` is constructed,
passing `proj_dim=config.proj_output_dim`.

### 3. Checkpoint compatibility

`projections` is a registered buffer, so it lands in `encoder`-adjacent state only
if `SIGReg` is checkpointed. Today the trainer saves `model_state_dict` (encoder),
`projector_state_dict`, EMA, and optimizer — **not** the SIGReg module. The frozen
projections are reproducible from `seed`, so no checkpoint change is strictly
required, **but** to guarantee identical subspaces across resume/eval, either (a)
fix `seed` in config and rebuild, or (b) add `sigreg_state_dict` to the saved
checkpoint. Recommend (a) for simplicity; it's deterministic and removes a failure
mode. Note SIGReg is SSL-only and discarded downstream, so this never touches the
fusion/JiT path.

### 4. Tests (`tests/test_sigreg.py`)

- **Orthonormality:** each `P_k @ P_kᵀ ≈ I_{d_s}`.
- **Determinism:** same `seed` ⇒ identical `projections`; different `seed` ⇒ different.
- **Reduction:** `SubspaceSIGReg(dim=D, n_subspaces=1, subspace_dim=D)` gives a loss
  in the same ballpark as `SlicingUnivariateTest(D)` on `N(0,I)` input (both ≈ small),
  and both blow up on a collapsed (rank-1) batch.
- **Anti-collapse preserved:** loss on a degenerate constant batch ≫ loss on `N(0,I)`.
- **Shape/finiteness:** finite scalar, correct device/dtype under autocast.

### 5. Suggested first sweep

`D = 128`, so candidate `(K, d_s)`: `(2, 64)`, `(4, 32)`, `(8, 16)`. Avoid very small
`d_s` (the paper's instability regime). Keep `M = sigreg_n_slices = 256` per subspace
(this multiplies total slices by `K` — cost is `K×` the slicing matmul, which is cheap
relative to the ViT forward). Hold `λ = 0.02` fixed first; it remains the single
loss-balancing knob.

## What to measure

The paper's headline diagnostic is **effective rank** (their Eq. 9):
`r_eff = exp(−Σ p_i log p_i)` over the normalized covariance eigenvalues `p_i` of the
embeddings. Our `lejepa_embeddings.ipynb` already plots the PCA explained-variance
spectrum — extend it to log `r_eff` for full-space vs each `(K, d_s)`. Expect Sub-JEPA
to *lower* `r_eff`. Whether lower `r_eff` helps **morphology retrieval** specifically is
the open question handed to the morphology plan.

## Risks / honest caveats

1. **Sub-JEPA optimizes compactness, not color/shape disentanglement.** Lower
   effective rank does not *by construction* move shape ahead of color. If color is
   the dominant axis of variation that survives our augmentations, a lower-rank space
   may simply keep color as PC1 with fewer trailing dims. The augmentation pipeline is
   the more direct lever for morphology (see companion doc).
2. **Domain transfer.** All Sub-JEPA evidence is on control envs with a temporal
   prediction loss. We're substituting multi-crop invariance for `L_pred`; the
   regularizer is identical but the surrounding objective is not.
3. **Small `d_s` instability** (see ablation note). Stay in the mid-range.

## References

- Zhao et al., *Sub-JEPA: Subspace Gaussian Regularization for Stable End-to-End World Models*, arXiv:2605.09241 (2026). https://github.com/intcomp/sub-jepa
- Maes et al., *LeWorldModel: Stable End-to-End JEPA from Pixels*, 2026 (the world-model leJEPA variant Sub-JEPA extends).
- Balestriero & LeCun, *LeJEPA: Provable and Scalable SSL Without the Heuristics*, 2025.
- Epps & Pulley, *A Test for Normality Based on the Empirical Characteristic Function*, Biometrika 70(3), 1983.
- Roy & Vetterli, *The Effective Rank: A Measure of Effective Dimensionality*, EUSIPCO 2007.
