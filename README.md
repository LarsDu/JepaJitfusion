# JepaJitfusion

Self-supervised image encoder meets flow-matching diffusion decoder, implemented in PyTorch.

Combines [LeJEPA](https://github.com/galilai-group/lejepa) (self-supervised ViT encoder with SIGReg loss) and [JiT](https://arxiv.org/abs/2511.13720) (Just Image Transformers — a plain ViT diffusion model with x-prediction and Heun ODE sampling) into a unified training pipeline.

The encoder learns rich image representations without labels. The decoder generates images, optionally conditioned on encoder embeddings. They can be trained independently or end-to-end.

## Features

* Reproducible environment with [`uv`](https://docs.astral.sh/uv/getting-started/installation). Get setup with a single command.
* Three training modes: **LeJEPA** (SSL encoder), **JiT** (diffusion decoder), **Fusion** (encoder → decoder).
* Automatic dataset download and preprocessing for Pokemon 11k, Tiny-ImageNet-200, and ImageNette.
* Hydra-based configuration with composable YAML configs for datasets, encoder architectures, and decoder architectures.
* Multi-crop augmentation for self-supervised training.
* Heun ODE sampler with classifier-free guidance support.
* Multi-decay EMA for improved sample quality.

## Architecture

```
                    ┌─────────────────┐
                    │  Input Image x  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │                             │
    ┌─────────▼─────────┐        ┌──────────▼──────────┐
    │  LeJEPA Encoder    │        │  JiT Decoder         │
    │  (ViT + SIGReg)   │        │  (ViT + x-prediction)│
    │                    │        │                      │
    │  Trained with SSL  │        │  Trained with v-loss │
    │  No labels needed  │        │  ± conditioning      │
    └─────────┬─────────┘        └──────────▲──────────┘
              │                             │
              │   CLS embedding             │
              └──────►JepaConditioner───────┘
                     (adaLN path)
```

**LeJEPA Encoder** — Standard ViT with CLS token. Trained via SIGReg: an invariance loss (MSE between views) plus a Gaussianity regularizer (Epps-Pulley characteristic function test via random slicing).

**JiT Decoder** — ViT with bottleneck patch embedding, RMSNorm, SwiGLU FFN, 2D RoPE, and adaLN-Zero conditioning. Predicts clean images directly (x-prediction) using flow-matching with logit-normal time sampling.

**Fusion** — Frozen encoder CLS embeddings are projected into the decoder's conditioning space via a learned MLP (JepaConditioner), replacing the class label path.

## Getting Started

### Setting up environment

This repo uses [`uv`](https://docs.astral.sh/uv/getting-started/installation) as the package/environment manager. Make sure to install it before proceeding.

```bash
# Install packages and setup virtual environment
uv sync

# Activate virtual environment

## Linux/macOS
. .venv/bin/activate

## Windows
. .venv/Scripts/activate
```

### Run the tests

```bash
uv run pytest tests/ -v
```

## Training

All training uses [Hydra](https://hydra.cc/) for configuration. Override any config value with `key=value` syntax.

Each training run is assigned a unique **run ID** (e.g., `jit_a1b2c3d4`) and writes all checkpoints and samples to its own directory under `checkpoints/<run_id>/`. This prevents runs from overwriting each other.

### Train a LeJEPA encoder (self-supervised)

```bash
uv run python -m jepajitfusion.train_lejepa_app
```

Train on Pokemon 11k with default ViT-Tiny encoder (256-dim, 6 layers, ~2.5M params). Uses multi-crop augmentation and SIGReg loss. No labels needed.

Override hyperparameters as needed:

```bash
uv run python -m jepajitfusion.train_lejepa_app \
  num_epochs=100 learning_rate=1e-4 batch_size=64
```

### Train a JiT diffusion decoder

```bash
uv run python -m jepajitfusion.train_jit_app
```

Train on Pokemon 11k with default JiT-Tiny decoder (384-dim, 8 layers, ~8M params). Unconditional generation using flow-matching with v-loss.

Use the smaller JiT-Micro config for faster iteration:

```bash
uv run python -m jepajitfusion.train_jit_app decoder=jit_micro
```

### Train with class conditioning (Tiny-ImageNet)

```bash
uv run python -m jepajitfusion.train_jit_app \
  dataset=imagenet_tiny decoder=jit_base \
  decoder.num_classes=200 decoder.conditioning_mode=label \
  batch_size=256 num_epochs=400
```

### Train the fusion pipeline (encoder → decoder)

First train a LeJEPA encoder, then use its checkpoint to condition the decoder:

```bash
uv run python -m jepajitfusion.train_fusion_app \
  encoder_checkpoint=checkpoints/<lejepa_run_id>/lejepa_last.pth
```

### Resume a training run

Pass the run ID from a previous run to resume from its latest checkpoint:

```bash
# The run ID is printed at startup, e.g. "Run: jit_a1b2c3d4"
uv run python -m jepajitfusion.train_jit_app decoder=jit_micro run_id=jit_a1b2c3d4
```

This restores model weights, optimizer state, and EMA models, and continues training from the last saved epoch. The learning rate schedule is also correctly resumed.

## Sampling

### Generate samples from a trained JiT model

```bash
uv run python -m jepajitfusion.sample_app \
  checkpoint_path=checkpoints/<run_id>/jit_last.pth \
  num_samples=32 \
  output_dir=samples/pokemon
```

### Generate class-conditioned samples

```bash
uv run python -m jepajitfusion.sample_app \
  checkpoint_path=checkpoints/<run_id>/jit_last.pth \
  num_samples=16 \
  class_label=42 \
  cfg_scale=2.0
```

## Datasets

| Dataset | Images | Resolution | Classes | Config |
|---------|--------|------------|---------|--------|
| Pokemon 11k | ~11,800 | 64×64 | — | `dataset=pokemon_64` |
| Tiny-ImageNet-200 | 100,000 | 64×64 | 200 | `dataset=imagenet_tiny` |
| ImageNette | ~13,000 | 64×64 | 10 | `dataset=imagenette` |

All datasets are downloaded automatically on first use.

## Model Configurations

### Encoder (LeJEPA)

| Config | embed_dim | depth | heads | params | Usage |
|--------|-----------|-------|-------|--------|-------|
| `encoder=vit_tiny` | 256 | 6 | 4 | ~2.5M | Pokemon, ImageNette |
| `encoder=vit_small` | 384 | 12 | 6 | ~21M | Tiny-ImageNet |

### Decoder (JiT)

| Config | dim | depth | heads | patch_size | params | Usage |
|--------|-----|-------|-------|------------|--------|-------|
| `decoder=jit_micro` | 128 | 6 | 4 | 8 | ~1.9M | Fast iteration, prototyping |
| `decoder=jit_tiny` | 384 | 8 | 6 | 4 | ~8M | Pokemon, ImageNette |
| `decoder=jit_base` | 768 | 12 | 12 | 4 | ~85M | Tiny-ImageNet |

## Project Structure

```
src/jepajitfusion/
  utils.py                     # get_device, set_seed, AMP helpers, LR schedule
  data/                        # Dataset download, transforms, DataLoader factory
  encoder/                     # ViT backbone, SIGReg loss, projection head, multi-crop
  decoder/                     # JiT model (RMSNorm, SwiGLU, RoPE, adaLN-Zero),
                               #   conditioning, diffusion utilities, Heun sampler
  models/                      # MultiEMA
  trainers/                    # BaseTrainer, LeJEPATrainer, JiTTrainer, FusionTrainer
  config/                      # Typed dataclass configs per pipeline
  conf/                        # Hydra YAML configs
  train_lejepa_app.py          # LeJEPA training entry point
  train_jit_app.py             # JiT training entry point
  train_fusion_app.py          # Fusion training entry point
  sample_app.py                # Sampling entry point
tests/                         # 45 unit tests across 7 test files
docs/proposals/                # Architecture design document
```

## Useful Resources

* [LeJEPA / SIGReg](https://github.com/galilai-group/lejepa) — Self-supervised ViT using the Epps-Pulley characteristic function test
* [JiT: Just Image Transformers](https://arxiv.org/abs/2511.13720) — Plain ViT for diffusion with x-prediction (Li & He, 2025)
* [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — Lipman et al. (2022)
* [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — Peebles & Xie (2023), introduces adaLN-Zero

## Developer Notes

`black`, `ruff`, `isort`, and `pre-commit` come as preinstalled dev packages in the virtual environment.

Install pre-commit hooks to ensure code consistency:

```bash
pre-commit install
```

### Future Goals

- [ ] Patch-level conditioning via cross-attention (not just CLS token)
- [ ] Joint fine-tuning (unfreeze encoder during fusion training)
- [ ] Progressive resolution training (64→128→256)
- [ ] DDP support for multi-GPU training
- [ ] Online linear probe during LeJEPA training
- [ ] Classification head at intermediate JiT layer (Appendix B.3)
