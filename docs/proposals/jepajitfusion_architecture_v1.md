# JepaJitfusion Architecture Plan v1

## Context

This project combines two cutting-edge techniques from early 2026:

1. **LeJEPA** (galilai-group/lejepa) — a self-supervised image encoder using the SIGReg loss (Epps-Pulley characteristic function test). Unlike teacher-student SSL methods (DINO, I-JEPA), LeJEPA uses a clean statistical approach: it tests whether each dimension of the embedding follows a standard normal distribution. No stop-gradient, no EMA teacher, no momentum encoder.

2. **JiT** (Li & He, 2025) — "Just Image Transformers" for diffusion. A plain ViT that directly predicts the clean image (x-prediction) rather than noise or velocity. Uses flow-matching with a linear schedule, bottleneck patch embedding, adaLN-Zero conditioning, and Heun ODE sampling.

The goal is a training pipeline where:
- A LeJEPA encoder learns rich image representations via SSL
- A JiT diffusion decoder generates images, optionally conditioned on LeJEPA embeddings
- These can be trained independently or end-to-end

**MVP dataset**: Pokemon 11k (64x64). **Additional datasets**: Tiny-ImageNet-200 (200 classes, 64x64, 100k images) and ImageNette (10 classes, ~13k images, resized to 64x64).

## Architecture Overview

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

## Key Design Decisions

See the full plan discussion in the project's planning transcript for detailed rationale on:
- Conditioning via adaLN (not cross-attention) for global CLS embeddings
- Patch sizes: encoder=8, decoder=4 at 64x64
- Model sizing: ViT-Tiny (~2.5M) / JiT-Tiny (~8M) for Pokemon
- x-prediction over epsilon-prediction
- noise_scale = img_size/256 = 0.25
