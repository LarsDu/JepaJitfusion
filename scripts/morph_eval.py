"""Compare LeJEPA checkpoints on the Pokemon morphology metrics.

Usage:
    python scripts/morph_eval.py [CHECKPOINT ...]

With no args, evaluates the default baseline + the two morphology experiment runs.
Each checkpoint is evaluated on the held-out test split using its EMA weights.
"""

import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from jepajitfusion.config import DataConfig, EncoderConfig
from jepajitfusion.data.datasets import get_dataset
from jepajitfusion.data.transforms import eval_transform
from jepajitfusion.encoder.vit import VisionTransformer
from jepajitfusion.eval.morphology import (
    color_descriptor,
    color_vs_embedding_correlation,
    effective_rank,
    parse_pokemon_filename,
    retrieval_recall_at_k,
    shiny_neighbor_ranks,
)
from jepajitfusion.utils import get_device

DEFAULT_CKPTS = [
    "checkpoints/lejepa_5eb15573/lejepa_last.pth",  # baseline (default aug)
    "checkpoints/lejepa_morph_colorjitter/lejepa_last.pth",
    "checkpoints/lejepa_morph_lab/lejepa_last.pth",
]


def evaluate(ckpt_path: str, device) -> dict | None:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        print(f"[skip] {ckpt_path} not found")
        return None

    ec = EncoderConfig(**ckpt["encoder_config"])
    dc = DataConfig(**ckpt["dataset_config"])
    enc = VisionTransformer(
        img_size=dc.img_size, patch_size=ec.patch_size, in_channels=dc.num_channels,
        embed_dim=ec.embed_dim, depth=ec.depth, num_heads=ec.num_heads,
        mlp_ratio=ec.mlp_ratio,
    ).to(device)
    state = ckpt["ema_state_dicts"][0] if ckpt.get("ema_state_dicts") else ckpt["model_state_dict"]
    enc.load_state_dict(state)
    enc.eval()

    _, test = get_dataset(dc.name, transform=eval_transform(dc.img_size),
                          data_dir=dc.data_dir, test_size=dc.test_size)
    parsed = [parse_pokemon_filename(p) for p, _ in test.samples]
    morph = np.array([p["morph_id"] for p in parsed])

    embs, imgs = [], []
    with torch.no_grad():
        for x, _ in DataLoader(test, batch_size=128, shuffle=False, num_workers=4):
            embs.append(enc(x.to(device)).cpu())
            imgs.append((x + 1) / 2)
    emb = torch.cat(embs).numpy()
    imgs01 = torch.cat(imgs)

    rec = retrieval_recall_at_k(emb, morph, ks=(1, 5, 10))
    shiny = shiny_neighbor_ranks(emb, parsed)
    corr = color_vs_embedding_correlation(emb, color_descriptor(imgs01))
    return {
        "run": ckpt.get("run_id", ckpt_path),
        "epoch": ckpt.get("epoch"),
        "morph_r@1": rec[1]["recall"],
        "morph_r@5": rec[5]["recall"],
        "shiny_med_rank": shiny["median_rank"],
        "shiny_r@10": shiny["recall_at_10"],
        "color_corr": corr["pearson"],
        "eff_rank": effective_rank(emb),
    }


def main():
    ckpts = sys.argv[1:] or DEFAULT_CKPTS
    device = get_device()
    rows = [r for r in (evaluate(c, device) for c in ckpts) if r is not None]
    if not rows:
        print("No checkpoints evaluated.")
        return

    hdr = ["run", "epoch", "morph_r@1", "morph_r@5", "shiny_med_rank",
           "shiny_r@10", "color_corr", "eff_rank"]
    print("\n" + " | ".join(f"{h:>14}" for h in hdr))
    print("-" * (len(hdr) * 17))
    for r in rows:
        cells = []
        for h in hdr:
            v = r[h]
            cells.append(f"{v:>14}" if isinstance(v, str) else
                         (f"{v:>14}" if isinstance(v, int) else f"{v:>14.3f}"))
        print(" | ".join(cells))
    print("\nLower shiny_med_rank & color_corr = less color-dominated (better morphology).")
    print("Higher shiny_r@10 = same creature's shiny variant is a near neighbor.")


if __name__ == "__main__":
    main()
