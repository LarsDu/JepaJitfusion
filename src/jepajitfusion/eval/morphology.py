"""Morphology-oriented evaluation for Pokemon embeddings.

These metrics quantify the failure mode motivating the color-augmentation
experiments: LeJEPA embeddings group *different* Pokemon with similar palettes
while pushing *the same* Pokemon's color variants (e.g. shiny) apart. They are
Pokemon-specific because they rely on the ``pokemon_11k`` filename convention to
recover ground-truth identity.

Filename convention (``downloads/pokemon_11k/.../class_0/<name>.png``)::

    <id>[-<forme>]_<game>_<gender>[_s][_fN].png
    e.g. 351-rainy_bw_m_s_f1.png, 4_hs_m_f2.png, 448_bw_m.png

- ``morph_id``  = ``<id>[-<forme>]`` — the *creature* (true morphology identity).
- ``species_id``= ``<id>`` — coarser identity (formes collapsed).
- ``shiny``     = the ``s`` token is present (a pure color variant).

Headline metrics:
- ``retrieval_recall_at_k`` keyed on ``morph_id`` — should go UP (same creature
  retrieved across renders/genders/frames/shiny).
- ``shiny_neighbor_ranks`` — rank of a sprite's shiny twin; should go DOWN.
- ``color_vs_embedding_correlation`` — correlation between embedding distance and
  palette distance; the color-dominance signature, should go DOWN.
- ``effective_rank`` — descriptive covariate (cf. Sub-JEPA).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

from jepajitfusion.encoder.color_transforms import rgb_to_lab

_FRAME_RE = re.compile(r"^f\d+$")


def parse_pokemon_filename(name: str) -> dict:
    """Parse a pokemon_11k sprite filename into identity/attribute fields.

    Returns a dict with keys: morph_id, species_id, shiny, game, gender, frame.
    Robust to missing tokens (some sprites omit gender/frame/shiny).
    """
    stem = Path(name).stem
    parts = stem.split("_")
    morph_id = parts[0]
    species_id = morph_id.split("-")[0]
    rest = parts[1:]
    return {
        "morph_id": morph_id,
        "species_id": species_id,
        # 2-letter game codes never equal the single-char 's', so this is unambiguous.
        "shiny": "s" in rest,
        "game": rest[0] if rest else None,
        "gender": next((p for p in rest if p in ("m", "f")), None),
        "frame": next((p for p in rest if _FRAME_RE.match(p)), None),
    }


def _pairwise_distances(emb: np.ndarray, metric: str) -> np.ndarray:
    """Dense (N, N) distance matrix. metric in {'cosine', 'euclidean'}."""
    x = np.asarray(emb, dtype=np.float64)
    if metric == "cosine":
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        sim = x @ x.T
        return 1.0 - sim
    if metric == "euclidean":
        sq = (x * x).sum(1)
        d2 = sq[:, None] + sq[None, :] - 2 * (x @ x.T)
        return np.sqrt(np.clip(d2, 0, None))
    raise ValueError(f"Unknown metric: {metric!r}")


def retrieval_recall_at_k(
    emb: np.ndarray,
    labels,
    ks=(1, 5, 10),
    metric: str = "cosine",
) -> dict:
    """Identity retrieval quality.

    For each query, neighbors are ranked by distance (self excluded). A query is
    scored only if at least one *other* sample shares its label.

    Returns ``{k: {"recall": any-hit rate, "precision": mean same-label fraction}}``.
    - recall@k: fraction of queries with >=1 same-label neighbor in the top k.
    - precision@k: mean (#same-label in top k) / k.
    """
    labels = np.asarray(labels)
    n = len(labels)
    dist = _pairwise_distances(emb, metric)
    np.fill_diagonal(dist, np.inf)
    order = np.argsort(dist, axis=1)  # nearest first
    neigh_labels = labels[order]  # (N, N-?) same-shape, last col is self (inf)

    same = neigh_labels == labels[:, None]  # (N, N) bool, ranked
    counts = {lab: c for lab, c in zip(*np.unique(labels, return_counts=True))}
    valid = np.array([counts[lab] > 1 for lab in labels])  # has a positive

    out = {}
    for k in ks:
        topk = same[:, :k]
        recall = topk[valid].any(1).mean() if valid.any() else float("nan")
        precision = topk[valid].sum(1).mean() / k if valid.any() else float("nan")
        out[int(k)] = {"recall": float(recall), "precision": float(precision)}
    return out


def effective_rank(emb: np.ndarray) -> float:
    """Effective rank r_eff = exp(-sum p_i log p_i) over normalized cov eigenvalues.

    A descriptive measure of how many dimensions the embedding actually uses
    (Roy & Vetterli 2007; the diagnostic Sub-JEPA correlates with downstream gains).
    """
    x = np.asarray(emb, dtype=np.float64)
    x = x - x.mean(0, keepdims=True)
    cov = (x.T @ x) / max(len(x) - 1, 1)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0, None)
    total = eig.sum()
    if total <= 0:
        return 0.0
    p = eig / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def color_descriptor(
    images: torch.Tensor, bg_thresh: float = 0.95
) -> np.ndarray:
    """Mean CIELAB color of the non-background pixels of each image.

    Args:
        images: (N, 3, H, W) sRGB tensor in [0, 1]. (Pokemon sprites are on a white
            background, which is masked out so the descriptor reflects the creature.)
        bg_thresh: pixels with all channels above this are treated as background.

    Returns:
        (N, 3) array of mean Lab. Falls back to the full-image mean if a sprite is
        entirely background.
    """
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError("images must be (N, 3, H, W) in [0, 1]")
    lab = rgb_to_lab(images)  # (N, 3, H, W)
    fg = ~(images > bg_thresh).all(dim=1)  # (N, H, W) True where creature
    out = np.empty((images.shape[0], 3), dtype=np.float64)
    for i in range(images.shape[0]):
        m = fg[i]
        sel = lab[i][:, m] if m.any() else lab[i].flatten(1)
        out[i] = sel.mean(dim=1).cpu().numpy()
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def color_vs_embedding_correlation(
    emb: np.ndarray,
    color_lab: np.ndarray,
    metric: str = "cosine",
    max_samples: int = 1500,
    seed: int = 0,
) -> dict:
    """Correlation between embedding distance and palette (Lab) distance.

    High correlation = color-dominated embedding (the failure mode). Returns both
    Pearson (on raw pairwise distances) and Spearman (Pearson on their ranks).
    Subsamples to ``max_samples`` rows for tractable O(n^2) pairwise computation.
    """
    rng = np.random.default_rng(seed)
    n = len(emb)
    if n > max_samples:
        idx = rng.choice(n, max_samples, replace=False)
        emb, color_lab = emb[idx], color_lab[idx]
    d_emb = _pairwise_distances(emb, metric)
    d_col = _pairwise_distances(color_lab, "euclidean")
    iu = np.triu_indices(len(d_emb), k=1)
    de, dc = d_emb[iu], d_col[iu]
    pearson = _pearson(de, dc)
    spearman = _pearson(
        np.argsort(np.argsort(de)).astype(float),
        np.argsort(np.argsort(dc)).astype(float),
    )
    return {"pearson": pearson, "spearman": spearman}


def shiny_neighbor_ranks(
    emb: np.ndarray,
    parsed: list[dict],
    metric: str = "cosine",
) -> dict:
    """Rank of each sprite's shiny twin among all other sprites.

    For every (morph_id, game, gender, frame) group that contains both a normal and
    a shiny sprite, query with the normal sprite and find the 1-indexed rank of its
    shiny twin in the distance ranking (self excluded). Lower is better — under a
    color-invariant encoder, the shiny twin should be a near neighbor.

    Returns ``{"ranks": [...], "median_rank": float, "recall_at_10": float,
    "n_pairs": int}``.
    """
    dist = _pairwise_distances(emb, metric)
    np.fill_diagonal(dist, np.inf)
    order = np.argsort(dist, axis=1)
    rank_of = np.argsort(order, axis=1)  # rank_of[q, j] = position of j for query q

    groups: dict[tuple, dict[bool, int]] = {}
    for i, p in enumerate(parsed):
        key = (p["morph_id"], p["game"], p["gender"], p["frame"])
        groups.setdefault(key, {})[p["shiny"]] = i

    ranks = []
    for members in groups.values():
        if True in members and False in members:
            q, twin = members[False], members[True]
            ranks.append(int(rank_of[q, twin]) + 1)  # 1-indexed
    ranks_arr = np.array(ranks, dtype=float)
    return {
        "ranks": ranks,
        "median_rank": float(np.median(ranks_arr)) if len(ranks) else float("nan"),
        "recall_at_10": float((ranks_arr <= 10).mean()) if len(ranks) else float("nan"),
        "n_pairs": len(ranks),
    }
