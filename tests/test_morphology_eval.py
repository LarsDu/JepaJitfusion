"""Tests for the Pokemon morphology evaluation metrics."""

import numpy as np
import torch

from jepajitfusion.eval.morphology import (
    color_descriptor,
    color_vs_embedding_correlation,
    effective_rank,
    parse_pokemon_filename,
    retrieval_recall_at_k,
    shiny_neighbor_ranks,
)


def test_parse_filename_variants():
    p = parse_pokemon_filename("227_pt_m_s_f1.png")
    assert p == {
        "morph_id": "227",
        "species_id": "227",
        "shiny": True,
        "game": "pt",
        "gender": "m",
        "frame": "f1",
    }
    p2 = parse_pokemon_filename("351-rainy_bw_f_f2.png")
    assert p2["morph_id"] == "351-rainy"
    assert p2["species_id"] == "351"
    assert p2["shiny"] is False
    assert p2["gender"] == "f" and p2["frame"] == "f2"
    p3 = parse_pokemon_filename("448_bw_m.png")
    assert p3["shiny"] is False and p3["frame"] is None and p3["gender"] == "m"


def test_effective_rank_bounds():
    rng = np.random.default_rng(0)
    iso = rng.standard_normal((2000, 16))
    r_iso = effective_rank(iso)
    assert 12 < r_iso <= 16  # near-isotropic uses most dims

    rank1 = np.outer(rng.standard_normal(2000), rng.standard_normal(16))
    assert effective_rank(rank1) < 1.5  # collapsed to ~1 direction


def test_retrieval_recall_perfect_and_random():
    # Two tight clusters with the same label -> perfect retrieval.
    a = np.array([[10.0, 0.0]]) + 0.01 * np.random.randn(10, 2)
    b = np.array([[-10.0, 0.0]]) + 0.01 * np.random.randn(10, 2)
    emb = np.vstack([a, b])
    labels = np.array([0] * 10 + [1] * 10)
    out = retrieval_recall_at_k(emb, labels, ks=(1, 5), metric="euclidean")
    assert out[1]["recall"] == 1.0
    assert out[5]["precision"] == 1.0


def test_retrieval_ignores_singleton_labels():
    emb = np.random.randn(6, 4)
    labels = np.array([0, 0, 1, 2, 3, 4])  # only label 0 has a positive pair
    out = retrieval_recall_at_k(emb, labels, ks=(1,), metric="cosine")
    assert not np.isnan(out[1]["recall"])  # computed over the valid query/queries


def test_color_descriptor_masks_background():
    # Red square on white background; descriptor should reflect red, not white.
    img = torch.ones(1, 3, 8, 8)
    img[0, :, 2:6, 2:6] = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
    lab = color_descriptor(img, bg_thresh=0.95)
    # Red has strongly positive a*; white would have a* ~ 0.
    assert lab[0, 1] > 20.0


def test_color_correlation_detects_color_structure():
    # Embedding == color -> high correlation; embedding random -> low.
    rng = np.random.default_rng(0)
    color = rng.standard_normal((300, 3)) * 30
    corr_same = color_vs_embedding_correlation(color.copy(), color, metric="euclidean")
    assert corr_same["pearson"] > 0.95
    rand_emb = rng.standard_normal((300, 16))
    corr_rand = color_vs_embedding_correlation(rand_emb, color, metric="euclidean")
    assert abs(corr_rand["pearson"]) < 0.3


def test_shiny_neighbor_ranks():
    # Build 3 creatures, each with a normal+shiny twin placed close together.
    parsed, rows = [], []
    for cid in range(3):
        center = np.zeros(4)
        center[cid] = 10.0
        for shiny in (False, True):
            parsed.append(
                {
                    "morph_id": str(cid),
                    "game": "bw",
                    "gender": "m",
                    "frame": "f1",
                    "shiny": shiny,
                }
            )
            rows.append(center + 0.01 * (1 if shiny else 0))
    emb = np.array(rows)
    out = shiny_neighbor_ranks(emb, parsed, metric="euclidean")
    assert out["n_pairs"] == 3
    assert out["median_rank"] == 1.0  # the twin is the nearest neighbor
    assert out["recall_at_10"] == 1.0
