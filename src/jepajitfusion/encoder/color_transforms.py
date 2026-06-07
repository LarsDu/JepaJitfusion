"""Color augmentations for LeJEPA SSL, with a focus on *morphology*.

Standard SSL color jitter leaves a sprite's palette largely intact, so the
invariance objective keeps absolute color as an easy discriminative cue and the
embedding space becomes color-dominated (different Pokemon with similar palettes
end up as nearest neighbors; shiny variants of the *same* Pokemon end up far
apart).

This module provides two opt-in color-augmentation modes that randomize the
*absolute* palette while preserving the *relative* color geometry of an image, so
the encoder is pushed to encode internal color layout + shape rather than absolute
hue:

- ``"colorjitter"``: torchvision ``ColorJitter`` with a wide hue range and **no**
  grayscale/solarize. Its brightness/contrast/saturation are spatially-uniform
  linear maps and its hue is an approximate gray-axis rotation, so relative color
  structure is approximately preserved.
- ``"lab"``: a custom, more principled transform that operates in CIELAB. A random
  rotation of the (a, b) chroma plane is an exact isometry in Lab (pairwise color
  distances are preserved), an additive L shift is a translation (also exact), and
  a bounded chroma scale preserves relative structure up to a global factor. This
  randomizes the palette (incl. shiny-like hue shifts) while keeping the relational
  color geometry intact.

``"default"`` reproduces the original ``ColorJitter(0.4, 0.4, 0.2, 0.1)`` baseline.

The factory returns ``(pil_stage, tensor_stage)``: a list of transforms that run on
the PIL image and a list that run on the ``[0, 1]`` CHW tensor (after ``ToTensor``),
so the caller can slot each into the right place in its augmentation pipeline.
"""

from __future__ import annotations

import torch
from torchvision import transforms

# --- sRGB <-> CIELAB (D65), pure torch, channels-first (..., 3, H, W) ----------

# Linear-sRGB -> XYZ (D65) and its inverse.
_RGB2XYZ = torch.tensor(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_XYZ2RGB = torch.linalg.inv(_RGB2XYZ)
# D65 reference white.
_WHITE = torch.tensor([0.95047, 1.00000, 1.08883])
_DELTA = 6.0 / 29.0


def _matmul_channels(mat: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    """Apply a 3x3 color matrix along the channel dim (-3) of a CHW/BCHW tensor."""
    mat = mat.to(img.dtype).to(img.device)
    return torch.einsum("ij,...jhw->...ihw", mat, img)


def _srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    return torch.where(c <= 0.04045, c / 12.92, ((c.clamp(min=0) + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: torch.Tensor) -> torch.Tensor:
    c = c.clamp(min=0.0)
    return torch.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Convert an sRGB tensor in [0, 1] (..., 3, H, W) to CIELAB.

    L in [0, 100], a/b roughly in [-128, 127].
    """
    xyz = _matmul_channels(_RGB2XYZ, _srgb_to_linear(rgb))
    white = _WHITE.to(rgb.dtype).to(rgb.device).view(3, 1, 1)
    t = xyz / white
    f = torch.where(t > _DELTA**3, t.clamp(min=0) ** (1 / 3), t / (3 * _DELTA**2) + 4 / 29)
    fx, fy, fz = f[..., 0, :, :], f[..., 1, :, :], f[..., 2, :, :]
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return torch.stack([L, a, b], dim=-3)


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """Convert CIELAB (..., 3, H, W) back to sRGB in [0, 1] (gamut-clamped)."""
    L, a, b = lab[..., 0, :, :], lab[..., 1, :, :], lab[..., 2, :, :]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    f = torch.stack([fx, fy, fz], dim=-3)
    t = torch.where(f > _DELTA, f**3, 3 * _DELTA**2 * (f - 4 / 29))
    white = _WHITE.to(lab.dtype).to(lab.device).view(3, 1, 1)
    xyz = t * white
    rgb = _linear_to_srgb(_matmul_channels(_XYZ2RGB, xyz))
    return rgb.clamp(0.0, 1.0)


class LabColorTransform:
    """Relational-preserving color augmentation in CIELAB space.

    Applies, per call, a spatially-uniform transform of the color coordinates:
      - hue: random rotation of the (a, b) chroma plane (exact Lab isometry),
      - lightness: additive shift of L (translation; exact),
      - chroma: bounded isotropic scale of (a, b) (preserves relative structure
        up to a global factor; kept away from 0 so we never approach grayscale).

    Operates on a CHW float tensor in [0, 1]; returns the same. Picklable
    (module-level class, no closures) for DataLoader workers.

    Args:
        hue_deg: chroma-plane rotation sampled from U(-hue_deg, +hue_deg) degrees.
            This is the absolute-palette knob: small (~45) keeps some absolute-hue
            (type) signal; 180 is fully hue-invariant.
        lightness: L shift sampled from U(-lightness, +lightness) * 100.
        chroma: chroma scale sampled from U(1 - chroma, 1 + chroma).
    """

    def __init__(self, hue_deg: float = 45.0, lightness: float = 0.1, chroma: float = 0.3):
        self.hue_deg = hue_deg
        self.lightness = lightness
        self.chroma = chroma

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        lab = rgb_to_lab(img)
        L, a, b = lab[0], lab[1], lab[2]

        theta = (torch.rand(()) * 2 - 1) * self.hue_deg * torch.pi / 180.0
        cos, sin = torch.cos(theta), torch.sin(theta)
        scale = 1.0 + (torch.rand(()) * 2 - 1) * self.chroma
        a_rot = (a * cos - b * sin) * scale
        b_rot = (a * sin + b * cos) * scale

        dL = (torch.rand(()) * 2 - 1) * self.lightness * 100.0
        L = (L + dL).clamp(0.0, 100.0)

        return lab_to_rgb(torch.stack([L, a_rot, b_rot], dim=0))


def build_color_augmentation(
    mode: str = "default",
    # colorjitter / default params
    brightness: float = 0.4,
    contrast: float = 0.4,
    saturation: float = 0.4,
    hue: float = 0.125,
    # lab params
    lab_hue_deg: float = 45.0,
    lab_lightness: float = 0.1,
    lab_chroma: float = 0.3,
) -> tuple[list, list]:
    """Build a color-augmentation stage.

    Returns ``(pil_stage, tensor_stage)`` lists of transforms. ``pil_stage`` runs
    on the PIL image (before ``ToTensor``); ``tensor_stage`` runs on the [0, 1] CHW
    tensor (after ``ToTensor``). Exactly one is non-empty for the current modes.

    Modes:
        "default":     ColorJitter(0.4, 0.4, 0.2, 0.1) — original baseline.
        "colorjitter": ColorJitter(brightness, contrast, saturation, hue), wide hue,
                       no grayscale. Relational-preserving (approx).
        "lab":         LabColorTransform — exact Lab-space relational preservation.
    """
    if mode == "default":
        return [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], []
    if mode == "colorjitter":
        return [transforms.ColorJitter(brightness, contrast, saturation, hue)], []
    if mode == "lab":
        return [], [LabColorTransform(lab_hue_deg, lab_lightness, lab_chroma)]
    raise ValueError(f"Unknown color_aug mode: {mode!r} (default|colorjitter|lab)")
