"""Multi-crop augmentation for LeJEPA SSL training."""

from torchvision import transforms

from jepajitfusion.encoder.color_transforms import build_color_augmentation


def _to_signed_unit_range(t):
    """Map a [0, 1] tensor to [-1, 1].

    Defined at module level (not a lambda) so MultiCropAugmentation transforms are
    picklable and can be used with DataLoader multiprocessing workers.
    """
    return (t * 2) - 1


class MultiCropAugmentation:
    """Generates multiple augmented crops from a PIL image.

    Produces n_global global crops (full resolution) and n_local local crops
    (smaller resolution) for self-supervised learning. Each crop gets
    random horizontal flip and a configurable color augmentation.

    Global crops: RandomResizedCrop(global_size, scale=global_scale)
    Local crops:  RandomResizedCrop(local_size, scale=local_scale)

    The color augmentation is selected by ``color_aug`` (see
    ``encoder.color_transforms.build_color_augmentation``). The morphology
    experiments use relational-preserving modes ("colorjitter"/"lab") that
    randomize the absolute palette while keeping relative color geometry, so the
    encoder learns shape/layout over absolute hue.
    """

    def __init__(
        self,
        n_global: int = 2,
        n_local: int = 4,
        global_size: int = 64,
        local_size: int = 32,
        global_scale: tuple[float, float] = (0.3, 1.0),
        local_scale: tuple[float, float] = (0.05, 0.3),
        color_aug: str = "default",
        cj_brightness: float = 0.4,
        cj_contrast: float = 0.4,
        cj_saturation: float = 0.4,
        cj_hue: float = 0.125,
        lab_hue_deg: float = 45.0,
        lab_lightness: float = 0.1,
        lab_chroma: float = 0.3,
    ):
        self.n_global = n_global
        self.n_local = n_local

        pil_color, tensor_color = build_color_augmentation(
            color_aug,
            brightness=cj_brightness,
            contrast=cj_contrast,
            saturation=cj_saturation,
            hue=cj_hue,
            lab_hue_deg=lab_hue_deg,
            lab_lightness=lab_lightness,
            lab_chroma=lab_chroma,
        )

        # PIL stage: flip + (optional PIL color jitter).
        # Tensor stage: ToTensor -> (optional tensor color transform) -> [-1, 1].
        common_pil = [transforms.RandomHorizontalFlip(), *pil_color]
        common_tensor = [
            transforms.ToTensor(),
            *tensor_color,
            transforms.Lambda(_to_signed_unit_range),  # normalize to [-1, 1]
        ]

        self.global_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(global_size, scale=global_scale),
                *common_pil,
                *common_tensor,
            ]
        )
        self.local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(local_size, scale=local_scale),
                *common_pil,
                *common_tensor,
            ]
        )

    def __call__(self, img) -> list:
        """Apply multi-crop augmentation.

        Args:
            img: PIL Image.

        Returns:
            List of tensors: [global_1, global_2, ..., local_1, local_2, ...]
        """
        crops = []
        for _ in range(self.n_global):
            crops.append(self.global_transform(img))
        for _ in range(self.n_local):
            crops.append(self.local_transform(img))
        return crops
