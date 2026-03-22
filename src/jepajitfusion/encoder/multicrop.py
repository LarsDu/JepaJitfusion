"""Multi-crop augmentation for LeJEPA SSL training."""

from torchvision import transforms


class MultiCropAugmentation:
    """Generates multiple augmented crops from a PIL image.

    Produces n_global global crops (full resolution) and n_local local crops
    (smaller resolution) for self-supervised learning. Each crop gets
    random horizontal flip and color jitter.

    Global crops: RandomResizedCrop(global_size, scale=global_scale)
    Local crops:  RandomResizedCrop(local_size, scale=local_scale)
    """

    def __init__(
        self,
        n_global: int = 2,
        n_local: int = 4,
        global_size: int = 64,
        local_size: int = 32,
        global_scale: tuple[float, float] = (0.3, 1.0),
        local_scale: tuple[float, float] = (0.05, 0.3),
    ):
        self.n_global = n_global
        self.n_local = n_local

        common_augs = [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),  # normalize to [-1, 1]
        ]

        self.global_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(global_size, scale=global_scale),
                *common_augs,
            ]
        )
        self.local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(local_size, scale=local_scale),
                *common_augs,
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
