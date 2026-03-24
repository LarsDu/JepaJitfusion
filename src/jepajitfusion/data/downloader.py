"""Data download utilities. Pokemon downloader adapted from DiffuMon."""

import hashlib
import os
import random
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Callable, Sequence

import py7zr
import requests
from PIL import Image
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def download_file(
    url: str,
    output_file: str | Path,
    headers: dict[str, str] | None = None,
    md5sum: str | None = None,
) -> Path:
    """Download a file from a URL. Skips if already exists."""
    output_file = Path(output_file)
    if output_file.exists():
        print(f"Found existing file at {output_file}")
        return output_file

    print(f"Downloading {url} to {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        pbar = tqdm(total=int(r.headers.get("Content-Length", 0)))
        with open(output_file, "wb") as f:
            for data in r.iter_content(chunk_size=1024):
                if data:
                    f.write(data)
                    pbar.update(len(data))
        pbar.close()

    if md5sum is not None:
        with open(output_file, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            if md5 != md5sum:
                raise ValueError(
                    f"MD5 mismatch for {output_file}. Expected {md5sum}, got {md5}"
                )
    return output_file


def unpack_7z(
    archive_file: str | Path,
    output_dir: str | Path,
    delete_archive: bool = False,
) -> None:
    """Unpack a 7z archive."""
    archive_file = Path(archive_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with py7zr.SevenZipFile(archive_file, mode="r") as z:
        z.extractall(output_dir)

    if delete_archive and archive_file.exists():
        os.remove(archive_file)


def unpack_tarball(
    tarball_path: str | Path,
    output_dir: str | Path,
    delete_archive: bool = False,
) -> None:
    """Unpack a tar.gz archive."""
    tarball_path = Path(tarball_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(output_dir)

    if delete_archive and tarball_path.exists():
        os.remove(tarball_path)


def unpack_zip(
    zip_path: str | Path,
    output_dir: str | Path,
    delete_archive: bool = False,
) -> None:
    """Unpack a zip archive."""
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)

    if delete_archive and zip_path.exists():
        os.remove(zip_path)


def convert_to_rgb_with_white_bg(
    input_path: str | Path, output_path: str | Path, output_format: str = "PNG"
) -> None:
    """Convert image with alpha transparency to RGB with white background."""
    image = Image.open(input_path)
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        bg_image = Image.new("RGB", image.size, (255, 255, 255))
        bg_image.paste(image, mask=image.split()[3])
    else:
        bg_image = image.convert("RGB")
    bg_image.save(output_path, format=output_format)


def download_pokemon_11k(
    transform: Callable | None = None,
    data_dir: str | Path = "downloads",
    test_size: float = 0.15,
    split_seed: int = 1999,
) -> tuple[ImageFolder, ImageFolder]:
    """Download the 11,779 Pokemon sprites dataset and split into train/test.

    Adapted from DiffuMon's download_pokemon_sprites_11k.
    """
    url = "https://raw.githubusercontent.com/jonasgrebe/tf-pokemon-generation/master/data/pokemon_sprite_dataset.7z"
    md5sum = "8b620579e0731115e8b30d24998b8c8b"

    output_dir = Path(data_dir) / "pokemon_11k"
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    if train_dir.exists() and test_dir.exists():
        print(f"Found existing train/test dirs in {output_dir}")
        return (
            ImageFolder(str(train_dir), transform=transform),
            ImageFolder(str(test_dir), transform=transform),
        )

    # Download and extract
    staging_dir = output_dir / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    archive_path = staging_dir / "pokemon_sprite_dataset.7z"
    download_file(url, archive_path, md5sum=md5sum)
    unpack_7z(archive_path, staging_dir, delete_archive=True)

    # Convert alpha to white background
    images = list(staging_dir.rglob("*.png"))
    print(f"Converting {len(images)} images to RGB with white background")
    for img_path in images:
        convert_to_rgb_with_white_bg(img_path, img_path)

    # Split into train/test
    (train_dir / "class_0").mkdir(parents=True, exist_ok=True)
    (test_dir / "class_0").mkdir(parents=True, exist_ok=True)

    random.seed(split_seed)
    random.shuffle(images)
    split_idx = int(test_size * len(images))
    test_images = images[:split_idx]
    train_images = images[split_idx:]

    print(f"Split: {len(train_images)} train, {len(test_images)} test")
    for img in train_images:
        if img.is_file():
            shutil.copy(img, train_dir / "class_0" / img.name)
    for img in test_images:
        if img.is_file():
            shutil.copy(img, test_dir / "class_0" / img.name)

    shutil.rmtree(staging_dir)

    return (
        ImageFolder(str(train_dir), transform=transform),
        ImageFolder(str(test_dir), transform=transform),
    )


def download_tiny_imagenet(
    transform: Callable | None = None,
    data_dir: str | Path = "downloads",
    **kwargs,
) -> tuple[ImageFolder, ImageFolder]:
    """Download Tiny-ImageNet-200 (64x64, 200 classes, 100k images)."""
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    output_dir = Path(data_dir) / "imagenet_tiny"

    train_dir = output_dir / "tiny-imagenet-200" / "train"
    val_dir = output_dir / "tiny-imagenet-200" / "val"

    if train_dir.exists() and val_dir.exists():
        # Reorganize val if needed (flat structure → class folders)
        _reorganize_tiny_imagenet_val(val_dir)
        return (
            ImageFolder(str(train_dir), transform=transform),
            ImageFolder(str(val_dir), transform=transform),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "tiny-imagenet-200.zip"
    download_file(url, archive_path)
    unpack_zip(archive_path, output_dir, delete_archive=True)

    _reorganize_tiny_imagenet_val(val_dir)

    return (
        ImageFolder(str(train_dir), transform=transform),
        ImageFolder(str(val_dir), transform=transform),
    )


def _reorganize_tiny_imagenet_val(val_dir: Path) -> None:
    """Reorganize Tiny-ImageNet val set from flat to class-folder structure."""
    annotations_file = val_dir / "val_annotations.txt"
    images_dir = val_dir / "images"

    if not annotations_file.exists() or not images_dir.exists():
        return  # Already reorganized or missing

    # Parse annotations: filename → class_id
    with open(annotations_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            fname, class_id = parts[0], parts[1]
            class_dir = val_dir / class_id / "images"
            class_dir.mkdir(parents=True, exist_ok=True)
            src = images_dir / fname
            if src.exists():
                shutil.move(str(src), str(class_dir / fname))

    # Clean up
    if images_dir.exists():
        shutil.rmtree(images_dir)
    if annotations_file.exists():
        annotations_file.unlink()


def download_imagenette(
    transform: Callable | None = None,
    data_dir: str | Path = "downloads",
    test_size: float = 0.15,
    **kwargs,
) -> tuple[ImageFolder, ImageFolder]:
    """Download ImageNette (10 classes, ~13k images, 160px version)."""
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    output_dir = Path(data_dir) / "imagenette"

    train_dir = output_dir / "imagenette2-160" / "train"
    val_dir = output_dir / "imagenette2-160" / "val"

    if train_dir.exists() and val_dir.exists():
        return (
            ImageFolder(str(train_dir), transform=transform),
            ImageFolder(str(val_dir), transform=transform),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "imagenette2-160.tgz"
    download_file(url, archive_path)
    unpack_tarball(archive_path, output_dir, delete_archive=True)

    return (
        ImageFolder(str(train_dir), transform=transform),
        ImageFolder(str(val_dir), transform=transform),
    )
