import os
import random
from typing import Dict, Iterable, List, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms


_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _list_images(root: str) -> List[str]:
    files: List[str] = []
    for d, _, names in os.walk(root):
        for n in names:
            if n.lower().endswith(_IMG_EXTS):
                files.append(os.path.join(d, n))
    return files


class ImageFolderDataset(Dataset):
    def __init__(self, files: Iterable[str], image_size: int = 64) -> None:
        self.files = list(files)
        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        return self.tf(img)


def build_curvton_difficulty_files(root_dir: str, gender: str = "all") -> Dict[str, List[str]]:
    files_by_diff: Dict[str, List[str]] = {}
    genders: Tuple[str, ...] = ("female", "male") if gender == "all" else (gender,)
    for diff in ("easy", "medium", "hard"):
        collected: List[str] = []
        for g in genders:
            tryon_dir = os.path.join(root_dir, diff, g, "tryon_image")
            if os.path.isdir(tryon_dir):
                collected.extend(_list_images(tryon_dir))
        files_by_diff[diff] = collected
    return files_by_diff


def subset_files(files: List[str], fraction: float, seed: int = 42) -> List[str]:
    if fraction >= 1.0:
        return files
    n = len(files)
    k = max(1, int(n * fraction))
    rng = random.Random(seed)
    return rng.sample(files, k)


def make_loader(
    files: List[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    world_size: int = 1,
    rank: int = 0,
    shuffle: bool = True,
) -> Tuple[DataLoader, DistributedSampler | None]:
    ds = ImageFolderDataset(files, image_size=image_size)
    sampler = (
        DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=True)
        if world_size > 1
        else None
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    return loader, sampler

