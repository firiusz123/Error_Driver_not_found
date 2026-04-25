import csv
from dataclasses import dataclass
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class Sample:
    image_path: str
    label: int


def read_manifest(path: str) -> List[Sample]:
    samples: List[Sample] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(Sample(image_path=row["image_path"], label=int(row["label"])))
    return samples


class DrowsinessDataset(Dataset):
    def __init__(self, manifest_path: str, image_size: int = 160, train: bool = False):
        self.samples = read_manifest(manifest_path)
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(8),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(sample.label, dtype=torch.long)
        return image, label
