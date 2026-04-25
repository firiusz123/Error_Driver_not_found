from dataclasses import dataclass


@dataclass
class TrainConfig:
    train_manifest: str = "data/manifests/train.csv"
    val_manifest: str = "data/manifests/val.csv"
    test_manifest: str = "data/manifests/test.csv"
    image_size: int = 160
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    model_name: str = "mobilenet_v2"  # or custom_cnn
    pretrained: bool = True
    dropout: float = 0.2
    device: str = "cuda"
    amp: bool = True
    seed: int = 42
    out_dir: str = "artifacts"
