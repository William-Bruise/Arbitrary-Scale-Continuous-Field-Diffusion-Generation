from dataclasses import dataclass, asdict
from pathlib import Path
import yaml

@dataclass
class TrainConfig:
    dataset: str = "ffhq"
    data_root: str = "./data"
    output_dir: str = "./runs/exp"
    image_size: int = 128
    batch_size: int = 8
    num_workers: int = 4
    steps: int = 20000
    lr: float = 2e-4
    weight_decay: float = 0.0
    seed: int = 42
    mixed_precision: bool = False
    diffusion_steps: int = 1000
    min_points: int = 512
    max_points: int = 4096
    use_irregular_ratio: float = 0.5
    global_latent_dim: int = 256
    hidden_dim: int = 256
    depth: int = 6
    consistency_weight: float = 0.2
    ckpt_every: int = 1000
    sample_every: int = 500


def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    return TrainConfig(**raw)


def save_config(cfg: TrainConfig, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
