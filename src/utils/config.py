import os
import sys
import tomllib
import torch
from pathlib import Path

PROJECT_ROOT = Path("/Users/cretuluca/uni/ro-doc-classification")


def load_config(config_path="config.toml"):
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
