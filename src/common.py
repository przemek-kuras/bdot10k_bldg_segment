from pathlib import Path
import yaml
import numpy as np
import torch
import segmentation_models_pytorch as smp

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def make_model(cfg):
    tr = cfg.get("training", {})
    return smp.Unet(
        encoder_name=tr.get("encoder_name", "resnet34"),
        encoder_weights=tr.get("encoder_weights", "imagenet"),
        in_channels=len(cfg.get("bands", [1, 2, 3])),
        classes=1,
        activation=None,
    )

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_image(arr):
    """arr: H,W,C uint8 or numeric. Robust per-band percentile normalization."""
    arr = np.asarray(arr, dtype=np.float32)
    out = np.zeros_like(arr, dtype=np.float32)
    for c in range(arr.shape[2]):
        band = arr[:, :, c]
        valid = np.isfinite(band)
        if not valid.any():
            continue
        lo, hi = np.percentile(band[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        out[:, :, c] = np.clip((band - lo) / (hi - lo), 0, 1)
        out[~valid, c] = 0
    return out

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
