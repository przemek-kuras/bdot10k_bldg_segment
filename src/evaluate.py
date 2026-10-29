import argparse, json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.common import load_config, make_model, device
from src.dataset import TileDataset
from src.train import scores_from_logits, metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])
    df = pd.read_csv(out / "splits.csv")
    test = df[df.split == "test"]
    if test.empty:
        raise ValueError("No test split. Provide at least 3 independent geographic raster areas and rerun preparation, or implement spatial-block splitting.")
    dev = device()
    model = make_model(cfg).to(dev)
    ckpt = torch.load(out / "checkpoints" / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model"]); model.eval()
    loader = DataLoader(TileDataset(test), batch_size=int(cfg.get("training", {}).get("batch_size", 8)),
                        shuffle=False, num_workers=int(cfg.get("training", {}).get("num_workers", 4)))
    tp = fp = fn = 0
    threshold = float(cfg.get("training", {}).get("threshold", 0.5))
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(dev))
            a,b,c = scores_from_logits(logits.cpu(), y, threshold)
            tp += a; fp += b; fn += c
    result = metrics(tp, fp, fn)
    result.update({"test_tiles": int(len(test)), "threshold": threshold,
                   "note": "Pixel-level metrics against BDOT10k-derived masks; not necessarily accuracy against true visible roof edges."})
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
