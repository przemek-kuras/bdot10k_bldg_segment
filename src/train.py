import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.common import load_config, make_model, device, ensure_dir
from src.dataset import TileDataset

def scores_from_logits(logits, y, threshold=0.5):
    p = torch.sigmoid(logits) >= threshold
    t = y >= 0.5
    tp = (p & t).sum().item()
    fp = (p & ~t).sum().item()
    fn = (~p & t).sum().item()
    return tp, fp, fn

def metrics(tp, fp, fn):
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}

def run_epoch(model, loader, optimizer, scaler, criterion, dev, train, amp, threshold):
    model.train(train)
    total_loss, n = 0.0, 0
    tp = fp = fn = 0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(dev), y.to(dev)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=(amp and dev.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * x.size(0); n += x.size(0)
        a, b, c = scores_from_logits(logits.detach(), y, threshold)
        tp += a; fp += b; fn += c
    return total_loss / max(n, 1), metrics(tp, fp, fn)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"]); ensure_dir(out / "checkpoints")
    df = pd.read_csv(out / "splits.csv")
    tr = df[df.split == "train"]
    va = df[df.split == "val"]
    if tr.empty or va.empty:
        raise ValueError("Need non-empty train and validation splits. Increase the study area or adjust the spatial block size / split proportions.")
    tc = cfg.get("training", {})
    dev = device()
    print("Device:", dev)
    train_loader = DataLoader(TileDataset(tr, augment=True), batch_size=int(tc.get("batch_size", 8)),
                              shuffle=True, num_workers=int(tc.get("num_workers", 4)), pin_memory=dev.type=="cuda")
    val_loader = DataLoader(TileDataset(va, augment=False), batch_size=int(tc.get("batch_size", 8)),
                            shuffle=False, num_workers=int(tc.get("num_workers", 4)), pin_memory=dev.type=="cuda")
    model = make_model(cfg).to(dev)
    # Dice + BCE helps with foreground/background imbalance.
    bce = torch.nn.BCEWithLogitsLoss()
    def criterion(logits, y):
        b = bce(logits, y)
        p = torch.sigmoid(logits)
        dims = (1,2,3)
        dice = 1 - ((2*(p*y).sum(dims)+1) / (p.sum(dims)+y.sum(dims)+1)).mean()
        return 0.5*b + 0.5*dice
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tc.get("learning_rate", 3e-4)),
                                  weight_decay=float(tc.get("weight_decay", 1e-4)))
    scaler = torch.amp.GradScaler("cuda", enabled=(bool(tc.get("amp", True)) and dev.type=="cuda"))
    best = -1; patience = int(tc.get("patience", 8)); stale = 0
    threshold = float(tc.get("threshold", 0.5))
    history = []
    for epoch in range(1, int(tc.get("epochs", 40))+1):
        train_loss, train_m = run_epoch(model, train_loader, optimizer, scaler, criterion, dev, True, bool(tc.get("amp", True)), threshold)
        with torch.no_grad():
            val_loss, val_m = run_epoch(model, val_loader, optimizer, scaler, criterion, dev, False, bool(tc.get("amp", True)), threshold)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               **{f"train_{k}":v for k,v in train_m.items()},
               **{f"val_{k}":v for k,v in val_m.items()}}
        history.append(row); print(json.dumps(row))
        if val_m["iou"] > best:
            best = val_m["iou"]; stale = 0
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_metrics": val_m},
                       out / "checkpoints" / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                print("Early stopping.")
                break
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    print("Best validation IoU:", best)

if __name__ == "__main__":
    main()
