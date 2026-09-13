from __future__ import annotations

import os

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader: DataLoader, optimizer, criterion, device) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader: DataLoader, criterion, device) -> dict:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "macro_f1": macro_f1,
        "preds": all_preds,
        "labels": all_labels,
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    epochs: int = 30,
    lr: float = 1e-4,
    patience: int = 7,
    checkpoint_path: str = "outputs/checkpoints/model.pt",
    device: torch.device | None = None,
) -> dict:
    device = device or get_device()
    model.to(device)
    criterion.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}
    best_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])

        scheduler.step(val_metrics["macro_f1"])

        print(
            f"Epoch {epoch:3d} | train_loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val_loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f} macro_f1 {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no val macro_f1 improvement for {patience} epochs)")
                break

    history["best_val_macro_f1"] = best_f1
    return history
