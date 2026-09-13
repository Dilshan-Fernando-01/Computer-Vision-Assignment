import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.augmentation.augment import compute_class_weights, get_training_augmentations
from src.datasets.dataset import IDRiDGradingDataset, IMAGES_DIR_TEST, IMAGES_DIR_TRAIN
from src.models.build_model import build_model
from src.training.train import evaluate, fit, get_device

IMAGE_SIZE = 512
BATCH_SIZE = 16
EPOCHS = 40
PATIENCE = 8
LR = 1e-4

SPLITS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "splits")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def main():
    device = get_device()
    print("device:", device)

    train_ds = IDRiDGradingDataset(
        os.path.join(SPLITS_DIR, "train.csv"), IMAGES_DIR_TRAIN, image_size=IMAGE_SIZE, transform=get_training_augmentations(IMAGE_SIZE)
    )
    val_ds = IDRiDGradingDataset(os.path.join(SPLITS_DIR, "val.csv"), IMAGES_DIR_TRAIN, image_size=IMAGE_SIZE)
    test_ds = IDRiDGradingDataset(os.path.join(SPLITS_DIR, "test.csv"), IMAGES_DIR_TEST, image_size=IMAGE_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    train_labels = [label for _, label in train_ds.samples]
    class_weights = compute_class_weights(train_labels)
    print("class weights:", class_weights)

    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    model = build_model("efficientnet_b0", num_classes=5, pretrained=True)

    checkpoint_path = os.path.join(OUTPUTS_DIR, "checkpoints", "variant_a_efficientnet_b0.pt")
    history = fit(
        model, train_loader, val_loader, criterion,
        epochs=EPOCHS, lr=LR, patience=PATIENCE,
        checkpoint_path=checkpoint_path, device=device,
    )

    history_dir = os.path.join(OUTPUTS_DIR, "history")
    os.makedirs(history_dir, exist_ok=True)
    with open(os.path.join(history_dir, "variant_a_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)

    report = classification_report(
        test_metrics["labels"], test_metrics["preds"],
        target_names=[f"Stage {i}" for i in range(5)], zero_division=0,
    )
    cm = confusion_matrix(test_metrics["labels"], test_metrics["preds"])

    print("\n=== Test set results (Variant A) ===")
    print("Test accuracy:", test_metrics["accuracy"])
    print("Test macro F1:", test_metrics["macro_f1"])
    print(report)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(np.array(cm))

    with open(os.path.join(history_dir, "variant_a_test_report.txt"), "w") as f:
        f.write(f"Test accuracy: {test_metrics['accuracy']}\n")
        f.write(f"Test macro F1: {test_metrics['macro_f1']}\n\n")
        f.write(report)
        f.write("\nConfusion matrix (rows=true, cols=predicted):\n")
        f.write(np.array2string(cm))


if __name__ == "__main__":
    main()
