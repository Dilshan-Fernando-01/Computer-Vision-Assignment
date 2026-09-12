import csv
import os

from sklearn.model_selection import train_test_split

GRADING_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "disease_grading", "B. Disease Grading")
TRAIN_CSV = os.path.join(GRADING_DIR, "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv")
TEST_CSV = os.path.join(GRADING_DIR, "2. Groundtruths", "b. IDRiD_Disease Grading_Testing Labels.csv")

SPLITS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "splits")


def load_labels(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Image name"].strip()
            if name:
                rows.append({"image_name": name, "grade": int(row["Retinopathy grade"])})
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "grade"])
        writer.writeheader()
        writer.writerows(rows)


def create_splits(val_fraction: float = 0.15, random_state: int = 42) -> None:
    all_train_rows = load_labels(TRAIN_CSV)
    test_rows = load_labels(TEST_CSV)

    labels = [r["grade"] for r in all_train_rows]
    train_rows, val_rows = train_test_split(
        all_train_rows,
        test_size=val_fraction,
        stratify=labels,
        random_state=random_state,
    )

    write_csv(os.path.join(SPLITS_DIR, "train.csv"), train_rows)
    write_csv(os.path.join(SPLITS_DIR, "val.csv"), val_rows)
    write_csv(os.path.join(SPLITS_DIR, "test.csv"), test_rows)

    print(f"train: {len(train_rows)}  val: {len(val_rows)}  test: {len(test_rows)}")


if __name__ == "__main__":
    create_splits()
