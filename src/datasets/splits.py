import csv
import os

from sklearn.model_selection import train_test_split

GRADING_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "ddr", "DR_grading.csv")
SPLITS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "splits")


def load_labels(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["id_code"].strip()
            if name:
                rows.append({"image_name": os.path.splitext(name)[0], "grade": int(row["diagnosis"])})
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "grade"])
        writer.writeheader()
        writer.writerows(rows)


def create_splits(val_fraction: float = 0.15, test_fraction: float = 0.15, random_state: int = 42) -> None:
    all_rows = load_labels(GRADING_CSV)
    labels = [r["grade"] for r in all_rows]

    train_rows, temp_rows = train_test_split(
        all_rows, test_size=val_fraction + test_fraction, stratify=labels, random_state=random_state
    )
    temp_labels = [r["grade"] for r in temp_rows]
    val_rows, test_rows = train_test_split(
        temp_rows, test_size=test_fraction / (val_fraction + test_fraction), stratify=temp_labels, random_state=random_state
    )

    write_csv(os.path.join(SPLITS_DIR, "train.csv"), train_rows)
    write_csv(os.path.join(SPLITS_DIR, "val.csv"), val_rows)
    write_csv(os.path.join(SPLITS_DIR, "test.csv"), test_rows)

    print(f"train: {len(train_rows)}  val: {len(val_rows)}  test: {len(test_rows)}")


if __name__ == "__main__":
    create_splits()
