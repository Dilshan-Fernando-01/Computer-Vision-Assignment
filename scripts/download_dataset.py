import os
import urllib.request
import zipfile

FILES = {
    "disease_grading": (
        "https://zenodo.org/records/17219542/files/B.%20Disease%20Grading.zip?download=1",
        "B. Disease Grading.zip",
    ),
    "segmentation": (
        "https://zenodo.org/records/17219542/files/A.%20Segmentation.zip?download=1",
        "A. Segmentation.zip",
    ),
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def download_and_extract(name: str, url: str, filename: str) -> None:
    target_dir = os.path.join(DATA_DIR, name)
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, filename)

    if not os.path.exists(zip_path):
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, zip_path)
    else:
        print(f"{filename} already downloaded, skipping download.")

    print(f"Extracting {filename} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    os.remove(zip_path)
    print(f"Done: {name} -> {target_dir}")


if __name__ == "__main__":
    for name, (url, filename) in FILES.items():
        download_and_extract(name, url, filename)
    print("\nAll done. Dataset is in data/raw/")
