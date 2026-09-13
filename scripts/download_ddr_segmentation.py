import os
import subprocess

DEST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ddr_segmentation")


def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "sunfish141/ddr-segmentation", "-p", DEST_DIR, "--unzip"],
        check=True,
    )
    print(f"Done. Dataset is in {DEST_DIR}")


if __name__ == "__main__":
    main()
