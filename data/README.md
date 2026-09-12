# Data

This folder holds the IDRiD dataset locally. Contents are not tracked in git (licensing + size) — only this README is.

## To populate this folder

Run the download script once it's implemented:

```bash
python scripts/download_dataset.py
```

Expected structure after download:

```
data/
  raw/
    disease_grading/     images + DR severity labels (0-4)
    segmentation/        images + lesion masks (microaneurysms, haemorrhages, exudates)
  processed/              preprocessed/augmented images (generated, not committed)
```
