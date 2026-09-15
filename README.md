# Diabetic Retinopathy Stage Detection

| | |
|---|---|
| **Batch** | BSc Hons in Computing 2024.2P |
| **Module** | Computer Vision |
| **Coursework** | Individual |
| **Student Name** | B N D U Fernando |
| **NIBM Student Index** | COBSCCOMP24.2P-035 |
| **Coventry Student Index** | 16110658 |

---

## Introduction

Diabetic Retinopathy (DR) is a complication of diabetes that damages the blood vessels in the retina and is a leading cause of preventable blindness worldwide. Early detection of its severity stage is critical, since treatment options and urgency differ significantly between mild and advanced disease. This project builds an image-based system that classifies a fundus (retina) photograph into one of five clinically recognized DR severity stages — No DR, Mild, Moderate, Severe, and Proliferative — following the International Clinical Diabetic Retinopathy (ICDR) grading scale.

Beyond a standard image classifier, this project runs a genuine ablation study: a transfer-learned CNN baseline, a CORAL-based ordinal loss variant (since DR stages are ordered, not unrelated categories), and a variant fusing classical computer-vision lesion detection (microaneurysms, haemorrhages, exudates) into the CNN. The baseline came out strongest overall — the lesion-fusion variant is a genuine negative result, diagnosed rather than hidden — so the baseline is what's deployed in the live demo. Model decisions are made interpretable through Grad-CAM visualizations, and low-confidence predictions are flagged for specialist referral rather than presented as certain.

---

## Core Features

- **Retina-Specific Preprocessing** — Circular cropping, illumination normalization, and contrast enhancement tailored to fundus photography
- **Data Augmentation & Class Balancing** — Rotation, flipping, zoom, and brightness/contrast adjustments, with class-weighted sampling to address the natural imbalance across DR severity stages
- **Lesion-Aware Hybrid Architecture** — Classical CV lesion segmentation (microaneurysms, haemorrhages, exudates) fused with a transfer-learned CNN backbone, rather than treating classification as a black box
- **Ordinal Stage Classification** — A CORAL-based ordinal loss reflecting that the five DR stages are ordered by severity, not independent classes
- **Explainability (Grad-CAM)** — Visual heatmaps showing which retinal regions drove each prediction, so results can be sanity-checked against actual lesions
- **Low-Confidence Referral Flagging** — Predictions below a confidence threshold are flagged for specialist review instead of being presented as definitive
- **Ablation Study** — Baseline, ordinal-loss, and lesion-fusion variants trained and compared on accuracy, precision, recall, F1-score, and confusion matrices — including an honest negative result (lesion fusion underperformed the baseline) with a diagnosed cause, not just a leaderboard
- **Live Public Demo** — An interactive Streamlit app, deployed publicly, where a fundus image can be uploaded to receive predicted stage probabilities, a confidence score, and a Grad-CAM overlay

---

## Results

Trained and evaluated on the DDR dataset (12,522 fundus images, stratified 70/15/15 split), held-out test set of 1,879 images:

| Variant | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| A — Baseline CNN (EfficientNet-B0) | **86.2%** | **0.669** | Strongest overall; used in the deployed demo |
| B — + Ordinal loss (CORAL) | 84.9% | 0.665 | Near-tie overall, but notably better Stage 3 (Severe) recall (0.54 vs 0.31) |
| C — + Classical lesion-fusion | 84.5% | 0.635 | Genuine negative result — the classical lesion detector's own accuracy against ground truth was weak (Dice 0.097), and the added channel appears to have introduced noise rather than signal |

A combined (B+C) variant was deliberately not trained, since C's fusion channel was shown to hurt rather than help — full reasoning in the project's development log.

---

## Project Structure

```
src/
  preprocessing/     retina-specific image preprocessing
  augmentation/       data augmentation pipeline
  datasets/           dataset loading, splits, class balancing
  models/             CNN backbones, ordinal loss (CORAL), lesion-fusion architecture
  lesion/             classical (non-learned) lesion candidate detection + evaluation
  training/           training loop, early stopping, checkpointing
  explainability/     Grad-CAM
scripts/              dataset download, preprocessing/lesion previews, local training reference
notebooks/            Kaggle training notebooks (Variants A/B/C)
demo/                 predictor.py (shared model/inference logic), app.py (Gradio, local testing),
                       streamlit_app.py (Streamlit, deployed live demo)
data/                 dataset (not tracked in git — see data/README.md)
outputs/              trained checkpoints, evaluation history, Grad-CAM outputs (mostly not tracked — see outputs/README.md)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Built on the [DDR](https://github.com/nkicsl/DDR-dataset) (Diabetic Retinopathy Dataset), 12,522 labeled fundus images including 757 with pixel-level lesion segmentation masks. See [`data/README.md`](data/README.md) for how to obtain and place it.

## Demo

Live demo: [computer-vision-assignment-d34cmkaxxs5nj9pshsztom.streamlit.app](https://computer-vision-assignment-d34cmkaxxs5nj9pshsztom.streamlit.app)

Upload a fundus photo (or pick one of the 5 built-in examples, one per stage) to get predicted stage probabilities, a confidence score, a low-confidence referral flag, and a Grad-CAM heatmap. Runs the baseline (Variant A) model — see [Results](#results) above for why.

## Report & Video

Full report and video walkthrough submitted separately per course requirements.
