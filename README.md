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

The system goes beyond a standard image classifier: it combines retina-specific preprocessing, a lesion-aware architecture that fuses classical computer vision lesion detection (microaneurysms, haemorrhages, exudates) with a transfer-learned CNN, and an ordinal classification approach that reflects the fact that DR stages are ordered rather than unrelated categories. Model decisions are made interpretable through Grad-CAM visualizations, and low-confidence predictions are flagged for specialist referral rather than presented as certain. The trained model is deployed as a live, publicly accessible demo.

---

## Core Features

- **Retina-Specific Preprocessing** — Circular cropping, illumination normalization, and contrast enhancement tailored to fundus photography
- **Data Augmentation & Class Balancing** — Rotation, flipping, zoom, and brightness/contrast adjustments, with class-weighted sampling to address the natural imbalance across DR severity stages
- **Lesion-Aware Hybrid Architecture** — Classical CV lesion segmentation (microaneurysms, haemorrhages, exudates) fused with a transfer-learned CNN backbone, rather than treating classification as a black box
- **Ordinal Stage Classification** — An ordinal loss function reflecting that the five DR stages are ordered by severity, not independent classes
- **Explainability (Grad-CAM)** — Visual heatmaps showing which retinal regions drove each prediction, so results can be sanity-checked against actual lesions
- **Confidence & Uncertainty Flagging** — Low-confidence predictions are flagged for specialist referral instead of being presented as definitive
- **Ablation Study** — Systematic comparison across baseline, ordinal-loss, lesion-fusion, and combined model variants, evaluated on accuracy, precision, recall, F1-score, and confusion matrices
- **Live Public Demo** — An interactive interface deployed on Hugging Face Spaces, where a fundus image can be uploaded to receive a predicted stage, confidence score, and Grad-CAM overlay

---

## Project Structure

```
src/
  preprocessing/     retina-specific image preprocessing
  augmentation/       data augmentation pipeline
  datasets/           dataset loading, splits, class balancing
  models/             CNN backbones, ordinal loss, lesion-fusion architecture
  training/           training loops, experiment configs
  evaluation/         metrics, plots, ablation study
  explainability/     Grad-CAM and confidence/uncertainty estimation
scripts/              dataset download and setup utilities
notebooks/            Colab training notebooks
demo/                 Gradio demo app (deployed to Hugging Face Spaces)
data/                 dataset (not tracked in git — see data/README.md)
outputs/              trained model outputs, plots (not tracked in git — see outputs/README.md)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Built on the [IDRiD](https://idrid.grand-challenge.org/) dataset. See [`data/README.md`](data/README.md) for how to obtain and place it.

## Demo

Live demo: _link added once deployed_

## Report & Video

Full report and video walkthrough submitted separately per course requirements.
