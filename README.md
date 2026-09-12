# Diabetic Retinopathy Stage Detection

A lesion-aware deep learning pipeline that classifies fundus (retina) photographs by Diabetic Retinopathy severity stage (0 – No DR, through 4 – Proliferative DR), following the ICDR clinical grading scale.

## Overview

Built on the [IDRiD](https://idrid.grand-challenge.org/) dataset, this project combines:
- Retina-specific image preprocessing
- A classical-CV lesion segmentation branch (microaneurysms, haemorrhages, exudates) fused with a CNN classifier
- Transfer learning with an ordinal classification loss, reflecting that DR stages are ordered rather than unrelated categories
- Grad-CAM explainability to visualize which retinal regions drove each prediction
- A live demo deployed on Hugging Face Spaces

## Project structure

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

See [`data/README.md`](data/README.md) for how to obtain and place the IDRiD dataset.

## Demo

Live demo: _link added once deployed_

## Report & video

Full report and video walkthrough submitted separately per course requirements.
