# Conformer for Aseptic Hip Implant Loosening Detection

Detecting **aseptic loosening of total hip replacement (THR) implants** from plain radiographs with a **parallel CNN–Transformer hybrid (Conformer)**, MURA-based transfer learning, and a full explainable-AI (XAI) toolkit.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?logo=jupyter&logoColor=white)

---

## Overview

Aseptic loosening is one of the leading causes of revision surgery after hip arthroplasty, and it is normally assessed by a radiologist reading follow-up X-rays — a process that can be slow and subject to inter-observer variability. This repository explores whether a single image model can classify a hip implant X-ray as **`loose`** or **`control`** (not loose), and just as importantly, whether its decisions can be *explained* in a way a clinician could inspect.

The model is the **Conformer** — an architecture that runs a convolutional branch and a Vision Transformer branch **in parallel**, continuously exchanging features between them so that local texture (CNN) and global context (self-attention) are fused throughout the network. Because the loosening dataset is small, the project leans heavily on **transfer learning**: the network is first pre-trained on a large musculoskeletal X-ray dataset (MURA), then fine-tuned on the loosening task. A from-scratch baseline is included so the benefit of that pre-training can be measured directly.

### What's inside

- **A reproducible two-stage transfer-learning pipeline** (pre-train on MURA → fine-tune on loose hips).
- **A from-scratch baseline** for an apples-to-apples comparison.
- **Dual-branch ensemble inference** — predictions average the CNN and Transformer heads.
- **A rigorous evaluation suite** — confusion matrix, per-class report, ROC/AUC, and bootstrapped confidence intervals.
- **Three complementary explainability methods** — Grad-CAM (CNN branch), Attention Rollout (Transformer branch), and Integrated Gradients (whole model).
- **A ready-to-run GPU environment** via Docker Compose.

> ⚠️ **Research / educational use only.** This code is a model-development experiment, not a medical device. It must not be used for diagnosis or any clinical decision-making.

---

## Approach at a glance

```
                Stage 1: Pre-training                Stage 2: Fine-tuning
        ┌──────────────────────────────┐     ┌──────────────────────────────────┐
        │  MURA (musculoskeletal X-ray) │     │  Aseptic Loose Hip Implant X-Ray  │
        │  7-class body-part task       │ ──▶ │  binary task: control vs loose    │
        │  → best_conformer_mura.pth    │     │  → best_conformer_..._pretrained  │
        └──────────────────────────────┘     └──────────────────────────────────┘
                                                              ▲
                                                              │  compared against
                                              ┌──────────────────────────────────┐
                                              │  Baseline: same task, random init │
                                              │  → best_conformer_scratch.pth     │
                                              └──────────────────────────────────┘
```

The Conformer is loaded as `Conformer_small_patch16` (224×224 input). Both its classification heads — the convolutional head (`conv_cls_head`) and the transformer head (`trans_cls_head`) — are retrained for the target task, and inference **ensembles the two branches** by averaging their softmax outputs.

---

## Repository structure

```
conformer-implant-loosening/
├── notebooks/
│   ├── parallel-vit-cnn-hybrid-mura.ipynb                       # Stage 1: pre-train on MURA
│   ├── parallel-vit-cnn-hybrid-loose-hips-mura-pretrained.ipynb # Stage 2: fine-tune (MURA-initialised)
│   ├── parallel-vit-cnn-hybrid-loose-hips-scratch.ipynb         # Baseline: train from scratch
│   └── Conformer/                                               # Vendored Conformer source (see Credits)
│       ├── models.py            # exposes Conformer_small_patch16
│       ├── conformer.py
│       └── ...
├── Dockerfile                   # GPU + Jupyter image (PyTorch on a TF base)
├── docker-compose.yml           # one-command launch, GPU + volume mounts
├── .gitignore                   # ignores data/ and *.pth checkpoints
├── LICENSE                      # MIT
└── README.md
```

Trained weights (`*.pth`) and the `data/` directory are intentionally **not** tracked by git.

---

## Datasets

You need to obtain both datasets yourself and place them under a local `data/` folder (git-ignored). Inside the container they are expected at `/tf/data/...`, which is what the `./data` volume mount maps to.

| Dataset | Used for | Task in this repo | Source |
|---|---|---|---|
| **MURA v1.1** | Stage 1 pre-training | 7-class upper-extremity **body-part** classification | [Stanford ML Group – MURA](https://stanfordmlgroup.github.io/competitions/mura/) |
| **Aseptic Loose Hip Implant X-Ray Database** | Stage 2 + baseline | Binary **control vs loose** | [Kaggle (Rahman et al.)](https://www.kaggle.com/datasets/tawsifurrahman/aseptic-loose-hip-implant-xray-database) |

The loosening dataset accompanies the *HipXNet* study (Rahman et al., *IEEE Access*, 2022). MURA is reused here purely as a domain-relevant pre-training corpus — the seven body-part classes (`XR_ELBOW`, `XR_FINGER`, `XR_FOREARM`, `XR_HAND`, `XR_HUMERUS`, `XR_SHOULDER`, `XR_WRIST`) teach the network general radiographic features before it ever sees a hip.

### Expected layout

```
data/
├── mura/                          # contents of the MURA-v1.1 release
│   ├── train_image_paths.csv      # paths inside start with "MURA-v1.1/..."
│   ├── train/
│   └── valid/
└── loose_hips/
    ├── control/                   # non-loose implant X-rays (.png/.jpg/.jpeg)
    └── loose/                     # loose implant X-rays
```

Notes:
- The MURA loader rewrites the `MURA-v1.1` prefix in `train_image_paths.csv` to `/tf/data/mura`, so simply copy the MURA-v1.1 contents into `data/mura/`.
- For the loosening data, sort the loosening-detection images into the `control/` and `loose/` subfolders shown above.
- MURA is split with **`GroupShuffleSplit` by patient ID** to prevent images from the same patient leaking across train/val/test.

---

## Getting started

### Prerequisites

- An **NVIDIA GPU** with recent drivers.
- **Docker** and **Docker Compose**, plus the **NVIDIA Container Toolkit** (so containers can see the GPU).

### Quick start (Docker Compose — recommended)

```bash
git clone https://github.com/nearlyphd/conformer-implant-loosening.git
cd conformer-implant-loosening

# place the datasets under ./data as described above, then:
docker compose up --build
```

This builds the image and starts JupyterLab on **http://localhost:8888** (the token is disabled in `docker-compose.yml` for local convenience). Open the `notebooks/` folder and run the notebooks in order.

The compose file mounts `./notebooks → /tf/notebooks` and `./data → /tf/data`, requests **all** GPUs, and enables TensorFlow GPU memory growth.

### Additional Python dependencies

The Docker image installs the core stack (PyTorch, `timm`, pandas, scikit-learn, matplotlib, seaborn, tqdm). The **explainability** sections additionally use libraries that aren't in the image, so install them in the running container (or add them to the `Dockerfile`):

```bash
pip install grad-cam captum opencv-python
```

| Library | Needed for |
|---|---|
| `grad-cam` (`pytorch_grad_cam`) | Grad-CAM on the CNN branch |
| `captum` | Integrated Gradients on the fused model |
| `opencv-python` (`cv2`) | resizing attention maps in Attention Rollout |

> The Conformer is a PyTorch model. TensorFlow appears only in a one-cell GPU-availability sanity check at the top of each notebook — it isn't used for modeling.

---

## Usage / workflow

Run the notebooks in this sequence; each one saves a checkpoint that the next can consume.

1. **`parallel-vit-cnn-hybrid-mura.ipynb`** — pre-trains the Conformer on MURA (7-class) and saves `best_conformer_mura.pth`.
2. **`parallel-vit-cnn-hybrid-loose-hips-mura-pretrained.ipynb`** — loads the MURA weights, swaps in binary heads, fine-tunes on the loosening data, and saves `best_conformer_loose_hips_mura_pretrained.pth`.
3. **`parallel-vit-cnn-hybrid-loose-hips-scratch.ipynb`** — the baseline: same binary task but from random initialization, saving `best_conformer_scratch.pth`.

Checkpoints are written relative to the notebook's working directory (`/tf/notebooks`). The Conformer source is imported from the adjacent `./Conformer` folder.

---

## Methodology details

**Preprocessing & augmentation.** Every image is run through a custom `LetterboxResize` to 224×224 that preserves aspect ratio by padding (important, since radiographs vary in size and shape). Training augmentations reflect real-world X-ray variation — random horizontal flip, ±15° rotation, random-resized crop (0.8–1.0 scale), and brightness/contrast jitter (saturation and hue are left at 0 because X-rays are grayscale) — followed by ImageNet normalization and `RandomErasing` for regularization. Validation and test images get only letterbox resize + normalization.

**Small-data handling.** The loosening training subset is **oversampled (≈20×)** with augmentation so each epoch sees enough varied examples; splits are **stratified 80/10/10** to keep the class balance consistent across train/val/test.

**Loss & optimization.** The Conformer emits two logits (CNN + Transformer), and the loss is the **sum of cross-entropy on both heads** (with label smoothing of 0.1 on the loosening task), as recommended by the Conformer authors. Optimization uses **AdamW**, gradient clipping (max-norm 1.0), and a **OneCycleLR** schedule.

**Two-stage fine-tuning (pre-trained run).** To protect the transferred features, fine-tuning runs in two phases:
1. **Warm-up** — freeze the backbone and train only the new heads.
2. **Unfreeze** — release the whole network with **differential learning rates** (heads ~1e-4, backbone ~1e-5) under OneCycleLR.

**Early stopping & checkpointing.** Training watches validation loss for early stopping (with patience) and checkpoints the weights at best validation accuracy.

---

## Evaluation

For each trained model the notebooks produce:

- a **confusion matrix** and a full **classification report** (precision / recall / F1 per class);
- **ROC curves with AUC**; and
- **bootstrapped 95% confidence intervals** (1,000 resamples) for Accuracy, AUC, F1, Precision, and Recall — giving an honest sense of how stable the metrics are on a small test set.

All metrics use the **branch-ensemble** prediction (CNN + Transformer averaged). The headline comparison the repo is built around is **MURA-pre-trained vs from-scratch**, which isolates how much the transfer learning actually helps.

> Numeric results depend on your data split, GPU, and run; reproduce them by executing the evaluation cells. *(Suggested table to fill in after running:)*

| Model | Accuracy | AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| From scratch | – | – | – | – | – |
| MURA pre-trained | – | – | – | – | – |

---

## Explainability (XAI)

Because a loosening call has to be *trustable*, each branch — and the model as a whole — is probed with a method suited to it:

- **Grad-CAM (CNN branch)** — class-activation heatmaps over the last convolutional layer, showing which regions of the implant drive the convolutional decision.
- **Attention Rollout (Transformer branch)** — propagates self-attention across layers (accounting for residual connections) and maps the `[CLS]` token's attention back onto a 14×14 patch grid, revealing where the Transformer "looks."
- **Integrated Gradients (whole model)** — via Captum, attributes the **fused** decision to individual input pixels against a black baseline, explaining the actual combined output rather than either branch alone.

---

## License

This project is released under the **MIT License** — see [`LICENSE`](LICENSE). Copyright © 2026 Anton Myshenin.

The bundled `notebooks/Conformer/` directory is third-party code distributed under its own (MIT) license; see `notebooks/Conformer/LICENSE`.

---

## Credits & references

- **Conformer** — Z. Peng, W. Huang, S. Gu, L. Xie, Y. Wang, J. Jiao, Q. Ye. *Conformer: Local Features Coupling Global Representations for Visual Recognition.* ICCV 2021. Original implementation: [`pengzhiliang/Conformer`](https://github.com/pengzhiliang/Conformer).
- **MURA** — P. Rajpurkar et al., Stanford ML Group. *MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs.*
- **Aseptic Loose Hip Implant X-Ray Database** — T. Rahman, A. Khandakar, et al. *HipXNet: Deep Learning Approaches to Detect Aseptic Loosening of Hip Implants Using X-Ray Images.* IEEE Access, 2022. Dataset on [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/aseptic-loose-hip-implant-xray-database).
- Explainability tooling: [`jacobgil/pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam) and [Captum](https://captum.ai/).
