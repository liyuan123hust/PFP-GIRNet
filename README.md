# 🚢 PFP-GIRNet

**PFP-GIRNet: Pseudo-future Perception-driven Gated Intention Residual Network for Multi-vessel Trajectory Prediction**

This repository provides the official implementation of **PFP-GIRNet: Pseudo-future Perception-driven Gated Intention Residual Network for Multi-vessel Trajectory Prediction**.

PFP-GIRNet is designed to improve the modeling of vessel interactions and potential maneuvering intentions from historical AIS trajectories. In our experiments, we evaluated both basic displacement features and enriched motion representations derived from historical trajectory information, with the latter showing stronger predictive performance. The current released version adopts the basic `dx` and `dy` displacement features as model inputs for a lightweight and straightforward implementation.

## 📄 Abstract

Trajectory prediction is a critical technology for collision avoidance warning and traffic management in maritime intelligent transportation systems (MITS). However, existing spatiotemporal prediction models often provide limited explicit characterization of maneuvering cues embedded in historical observations. To address this issue, we propose a **Pseudo-future Perception-driven Gated Intention Residual Network (PFP-GIRNet)**.

PFP-GIRNet consists of three main components. An **Adaptive Multi-Graph Convolution (AMG)** module models complex vessel interaction dependencies. A **Spatial-Spectral Feature Fusion (SSF)** module jointly captures interaction patterns and motion details through spatial attention and discrete wavelet transform. Furthermore, a **Gated Intention Fusion (GIF)** module generates a pseudo-future representation from historical features and adaptively integrates it with the historical motion representation to enhance nonlinear maneuver modeling.

Experimental results demonstrate that PFP-GIRNet achieves superior trajectory prediction performance across different maritime scenarios and prediction horizons.

## 🧪 Released Model

The current release provides the **30 → 12** prediction setting using `dx` and `dy` as the historical motion inputs.

The pretrained checkpoint is provided at:

```text
checkpoints/pfp_girnet_skagen_region/30-12/best_model.pth
```

## 💻 Usage

Run the released model using:

```bash
python train_and_test/test.py
```

The test script loads the provided checkpoint and evaluates PFP-GIRNet on the released test set.

## 📂 Data

The released files include:

```text
dataset/
├── norm_params.pkl
└── test.pkl
```

`norm_params.pkl` contains the normalization statistics required for inference, while `test.pkl` contains the test samples used for evaluation.

## ⚙️ Environment

The code was developed and tested with the following environment:

- Python 3.12.9
- PyTorch 2.5.1 + CUDA 12.1
- torchvision 0.20.1
- torchaudio 2.5.1
- PyTorch Geometric 2.6.1
- NumPy 2.4.6
- SciPy 1.17.1
- pandas 2.3.0
- tqdm 4.66.1
- NetworkX 3.6.1
- einops 0.8.1
