# PFP-GIRNet

**PFP-GIRNet: Pseudo-future Perception-driven Gated Intention Residual Network for Multi-vessel Trajectory Prediction**

PFP-GIRNet is a multi-vessel trajectory prediction framework designed to improve the modeling of vessel interactions and potential maneuvering intentions from historical AIS trajectories. The model uses two historical motion features, `dx` and `dy`, as input. The positional-encoding channel at index 0 of the raw `v_obs` tensor is excluded, and the model input is constructed using `v_obs[..., 1:3]` with `input_dim=2`.

## Abstract

Trajectory prediction is a critical technology for collision avoidance warning and traffic management in maritime intelligent transportation systems (MITS). However, existing spatiotemporal prediction models often provide limited explicit characterization of maneuvering cues embedded in historical observations. To address this issue, we propose a **Pseudo-future Perception-driven Gated Intention Residual Network (PFP-GIRNet)**.

PFP-GIRNet consists of three main components. An **Adaptive Multi-Graph Convolution (AMG)** module models complex vessel interaction dependencies. A **Spatial-Spectral Feature Fusion (SSF)** module jointly captures interaction patterns and motion details through spatial attention and discrete wavelet transform. Furthermore, a **Gated Intention Fusion (GIF)** module generates a pseudo-future representation from historical features and adaptively integrates it with the historical motion representation to enhance nonlinear maneuver modeling.

Experimental results demonstrate that PFP-GIRNet achieves superior trajectory prediction performance across different maritime scenarios and prediction horizons.

## Commands

```bash
python train_and_test/train.py
python train_and_test/test.py
```

Training saves new two-feature checkpoints under `checkpoints/pfp_girnet_skagen_region/<obs_len>-<pred_len>/`. Test defaults to the matching `best_model.pth`.

## Environment

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
