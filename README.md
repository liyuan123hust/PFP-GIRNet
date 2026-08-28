# PFP-GIRNet

PFP-GIRNet predicts vessel trajectories from two historical motion features: `dx` and `dy`. The raw `v_obs` positional-encoding channel at index 0 is excluded; model inputs use `v_obs[..., 1:3]` and `input_dim=2`.

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