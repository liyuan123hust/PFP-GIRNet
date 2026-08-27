# 🚢 PFP-GIRNet

Official PyTorch implementation of **PFP-GIRNet** for multi-vessel trajectory prediction.

## 📁 Project Structure

```text
PFP-GIRNet/
├── checkpoints/        # Pretrained model
├── dataset/            # Test data and normalization parameters
├── model/              # PFP-GIRNet model
├── train_and_test/     # Testing script
├── Environment.md      # Environment information
└── command.md          # Running command
```

## ⚙️ Environment

Main dependencies:

- Python 3.12
- PyTorch 2.5.1
- CUDA 12.1
- PyTorch Geometric 2.6.1
- NumPy
- SciPy
- einops

More details are provided in `Environment.md`.

## ▶️ Test

Run:

```bash
python train_and_test/test.py
```

The test script automatically loads:

- `dataset/test.pkl`
- `dataset/norm_params.pkl`
- `checkpoints/vig_bimamba_skagen_region/30-24/best_model.pth`

## 📌 Model

The released checkpoint corresponds to the **30 → 24** trajectory prediction task.

## 📄 Citation

If this code is useful for your research, please consider citing our paper.

Citation information will be updated after publication.
