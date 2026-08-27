"""
PFP-GIRNet Testing Script
=========================
Evaluate PFP-GIRNet on the Oresund Region test set.

Usage:
    python train_and_test/test.py
"""

import argparse
import importlib.util
import os
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "model" / "PFP-GIRNet.py"
DATASET_DIR = PROJECT_ROOT / "dataset"
REQUESTED_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "vig_bimamba_skagen_region"
    / "30-24"
    / "best_model.pth"
)
PACKAGED_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "vig_bimamba_skagen_region"
    / "30-24"
    / "best_model.pth"
)


def load_model_class():
    """Load ShipTrajectoryRefiner from the hyphenated PFP-GIRNet.py filename."""
    spec = importlib.util.spec_from_file_location("pfp_girnet_model", MODEL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load model definition from: {MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ShipTrajectoryRefiner


ShipTrajectoryRefiner = load_model_class()


# ===============================================================
# 1. Batch collation
# ===============================================================
def seq_collate(data):
    """
    Collate samples with different numbers of vessels.

    The model input is arranged as [B, N, T, D]. Three graph structures are
    included: Social, EA, and TCPA.
    """
    max_nodes = max(d["obs_traj"].shape[1] for d in data)
    batch_size = len(data)
    obs_len = data[0]["obs_traj"].shape[0]
    pred_len = data[0]["pred_traj"].shape[0]

    obs_traj_batch = torch.zeros(batch_size, max_nodes, obs_len, 2)
    pred_traj_batch = torch.zeros(batch_size, max_nodes, pred_len, 2)
    v_obs_batch = torch.zeros(batch_size, max_nodes, obs_len, data[0]["v_obs"].shape[2])
    v_pred_batch = torch.zeros(batch_size, max_nodes, pred_len, 2)
    mask_batch = torch.zeros(batch_size, max_nodes)

    graph_social_batch = torch.zeros(batch_size, obs_len, max_nodes, max_nodes)
    graph_ea_batch = torch.zeros(batch_size, obs_len, max_nodes, max_nodes)
    graph_tcpa_batch = torch.zeros(batch_size, obs_len, max_nodes, max_nodes)

    for i, sample in enumerate(data):
        obs_traj = sample["obs_traj"].permute(1, 0, 2)
        pred_traj = sample["pred_traj"].permute(1, 0, 2)
        v_obs = sample["v_obs"].permute(1, 0, 2)
        v_pred = sample["v_pred"].permute(1, 0, 2)

        num_nodes = obs_traj.shape[0]

        obs_traj_batch[i, :num_nodes, :, :] = obs_traj
        pred_traj_batch[i, :num_nodes, :, :] = pred_traj
        v_obs_batch[i, :num_nodes, :, :] = v_obs
        v_pred_batch[i, :num_nodes, :, :] = v_pred
        mask_batch[i, :num_nodes] = 1.0

        if "graph_social" in sample:
            graph_social_batch[i, :, :num_nodes, :num_nodes] = sample["graph_social"]
            graph_ea_batch[i, :, :num_nodes, :num_nodes] = sample["graph_EA"]
            graph_tcpa_batch[i, :, :num_nodes, :num_nodes] = sample["graph_TCPA"]

    return {
        "obs_traj": obs_traj_batch,
        "pred_traj": pred_traj_batch,
        "v_obs": v_obs_batch,
        "v_pred": v_pred_batch,
        "mask": mask_batch,
        "graphs": {
            "social": graph_social_batch,
            "ea": graph_ea_batch,
            "tcpa": graph_tcpa_batch,
        },
    }


# ===============================================================
# 2. Utilities
# ===============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def relative_to_abs(rel_traj, start_pos):
    """
    Convert relative displacements to absolute coordinates.

    rel_traj: [B, N, T, 2]
    start_pos: [B, N, 2]
    """
    abs_traj = torch.zeros_like(rel_traj)
    abs_traj[:, :, 0, :] = start_pos + rel_traj[:, :, 0, :]
    for i in range(1, rel_traj.shape[2]):
        abs_traj[:, :, i, :] = abs_traj[:, :, i - 1, :] + rel_traj[:, :, i, :]
    return abs_traj


def compute_masked_metric_sums(
    error,
    mask,
    pred_len,
    mr_fde_threshold=30.0,
    mr_ade_threshold=30.0,
):
    """Accumulate ADE, FDE, MDE, and miss-rate statistics over valid vessels."""
    valid_mask = mask.bool()
    valid_error = error[valid_mask]
    total_valid_nodes = int(valid_error.shape[0])

    if total_valid_nodes == 0:
        return {
            "ade_error_sum": 0.0,
            "fde_error_sum": 0.0,
            "mde_error_sum": 0.0,
            "miss_count_fde": 0,
            "miss_count_ade": 0,
            "total_valid_nodes": 0,
        }

    if valid_error.shape[-1] != pred_len:
        raise ValueError(
            f"Prediction length mismatch: expected {pred_len}, got {valid_error.shape[-1]}."
        )

    ade_per_vessel = valid_error.mean(dim=-1)
    fde_per_vessel = valid_error[:, -1]
    mde_per_vessel = valid_error.max(dim=-1).values

    return {
        "ade_error_sum": float(valid_error.sum().item()),
        "fde_error_sum": float(fde_per_vessel.sum().item()),
        "mde_error_sum": float(mde_per_vessel.sum().item()),
        "miss_count_fde": int((fde_per_vessel > mr_fde_threshold).sum().item()),
        "miss_count_ade": int((ade_per_vessel > mr_ade_threshold).sum().item()),
        "total_valid_nodes": total_valid_nodes,
    }


def finalize_masked_metrics(
    ade_error_sum,
    fde_error_sum,
    mde_error_sum,
    total_valid_nodes,
    pred_len,
    miss_count_fde=0,
    miss_count_ade=0,
):
    """Convert accumulated metric sums to dataset-level averages."""
    if total_valid_nodes == 0:
        return {
            "ade": 0.0,
            "fde": 0.0,
            "mde": 0.0,
            "miss_rate_fde": 0.0,
            "miss_rate_ade": 0.0,
        }

    return {
        "ade": ade_error_sum / (total_valid_nodes * pred_len),
        "fde": fde_error_sum / total_valid_nodes,
        "mde": mde_error_sum / total_valid_nodes,
        "miss_rate_fde": miss_count_fde / total_valid_nodes,
        "miss_rate_ade": miss_count_ade / total_valid_nodes,
    }


class ShipTrajDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


# ===============================================================
# 3. Evaluation
# ===============================================================
def test(
    model,
    loader,
    mean,
    std,
    device,
    pred_len,
    mr_fde_threshold=30.0,
    mr_ade_threshold=30.0,
):
    """Evaluate PFP-GIRNet and return ADE, FDE, MDE, and miss-rate metrics."""
    model.eval()
    ade_error_sum, fde_error_sum, mde_error_sum = 0.0, 0.0, 0.0
    miss_count_fde, miss_count_ade, total_valid_nodes = 0, 0, 0

    loop = tqdm(loader, desc="Test", leave=False)

    with torch.no_grad():
        for batch in loop:
            obs_traj = batch["obs_traj"].to(device)
            pred_traj = batch["pred_traj"].to(device)
            v_obs = batch["v_obs"].to(device)
            mask = batch["mask"].to(device)

            graphs = {
                "social": batch["graphs"]["social"].to(device),
                "ea": batch["graphs"]["ea"].to(device),
                "tcpa": batch["graphs"]["tcpa"].to(device),
            }

            # The original input contains eight dimensions including positional encoding.
            # PFP-GIRNet uses dimensions 1-7 as the seven model input features.
            v_obs_7d = v_obs[..., 1:8]
            obs_input = (v_obs_7d - mean) / std

            y_pred, _, _, _ = model(obs_input, graphs=graphs)

            pred_rel_real = y_pred * std[:2] + mean[:2]
            start_pos = obs_traj[:, :, -1, :]
            pred_abs = relative_to_abs(pred_rel_real, start_pos)

            diff = pred_abs - pred_traj
            error = torch.sqrt((diff ** 2).sum(dim=-1))

            metric_sums = compute_masked_metric_sums(
                error,
                mask,
                pred_len=pred_len,
                mr_fde_threshold=mr_fde_threshold,
                mr_ade_threshold=mr_ade_threshold,
            )
            ade_error_sum += metric_sums["ade_error_sum"]
            fde_error_sum += metric_sums["fde_error_sum"]
            mde_error_sum += metric_sums["mde_error_sum"]
            miss_count_fde += metric_sums["miss_count_fde"]
            miss_count_ade += metric_sums["miss_count_ade"]
            total_valid_nodes += metric_sums["total_valid_nodes"]

    if total_valid_nodes == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0

    metrics = finalize_masked_metrics(
        ade_error_sum=ade_error_sum,
        fde_error_sum=fde_error_sum,
        mde_error_sum=mde_error_sum,
        total_valid_nodes=total_valid_nodes,
        pred_len=pred_len,
        miss_count_fde=miss_count_fde,
        miss_count_ade=miss_count_ade,
    )

    return (
        metrics["ade"],
        metrics["fde"],
        metrics["mde"],
        metrics["miss_rate_fde"],
        int(miss_count_fde),
        metrics["miss_rate_ade"],
        int(miss_count_ade),
        int(total_valid_nodes),
    )


# ===============================================================
# 4. Main
# ===============================================================
def resolve_checkpoint_path(checkpoint_path):
    """Use the requested local checkpoint and fall back to the packaged copy."""
    requested = Path(checkpoint_path)
    if requested.is_file():
        return requested
    if PACKAGED_CHECKPOINT_PATH.is_file():
        return PACKAGED_CHECKPOINT_PATH
    raise FileNotFoundError(
        "Checkpoint not found. Checked:\n"
        f"  {requested}\n"
        f"  {PACKAGED_CHECKPOINT_PATH}"
    )


def main(
    obs_len,
    pred_len,
    batch_size,
    d_model,
    num_layers,
    data_dir,
    checkpoint_path,
    mr_fde_threshold=30.0,
    mr_ade_threshold=30.0,
    result_dir=None,
    mamba_layers=2,
    mamba_d_state=16,
    mamba_d_conv=4,
    mamba_expand=2,
    mamba_dropout=0.1,
    seed=None,
):
    if seed is not None:
        set_seed(seed)

    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(data_dir)
    norm_path = data_dir / "norm_params.pkl"
    if not norm_path.is_file():
        raise FileNotFoundError(f"Normalization file not found: {norm_path}")

    with norm_path.open("rb") as file_obj:
        norm_params = pickle.load(file_obj)

    mean = torch.tensor(norm_params["mean"]).float().to(device)
    std = torch.tensor(norm_params["std"]).float().to(device)
    print("Normalization parameters loaded.")

    test_file_path = data_dir / "test.pkl"
    if not test_file_path.is_file():
        raise FileNotFoundError(f"Test set not found: {test_file_path}")

    with test_file_path.open("rb") as file_obj:
        test_set = ShipTrajDataset(pickle.load(file_obj))

    print(f"Test set loaded. Samples: {len(test_set)}")

    loader_test = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=seq_collate,
        num_workers=0,
        pin_memory=True,
    )

    model = ShipTrajectoryRefiner(
        num_ships=20,
        input_dim=7,
        d_model=d_model,
        hist_len=obs_len,
        pred_len=pred_len,
        num_layers=num_layers,
        mamba_layers=mamba_layers,
        mamba_d_state=mamba_d_state,
        mamba_d_conv=mamba_d_conv,
        mamba_expand=mamba_expand,
        mamba_dropout=mamba_dropout,
    ).to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"PFP-GIRNet parameters: {total_params:,}")

    print("\nStarting evaluation...")
    metrics = test(
        model,
        loader_test,
        mean,
        std,
        device,
        pred_len,
        mr_fde_threshold,
        mr_ade_threshold,
    )
    (
        ade,
        fde,
        mde,
        miss_rate_fde,
        miss_count_fde,
        miss_rate_ade,
        miss_count_ade,
        total_nodes,
    ) = metrics

    print("=" * 60)
    print("PFP-GIRNet Test Results")
    print("=" * 60)
    print(f"ADE: {ade:.4f} m (Average Displacement Error)")
    print(f"FDE: {fde:.4f} m (Final Displacement Error)")
    print(f"MDE: {mde:.4f} m (Maximum Displacement Error)")
    print("-" * 60)
    print(f"Miss Rate - FDE (threshold: {mr_fde_threshold} m)")
    print(f"Miss Count: {miss_count_fde} / {total_nodes}")
    print(f"Miss Rate (FDE): {miss_rate_fde * 100:.2f}%")
    print("-" * 60)
    print(f"Miss Rate - ADE (threshold: {mr_ade_threshold} m)")
    print(f"Miss Count: {miss_count_ade} / {total_nodes}")
    print(f"Miss Rate (ADE): {miss_rate_ade * 100:.2f}%")
    print("=" * 60)

    result_root = Path(result_dir) if result_dir else PROJECT_ROOT / "result" / "PFP-GIRNet_test_results"
    result_output_dir = result_root / data_dir.name / f"{obs_len}-{pred_len}"
    result_output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = result_output_dir / f"pfp_girnet_test_metrics_{timestamp}.txt"
    result_path.write_text(
        "\n".join(
            [
                "PFP-GIRNet Test Metrics",
                f"checkpoint_path: {checkpoint_path}",
                f"data_dir: {data_dir}",
                f"obs_len: {obs_len}",
                f"pred_len: {pred_len}",
                f"mamba_layers: {mamba_layers}",
                f"mamba_d_state: {mamba_d_state}",
                f"mamba_d_conv: {mamba_d_conv}",
                f"mamba_expand: {mamba_expand}",
                f"mamba_dropout: {mamba_dropout}",
                f"ADE: {ade:.6f}",
                f"FDE: {fde:.6f}",
                f"MDE: {mde:.6f}",
                f"miss_rate_fde: {miss_rate_fde:.6f}",
                f"miss_count_fde: {miss_count_fde}",
                f"miss_rate_ade: {miss_rate_ade:.6f}",
                f"miss_count_ade: {miss_count_ade}",
                f"total_nodes: {total_nodes}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Results saved to: {result_path}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Test PFP-GIRNet on the Oresund Region dataset.")
    parser.add_argument("--obs_len", type=int, default=30)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--mamba_layers", type=int, default=2)
    parser.add_argument("--mamba_d_state", type=int, default=16)
    parser.add_argument("--mamba_d_conv", type=int, default=4)
    parser.add_argument("--mamba_expand", type=int, default=2)
    parser.add_argument("--mamba_dropout", type=float, default=0.10)
    parser.add_argument("--mr_fde_threshold", type=float, default=30.0)
    parser.add_argument("--mr_ade_threshold", type=float, default=30.0)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DATASET_DIR),
        help="Directory containing test.pkl and norm_params.pkl.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=str(REQUESTED_CHECKPOINT_PATH),
        help="Path to best_model.pth. The packaged checkpoint is used as a fallback.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=str(PROJECT_ROOT / "result" / "PFP-GIRNet_test_results"),
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint_path,
        mr_fde_threshold=args.mr_fde_threshold,
        mr_ade_threshold=args.mr_ade_threshold,
        result_dir=args.result_dir,
        mamba_layers=args.mamba_layers,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        mamba_dropout=args.mamba_dropout,
        seed=args.seed,
    )
