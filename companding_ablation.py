import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as f_ta
from pesq import pesq
from pystoi import stoi
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dyn_encoder import (
    dyn_encode_block,
    dyn_decode_block,
    dyn_encode_per_sample,
    dyn_decode_per_sample,
)


def mu_law_roundtrip(x_norm: torch.Tensor):
    encoded = f_ta.mu_law_encoding(x_norm, quantization_channels=256)
    encoded = encoded.to(torch.uint8)
    decoded = f_ta.mu_law_decoding(encoded, quantization_channels=256)
    return decoded


def compute_pesq_np(ref_np: np.ndarray, deg_np: np.ndarray, sr: int):
    try:
        return float(pesq(sr, ref_np, deg_np))
    except Exception:
        return np.nan


def compute_stoi_np(ref_np: np.ndarray, deg_np: np.ndarray, sr: int, extended: bool):
    try:
        return float(stoi(ref_np, deg_np, sr, extended=extended))
    except Exception:
        return np.nan


def reconstruction_error_np(ref_np: np.ndarray, deg_np: np.ndarray):
    diff = ref_np.astype(np.float32) - deg_np.astype(np.float32)
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    return mse, mae


def update_values(
    values: dict, method: str, block_size: int | None, metrics: dict
):
    key = (method, None if block_size is None else int(block_size))
    if key not in values:
        values[key] = {
            "sum_pesq": 0.0,
            "cnt_pesq": 0,
            "sum_stoi": 0.0,
            "cnt_stoi": 0,
            "sum_estoi": 0.0,
            "cnt_estoi": 0,
            "sum_mse": 0.0,
            "cnt_mse": 0,
            "sum_mae": 0.0,
            "cnt_mae": 0,
            "num_items": 0,
        }
    agg = values[key]
    agg["num_items"] += 1
    for name in ("pesq", "stoi", "estoi", "mse", "mae"):
        val = metrics.get(name, np.nan)
        if val is not None and np.isfinite(val):
            agg[f"sum_{name}"] += float(val)
            agg[f"cnt_{name}"] += 1


def safe_avg(sum_key: str, cnt_key: str):
    return (agg[sum_key] / agg[cnt_key]) if agg[cnt_key] > 0 else np.nan


if __name__ == "__main__":
    root = "./data"
    os.makedirs(root, exist_ok=True)
    dataset = torchaudio.datasets.LIBRISPEECH(
        root=root, url="dev-clean", download=True
    )

    block_sizes = [16, 64, 256, 1024]
    compute_extended_stoi = False
    max_seconds_for_metrics = None

    loader = DataLoader(dataset, batch_size=None, num_workers=4)

    aggregates = {}

    with torch.no_grad():
        for idx, item in enumerate(
            tqdm(loader, desc="Companding metrics", unit="file")
        ):
            waveform, sr, *_ = item

            x = waveform.squeeze(0).to(torch.float32)
            x = torch.clamp(x, -1.0, 1.0)

            if max_seconds_for_metrics is not None:
                max_len = int(sr * max_seconds_for_metrics)
                if x.numel() > max_len:
                    x = x[:max_len]

            duration_sec = float(x.numel()) / float(sr)

            x_np = x.detach().cpu().numpy().astype(np.float32)

            recon_variants = []
            try:
                y = mu_law_roundtrip(x)
                recon_variants.append(("mu-law", None, y))
            except Exception:
                recon_variants.append(("mu-law", None, None))

            try:
                m_t, g_t = dyn_encode_per_sample(x)
                y = dyn_decode_per_sample(m_t, g_t)
                recon_variants.append(("dyn-per-sample", None, y))
            except Exception:
                recon_variants.append(("dyn-per-sample", None, None))

            for N in block_sizes:
                try:
                    s_b, g_b = dyn_encode_block(x, block_size=N)
                    y = dyn_decode_block(s_b, g_b)
                    recon_variants.append(("dyn-block", N, y))
                except Exception:
                    recon_variants.append(("dyn-block", N, None))

            for method, block_size, y in recon_variants:
                metrics = {
                    "pesq": np.nan,
                    "stoi": np.nan,
                    "estoi": np.nan,
                    "mse": np.nan,
                    "mae": np.nan,
                }
                if y is not None:
                    y_np = y.detach().cpu().numpy().astype(np.float32)
                    metrics["pesq"] = compute_pesq_np(x_np, y_np, sr)
                    metrics["stoi"] = compute_stoi_np(x_np, y_np, sr, extended=False)
                    if compute_extended_stoi:
                        metrics["estoi"] = compute_stoi_np(
                            x_np, y_np, sr, extended=True
                        )
                    mse, mae = reconstruction_error_np(x_np, y_np)
                    metrics["mse"], metrics["mae"] = mse, mae
                update_values(aggregates, method, block_size, metrics)

    rows = []
    for (method, block_size), agg in aggregates.items():

        rows.append(
            {
                "method": method,
                "block_size": block_size,
                "num_items": int(agg["num_items"]),
                "avg_pesq": safe_avg("sum_pesq", "cnt_pesq"),
                "avg_stoi": safe_avg("sum_stoi", "cnt_stoi"),
                "avg_estoi": safe_avg("sum_estoi", "cnt_estoi"),
                "avg_mse": safe_avg("sum_mse", "cnt_mse"),
                "avg_mae": safe_avg("sum_mae", "cnt_mae"),
            }
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "method",
            "block_size",
            "num_items",
            "avg_pesq",
            "avg_stoi",
            "avg_estoi",
            "avg_mse",
            "avg_mae",
        ],
    )
    out_csv = "librispeech_dev_clean_companding_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} averaged rows to {out_csv}")
