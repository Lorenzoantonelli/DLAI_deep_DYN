#!/usr/bin/env python3

import argparse
import csv
import os

import torch
import torch.nn as nn
import torchaudio
from pesq import pesq as pesq_metric
from pystoi.stoi import stoi as stoi_metric
from torchaudio.functional import mu_law_encoding
from tqdm import tqdm

from datasets import Libri2sPCM
from dyn_encoder import dyn_encode_block
from models import (
    DualStreamCAE,
    DualStreamCAESharedEncoder,
    DualStreamCAE_ShareEncDec,
    DualStreamCAE_TwoEnc_DirectX,
    DualStreamCAE_ShareEnc_DirectX,
    BaselineCAE,
)

MODEL_REGISTRY = {
    "DualStreamCAE": DualStreamCAE,
    "DualStreamCAESharedEncoder": DualStreamCAESharedEncoder,
    "DualStreamCAE_ShareEncDec": DualStreamCAE_ShareEncDec,
    "DualStreamCAE_TwoEnc_DirectX": DualStreamCAE_TwoEnc_DirectX,
    "DualStreamCAE_ShareEnc_DirectX": DualStreamCAE_ShareEnc_DirectX,
    "BaselineCAE": BaselineCAE,
}


def make_test_loader(root: str, seconds: float, sr: int, batch_size: int, workers: int):
    ds = Libri2sPCM(root, "test-clean", seconds=seconds, sr=sr, train=False)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        drop_last=False,
    )


def is_baseline_model(model: nn.Module):
    return isinstance(
        model, BaselineCAE
    ) or model.__class__.__name__.lower().startswith("baseline")


def call_model(
    model: nn.Module, x: torch.Tensor, s_norm: torch.Tensor, g_norm: torch.Tensor
):
    if is_baseline_model(model):
        out = model(x)
    else:
        out = model(s_norm, g_norm)

    if isinstance(out, (tuple, list)):
        x_hat = out[0]
    else:
        x_hat = out
    return x_hat


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader,
    device,
    sample_rate: int,
    block_size: int = 256,
    save_wavs: bool = False,
    recons_dir: str = "",
    max_recons: int = 5,
    limit_batches: int = None,
    mu_law_baseline: bool = False,
):
    model.eval()

    pesq_sum = 0.0
    stoi_sum = 0.0
    pesq_count = 0
    stoi_count = 0
    num_batches = 0
    saved_count = 0

    pbar = tqdm(loader, desc="Testing", leave=False)
    for b_idx, x in enumerate(pbar):
        if limit_batches is not None and num_batches >= limit_batches:
            break

        x = x.to(device, non_blocking=True)  # [B, 1, T]

        sample, gain = dyn_encode_block(x, block_size=block_size, keep_shape=True)
        sample = sample.to(device)
        gain = gain.to(device)
        sample_norm = sample.float() / 128.0
        gain_norm = gain.float() / 63.0

        if is_baseline_model(model) and mu_law_baseline:
            with torch.no_grad():  # mu-law edge case support
                x_mu = mu_law_encoding(x, quantization_channels=256)
                x_in = (x_mu.to(torch.float32) / 255.0) * 2.0 - 1.0
            out = model(x_in)
            x_hat = out[0] if isinstance(out, (tuple, list)) else out
        else:
            x_hat = call_model(model, x, sample_norm, gain_norm)

        num_batches += 1

        x_np = x.detach().cpu().numpy()
        xhat_np = x_hat.detach().cpu().numpy()

        for i in range(x.shape[0]):
            orig = x_np[i, 0]
            recon = xhat_np[i, 0]
            try:
                pesq_sum += pesq_metric(sample_rate, orig, recon, "wb")
                pesq_count += 1
            except Exception:
                pass
            try:
                stoi_sum += stoi_metric(orig, recon, sample_rate, extended=False)
                stoi_count += 1
            except Exception:
                pass

            if save_wavs and saved_count < max_recons:
                os.makedirs(recons_dir, exist_ok=True)
                out_path = os.path.join(recons_dir, f"recon_{saved_count:04d}.wav")
                torchaudio.save(
                    out_path, torch.from_numpy(recon).unsqueeze(0), sample_rate
                )
                saved_count += 1

    avg_pesq = (pesq_sum / pesq_count) if pesq_count > 0 else 0.0
    avg_stoi = (stoi_sum / stoi_count) if stoi_count > 0 else 0.0

    return {
        "pesq": avg_pesq,
        "stoi": avg_stoi,
        "pesq_count": pesq_count,
        "stoi_count": stoi_count,
        "num_batches": num_batches,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data", help="LibriSpeech root")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--model",
        type=str,
        default="DualStreamCAE_TwoEnc_DirectX",
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Model to test",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint .pt file",
    )
    parser.add_argument(
        "--reconstructions_dir", type=str, default="test_reconstructions", help="Directory to save reconstructions"
    )
    parser.add_argument("--save_metrics_path", type=str, default="test_metrics.csv", help="Path to save metrics")
    parser.add_argument("--save_wavs", action="store_true", help="Save reconstructed waveforms")
    parser.add_argument("--max_recons_per_model", type=int, default=5, help="Maximum number of reconstructions to save per model")
    parser.add_argument("--limit_batches", type=int, default=None, help="Limit the number of batches to test")
    parser.add_argument("--block_size", type=int, default=256, help="Block size for DYN encoding")
    parser.add_argument(
        "--mu_law_baseline",
        action="store_true",
        help="For baseline models trained on mu-law input, enable mu-law preprocessing",
    )
    return parser.parse_args()

def run_single_test(
    model_name: str,
    model_cls,
    checkpoint_path: str,
    args,
    device,
    test_loader,
):
    if not os.path.isfile(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None

    print(f"=== Evaluating {model_name} ===")
    print(f"Checkpoint: {checkpoint_path}")

    model = model_cls().to(device)
    try:
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    except Exception as e:
        print(f"Error loading checkpoint for {model_name}: {e}")
        return None

    metrics = evaluate_model(
        model,
        test_loader,
        device,
        sample_rate=args.sr,
        block_size=args.block_size,
        save_wavs=args.save_wavs,
        recons_dir=os.path.join(args.reconstructions_dir, model_name),
        max_recons=args.max_recons_per_model,
        limit_batches=args.limit_batches,
        mu_law_baseline=args.mu_law_baseline,
    )

    print(
        f"{model_name} -> "
        f"PESQ: {metrics['pesq']:.3f} ({metrics['pesq_count']} clips), "
        f"STOI: {metrics['stoi']:.3f} ({metrics['stoi_count']} clips)"
    )

    row = {
        "model": model_name,
        "checkpoint_file": os.path.basename(checkpoint_path),
        "pesq": f"{metrics['pesq']:.6f}",
        "stoi": f"{metrics['stoi']:.6f}",
        "pesq_count": metrics["pesq_count"],
        "stoi_count": metrics["stoi_count"],
        "num_batches": metrics["num_batches"],
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


def main():
    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    os.makedirs(args.reconstructions_dir, exist_ok=True)
    os.makedirs(args.root, exist_ok=True)
    
    test_loader = make_test_loader(
        root=args.root,
        seconds=args.seconds,
        sr=args.sr,
        batch_size=args.batch_size,
        workers=args.num_workers,
    )
    print(f"Test batches: {len(test_loader)}")

    model_cls = MODEL_REGISTRY[args.model]
    row = run_single_test(
        model_name=args.model,
        model_cls=model_cls,
        checkpoint_path=args.checkpoint,
        args=args,
        device=device,
        test_loader=test_loader,
    )

    if row is None:
        print("No results to save (evaluation failed).")
        return

    out_csv = args.save_metrics_path
    write_header = not os.path.isfile(out_csv)
    with open(out_csv, "a" if not write_header else "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "checkpoint_file",
                "pesq",
                "stoi",
                "pesq_count",
                "stoi_count",
                "num_batches",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Saved metrics to: {out_csv}")


if __name__ == "__main__":
    main()
