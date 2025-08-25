import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.functional import mu_law_encoding
from torch.utils.data import DataLoader
from tqdm import tqdm
from pesq import pesq as pesq_metric
from pystoi.stoi import stoi as stoi_metric
import wandb

from datasets import Libri2sPCM
from models import BaselineCAE
from loss import CombinedAudioLoss
from utils.misc import set_seed, count_parameters
from utils.early_stopping import EarlyStopping


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data", help="LibriSpeech root")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints_baseline",
        help="Directory to save model checkpoints",
    )
    parser.add_argument("--sisdr_weight", type=float, default=0.1, help="SI-SDR weight")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument(
        "--loss",
        type=str,
        default="waveform",
        choices=["waveform", "l1"],
        help="Training loss: 'waveform' (default), or L1",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="pcm16",
        choices=["pcm16", "mu_law_8bit"],
        help="Baseline input variant: 16-bit PCM or 8-bit mu-law companded",
    )
    return parser.parse_args()


def make_loaders(root: str, seconds: float, sr: int, batch_size: int, workers: int = 4):
    train_ds = Libri2sPCM(root, "train-clean-100", seconds=seconds, sr=sr, train=True)
    val_ds = Libri2sPCM(root, "dev-clean", seconds=seconds, sr=sr, train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, val_loader


def _preprocess_input(x: torch.Tensor, variant: str):
    if variant == "mu_law_8bit":
        x_mu = mu_law_encoding(x, quantization_channels=256)
        return (x_mu.to(torch.float32) / 255.0) * 2.0 - 1.0
    return x


def train_one_epoch(model, loader, opt, device, criterion, variant: str):
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Training")
    for batch_idx, x in enumerate(pbar):
        x = x.to(device, non_blocking=True)

        opt.zero_grad()

        with torch.no_grad():
            x_in = _preprocess_input(x, variant)

        x_hat = model(x_in)

        loss = criterion(x, x_hat)

        loss.backward()
        opt.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return {
        "total_loss": total_loss / num_batches,
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    epoch: int,
    sample_rate: int,
    criterion: nn.Module,
    save_reconstructions: bool = True,
    reconstructions_to_save: int = 5,
    reconstructions_dir: str = "reconstructions",
    variant: str = "pcm16",
):
    model.eval()

    total_loss = 0.0
    pesq_sum = 0.0
    stoi_sum = 0.0
    pesq_count = 0
    stoi_count = 0
    num_batches = 0
    saved_count = 0

    if save_reconstructions:
        epoch_dir = os.path.join(reconstructions_dir, f"epoch_{epoch:03d}")
        os.makedirs(epoch_dir, exist_ok=True)

    pbar = tqdm(loader, desc="Evaluating")
    for batch_idx, x in enumerate(pbar):
        x = x.to(device)

        with torch.no_grad():
            x_in = _preprocess_input(x, variant)

        x_hat = model(x_in)

        loss = criterion(x, x_hat)

        total_loss += loss.item()
        num_batches += 1

        x_np = x.cpu().numpy()
        x_recon_np = x_hat.cpu().numpy()

        for i in range(x.shape[0]):
            orig = x_np[i, 0]
            recon = x_recon_np[i, 0]

            try:
                pesq_score = pesq_metric(sample_rate, orig, recon, "wb")
                pesq_sum += pesq_score
                pesq_count += 1
            except Exception:
                pass

            try:
                stoi_score = stoi_metric(orig, recon, sample_rate, extended=False)
                stoi_sum += stoi_score
                stoi_count += 1
            except Exception:
                pass

            if save_reconstructions and saved_count < reconstructions_to_save:
                recon_path = os.path.join(epoch_dir, f"val_{saved_count:02d}_recon.wav")
                recon_tensor = torch.from_numpy(recon).unsqueeze(0)
                torchaudio.save(recon_path, recon_tensor, sample_rate)
                saved_count += 1

        avg_loss = total_loss / num_batches
        pbar.set_postfix({"loss": f"{avg_loss:.6f}"})

    avg_loss = total_loss / num_batches
    avg_pesq = pesq_sum / pesq_count if pesq_count > 0 else 0.0
    avg_stoi = stoi_sum / stoi_count if stoi_count > 0 else 0.0

    return {
        "loss": avg_loss,
        "pesq": avg_pesq,
        "stoi": avg_stoi,
        "pesq_count": pesq_count,
        "stoi_count": stoi_count,
    }


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    criterion,
    optimizer,
    scheduler,
    early_stopping,
    epochs: int,
    sample_rate: int,
    variant: str,
    use_wandb: bool,
    save_dir: str,
    model_name: str,
):
    best_val_loss = float("inf")
    best_pesq = float("-inf")
    best_stoi = float("-inf")

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 50)

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, criterion, variant
        )
        train_loss = train_metrics["total_loss"]

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            epoch=epoch,
            sample_rate=sample_rate,
            criterion=criterion,
            save_reconstructions=True,
            reconstructions_to_save=5,
            variant=variant,
        )
        val_loss = val_metrics["loss"]
        pesq_score = val_metrics["pesq"]
        stoi_score = val_metrics["stoi"]

        print(f"Val Loss: {val_loss:.6f}")
        print(
            f"PESQ: {pesq_score:.3f} (computed on {val_metrics['pesq_count']} samples)"
        )
        print(
            f"STOI: {stoi_score:.3f} (computed on {val_metrics['stoi_count']} samples)"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if val_metrics["pesq_count"] > 0 and pesq_score > best_pesq:
            best_pesq = pesq_score
            pesq_path = os.path.join(save_dir, f"best_pesq_{model_name}_{variant}.pt")
            torch.save(model.state_dict(), pesq_path)
            print(f"New best PESQ {best_pesq:.3f}. Saved model to {pesq_path}")

        if val_metrics["stoi_count"] > 0 and stoi_score > best_stoi:
            best_stoi = stoi_score
            stoi_path = os.path.join(save_dir, f"best_stoi_{model_name}_{variant}.pt")
            torch.save(model.state_dict(), stoi_path)
            print(f"New best STOI {best_stoi:.3f}. Saved model to {stoi_path}")

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "pesq": pesq_score,
                    "stoi": stoi_score,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        scheduler.step(val_loss)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best PESQ: {best_pesq:.3f}")
    print(f"Best STOI: {best_stoi:.3f}")

    return {
        "best_val_loss": best_val_loss,
        "best_pesq": best_pesq,
        "best_stoi": best_stoi,
    }


def run_with_args(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.root, exist_ok=True)

    if args.wandb:
        wandb.init(project="pcm_cae", config=args)

    seconds = 2.0
    sr = 16000

    train_loader, val_loader = make_loaders(
        root=args.root,
        seconds=seconds,
        sr=sr,
        batch_size=args.batch_size,
        workers=args.num_workers,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = BaselineCAE().to(device)

    if args.loss == "waveform":
        criterion = CombinedAudioLoss(
            sisdr_weight=args.sisdr_weight, sample_rate=sr
        ).to(device)
    elif args.loss == "l1":
        criterion = nn.L1Loss().to(device)
    else:
        raise ValueError(f"Unsupported loss: {args.loss}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    loss_suffix = "" if args.loss == "waveform" else f"_{args.loss}"
    early_stop_path = os.path.join(
        args.save_dir, f"early_stop_BaselineCAE{loss_suffix}_{args.variant}.pt"
    )
    early_stopping = EarlyStopping(
        patience=10, delta=0.0, path=early_stop_path, mode="min"
    )

    total_params = count_parameters(model)
    print(f"Model parameters: {total_params:,} total")

    model_name = f"BaselineCAE{loss_suffix}"

    _ = train_model(
        model,
        train_loader,
        val_loader,
        device,
        criterion,
        optimizer,
        scheduler,
        early_stopping,
        epochs=args.epochs,
        sample_rate=sr,
        variant=args.variant,
        use_wandb=args.wandb,
        save_dir=args.save_dir,
        model_name=model_name,
    )


def main():
    args = parse_args()
    run_with_args(args)


if __name__ == "__main__":
    main()
