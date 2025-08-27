import torch
import torch.nn as nn
import auraloss


class CombinedAudioLoss(nn.Module):
    """
    Combined SI-SDR + Multi-Resolution STFT loss.
    """

    def __init__(self, sisdr_weight: float = 0.1, sample_rate: int = 16000) -> None:
        super().__init__()
        self.sisdr_weight = sisdr_weight
        self.sisdr = auraloss.time.SISDRLoss()
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[512, 1024, 2048],
            hop_sizes=[128, 256, 512],
            win_lengths=[512, 1024, 2048],
            n_bins=128,
            sample_rate=sample_rate,
            perceptual_weighting=True,
        )

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        stft_loss = self.mrstft(y_pred, y_true)
        sisdr_loss = self.sisdr(y_pred, y_true)
        return stft_loss + self.sisdr_weight * sisdr_loss
