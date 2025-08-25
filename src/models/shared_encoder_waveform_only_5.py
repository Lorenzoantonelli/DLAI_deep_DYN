import torch
import torch.nn as nn

from .model_config import KERNEL_SIZE, STRIDE, PADDING, LEAKY_RELU_SLOPE, OUTPUT_PADDING
from .helpers import _encoder_block, _decoder_block


class DualStreamCAE_ShareEnc_DirectX(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.enc1 = _encoder_block(1, 16)
        self.enc2 = _encoder_block(16, 32)
        self.enc3 = _encoder_block(32, 64)
        self.enc4 = _encoder_block(64, 64)
        self.enc5 = _encoder_block(64, 64)
        self.enc6 = _encoder_block(64, 64)

        self.joint_bottleneck = nn.Sequential(
            nn.Conv1d(
                in_channels=128,
                out_channels=32,
                kernel_size=KERNEL_SIZE,
                stride=STRIDE,
                padding=PADDING,
            ),
            nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=True),
        )

        self.dec1 = _decoder_block(32, 64)
        self.dec2 = _decoder_block(64, 64)
        self.dec3 = _decoder_block(64, 64)
        self.dec4 = _decoder_block(64, 64)
        self.dec5 = _decoder_block(64, 32)
        self.dec6 = _decoder_block(32, 16)
        self.x_final = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=16,
                out_channels=1,
                kernel_size=KERNEL_SIZE,
                stride=STRIDE,
                padding=PADDING,
                output_padding=OUTPUT_PADDING,
            ),
            nn.Tanh(),
        )

    def forward(self, s_in: torch.Tensor, g_in: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s_in: [B, 1, T] sample stream (normalized to [-1,1])
            g_in: [B, 1, T] gain stream (normalized to [0,1])
        Returns:
            x_hat: [B, 1, T] reconstructed waveform (x_hat = s_hat * g_hat) normalized to [-1,1]
        """
        se6 = self._encode_shared(s_in)
        ge6 = self._encode_shared(g_in)

        joint = torch.cat([se6, ge6], dim=1)
        z = self.joint_bottleneck(joint)

        f = self.dec1(z)
        f = self.dec2(f)
        f = self.dec3(f)
        f = self.dec4(f)
        f = self.dec5(f)
        f = self.dec6(f)
        x_hat = self.x_final(f)

        return x_hat

    def _encode_shared(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.enc6(x)
        return x


if __name__ == "__main__":
    model = DualStreamCAE_ShareEnc_DirectX()
    s_in = torch.randn(8, 1, 32000)
    g_in = torch.randn(8, 1, 32000)
    x_hat = model(s_in, g_in)
    print(x_hat.shape)
