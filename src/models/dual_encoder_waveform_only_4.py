import torch
import torch.nn as nn

from .helpers import _encoder_block, _decoder_block
from .model_config import KERNEL_SIZE, STRIDE, PADDING, LEAKY_RELU_SLOPE, OUTPUT_PADDING


class DualStreamCAE_TwoEnc_DirectX(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.s_enc1 = _encoder_block(1, 16)
        self.s_enc2 = _encoder_block(16, 32)
        self.s_enc3 = _encoder_block(32, 64)
        self.s_enc4 = _encoder_block(64, 64)
        self.s_enc5 = _encoder_block(64, 64)
        self.s_enc6 = _encoder_block(64, 64)

        self.g_enc1 = _encoder_block(1, 16)
        self.g_enc2 = _encoder_block(16, 32)
        self.g_enc3 = _encoder_block(32, 64)
        self.g_enc4 = _encoder_block(64, 64)
        self.g_enc5 = _encoder_block(64, 64)
        self.g_enc6 = _encoder_block(64, 64)

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
        se = self._encode_s(s_in)
        ge = self._encode_g(g_in)

        joint = torch.cat([se, ge], dim=1)
        z = self.joint_bottleneck(joint)

        f = self.dec1(z)
        f = self.dec2(f)
        f = self.dec3(f)
        f = self.dec4(f)
        f = self.dec5(f)
        f = self.dec6(f)
        x_hat = self.x_final(f)

        return x_hat

    def _encode_s(self, x: torch.Tensor) -> torch.Tensor:
        x = self.s_enc1(x)
        x = self.s_enc2(x)
        x = self.s_enc3(x)
        x = self.s_enc4(x)
        x = self.s_enc5(x)
        x = self.s_enc6(x)
        return x

    def _encode_g(self, x: torch.Tensor) -> torch.Tensor:
        x = self.g_enc1(x)
        x = self.g_enc2(x)
        x = self.g_enc3(x)
        x = self.g_enc4(x)
        x = self.g_enc5(x)
        x = self.g_enc6(x)
        return x


if __name__ == "__main__":
    model = DualStreamCAE_TwoEnc_DirectX()
    s_in = torch.randn(8, 1, 32000)
    g_in = torch.randn(8, 1, 32000)
    x_hat = model(s_in, g_in)
    print(x_hat.shape)
