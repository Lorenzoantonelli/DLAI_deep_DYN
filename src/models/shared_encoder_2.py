import torch
import torch.nn as nn

from .model_config import KERNEL_SIZE, STRIDE, PADDING, LEAKY_RELU_SLOPE
from .helpers import _encoder_block, _decoder_block, _final_block


class DualStreamCAESharedEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # The encoder is shared for both sample and gain
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

        self.s_dec1 = _decoder_block(32, 64)
        self.s_dec2 = _decoder_block(64, 64)
        self.s_dec3 = _decoder_block(64, 64)
        self.s_dec4 = _decoder_block(64, 64)
        self.s_dec5 = _decoder_block(64, 32)
        self.s_dec6 = _decoder_block(32, 16)
        self.s_final = _final_block(16, 1, activation="tanh")

        self.g_dec1 = _decoder_block(32, 64)
        self.g_dec2 = _decoder_block(64, 64)
        self.g_dec3 = _decoder_block(64, 64)
        self.g_dec4 = _decoder_block(64, 64)
        self.g_dec5 = _decoder_block(64, 32)
        self.g_dec6 = _decoder_block(32, 16)
        self.g_final = _final_block(16, 1, activation="sigmoid")

    def forward(self, s_in: torch.Tensor, g_in: torch.Tensor):
        """
        Args:
            s_in: [B, 1, T] sample stream (normalized to [-1,1])
            g_in: [B, 1, T] gain stream (normalized to [0,1])
        Returns:
            x_hat: [B, 1, T] reconstructed waveform (x_hat = s_hat * g_hat) normalized to [-1,1]
            s_hat: [B, 1, T] reconstructed sample stream ([-1,1])
            g_hat: [B, 1, T] reconstructed gain stream ([0,1])
        """
        se = self._encode_shared(s_in)
        ge = self._encode_shared(g_in)

        joint = torch.cat([se, ge], dim=1)
        z = self.joint_bottleneck(joint)

        sd1 = self.s_dec1(z)
        sd2 = self.s_dec2(sd1)
        sd3 = self.s_dec3(sd2)
        sd4 = self.s_dec4(sd3)
        sd5 = self.s_dec5(sd4)
        sd6 = self.s_dec6(sd5)
        s_hat = self.s_final(sd6)

        gd1 = self.g_dec1(z)
        gd2 = self.g_dec2(gd1)
        gd3 = self.g_dec3(gd2)
        gd4 = self.g_dec4(gd3)
        gd5 = self.g_dec5(gd4)
        gd6 = self.g_dec6(gd5)
        g_hat = self.g_final(gd6)

        x_hat = s_hat * g_hat

        return x_hat, s_hat, g_hat

    def _encode_shared(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.enc6(x)
        return x


if __name__ == "__main__":
    model = DualStreamCAESharedEncoder()
    s_in = torch.randn(8, 1, 32000)
    g_in = torch.randn(8, 1, 32000)

    x_hat, s_hat, g_hat = model(s_in, g_in)
    print(x_hat.shape)
    print(s_hat.shape)
    print(g_hat.shape)
