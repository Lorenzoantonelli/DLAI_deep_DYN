import torch
import torch.nn as nn

from .helpers import _encoder_block, _decoder_block, _final_block


class BaselineCAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.enc1 = _encoder_block(1, 16)
        self.enc2 = _encoder_block(16, 32)
        self.enc3 = _encoder_block(32, 64)
        self.enc4 = _encoder_block(64, 64)
        self.enc5 = _encoder_block(64, 64)
        self.enc6 = _encoder_block(64, 64)

        self.bottleneck = _encoder_block(64, 32)

        self.dec1 = _decoder_block(32, 64)
        self.dec2 = _decoder_block(64, 64)
        self.dec3 = _decoder_block(64, 64)
        self.dec4 = _decoder_block(64, 64)
        self.dec5 = _decoder_block(64, 32)
        self.dec6 = _decoder_block(32, 16)

        self.final_layer = _final_block(16, 1, activation="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, T] waveform (normalized to [-1,1])
        Returns:
            output: [B, 1, T] reconstructed waveform (normalized to [-1,1])
        """
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)

        z = self.bottleneck(e6)

        d1 = self.dec1(z)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        d4 = self.dec4(d3)
        d5 = self.dec5(d4)
        d6 = self.dec6(d5)

        output = self.final_layer(d6)

        return output


if __name__ == "__main__":
    model = BaselineCAE()
    x = torch.randn(8, 1, 32000)
    output = model(x)
    print(output.shape)