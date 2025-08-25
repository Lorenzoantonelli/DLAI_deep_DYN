import torch.nn as nn

from .model_config import KERNEL_SIZE, STRIDE, PADDING, OUTPUT_PADDING, LEAKY_RELU_SLOPE


def _encoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=PADDING,
        ),
        nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=True),
    )


def _decoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=PADDING,
            output_padding=OUTPUT_PADDING,
        ),
        nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=True),
    )


def _final_block(
    in_channels: int, out_channels: int, activation: str = "tanh"
) -> nn.Sequential:
    if activation == "tanh":
        act = nn.Tanh()
    elif activation == "sigmoid":
        act = nn.Sigmoid()
    else:
        raise ValueError("activation must be 'tanh' or 'sigmoid'")
    return nn.Sequential(
        nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=PADDING,
            output_padding=OUTPUT_PADDING,
        ),
        act,
    )
