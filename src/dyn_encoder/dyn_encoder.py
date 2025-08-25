import torch
import torch.nn.functional as F


def dyn_encode_per_sample(x: torch.Tensor):
    ax = torch.abs(x)
    g = torch.ceil(63.0 * ax).clamp(0, 63).to(torch.uint8)

    gain_scale = torch.where(
        g == 0,
        torch.tensor(1.0, dtype=torch.float32, device=x.device),
        g.to(torch.float32) / 63.0,
    )

    s_float = torch.where(
        g == 0,
        torch.tensor(0.0, dtype=torch.float32, device=x.device),
        (x / gain_scale) * 128.0,
    )
    s = torch.round(s_float).to(torch.int32)
    s = torch.clamp(s, -128, 127).to(torch.int8)
    return s, g


def dyn_decode_per_sample(s: torch.Tensor, g: torch.Tensor):
    gain_scale = g.to(torch.float32) / 63.0
    return (s.to(torch.float32) / 128.0) * gain_scale


def dyn_encode_block(x: torch.Tensor, block_size: int = 64, keep_shape: bool = True):
    *prefix, L = x.shape
    if block_size <= 0:
        raise ValueError("block_size must be ≥ 1")

    N = int(torch.tensor(prefix).prod()) if prefix else 1
    flat = x.reshape(N, L)
    pad = (block_size - (L % block_size)) % block_size
    if pad:
        flat = F.pad(flat, (0, pad))
    L_pad = flat.size(1)
    n_blocks = L_pad // block_size

    blocks = flat.view(N, n_blocks, block_size)

    max_abs = blocks.abs().amax(dim=2)
    g_blocks = torch.clamp(torch.ceil(63.0 * max_abs), 0, 63).to(torch.int64)

    gain_scale = torch.where(
        g_blocks == 0, torch.ones_like(max_abs), g_blocks.to(torch.float32) / 63.0
    )

    s_blocks = torch.round((blocks / gain_scale.unsqueeze(-1)) * 128.0).to(torch.int32)
    s_blocks = torch.clamp(s_blocks, -128, 127).to(torch.int8)

    s_int = s_blocks.view(N, L_pad)[:, :L].reshape(*prefix, L)

    if keep_shape:
        g_rep = g_blocks.repeat_interleave(block_size, dim=1)[:, :L].to(torch.uint8)
        g_int = g_rep.reshape(*prefix, L)
    else:
        g_int = g_blocks.to(torch.uint8).reshape(*prefix, n_blocks)

    return s_int, g_int


def dyn_decode_block(
    s_int: torch.Tensor,
    g_int: torch.Tensor,
    block_size: int = 64,
    keep_shape: bool = True,
):
    s = s_int.to(torch.float32) / 128.0

    if keep_shape:
        gain_scale = g_int.to(torch.float32) / 63.0
    else:
        *_, L = s.shape
        g_blocks = g_int.to(torch.float32) / 63.0
        gain_scale = torch.repeat_interleave(g_blocks, block_size, dim=-1)[..., :L]

    return s * gain_scale
