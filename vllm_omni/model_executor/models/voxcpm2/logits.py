# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Materialize MRV2 stop logits directly from request slots."""

from vllm.triton_utils import tl, triton


@triton.jit
def write_stop_logits(
    stop_logits,
    metadata,
    output,
    stop_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    block: tl.constexpr,
    active,
    use_active: tl.constexpr,
    force_argmax_stop: tl.constexpr,
):
    row = tl.program_id(0)
    tokens = tl.program_id(1) * block + tl.arange(0, block)
    slot = tl.load(metadata + row * 2)
    mode = tl.load(metadata + row * 2 + 1)
    if use_active:
        alive = tl.load(active + slot, mask=slot >= 0, other=True)
        if force_argmax_stop:
            stop0 = tl.load(stop_logits + slot * stop_stride, mask=(slot >= 0) & (mode == 1), other=0)
            stop1 = tl.load(stop_logits + slot * stop_stride + 1, mask=(slot >= 0) & (mode == 1), other=0)
            mode = tl.where((mode == 1) & (stop1 > stop0), 2, mode)
        mode = tl.where((slot >= 0) & ~alive, 2, mode)
    # -1: unfilled row, 0: default, 1: model logits, 2: stopping, 3: prefill done.
    model_value = tl.load(
        stop_logits + slot * stop_stride + tokens,
        mask=(mode == 1) & (tokens < 2),
        other=0,
    )
    value = tl.full((block,), float("-inf"), tl.float32)
    value = tl.where((mode == 1) & (tokens < 2), model_value, value)
    value = tl.where((mode == 0) & (tokens == 0), 1.0, value)
    value = tl.where((mode == 2) & (tokens == 0), 0.0, value)
    value = tl.where(((mode == 2) | (mode == 3)) & (tokens == 1), 1.0, value)
    tl.store(output + row * vocab_size + tokens, value, mask=tokens < vocab_size)
