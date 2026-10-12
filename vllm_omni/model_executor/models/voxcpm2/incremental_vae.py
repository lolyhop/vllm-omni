# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-slot feature caches for the native causal AudioVAE decoder."""

from contextlib import contextmanager

import torch
from torch import nn


@contextmanager
def incremental_vae_context(vae, caches, slots, valid, num_slots):
    """Replace causal padding with cached inputs, computing only new outputs.

    Cache storage stays fixed across graph replays. Overrides are installed only
    while decoding/capturing; ordinary full-sequence decode remains available.
    """
    originals = []

    def cached_input(name, x, history):
        cache = caches.get(name)
        if cache is None:
            cache = x.new_zeros(num_slots, x.shape[1], history)
            caches[name] = cache
        old = cache.index_select(0, slots)
        joined = torch.cat([old, x], dim=-1)
        cache.index_copy_(0, slots, torch.where(valid[:, None, None], joined[..., -history:], old))
        return joined

    try:
        for name, module in vae.decoder.named_modules():
            if type(module).__name__ == "NoiseBlock":
                raise ValueError("Incremental VoxCPM2 VAE does not support stochastic NoiseBlock")
            if isinstance(module, nn.ConvTranspose1d):
                if type(module).__name__ != "CausalTransposeConv1d":
                    raise ValueError("Incremental VoxCPM2 VAE requires causal transpose convolutions")
                history = (module.kernel_size[0] - 1) // module.stride[0]
                trim = module._CausalTransposeConv1d__padding * 2 - module._CausalTransposeConv1d__output_padding
                if not history:
                    continue

                def forward(x, mod=module, key=name, size=history, right=trim):
                    joined = cached_input(key, x, size)
                    out = nn.ConvTranspose1d.forward(mod, joined)
                    left = size * mod.stride[0]
                    return out[..., left:-right] if right else out[..., left:]

            elif isinstance(module, nn.Conv1d):
                if type(module).__name__ != "CausalConv1d":
                    raise ValueError("Incremental VoxCPM2 VAE requires causal convolutions")
                if module.stride[0] != 1:
                    raise ValueError("Incremental VoxCPM2 VAE requires stride-one decoder convolutions")
                history = module._CausalConv1d__padding * 2 - module._CausalConv1d__output_padding
                if not history:
                    continue

                def forward(x, mod=module, key=name, size=history):
                    return nn.Conv1d.forward(mod, cached_input(key, x, size))

            else:
                continue
            originals.append((module, module.forward))
            module.forward = forward
        yield
    finally:
        for module, original in reversed(originals):
            module.forward = original


def incremental_vae_decode(vae, patch, caches, slots, valid, num_slots, sr_cond):
    # Bypass an existing full-sequence compile wrapper: this path is captured by
    # the model-owned graph, including all per-layer feature-cache operations.
    decode = getattr(vae.decode, "_torchdynamo_orig_callable", vae.decode)
    with incremental_vae_context(vae, caches, slots, valid, num_slots):
        return decode(patch, sr_cond=sr_cond)
