# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fixed-capacity causal VAE inputs and asynchronous per-patch PCM output."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm_omni.worker_v2.streaming_audio import AudioRequest


def pack_patch_context(context: torch.Tensor, lengths: torch.Tensor, patch: torch.Tensor):
    """Right-pad after valid context + patch; never cache the padding."""
    batch, capacity, dim = context.shape
    patch_frames = patch.shape[1]
    frames = torch.arange(capacity + patch_frames, device=patch.device)[None, :]
    old = context.gather(1, frames.clamp(max=capacity - 1).expand(batch, -1)[..., None].expand(-1, -1, dim))
    indices = (frames - lengths[:, None]).clamp(0, patch_frames - 1)
    new = patch.gather(1, indices[..., None].expand(-1, -1, dim))
    valid_lengths = lengths + patch_frames
    combined = torch.where((frames < lengths[:, None])[..., None], old, new)
    combined = combined.masked_fill((frames >= valid_lengths[:, None])[..., None], 0)
    next_lengths = valid_lengths.clamp(max=capacity)
    tail_indices = valid_lengths[:, None] - next_lengths[:, None] + torch.arange(capacity, device=patch.device)[None, :]
    tail = combined.gather(1, tail_indices.clamp(max=capacity + patch_frames - 1)[..., None].expand(-1, -1, dim))
    tail = tail.masked_fill(
        (torch.arange(capacity, device=patch.device)[None, :] >= next_lengths[:, None])[..., None], 0
    )
    return combined.transpose(1, 2).contiguous(), tail, next_lengths


@dataclass
class PerPatchPCMOutput:
    wav: torch.Tensor
    valid: torch.Tensor
    ended: torch.Tensor
    sample_rate: torch.Tensor
    requests: list[AudioRequest]
    copy_done: torch.cuda.Event | None = None

    def to_cpu(self, copy_stream, copy_tensor: Callable[[torch.Tensor], torch.Tensor]):
        output = PerPatchPCMOutput(
            copy_tensor(self.wav),
            copy_tensor(self.valid),
            copy_tensor(self.ended),
            self.sample_rate,
            self.requests,
        )
        if self.copy_done is not None:
            self.copy_done.record(copy_stream)
        return output

    def get_output(self):
        # The runner has already waited for its existing async output event.
        valid, ended = self.valid.tolist(), self.ended.tolist()
        output = []
        for row, request in enumerate(self.requests):
            pcm = self.wav[row].reshape(-1) if valid[row] else self.wav.new_empty(0)
            chunk = request.push(pcm, bool(ended[row]))
            output.append({"model_outputs": chunk, "sr": self.sample_rate} if chunk is not None else None)
        return output
