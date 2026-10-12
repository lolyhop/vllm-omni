# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-patch graph layout and delayed API emission contracts."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.voxcpm2.model_state import VoxCPM2ModelState
from vllm_omni.model_executor.models.voxcpm2.per_patch_audio import PerPatchPCMOutput, pack_patch_context
from vllm_omni.worker_v2.streaming_audio import AudioRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("lengths", [[0, 2, 4], [4, 4, 4]])
def test_patch_context_right_padding_and_tail(lengths):
    context = torch.arange(24.0).reshape(3, 4, 2)
    patches = torch.arange(12.0).reshape(3, 2, 2) + 100
    packed, tails, next_lengths = pack_patch_context(context, torch.tensor(lengths), patches)
    for row, length in enumerate(lengths):
        valid = torch.cat([context[row, :length], patches[row]])
        expected = torch.nn.functional.pad(valid.T, (0, 6 - len(valid)))
        torch.testing.assert_close(packed[row], expected)
        n = min(4, len(valid))
        torch.testing.assert_close(tails[row, :n], valid[-n:])
        assert not tails[row, n:].any()
        assert next_lengths[row] == n


def test_per_patch_pcm_accumulates_until_interval_or_tail():
    request = AudioRequest(samples_per_frame=2, chunk_frames=3, emitted=True)

    def output(value, ended=False, valid=True):
        return PerPatchPCMOutput(
            torch.full((1, 2), float(value)),
            torch.tensor([valid]),
            torch.tensor([ended]),
            torch.tensor(48000),
            [request],
        ).get_output()[0]

    assert output(1) is None
    assert output(2) is None
    assert output(3)["model_outputs"].tolist() == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    assert output(4, ended=True)["model_outputs"].tolist() == [4.0, 4.0]
    assert output(5, valid=False) is None  # Already queued execution cannot emit again.


def test_per_patch_copy_records_completion_after_pcm_and_metadata():
    events = []
    output = PerPatchPCMOutput(
        torch.ones(2, 4),
        torch.tensor([True, False]),
        torch.tensor([False, True]),
        torch.tensor(48000),
        [],
        SimpleNamespace(record=lambda stream: events.append("event")),
    )

    def copy(value):
        events.append(value.shape)
        return value.clone()

    output.to_cpu(None, copy)
    assert events == [torch.Size([2, 4]), torch.Size([2]), torch.Size([2]), "event"]


def test_graph_vae_updates_only_active_context_and_keeps_last_patch():
    owner = object.__new__(VoxCPM2ModelState)
    owner.graph_context = torch.zeros(3, 4, 2)
    owner.graph_context[1].fill_(9)
    owner.graph_context_lengths = torch.tensor([0, 4, 0])
    owner.graph_active = torch.tensor([True, False, False])
    owner.model = SimpleNamespace(
        _patch_size=2,
        _get_vae_decode_sr_cond=lambda _: None,
        tts=SimpleNamespace(
            audio_vae=SimpleNamespace(
                decode_chunk_size=3, decode=lambda x, **kwargs: x.sum(1).repeat_interleave(3, dim=1)
            )
        ),
    )
    pcm, valid = owner.decode_patch_in_graph(
        torch.tensor([0, 1, 2]), torch.tensor([True, True, False]), torch.ones(3, 2, 2)
    )
    assert pcm.shape == (3, 6)
    assert valid.tolist() == [True, False, False]
    assert pcm[0].tolist() == [2.0] * 6
    assert owner.graph_context_lengths.tolist() == [2, 4, 0]
    assert owner.graph_context[1].tolist() == [[9.0, 9.0]] * 4
    assert not owner.graph_context[2].any()


@pytest.mark.parametrize("length_limit", [2, 100])
def test_post_sample_stops_only_valid_patch_rows(monkeypatch, length_limit):
    import numpy as np

    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

    owner = object.__new__(VoxCPM2ModelState)
    owner.per_patch_audio = True
    owner.graph_active = torch.tensor([True, True, True, False])
    owner.slots = [SimpleNamespace(is_stopping=False) for _ in range(3)]
    owner._length_limits = np.array([length_limit, 100, 100])
    owner._step_pcm = (torch.zeros(3, 2), torch.tensor([True, True, False]), None)
    owner.model = SimpleNamespace(
        _audio_emit_every=1,
        _audio_length_limit_reached=lambda computed, scheduled, limits: computed + scheduled + 1 >= limits,
    )
    batch = SimpleNamespace(
        num_reqs=3,
        idx_mapping=torch.tensor([2, 0, 1]),
        idx_mapping_np=np.array([2, 0, 1]),
        num_computed_tokens_np=np.zeros(3),
        num_scheduled_tokens=[1, 1, 1],
    )
    monkeypatch.setattr(OmniModelState, "run_eager_mtp", lambda *_args: None)
    # CPU reference test: bypass pinning, not a CUDA synchronization test.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    owner.run_eager_mtp(batch, None, torch.tensor([[1], [0], [1]]), {})
    assert owner.graph_active.tolist() == [length_limit != 2, True, False, False]
    assert owner._step_ended.tolist() == [True, length_limit == 2, False]


def test_graph_commit_preserves_inactive_request_state():
    owner = object.__new__(VoxCPM2ModelState)
    owner.next_embed = torch.full((3, 4), 7.0)
    owner.prefix_feat = torch.full((3, 2, 2), 9.0)
    owner.stop_logits = torch.zeros(3, 2)
    owner.commit_patch_in_graph(
        torch.tensor([0, 1, 2]),
        torch.tensor([True, False, False]),
        torch.ones(3, 4),
        torch.ones(3, 2, 2),
        torch.full((3, 2), 3.0),
    )
    assert owner.next_embed[:, 0].tolist() == [1.0, 7.0, 7.0]
    assert owner.prefix_feat[:, 0, 0].tolist() == [1.0, 9.0, 9.0]
    assert owner.stop_logits.tolist() == [[3.0, 3.0]] * 3
