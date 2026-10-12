# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed-to-request layout and admission contracts for mixed VoxCPM2 graphs."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.voxcpm2.scheduler import VoxCPM2OmniARAsyncScheduler
from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import VoxCPM2TalkerForConditionalGeneration, _RequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _model():
    model = VoxCPM2TalkerForConditionalGeneration.__new__(VoxCPM2TalkerForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model._enable_unified_decode_graph = True
    model._mrv2_model_state = None
    model.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(enable_chunked_prefill=False, max_num_batched_tokens=16)
    )
    return model


@pytest.mark.parametrize("chunked,uses_v2,expected", [(False, True, True), (True, True, False), (False, False, False)])
def test_mixed_graph_requires_v2_full_prefill(chunked, uses_v2, expected):
    model = _model()
    model._mrv2_model_state = object() if uses_v2 else None
    model.vllm_config.scheduler_config.enable_chunked_prefill = chunked
    assert model._mixed_unified_graph_enabled() is expected


def test_mixed_token_capacity_bounds_and_buckets():
    model = _model()
    assert model._mixed_token_capacity(5, 4) == 8
    assert model._mixed_token_capacity(1, 4) == 4
    assert model._mixed_token_capacity(16, 4) == 16
    assert model._mixed_token_capacity(17, 4) is None


def test_mixed_graph_packs_prefill_masks_and_decode_conditions(monkeypatch):
    model = _model()
    prefill = _RequestState(request_id="prefill")
    prefill.prefill_masks = (
        torch.tensor([[1, 0, 1]]),
        torch.tensor([[0, 1, 0]]),
        torch.arange(12.0).reshape(1, 3, 2, 2),
        torch.full((1, 3, 4), 5.0),
    )
    decode = _RequestState(request_id="decode")
    decode.prev_feat_embed = torch.full((1, 4), 7.0)
    decode.curr_prefix_feat_cond = torch.full((2, 2), 9.0)
    model._pending_requests = [(prefill, True, None, 3), (decode, False, None, 1)]
    graph = SimpleNamespace(
        batch_size=4,
        text_mask=torch.empty(8, 1),
        prev_feat_embed=torch.empty(8, 4),
        prefix_feat_cond=torch.empty(4, 2, 2),
        last_token_indices=torch.empty(4, dtype=torch.long),
    )
    model._prepare_mixed_graph_inputs(graph, 2)
    assert graph.text_mask[:, 0].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert graph.prev_feat_embed[:, 0].tolist() == [0.0, 5.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0]
    assert graph.last_token_indices.tolist() == [2, 3, 3, 3]
    torch.testing.assert_close(graph.prefix_feat_cond[0], prefill.prefill_masks[2][0, -1])
    assert graph.prefix_feat_cond[1].tolist() == [[9.0, 9.0]] * 2
    assert not graph.prefix_feat_cond[2:].any()


@pytest.mark.parametrize("uses_v2,chunked,defer", [(True, False, False), (True, True, True), (False, False, True)])
def test_mixed_admission_removes_only_graph_specific_restriction(monkeypatch, uses_v2, chunked, defer):
    from vllm.v1.request import RequestStatus

    scheduler = object.__new__(VoxCPM2OmniARAsyncScheduler)
    scheduler.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(use_v2_model_runner=uses_v2),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=chunked),
    )
    scheduler.waiting = [object()]
    scheduler.kv_holding_waiting = []
    scheduler.running = [SimpleNamespace(status=RequestStatus.RUNNING, is_finished=lambda: False, num_prompt_tokens=5)]
    monkeypatch.setattr(scheduler, "_unified_decode_graph_enabled", lambda: True)
    monkeypatch.setattr(scheduler, "_get_confirmed_num_computed_tokens", lambda _: 5)
    assert scheduler._should_defer_waiting_admission() is defer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mixed_graph_capture_replays_changed_packed_layout(monkeypatch):
    """Exercise actual CUDA capture; attention is stubbed, so this is not FA2 E2E."""
    from contextlib import nullcontext

    from vllm_omni.model_executor.models.voxcpm2 import voxcpm2_talker as module

    model = _model()
    model.config = SimpleNamespace(hidden_size=4)
    model._device = torch.device("cuda")
    model._side_dtype = torch.float32
    model._feat_dim = model._patch_size = 2
    model._inference_timesteps = 1
    model._cfg_value = model._cfg_cutoff_ratio = 1.0
    model._unified_graph_stats = SimpleNamespace(captures=0)

    class Backbone:
        def precompute_fused_qkv(self):
            pass

        def __call__(self, **kwargs):
            return kwargs["inputs_embeds"] * 2

    model.model = Backbone()
    model.residual_model = Backbone()
    projection = torch.nn.Linear(4, 2, device="cuda")
    tts = SimpleNamespace(
        lm_to_dit_proj=projection,
        res_to_dit_proj=projection,
        fsq_layer=lambda x: x + 10,
        fusion_concat_proj=lambda x: x[:, :4] + x[:, 4:],
        feat_decoder=SimpleNamespace(estimator=torch.nn.Identity()),
        feat_encoder=lambda x: x.mean(dim=2),
        enc_to_lm_proj=lambda x: torch.cat([x, x], dim=-1),
    )
    model._tts = tts
    model._dit_proj_fn = lambda lm, res: torch.cat([projection(lm), projection(res)], dim=-1)
    monkeypatch.setattr(model, "_setup_cfm_buffers", lambda: None)
    monkeypatch.setattr(model, "_setup_execution_optimizations", lambda: None)
    monkeypatch.setattr(model, "_voxcpm2_compile_unified_capture_estimator", lambda x: x)
    monkeypatch.setattr(model, "_voxcpm2_compile_unified_capture_feat_encoder", lambda x: x)
    monkeypatch.setattr(module, "_CFMBufferManager", lambda **kwargs: None)
    monkeypatch.setattr(module, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None))
    monkeypatch.setattr(module, "override_forward_context", lambda _: nullcontext())
    monkeypatch.setattr(
        module,
        "_optimized_solve_euler_with_noise",
        lambda decoder, mu, patch, cond, *args, **kwargs: mu[:, :2, None] + cond,
    )
    graph = model._capture_unified_decode_graph(4, 8)
    for order, length in (("prefill-first", 3), ("decode-first", 5)):
        prefill = _RequestState(request_id="prefill")
        prefill.prefill_masks = (
            torch.ones(1, length, device="cuda"),
            torch.zeros(1, length, device="cuda"),
            torch.ones(1, length, 2, 2, device="cuda"),
            torch.zeros(1, length, 4, device="cuda"),
        )
        decode = _RequestState(
            request_id="decode",
            prev_feat_embed=torch.ones(1, 4, device="cuda"),
            curr_prefix_feat_cond=torch.ones(2, 2, device="cuda"),
        )
        entries = [(prefill, True, None, length), (decode, False, None, 1)]
        if order == "decode-first":
            entries.reverse()
        model._mrv2_model_state = None  # Use request tensors for this layout-only test.
        model._pending_requests = entries
        model._prepare_mixed_graph_inputs(graph, 2)
        graph.input_embeds.copy_(torch.arange(32.0, device="cuda").reshape(8, 4))
        graph.graph.replay()
        scaffold_h = graph.input_embeds * 2
        expected = (scaffold_h + 10) * (1 - graph.text_mask) + scaffold_h * graph.text_mask
        expected = expected.index_select(0, graph.last_token_indices)
        torch.testing.assert_close(graph.lm_hidden, expected)
        assert graph.next_feat_embed.shape == (4, 4)


def test_mixed_metadata_request_uses_token_and_request_buckets():
    model = _model()
    model._mrv2_model_state = object()
    model._enable_cuda_graph = True
    model._tts = SimpleNamespace(feat_decoder=SimpleNamespace(estimator=SimpleNamespace(_compiled=True)))
    model._cuda_graph_warmup_steps = model._cuda_graph_warmup_threshold = 3
    model._max_cached_graphs = 8
    model._unified_decode_graph_max_batch_size = 4
    model._mixed_graphs = {}
    model._select_unified_graph_bucket_size = lambda count: 4
    request = model.get_runner_assisted_full_attention_metadata_request(
        req_ids=["prefill", "decode"],
        num_reqs=2,
        num_scheduled_tokens=[5, 1],
        num_computed_tokens=[0, 20],
        max_num_scheduled_tokens=5,
    )
    assert request.num_reqs_padded == 4
    assert request.num_tokens_padded == 8
    assert request.for_cudagraph_capture
    model._mixed_graphs[(4, 8)] = object()
    replay = model.get_runner_assisted_full_attention_metadata_request(
        req_ids=["a", "b"],
        num_reqs=2,
        num_scheduled_tokens=[1, 7],
        num_computed_tokens=[20, 0],
        max_num_scheduled_tokens=7,
    )
    assert not replay.for_cudagraph_capture
    model._cuda_graph_warmup_steps = 0
    assert (
        model.get_runner_assisted_full_attention_metadata_request(
            req_ids=["a", "b"],
            num_reqs=2,
            num_scheduled_tokens=[5, 1],
            num_computed_tokens=[0, 20],
            max_num_scheduled_tokens=5,
        )
        is None
    )


def test_all_decode_keeps_uniform_metadata_when_mixed_graph_enabled():
    model = _model()
    # Metadata selection needs slot readiness, not the GPU buffers themselves.
    model._mrv2_model_state = SimpleNamespace(tensor_for=lambda *_: torch.ones(1))
    model._enable_cuda_graph = True
    model._tts = SimpleNamespace(feat_decoder=SimpleNamespace(estimator=SimpleNamespace(_compiled=True)))
    model._cuda_graph_warmup_steps = model._cuda_graph_warmup_threshold = 3
    model._max_cached_graphs = 8
    model._unified_decode_graph_max_batch_size = 4
    model._mixed_graphs = {}
    model._unified_graphs = {}
    model._select_unified_graph_bucket_size = lambda count: 4
    model._active_states = {req_id: _RequestState(request_id=req_id, prefill_completed=True) for req_id in ("a", "b")}
    request = model.get_runner_assisted_full_attention_metadata_request(
        req_ids=["a", "b"],
        num_reqs=2,
        num_scheduled_tokens=[1, 1],
        num_computed_tokens=[20, 30],
        max_num_scheduled_tokens=1,
    )
    assert request.num_reqs_padded == 4
    assert request.num_tokens_padded is None
    assert request.for_cudagraph_capture
    model._unified_graphs[4] = object()
    replay = model.get_runner_assisted_full_attention_metadata_request(
        req_ids=["a", "b"],
        num_reqs=2,
        num_scheduled_tokens=[1, 1],
        num_computed_tokens=[20, 30],
        max_num_scheduled_tokens=1,
    )
    assert not replay.for_cudagraph_capture
