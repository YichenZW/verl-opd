# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")

from verl.workers.rollout.vllm_rollout import utils as vllm_rollout_utils
from verl.workers.rollout.vllm_rollout.utils import (  # noqa: E402
    _compute_prompt_logprobs_for_token_ids,
    _get_vllm_logprobs_mode,
    _patch_v2_prompt_logprobs_worker,
)


class _DummyPromptLogprobsWorker:
    def __init__(self):
        self.logprobs_mode = "raw_logprobs"
        self.uses_prompt_logprobs = np.zeros(1, dtype=bool)
        self.num_prompt_logprobs = np.zeros(1, dtype=np.int32)
        self.in_progress_prompt_logprobs = {}
        self.original_arg_lengths = []

    def add_request(self, req_id, req_idx, sampling_params):
        self.uses_prompt_logprobs[req_idx] = sampling_params.prompt_logprobs is not None
        self.num_prompt_logprobs[req_idx] = sampling_params.prompt_logprobs or 0
        if sampling_params.prompt_logprobs is not None:
            self.in_progress_prompt_logprobs[req_id] = []

    def remove_request(self, req_id):
        self.in_progress_prompt_logprobs.pop(req_id, None)

    def compute_prompt_logprobs(self, *args):
        self.original_arg_lengths.append(len(args))
        return {"arg_length": len(args)}


def _make_prompt_logprobs_input_batch():
    return SimpleNamespace(
        idx_mapping_np=np.array([0], dtype=np.int32),
        req_ids=["req-0"],
        prefill_len_np=np.array([4], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0], dtype=np.int32),
    )


def test_targeted_prompt_logprobs_patch_accepts_vllm_026_decode_signature():
    worker = _DummyPromptLogprobsWorker()
    _patch_v2_prompt_logprobs_worker(worker)
    worker.add_request("req-0", 0, SimpleNamespace(prompt_logprobs=None, extra_args=None))

    result = worker.compute_prompt_logprobs(
        None,
        None,
        _make_prompt_logprobs_input_batch(),
        None,
        None,
        np.array([4], dtype=np.int32),
    )

    assert result == {}
    assert worker.original_arg_lengths == []


def test_targeted_prompt_logprobs_patch_preserves_vllm_026_regular_prompt_logprobs():
    worker = _DummyPromptLogprobsWorker()
    _patch_v2_prompt_logprobs_worker(worker)
    worker.add_request("req-0", 0, SimpleNamespace(prompt_logprobs=2, extra_args=None))

    result = worker.compute_prompt_logprobs(
        None,
        None,
        _make_prompt_logprobs_input_batch(),
        None,
        None,
        np.array([4], dtype=np.int32),
    )

    assert result == {"arg_length": 6}
    assert worker.original_arg_lengths == [6]


def test_targeted_prompt_logprobs_patch_preserves_vllm_022_regular_prompt_logprobs():
    worker = _DummyPromptLogprobsWorker()
    _patch_v2_prompt_logprobs_worker(worker)
    worker.add_request("req-0", 0, SimpleNamespace(prompt_logprobs=2, extra_args=None))

    result = worker.compute_prompt_logprobs(
        None,
        None,
        SimpleNamespace(idx_mapping_np=np.array([0], dtype=np.int32), req_ids=["req-0"]),
        None,
        None,
        np.array([4], dtype=np.int32),
        np.array([4], dtype=np.int32),
        np.array([0], dtype=np.int32),
    )

    assert result == {"arg_length": 8}
    assert worker.original_arg_lengths == [8]


def test_targeted_prompt_logprobs_patch_computes_targeted_vllm_026_batch(monkeypatch):
    target_ids_a = torch.tensor([[11, 12], [13, 14], [15, 16], [17, 18]], dtype=torch.int64)
    target_ids_b = torch.tensor([[21, 22], [23, 24], [25, 26], [27, 28], [29, 30]], dtype=torch.int64)
    captured = {}

    def fake_get_prompt_logprobs_token_ids(num_tokens, *args):
        assert num_tokens == 9
        return torch.arange(100, 109, dtype=torch.int64)

    def fake_compute_prompt_logprobs_for_token_ids(
        prompt_token_ids,
        target_token_ids,
        prompt_hidden_states,
        logits_fn,
        num_prompt_logprobs,
        logprobs_mode,
    ):
        captured["target_token_ids"] = target_token_ids.clone()
        captured["logprobs_mode"] = logprobs_mode
        assert prompt_hidden_states.shape == (9, 1)
        assert num_prompt_logprobs == 2
        token_ids = torch.cat((prompt_token_ids.unsqueeze(-1), target_token_ids), dim=-1)
        logprobs = token_ids.to(torch.float32) * -0.01
        ranks = torch.full((token_ids.shape[0],), 3, dtype=torch.int64)
        return token_ids, logprobs, ranks

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.sample.prompt_logprob.get_prompt_logprobs_token_ids",
        fake_get_prompt_logprobs_token_ids,
    )
    monkeypatch.setattr(
        vllm_rollout_utils,
        "_compute_prompt_logprobs_for_token_ids",
        fake_compute_prompt_logprobs_for_token_ids,
    )

    worker = _DummyPromptLogprobsWorker()
    worker.uses_prompt_logprobs = np.zeros(2, dtype=bool)
    worker.num_prompt_logprobs = np.zeros(2, dtype=np.int32)
    _patch_v2_prompt_logprobs_worker(worker)
    worker.add_request(
        "req-a",
        1,
        SimpleNamespace(
            prompt_logprobs=2,
            extra_args={"prompt_logprob_token_ids": target_ids_a.tolist()},
        ),
    )
    worker.add_request(
        "req-b",
        0,
        SimpleNamespace(
            prompt_logprobs=2,
            extra_args={"prompt_logprob_token_ids": target_ids_b.tolist()},
        ),
    )

    input_batch = SimpleNamespace(
        idx_mapping_np=np.array([1, 0], dtype=np.int32),
        idx_mapping=torch.tensor([1, 0], dtype=torch.int32),
        req_ids=["req-a", "req-b"],
        prefill_len_np=np.array([4, 5], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0, 0], dtype=np.int32),
        num_tokens=9,
        num_scheduled_tokens=np.array([4, 5], dtype=np.int32),
        query_start_loc=torch.tensor([0, 4, 9], dtype=torch.int32),
        query_start_loc_np=np.array([0, 4, 9], dtype=np.int32),
    )

    result = worker.compute_prompt_logprobs(
        lambda hidden_states: hidden_states,
        torch.zeros((9, 1)),
        input_batch,
        torch.zeros((2, 10), dtype=torch.int64),
        torch.zeros(2, dtype=torch.int64),
        np.array([5, 4], dtype=np.int32),
    )

    assert captured["target_token_ids"].tolist() == (target_ids_a.tolist() + target_ids_b.tolist())
    assert captured["logprobs_mode"] == "raw_logprobs"
    assert set(result) == {"req-a", "req-b"}
    assert result["req-a"].logprob_token_ids[:, 1:].tolist() == target_ids_a[:-1].tolist()
    assert result["req-b"].logprob_token_ids[:, 1:].tolist() == target_ids_b[:-1].tolist()
    assert worker.original_arg_lengths == []
    assert worker._verl_prompt_logprob_token_ids == {}


def test_targeted_prompt_logprobs_helper_respects_raw_logits_mode():
    prompt_token_ids = torch.tensor([1, 2], dtype=torch.int64)
    target_token_ids = torch.tensor([[0, 2], [1, 3]], dtype=torch.int64)
    hidden_states = torch.zeros((2, 1))

    def logits_fn(_hidden_states):
        return torch.tensor([[0.1, 0.2, 0.3, 0.4], [1.0, 2.0, 3.0, 4.0]])

    token_ids, scores, ranks = _compute_prompt_logprobs_for_token_ids(
        prompt_token_ids,
        target_token_ids,
        hidden_states,
        logits_fn,
        num_prompt_logprobs=2,
        logprobs_mode="raw_logits",
    )

    assert token_ids.tolist() == [[1, 0, 2], [2, 1, 3]]
    torch.testing.assert_close(scores, torch.tensor([[0.2, 0.1, 0.3], [3.0, 2.0, 4.0]]))
    assert ranks.tolist() == [3, 3]


def test_vllm_logprobs_mode_prefers_sampler_then_model_config():
    assert _get_vllm_logprobs_mode(SimpleNamespace(logprobs_mode="processed_logits")) == "processed_logits"
    assert _get_vllm_logprobs_mode(
        SimpleNamespace(sampler=SimpleNamespace(logprobs_mode="raw_logits"), model_config=SimpleNamespace())
    ) == "raw_logits"
    assert _get_vllm_logprobs_mode(
        SimpleNamespace(sampler=SimpleNamespace(), model_config=SimpleNamespace(logprobs_mode="processed_logits"))
    ) == "processed_logits"
    assert _get_vllm_logprobs_mode(SimpleNamespace()) == "raw_logprobs"
