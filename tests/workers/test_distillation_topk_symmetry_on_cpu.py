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
"""Regression guard for verl#6293.

The use_remove_padding=False branch of
FSDPEngineWithLMHead.prepare_model_outputs previously lacked the
distillation_use_topk handling that the use_remove_padding=True branch had,
so distillation outputs were silently dropped from model_output and the
downstream loss raised KeyError. This test invokes prepare_model_outputs on
a stub engine for both branches with distillation_use_topk=True and asserts
the distillation keys produced by logits_processor_func are propagated into
model_output as nested tensors in both cases.

``logprobs_from_logits`` is patched out: in CI environments where flash-attn
is installed, it dispatches to a Triton CrossEntropyLoss kernel that cannot
operate on CPU tensors. The substitute returns a dummy ``log_probs`` tensor
of the right shape, which is sufficient for this test — the contract under
test is the propagation of distillation keys, not the numerical correctness
of log-prob computation.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from verl.experimental.teacher_loop.teacher_manager import _get_teacher_sampling_params
from verl.trainer.distillation.fsdp.losses import compute_forward_kl_topk as compute_fsdp_forward_kl_topk
from verl.trainer.distillation.fsdp.losses import (
    compute_reverse_kl_topk as compute_fsdp_reverse_kl_topk,
)
from verl.trainer.distillation.losses import compute_forward_kl_topk as collect_forward_kl_topk_metrics
from verl.trainer.distillation.losses import compute_topk_loss
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.config.distillation import DistillationTeacherModelConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

_VOCAB_SIZE = 8
_DISTILLATION_KEYS = ("distillation_losses", "student_mass", "overlap_count", "overlap_token_advantage")


def _make_engine_stub():
    """Bypass FSDPEngineWithLMHead.__init__; set only attributes that
    prepare_model_outputs touches in this test path (no SP, no fused kernels,
    no entropy)."""
    eng = object.__new__(FSDPEngineWithLMHead)
    eng.use_ulysses_sp = False

    class _EngineCfg:
        entropy_checkpointing = False

    eng.engine_config = _EngineCfg()
    return eng


def _make_logits_processor(keys):
    """Fake top-k distillation processor: returns one (1, total_nnz) tensor per key.

    The real processor (verl/trainer/distillation/losses.py) returns
    student_logits.shape[:2]; we mimic that contract.
    """

    def _proc(student_logits, data):
        n = student_logits.shape[1]
        return {k: torch.full((1, n), float(i + 1)) for i, k in enumerate(keys)}

    return _proc


@pytest.mark.parametrize("use_remove_padding", [True, False])
@pytest.mark.parametrize("distillation_only", [False, True])
def test_distillation_outputs_emitted_in_both_padding_modes(use_remove_padding, distillation_only):
    """distillation_use_topk=True must populate distillation outputs into
    model_output regardless of use_remove_padding. See verl#6293.

    When distillation_only=True, log_probs must be omitted (supervised top-k path)."""
    bsz = 2
    seq_lengths_list = [3, 2]
    seq_lengths = torch.tensor(seq_lengths_list, dtype=torch.int64)
    total_nnz = int(seq_lengths.sum())

    cu_seqlens = torch.cat([torch.tensor([0]), seq_lengths.cumsum(0)]).to(torch.int64)

    flat_input_ids = torch.randint(0, _VOCAB_SIZE, (total_nnz,))
    input_ids_nested = torch.nested.nested_tensor_from_jagged(flat_input_ids, offsets=cu_seqlens)

    input_ids_rmpad_rolled = torch.randint(0, _VOCAB_SIZE, (total_nnz,))

    class _Output:
        pass

    output = _Output()

    if use_remove_padding:
        # True branch: output.logits shape (1, total_nnz, V), squeeze(0) -> (total_nnz, V).
        output.logits = torch.randn(1, total_nnz, _VOCAB_SIZE)
        output_args = {
            "input_ids_rmpad_rolled": input_ids_rmpad_rolled,
            "temperature_rmpad": torch.ones(total_nnz),
            # No SP and no static pad_to_length here, so nothing to trim off the packed tail.
            "pad_size": 0,
        }
    else:
        # False branch: output.logits shape (bsz, max_seqlen, V).
        max_seqlen = max(seq_lengths_list)
        output.logits = torch.randn(bsz, max_seqlen, _VOCAB_SIZE)
        output_args = {
            "input_ids_rmpad_rolled": input_ids_rmpad_rolled,
            "temperature": torch.ones(bsz),
        }

    micro_batch = TensorDict({"input_ids": input_ids_nested}, batch_size=[])
    tu.assign_non_tensor(
        micro_batch,
        use_remove_padding=use_remove_padding,
        pad_mode=DatasetPadMode.NO_PADDING,
        use_fused_kernels=False,
        calculate_entropy=False,
        calculate_sum_pi_squared=False,
        distillation_use_topk=True,
        distillation_only=distillation_only,
        max_response_length=max(seq_lengths_list),
    )

    eng = _make_engine_stub()

    # Patch logprobs_from_logits because flash-attn's Triton CrossEntropyLoss
    # cannot operate on CPU tensors. The shape is what downstream code asserts
    # against (v.shape == log_probs.shape), and prepare_model_outputs reduces
    # both branches to a (total_nnz,) log_probs over the rmpad'ed logits.
    with patch(
        "verl.workers.engine.fsdp.transformer_impl.logprobs_from_logits",
        return_value=torch.zeros(total_nnz),
    ):
        model_output = FSDPEngineWithLMHead.prepare_model_outputs(
            eng,
            output=output,
            output_args=output_args,
            micro_batch=micro_batch,
            logits_processor_func=_make_logits_processor(_DISTILLATION_KEYS),
        )

    if distillation_only:
        assert "log_probs" not in model_output, (
            f"log_probs should be omitted when distillation_only=True "
            f"(use_remove_padding={use_remove_padding}); keys: {list(model_output.keys())}"
        )
    else:
        assert "log_probs" in model_output, (
            f"log_probs missing (use_remove_padding={use_remove_padding}); keys: {list(model_output.keys())}"
        )

    for k in _DISTILLATION_KEYS:
        assert k in model_output, (
            f"Distillation key '{k}' missing from model_output "
            f"(use_remove_padding={use_remove_padding}); "
            f"keys: {list(model_output.keys())}"
        )
        assert model_output[k].is_nested, (
            f"Expected '{k}' to be a nested tensor (use_remove_padding={use_remove_padding}); "
            f"got {type(model_output[k])}"
        )


def _nested_from_rows(rows):
    values = torch.tensor(rows)
    offsets = torch.tensor([0, len(rows)], dtype=torch.int64)
    return torch.nested.nested_tensor_from_jagged(values, offsets=offsets)


def test_forward_kl_topk_emits_overlap_metrics():
    logits = torch.tensor(
        [
            [0.0, 9.0, 8.0, 1.0, 0.0, 0.0],
            [8.0, 7.0, 0.0, 0.0, 9.0, 0.0],
            [9.0, 8.0, 7.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    teacher_ids = _nested_from_rows([[1, 2], [4, 5], [3, 4]]).to(torch.int64)
    teacher_logprobs = _nested_from_rows(
        [
            [torch.log(torch.tensor(0.7)), torch.log(torch.tensor(0.2))],
            [torch.log(torch.tensor(0.6)), torch.log(torch.tensor(0.3))],
            [torch.log(torch.tensor(0.5)), torch.log(torch.tensor(0.4))],
        ]
    ).to(torch.float32)
    config = SimpleNamespace(distillation_loss=SimpleNamespace(log_prob_min_clamp=None))

    output = compute_fsdp_forward_kl_topk(
        student_logits=logits,
        teacher_topk_log_probs=teacher_logprobs,
        teacher_topk_ids=teacher_ids,
        config=config,
        data_format="thd",
    )

    torch.testing.assert_close(output["overlap_count"], torch.tensor([[2, 1, 0]]))

    student_log_probs = torch.log_softmax(logits, dim=-1)
    gathered_student = torch.gather(student_log_probs, dim=-1, index=teacher_ids.values().unsqueeze(0))
    teacher_log_probs = teacher_logprobs.values().unsqueeze(0)
    token_adv = -(teacher_log_probs.exp() * (teacher_log_probs - gathered_student))
    expected_ota = torch.tensor(
        [[token_adv[0, 0].mean(), token_adv[0, 1, 0], 0.0]],
        dtype=output["overlap_token_advantage"].dtype,
    )
    torch.testing.assert_close(output["overlap_token_advantage"], expected_ota)


def test_forward_kl_topk_metric_aggregation_for_overlap_outputs():
    data = TensorDict(
        {
            "prompts": torch.tensor([[101]]),
            "responses": torch.tensor([[11, 12, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "response_mask": torch.tensor([[1, 1, 0]], dtype=torch.bool),
        },
        batch_size=[1],
    )
    model_output = {
        "distillation_losses": torch.tensor([0.1, 0.2, 0.3]),
        "student_mass": torch.tensor([0.9, 0.8, 0.7]),
        "teacher_mass": torch.tensor([0.95, 0.85, 0.75]),
        "overlap_count": torch.tensor([2, 1, 0]),
        "overlap_token_advantage": torch.tensor([-0.2, -0.4, 0.0]),
    }
    distillation_config = SimpleNamespace(distillation_loss=SimpleNamespace(topk=2))

    _, metrics = collect_forward_kl_topk_metrics(
        config=SimpleNamespace(),
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )

    assert metrics["distillation/overlap_ratio"] == pytest.approx(0.75)
    assert metrics["distillation/overlap_token_advantage"] == pytest.approx(-0.3)


def test_reverse_kl_topk_uses_student_topk_and_full_teacher_logprobs():
    logits = torch.tensor([[[3.0, 1.0, 2.0, 0.0]]], dtype=torch.float32)
    teacher_ids = _nested_from_rows([[2, 0, 3, 1]]).to(torch.int64)
    teacher_logprobs = _nested_from_rows(
        [
            [
                torch.log(torch.tensor(0.2)),
                torch.log(torch.tensor(0.5)),
                torch.log(torch.tensor(0.1)),
                torch.log(torch.tensor(0.2)),
            ]
        ]
    ).to(torch.float32)
    config = SimpleNamespace(distillation_loss=SimpleNamespace(topk=2, log_prob_min_clamp=None, use_chunked_topk=False))

    output = compute_fsdp_reverse_kl_topk(
        student_logits=logits,
        teacher_topk_log_probs=teacher_logprobs,
        teacher_topk_ids=teacher_ids,
        config=config,
        data_format="thd",
    )

    student_log_probs = torch.log_softmax(logits, dim=-1)
    student_topk_log_probs, student_topk_ids = torch.topk(student_log_probs, k=2, dim=-1)
    dense_teacher_logprobs = torch.empty_like(student_log_probs)
    dense_teacher_logprobs.scatter_(
        dim=-1, index=teacher_ids.values().unsqueeze(0), src=teacher_logprobs.values().unsqueeze(0)
    )
    teacher_on_student_topk = torch.gather(dense_teacher_logprobs, dim=-1, index=student_topk_ids)
    expected_loss = (student_topk_log_probs.exp() * (student_topk_log_probs - teacher_on_student_topk)).sum(dim=-1)

    torch.testing.assert_close(output["distillation_losses"], expected_loss)
    torch.testing.assert_close(output["student_mass"], student_topk_log_probs.exp().sum(dim=-1))
    torch.testing.assert_close(output["teacher_mass"], teacher_on_student_topk.exp().sum(dim=-1))


def test_reverse_kl_topk_accepts_targeted_teacher_logprobs():
    logits = torch.tensor([[[3.0, 1.0, 2.0, 0.0]]], dtype=torch.float32)
    config = SimpleNamespace(distillation_loss=SimpleNamespace(topk=2, log_prob_min_clamp=None, use_chunked_topk=False))

    student_log_probs = torch.log_softmax(logits, dim=-1)
    student_topk_log_probs, student_topk_ids = torch.topk(student_log_probs, k=2, dim=-1)
    teacher_full_logprobs = torch.log(torch.tensor([[[0.2, 0.1, 0.5, 0.2]]], dtype=torch.float32))
    teacher_on_student_topk = torch.gather(teacher_full_logprobs, dim=-1, index=student_topk_ids)

    output = compute_fsdp_reverse_kl_topk(
        student_logits=logits,
        teacher_topk_log_probs=_nested_from_rows(teacher_on_student_topk.squeeze(0).tolist()).to(torch.float32),
        teacher_topk_ids=_nested_from_rows(student_topk_ids.squeeze(0).tolist()).to(torch.int64),
        config=config,
        data_format="thd",
    )

    expected_loss = (student_topk_log_probs.exp() * (student_topk_log_probs - teacher_on_student_topk)).sum(dim=-1)
    torch.testing.assert_close(output["distillation_losses"], expected_loss)
    torch.testing.assert_close(output["student_mass"], student_topk_log_probs.exp().sum(dim=-1))
    torch.testing.assert_close(output["teacher_mass"], teacher_on_student_topk.exp().sum(dim=-1))


def test_reverse_kl_topk_rejects_non_targeted_partial_teacher_payload():
    logits = torch.tensor([[[3.0, 1.0, 2.0, 0.0]]], dtype=torch.float32)
    teacher_ids = _nested_from_rows([[0, 2, 3]]).to(torch.int64)
    teacher_logprobs = _nested_from_rows(
        [[torch.log(torch.tensor(0.5)), torch.log(torch.tensor(0.2)), torch.log(torch.tensor(0.1))]]
    ).to(torch.float32)
    config = SimpleNamespace(distillation_loss=SimpleNamespace(topk=2, log_prob_min_clamp=None, use_chunked_topk=False))

    with pytest.raises(ValueError, match="requires either teacher log-probs on the student top-k support"):
        compute_fsdp_reverse_kl_topk(
            student_logits=logits,
            teacher_topk_log_probs=teacher_logprobs,
            teacher_topk_ids=teacher_ids,
            config=config,
            data_format="thd",
        )


def test_reverse_kl_topk_teacher_sampling_requires_target_ids():
    teacher_config = SimpleNamespace(inference=SimpleNamespace(temperature=1.0), get_vocab_size=lambda: 7)
    loss_config = SimpleNamespace(
        loss_mode="reverse_kl_topk",
        topk=2,
        loss_settings=SimpleNamespace(use_topk=True),
    )

    with pytest.raises(ValueError, match="student top-k target token ids"):
        _get_teacher_sampling_params(teacher_config, loss_config)


def test_reverse_kl_topk_teacher_sampling_can_request_targeted_ids():
    teacher_config = SimpleNamespace(inference=SimpleNamespace(temperature=1.0), get_vocab_size=lambda: 7)
    loss_config = SimpleNamespace(
        loss_mode="reverse_kl_topk",
        topk=2,
        loss_settings=SimpleNamespace(use_topk=True),
    )
    target_ids = torch.tensor([[1, 2], [3, 4], [0, 0]], dtype=torch.int32)

    sampling_params = _get_teacher_sampling_params(teacher_config, loss_config, target_token_ids=target_ids)

    assert sampling_params["prompt_logprobs"] == 2
    assert sampling_params["detokenize"] is False
    assert sampling_params["flat_logprobs"] is True
    assert sampling_params["extra_args"]["prompt_logprob_token_ids"] == target_ids.tolist()


def test_teacher_vocab_size_cache_is_mutable(monkeypatch):
    calls = []

    class _FakeAutoConfig:
        vocab_size = 13

    def _from_pretrained(model_path):
        calls.append(model_path)
        return _FakeAutoConfig()

    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", _from_pretrained)

    teacher_config = DistillationTeacherModelConfig(model_path="dummy-teacher")

    assert teacher_config.get_vocab_size() == 13
    assert teacher_config.get_vocab_size() == 13
    assert calls == ["dummy-teacher"]


def test_megatron_reverse_kl_topk_dispatches_named_backend(monkeypatch):
    calls = []

    def _forward_backend(**kwargs):
        calls.append("forward")
        raise AssertionError("forward_kl_topk backend should not be selected")

    def _reverse_backend(student_logits, **kwargs):
        calls.append("reverse")
        return {
            "distillation_losses": torch.zeros(student_logits.shape[:2]),
            "student_mass": torch.zeros(student_logits.shape[:2]),
            "teacher_mass": torch.zeros(student_logits.shape[:2]),
        }

    fake_megatron_losses = SimpleNamespace(
        compute_forward_kl_topk=_forward_backend,
        compute_reverse_kl_topk=_reverse_backend,
    )
    fake_megatron_package = SimpleNamespace(losses=fake_megatron_losses)
    monkeypatch.setitem(sys.modules, "verl.trainer.distillation.megatron", fake_megatron_package)
    monkeypatch.setitem(sys.modules, "verl.trainer.distillation.megatron.losses", fake_megatron_losses)

    student_logits = torch.zeros(1, 2, 4)
    data = TensorDict(
        {
            "teacher_logprobs": torch.zeros(1, 2, 4),
            "teacher_ids": torch.arange(4, dtype=torch.long).view(1, 1, 4).expand(1, 2, 4),
        },
        batch_size=[1, 2],
    )
    outputs = compute_topk_loss(
        config=SimpleNamespace(strategy="megatron"),
        distillation_config=SimpleNamespace(distillation_loss=SimpleNamespace(loss_mode="reverse_kl_topk")),
        data=data,
        student_logits=student_logits,
        data_format="thd",
    )

    assert calls == ["reverse"]
    assert outputs["distillation_losses"].shape == student_logits.shape[:2]
