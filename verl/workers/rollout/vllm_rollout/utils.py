# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import ctypes
import dataclasses
import functools
import json
import logging
import os
import platform
import signal
import threading
from collections.abc import Callable, Mapping
from types import MethodType
from typing import Any, Literal, Optional, get_args

import numpy as np
import torch
from vllm.outputs import RequestOutput

from verl.utils.device import get_device_name, is_npu_available
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, resolve_weight_name
from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
from verl.utils.vllm.vllm_quant_utils import apply_vllm_quant_patches, is_fp8_model, load_quanted_weights
from verl.workers.rollout.vllm_rollout.weight_update_utils import apply_buffer_updates, split_buffer_updates

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# magic numbers that ensure we are using the same LoRA adapter during the rollout and training process
VLLM_LORA_INT_ID = 123
VLLM_LORA_NAME = "123"
VLLM_LORA_PATH = "simon_lora_path"

VLLM_ASCEND_REQUIRED_ENV_VARS = {"VLLM_ALL2ALL_BACKEND": "flashinfer_all2allv", "VLLM_ASCEND_ENABLE_NZ": "0"}
_PROMPT_LOGPROB_TOKEN_IDS_EXTRA_ARG = "prompt_logprob_token_ids"


def _resolve_vllm_weight_sync_local_rank(worker_local_rank: int, parallel_config: Any) -> int:
    worker_local_rank = int(worker_local_rank)
    if parallel_config is None:
        return worker_local_rank

    tp_size = max(int(getattr(parallel_config, "tensor_parallel_size", 1) or 1), 1)
    dp_size = int(getattr(parallel_config, "data_parallel_size", 1) or 1)
    dp_local_size = int(getattr(parallel_config, "data_parallel_size_local", 1) or 1)
    if dp_size <= 1 and dp_local_size <= 1:
        return worker_local_rank

    dp_local_rank = getattr(parallel_config, "data_parallel_rank_local", None)
    if dp_local_rank is None:
        dp_rank = getattr(parallel_config, "data_parallel_rank", None)
        if dp_rank is None:
            dp_rank = getattr(parallel_config, "data_parallel_index", None)
        if dp_rank is not None and dp_local_size > 0:
            dp_local_rank = int(dp_rank) % dp_local_size

    if dp_local_rank is None:
        return worker_local_rank

    tp_rank = worker_local_rank % tp_size
    return int(dp_local_rank) * tp_size + tp_rank


def set_death_signal():
    """Kill the current process when the parent process exits."""
    if platform.system() != "Linux":
        return
    libc = ctypes.CDLL("libc.so.6")
    libc.prctl(1, signal.SIGKILL)
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGKILL)


def get_vllm_max_lora_rank(lora_rank: int):
    """
    For vLLM, automatically adjusts the `max_lora_rank` to the nearest allowed value.
    The allowed values are retrieved from vLLM's MaxLoRARanks type definition.
    """
    assert lora_rank > 0, f"lora_rank must be greater than 0, get {lora_rank}"

    try:
        from vllm.config.lora import MaxLoRARanks
    except Exception:
        # FIXME: migrate vllm version https://github.com/vllm-project/vllm/blob/main/vllm/config/lora.py#L25
        MaxLoRARanks = Literal[1, 8, 16, 32, 64, 128, 256, 320, 512]

    vllm_max_lora_ranks = sorted(get_args(MaxLoRARanks))
    if lora_rank > vllm_max_lora_ranks[-1]:
        raise ValueError(f"lora_rank must be less than or equal to {vllm_max_lora_ranks[-1]}, but got {lora_rank}")

    for rank in vllm_max_lora_ranks:
        if lora_rank <= rank:
            return rank


# https://github.com/vllm-project/vllm/issues/13175
def monkey_patch_compute_logits(model, vocab_size: int, banned_token_ids: Optional[list[int]] = None):
    """Mask the tokens the sampler must never pick.

    Beyond the out-of-vocabulary tail, `banned_token_ids` covers tokens that live *inside* the
    vocabulary yet are still illegal to generate: the vision placeholders, which are meaningless
    unless a real image or video sits behind them. See `get_vision_placeholder_token_ids`.
    """
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        if banned_token_ids:
            logits[..., banned_token_ids] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


def _get_prompt_logprob_token_ids(sampling_params: Any) -> Any:
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    return extra_args.get(_PROMPT_LOGPROB_TOKEN_IDS_EXTRA_ARG)


def _to_cpu_target_ids(token_ids: Any) -> torch.Tensor:
    return torch.as_tensor(token_ids, dtype=torch.int64, device="cpu")


def _validate_target_ids(target_ids: torch.Tensor, prompt_len: int, topk: int, req_id: str) -> torch.Tensor:
    if target_ids.ndim != 2:
        raise RuntimeError(f"{_PROMPT_LOGPROB_TOKEN_IDS_EXTRA_ARG} for request {req_id} must be 2D.")
    if target_ids.shape[0] != prompt_len:
        raise RuntimeError(
            f"{_PROMPT_LOGPROB_TOKEN_IDS_EXTRA_ARG} for request {req_id} has {target_ids.shape[0]} rows, "
            f"expected prompt length {prompt_len}."
        )
    if target_ids.shape[1] < topk:
        raise RuntimeError(
            f"{_PROMPT_LOGPROB_TOKEN_IDS_EXTRA_ARG} for request {req_id} has width {target_ids.shape[1]}, "
            f"expected at least {topk}."
        )
    return target_ids[:, :topk]


def _compute_prompt_logprobs_for_token_ids(
    prompt_token_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    prompt_hidden_states: torch.Tensor,
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    num_prompt_logprobs: int,
    logprobs_mode: str = "raw_logprobs",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

    chunk_size = 1024
    logits_mode = logprobs_mode in ("raw_logits", "processed_logits")
    token_ids, logprobs, ranks = [], [], []
    prompt_token_ids = prompt_token_ids.to(torch.int64)
    target_token_ids = target_token_ids.to(torch.int64)
    for start_idx in range(0, prompt_token_ids.shape[0], chunk_size):
        end_idx = start_idx + chunk_size
        logits = logits_fn(prompt_hidden_states[start_idx:end_idx])
        sampled_ids = prompt_token_ids[start_idx:end_idx].to(logits.device, non_blocking=True).unsqueeze(-1)
        requested_ids = target_token_ids[start_idx:end_idx].to(logits.device, non_blocking=True)
        logprob_token_ids = torch.cat((sampled_ids, requested_ids), dim=-1)
        token_ids.append(logprob_token_ids)
        if logits_mode:
            logprobs.append(logits.gather(-1, logprob_token_ids).to(torch.float32))
        else:
            logprobs.append(compute_token_logprobs(logits, logprob_token_ids))
        ranks.append(
            torch.full(
                (logprob_token_ids.shape[0],),
                num_prompt_logprobs + 1,
                dtype=torch.int64,
                device=logprob_token_ids.device,
            )
        )

    token_ids = torch.cat(token_ids, dim=0) if len(token_ids) > 1 else token_ids[0]
    logprobs = torch.cat(logprobs, dim=0) if len(logprobs) > 1 else logprobs[0]
    ranks = torch.cat(ranks, dim=0) if len(ranks) > 1 else ranks[0]
    return token_ids, logprobs, ranks


def _get_vllm_logprobs_mode(obj: Any) -> str:
    for source in (obj, getattr(obj, "sampler", None), getattr(obj, "model_config", None)):
        mode = getattr(source, "logprobs_mode", None)
        if mode is not None:
            return mode
    return "raw_logprobs"


def _patch_v2_prompt_logprobs_worker(worker: Any) -> None:
    if worker is None or getattr(worker, "_verl_target_prompt_logprobs_patched", False):
        return

    original_add_request = worker.add_request
    original_remove_request = worker.remove_request
    original_compute_prompt_logprobs = worker.compute_prompt_logprobs
    worker._verl_prompt_logprob_token_ids = {}

    def add_request(self, req_id: str, req_idx: int, sampling_params: Any):
        original_add_request(req_id, req_idx, sampling_params)
        target_ids = _get_prompt_logprob_token_ids(sampling_params)
        if target_ids is not None:
            self._verl_prompt_logprob_token_ids[req_id] = _to_cpu_target_ids(target_ids)

    def remove_request(self, req_id: str) -> None:
        self._verl_prompt_logprob_token_ids.pop(req_id, None)
        original_remove_request(req_id)

    def compute_prompt_logprobs(
        self,
        logits_fn: Callable[[torch.Tensor], torch.Tensor],
        hidden_states: torch.Tensor,
        input_batch: Any,
        all_token_ids: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        prompt_lens: np.ndarray,
        prefill_lens: Optional[np.ndarray] = None,
        num_computed_prefill_tokens: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        prompt_lens_all = prompt_lens
        use_input_batch_prefill_state = prefill_lens is None or num_computed_prefill_tokens is None
        if use_input_batch_prefill_state:
            prefill_lens_all = input_batch.prefill_len_np
            num_computed_prefill_tokens_all = input_batch.num_computed_prefill_tokens_np
        else:
            prefill_lens_all = prefill_lens
            num_computed_prefill_tokens_all = num_computed_prefill_tokens
        idx_mapping_np = input_batch.idx_mapping_np
        needs_prompt_logprobs = self.uses_prompt_logprobs[idx_mapping_np]
        if not np.any(needs_prompt_logprobs):
            return {}

        num_prompt_logprobs = self.num_prompt_logprobs[idx_mapping_np]
        prompt_lens = prompt_lens[idx_mapping_np]
        if use_input_batch_prefill_state:
            prefill_lens = prefill_lens_all
            computed_prefill = num_computed_prefill_tokens_all
        else:
            prefill_lens = prefill_lens_all[idx_mapping_np]
            computed_prefill = num_computed_prefill_tokens_all[idx_mapping_np]
        includes_prompt = computed_prefill < prompt_lens
        resumed_after_prompt = prompt_lens < prefill_lens
        needs_prompt_logprobs &= includes_prompt & ~resumed_after_prompt
        if not np.any(needs_prompt_logprobs):
            return {}

        targeted_reqs = [
            req_id
            for i, req_id in enumerate(input_batch.req_ids)
            if needs_prompt_logprobs[i] and req_id in self._verl_prompt_logprob_token_ids
        ]
        if not targeted_reqs:
            if use_input_batch_prefill_state:
                return original_compute_prompt_logprobs(
                    logits_fn,
                    hidden_states,
                    input_batch,
                    all_token_ids,
                    num_computed_tokens,
                    prompt_lens_all,
                )
            else:
                return original_compute_prompt_logprobs(
                    logits_fn,
                    hidden_states,
                    input_batch,
                    all_token_ids,
                    num_computed_tokens,
                    prompt_lens_all,
                    prefill_lens_all,
                    num_computed_prefill_tokens_all,
                )

        missing_reqs = [
            req_id
            for i, req_id in enumerate(input_batch.req_ids)
            if needs_prompt_logprobs[i] and req_id not in self._verl_prompt_logprob_token_ids
        ]
        if missing_reqs:
            raise RuntimeError(f"Cannot mix targeted and regular prompt logprobs in one vLLM batch: {missing_reqs}.")

        requested_num_prompt_logprobs = num_prompt_logprobs[needs_prompt_logprobs]
        if np.any(requested_num_prompt_logprobs <= 0):
            raise RuntimeError("Targeted prompt logprobs require prompt_logprobs > 0.")
        max_num_prompt_logprobs = int(requested_num_prompt_logprobs.max())

        from vllm.v1.outputs import LogprobsTensors
        from vllm.v1.worker.gpu.sample.prompt_logprob import get_prompt_logprobs_token_ids

        target_token_ids = torch.zeros(
            (input_batch.num_tokens, max_num_prompt_logprobs),
            dtype=torch.int64,
            device=hidden_states.device,
        )
        query_start_loc_np = input_batch.query_start_loc_np
        for i, req_id in enumerate(input_batch.req_ids):
            if not needs_prompt_logprobs[i]:
                continue
            req_topk = int(num_prompt_logprobs[i])
            req_target_ids = _validate_target_ids(
                self._verl_prompt_logprob_token_ids[req_id],
                prompt_len=int(prompt_lens[i]),
                topk=req_topk,
                req_id=req_id,
            )
            start_idx = int(query_start_loc_np[i])
            end_idx = int(query_start_loc_np[i + 1])
            token_offset = int(computed_prefill[i])
            rows = req_target_ids[token_offset : token_offset + end_idx - start_idx]
            target_token_ids[start_idx:end_idx, :req_topk] = rows.to(hidden_states.device, non_blocking=True)

        prompt_logprobs_token_ids = get_prompt_logprobs_token_ids(
            input_batch.num_tokens,
            input_batch.query_start_loc,
            input_batch.idx_mapping,
            num_computed_tokens,
            all_token_ids,
        )
        prompt_token_ids, prompt_logprobs, prompt_ranks = _compute_prompt_logprobs_for_token_ids(
            prompt_logprobs_token_ids,
            target_token_ids,
            hidden_states[: input_batch.num_tokens],
            logits_fn,
            max_num_prompt_logprobs,
            _get_vllm_logprobs_mode(self),
        )

        pos_after_step = computed_prefill + input_batch.num_scheduled_tokens
        is_prompt_chunked = pos_after_step < prompt_lens

        prompt_logprobs_dict = {}
        for i, req_id in enumerate(input_batch.req_ids):
            if not needs_prompt_logprobs[i]:
                continue

            req_is_prompt_chunked = is_prompt_chunked[i]
            req_num_prompt_logprobs = int(num_prompt_logprobs[i])
            start_idx = int(query_start_loc_np[i])
            end_idx = int(query_start_loc_np[i + 1])
            if not req_is_prompt_chunked:
                end_idx -= 1

            width = req_num_prompt_logprobs + 1
            logprobs = (
                None
                if start_idx >= end_idx
                else LogprobsTensors(
                    logprob_token_ids=prompt_token_ids[start_idx:end_idx, :width],
                    logprobs=prompt_logprobs[start_idx:end_idx, :width],
                    selected_token_ranks=prompt_ranks[start_idx:end_idx],
                )
            )

            prompt_logprobs_list = self.in_progress_prompt_logprobs[req_id]
            if logprobs is not None and (req_is_prompt_chunked or prompt_logprobs_list):
                prompt_logprobs_list.append(logprobs)
            if req_is_prompt_chunked:
                continue

            if prompt_logprobs_list:
                logprobs = LogprobsTensors(
                    logprob_token_ids=torch.cat([x.logprob_token_ids for x in prompt_logprobs_list]),
                    logprobs=torch.cat([x.logprobs for x in prompt_logprobs_list]),
                    selected_token_ranks=torch.cat([x.selected_token_ranks for x in prompt_logprobs_list]),
                )
                prompt_logprobs_list.clear()

            if logprobs is not None:
                prompt_logprobs_dict[req_id] = logprobs
                self._verl_prompt_logprob_token_ids.pop(req_id, None)
        return prompt_logprobs_dict

    worker.add_request = MethodType(add_request, worker)
    worker.remove_request = MethodType(remove_request, worker)
    worker.compute_prompt_logprobs = MethodType(compute_prompt_logprobs, worker)
    worker._verl_target_prompt_logprobs_patched = True


def _patch_v1_model_runner_prompt_logprobs(model_runner: Any) -> None:
    if getattr(model_runner, "_verl_target_prompt_logprobs_patched", False) or not hasattr(
        model_runner, "_get_prompt_logprobs_dict"
    ):
        return

    original_get_prompt_logprobs_dict = model_runner._get_prompt_logprobs_dict

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, Any]:
        if not self.num_prompt_logprobs:
            return {}

        has_targeted_req = False
        for req_id in self.num_prompt_logprobs:
            if (
                req_id in num_scheduled_tokens
                and _get_prompt_logprob_token_ids(self.requests[req_id].sampling_params) is not None
            ):
                has_targeted_req = True
                break
        if not has_targeted_req:
            return original_get_prompt_logprobs_dict(hidden_states, num_scheduled_tokens)

        from vllm.v1.outputs import LogprobsTensors

        prompt_logprobs_dict = {}
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in self.num_prompt_logprobs.items():
            num_tokens = num_scheduled_tokens.get(req_id)
            if num_tokens is None:
                continue

            request = self.requests[req_id]
            if request.prompt_token_ids is None:
                continue

            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(self.device, non_blocking=True)

            logprobs_tensors = request.in_progress_prompt_logprobs_cpu
            if logprobs_tensors is None:
                logprobs_tensors = LogprobsTensors.empty_cpu(num_prompt_tokens - 1, num_prompt_logprobs + 1)
                request.in_progress_prompt_logprobs_cpu = logprobs_tensors

            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                num_logits = num_tokens
            else:
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                continue

            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc.np[req_idx].item()
            prompt_hidden_states = hidden_states[offset : offset + num_logits]

            tgt_token_ids = prompt_token_ids[start_tok : start_tok + num_logits]
            target_ids = _get_prompt_logprob_token_ids(request.sampling_params)
            if target_ids is None:
                logits = self.model.compute_logits(prompt_hidden_states)
                logprobs = self.sampler.compute_logprobs(logits)
                token_ids, logprobs, ranks, _ = self.sampler.gather_logprobs(
                    logprobs, num_prompt_logprobs, tgt_token_ids
                )
            else:
                target_ids = _validate_target_ids(
                    _to_cpu_target_ids(target_ids),
                    prompt_len=num_prompt_tokens,
                    topk=num_prompt_logprobs,
                    req_id=req_id,
                )
                token_ids, logprobs, ranks = _compute_prompt_logprobs_for_token_ids(
                    tgt_token_ids,
                    target_ids[start_idx : start_idx + num_logits].to(self.device, non_blocking=True),
                    prompt_hidden_states,
                    self.model.compute_logits,
                    num_prompt_logprobs,
                    _get_vllm_logprobs_mode(self),
                )

            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(token_ids, non_blocking=True)
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs, non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(ranks, non_blocking=True)

        for req_id in completed_prefill_reqs:
            del self.num_prompt_logprobs[req_id]
            self.requests[req_id].in_progress_prompt_logprobs_cpu = None

        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    model_runner._get_prompt_logprobs_dict = MethodType(_get_prompt_logprobs_dict, model_runner)
    model_runner._verl_target_prompt_logprobs_patched = True


def patch_prompt_logprob_target_ids(model_runner: Any) -> None:
    _patch_v2_prompt_logprobs_worker(getattr(model_runner, "prompt_logprobs_worker", None))
    _patch_v1_model_runner_prompt_logprobs(model_runner)


class vLLMColocateWorkerExtension:
    """
    The class for vLLM's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.

    Feature support:
    1. LoRA
    2. Online FP8 quantization
    """

    def __new__(cls, **kwargs):
        set_death_signal()

        if os.environ.get("VERL_FULL_DETERMINISM", "0") == "1":
            from verl.workers.engine.utils import enable_full_determinism

            # VERL_SEED is set by vLLMHttpServer.__init__ only when the
            # rollout config has full_determinism=true.  Worker sub-processes
            # inherit their parent's env, so rollout workers will see it but
            # RM workers (whose parent vLLMHttpServer does not set it) won't.
            # If VERL_SEED is missing, skip — RM doesn't need the determinism
            # patch, only rollout does.
            verl_seed = os.environ.get("VERL_SEED")
            if verl_seed is not None:
                enable_full_determinism(seed=int(verl_seed))

        # 1. patch for Lora
        VLLMHijack.hijack()
        vllm_config = kwargs.get("vllm_config")
        # 2. patch online fp8 quant. Some models, including DeepSeek-V4, get
        # fp8 from the HF config rather than an explicit rollout quantization arg.
        if os.environ.get("VERL_VLLM_FP8_QUANT_ENABLED", "0") == "1" or is_fp8_model(vllm_config):
            apply_vllm_quant_patches()
        # 3. patch QAT (compressed-tensors NVFP4) for dynamic weight loading
        quant_config = getattr(vllm_config, "quant_config", None) if vllm_config else None
        _is_qat_model = getattr(quant_config, "quant_format", None) == "nvfp4-pack-quantized"
        _is_modelopt_qat = type(quant_config).__name__ == "ModelOptNvFp4Config"
        if _is_qat_model:
            from verl.utils.qat import apply_qat_patches

            apply_qat_patches()
            logger.info("Applied QAT (compressed-tensors) patches in vLLM worker subprocess")
        elif _is_modelopt_qat:
            from verl.utils.modelopt import apply_modelopt_nvfp4_patches

            apply_modelopt_nvfp4_patches()
            logger.info("Applied ModelOpt NVFP4 patches in vLLM worker subprocess")

        # TODO: For ascend NPU, when the corresponding vllm-ascend version is upgraded to v0.13.0,
        # please remove the VLLM_ASCEND_REQUIRED_ENV_VARS variable replacement action.
        # This is only a fix for vllm version < v0.13.0.
        if is_npu_available:
            for k in VLLM_ASCEND_REQUIRED_ENV_VARS:
                if k not in os.environ:
                    os.environ[k] = VLLM_ASCEND_REQUIRED_ENV_VARS[k]

        instance = super().__new__(cls)
        instance._is_qat_model = _is_qat_model
        instance._is_modelopt_qat = _is_modelopt_qat
        return instance

    def _get_drafter_model(self):
        """Return the drafter's model object, or None if unavailable."""
        drafter = getattr(self.model_runner, "drafter", None)
        return drafter.model if drafter is not None and hasattr(drafter, "model") else None

    def _get_draft_model_config(self):
        """Return the draft model config from speculative_config, or None."""
        spec = self.model_runner.vllm_config.speculative_config
        return spec.draft_model_config if spec is not None and spec.draft_model_config is not None else None

    def _use_mtp_drafter_weight_sync(self):
        """Return whether the vLLM MTP drafter should receive actor weights."""
        spec = self.model_runner.vllm_config.speculative_config
        return spec is not None and spec.method == "mtp" and self._get_drafter_model() is not None

    def _iter_all_models(self):
        """Yield models that need weight updates.

        Only vLLM MTP drafter sync is supported for now. Independent non-MTP
        draft models are not compatible with actor weight loading through this path.
        """
        yield self.model_runner.model
        if self._use_mtp_drafter_weight_sync():
            yield self._get_drafter_model()

    def _iter_all_models_with_config(self):
        """Yield (model, model_config) for models that need post-processing."""
        yield self.model_runner.model, self.model_runner.vllm_config.model_config
        if self._use_mtp_drafter_weight_sync():
            draft_cfg = self._get_draft_model_config()
            if draft_cfg is not None:
                yield self._get_drafter_model(), draft_cfg

    def monkey_patch_model(self, vocab_size: int, banned_token_ids: Optional[list[int]] = None):
        patch_prompt_logprob_target_ids(self.model_runner)
        for model in self._iter_all_models():
            # patch compute_logits to avoid sampling OOV and other illegal tokens
            monkey_patch_compute_logits(model, vocab_size, banned_token_ids)
            # patch weight loader to support MoE model
            patch_vllm_moe_model_weight_loader(model)

    def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False):
        """Update the weights of the rollout model."""
        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        if self.device is None:
            # vLLM workers may leave self.device unset on non-CUDA platforms (e.g. NPU);
            # fall back to the worker's local rank on the current accelerator.
            self.device = torch.device(f"{get_device_name()}:{self.local_rank}")

        # =========================== step 1: prepare for weight loading ===========================
        quant_reload_states = None

        # The engine came up on dummy weights, whose init zeroes integer buffers on
        # ROCm -- including the expert-parallel routing maps, which no weight stream
        # restores. Repair them before the reload so the rollout routes correctly.
        if torch.version.hip is not None:
            from verl.utils.vllm.rocm_vllm_moe_expert_map import restore_moe_expert_maps

            for model in self._iter_all_models():
                restore_moe_expert_maps(model)

        if self._is_qat_model:
            # QAT (compressed-tensors): Prepare for weight loading BEFORE receiving any buckets
            from verl.utils.qat import prepare_qat_for_load_weights

            for model in self._iter_all_models():
                prepare_qat_for_load_weights(model, device=self.device)
            logger.info("QAT: prepare_qat_for_load_weights completed")
        elif self._is_modelopt_qat:
            from verl.utils.modelopt.vllm_modelopt_patch import prepare_modelopt_for_weight_reload

            prepare_modelopt_for_weight_reload(self.model_runner.model, device=self.device)
            logger.info("ModelOpt: prepare_modelopt_for_weight_reload completed")
        elif peft_config and base_sync_done:
            # Remove the old LoRA before the new one arrives (applied after is_last below).
            self.remove_lora(VLLM_LORA_INT_ID)
            logger.info("LoRA adapter sync: remove old lora and prepare new lora")
        elif is_fp8_model(self.model_runner.vllm_config):
            from verl.utils.vllm.vllm_quant_utils import prepare_quanted_weights_for_loading

            quant_reload_states = [
                (model, prepare_quanted_weights_for_loading(model)) for model in self._iter_all_models()
            ]
        else:
            # TODO(wuxibin): not need anymore for newer vllm version.
            for model in self._iter_all_models():
                patch_vllm_moe_model_weight_loader(model)

        # =========================== step 2: receive weights and update ===========================
        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(),
            device=self.device,
            use_shm=use_shm,
        )
        # LoRA adapters need a single complete tensor dict per ``add_lora``, but
        # the bucketed transport may split one across buckets. Accumulate and
        # apply only after ``is_last``; standard base weights load per bucket.
        lora_weights: dict[str, torch.Tensor] | None = {} if (peft_config and base_sync_done) else None

        def on_bucket_received(weights: list[tuple[str, torch.Tensor]], is_last: bool) -> None:
            if lora_weights is not None:
                # Clone: add_lora keeps these past the callback (reused IPC buffer, #6454).
                lora_weights.update((name, tensor.clone()) for name, tensor in weights)
                if not is_last:
                    return
                self._update_weights(
                    list(lora_weights.items()),
                    peft_config=peft_config,
                    base_sync_done=base_sync_done,
                )
                lora_weights.clear()
                return
            self._update_weights(
                weights,
                peft_config=peft_config,
                base_sync_done=base_sync_done,
            )

        receiver.receive_weights(on_bucket_received=on_bucket_received)

        # =========================== step 3: process weights after loading ===========================
        if self._is_qat_model:
            # QAT (compressed-tensors): call process_weights_after_loading AFTER all buckets are received
            from verl.utils.qat import manual_process_weights_after_loading

            for model in self._iter_all_models():
                manual_process_weights_after_loading(model)
            logger.info("QAT: process_weights_after_loading completed")
        elif self._is_modelopt_qat:
            from verl.utils.modelopt.vllm_modelopt_patch import modelopt_process_weights_after_loading

            modelopt_process_weights_after_loading(self.model_runner.model)
            logger.info("ModelOpt QAT: process_weights_after_loading completed")
        elif peft_config and base_sync_done:
            logger.info("LoRA adapter sync, no post-process needed")
        elif is_fp8_model(self.model_runner.vllm_config):
            from verl.utils.vllm.vllm_quant_utils import process_quanted_weights_after_loading

            for model, reload_state in quant_reload_states:
                process_quanted_weights_after_loading(model, reload_state)
        else:
            # Some post-load transforms are non-idempotent; run once after all buckets.
            from vllm.model_executor.model_loader.utils import process_weights_after_loading

            for model, model_config in self._iter_all_models_with_config():
                process_weights_after_loading(model, model_config, self.device)

    def _apply_buffer_updates_all_models(self, buffer_updates, main_named_buffers):
        """Apply buffer updates to the main model and any synced MTP drafter.

        The main model (yielded first) reuses the prebuilt ``named_buffers`` map;
        the drafter builds its own. Returns buffers applied to the main model.
        """
        models = list(self._iter_all_models())
        loaded = apply_buffer_updates(models[0], buffer_updates, named_buffers=main_named_buffers)
        for model in models[1:]:
            apply_buffer_updates(model, buffer_updates)
        return loaded

    def _update_weights(
        self,
        weights: list[tuple[str, torch.Tensor]],
        peft_config: dict,
        base_sync_done: bool,
    ):
        if peft_config and base_sync_done:
            # Clone out of the receiver's reused IPC bucket buffer: add_lora keeps these tensors
            # past this callback, so views into the freed/overwritten buffer crash later (#6454).
            weights = {name: tensor.clone() for name, tensor in weights}
            lora_request = TensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=peft_config,
                lora_tensors=weights,
            )
            self.add_lora(lora_request)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            param_updates, buffer_updates, named_buffers = split_buffer_updates(self.model_runner.model, weights)
            # Add the FP8 related logic here as sharding manager has been deprecated.
            # Check if FP8 quantization is enabled and apply appropriate weight loading
            if is_fp8_model(self.model_runner.vllm_config):
                logger.info(f"FP8 model detected (async): {self.model_runner.vllm_config.quant_config}")
                # Convert bf16 weights to fp8 format before loading
                loaded_params = load_quanted_weights(param_updates, self.model_runner) if param_updates else []
                # Keep the draft model in sync when present.
                if self._use_mtp_drafter_weight_sync() and param_updates:
                    load_quanted_weights(param_updates, self.model_runner, is_drafter=True)
                loaded_buffers = self._apply_buffer_updates_all_models(buffer_updates, named_buffers)
                logger.info(
                    f"FP8 weights loaded (async), loaded_params: {len(loaded_params)}, loaded_buffers: {loaded_buffers}"
                )
            else:
                if param_updates:
                    for model in self._iter_all_models():
                        if peft_config is None:
                            model.load_weights(param_updates)
                        else:
                            names = {n for n, _ in model.named_parameters(remove_duplicate=False)}
                            names.update(n for n, _ in model.named_buffers())
                            model.load_weights((resolve_weight_name(model, n, names), t) for n, t in param_updates)
                loaded_buffers = self._apply_buffer_updates_all_models(buffer_updates, named_buffers)
                logger.info(
                    f"Loading standard weights (non-FP8, async), "
                    f"loaded_params: {len(param_updates)}, loaded_buffers: {loaded_buffers}"
                )

    def _get_zmq_handle(self) -> str:
        """Get ZMQ handle for communication.

        Uses Ray job id + replica_rank + rollout-local rank to match the sender
        side and avoid cross-job collisions on shared hosts.
        In PD mode, each engine actor's local ranks start at 0; the optional
        VERL_ZMQ_BASE_TRAINER_RANK offset maps them back to trainer ranks.
        """
        replica_rank = os.environ.get("VERL_REPLICA_RANK", "0")
        job_id = os.environ.get("VERL_RAY_JOB_ID", "0")
        vllm_config = getattr(self.model_runner, "vllm_config", None)
        parallel_config = getattr(vllm_config, "parallel_config", None)
        local_rank = _resolve_vllm_weight_sync_local_rank(self.local_rank, parallel_config)
        trainer_rank_base = os.environ.get("VERL_ZMQ_BASE_TRAINER_RANK")
        trainer_rank = int(trainer_rank_base) + local_rank if trainer_rank_base is not None else local_rank
        return f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{trainer_rank}.sock"


class SuppressSignalInThread:
    def __enter__(self):
        self.original_signal = signal.signal

        def no_op_signal(sig, action):
            if threading.current_thread() is not threading.main_thread():
                print(f"Ignored signal {sig} in thread {threading.current_thread().name}")
                return
            return self.original_signal(sig, action)

        signal.signal = no_op_signal
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal = self.original_signal


@functools.lru_cache(maxsize=1)
def _optional_bool_vllm_args() -> set[str]:
    """Return the names of vLLM `AsyncEngineArgs` fields typed exactly `bool | None`.

    For such fields an omitted flag leaves the None default, which vLLM can
    resolve to True at engine-config time (e.g. `enable_prefix_caching`), so
    an explicit False must be serialized as `--no-<flag>` instead of being
    dropped.
    """
    from vllm.engine.arg_utils import AsyncEngineArgs

    return {f.name for f in dataclasses.fields(AsyncEngineArgs) if set(get_args(f.type)) == {bool, type(None)}}


def build_cli_args_from_config(config: dict[str, Any]) -> list[str]:
    """
    Convert a config dictionary to CLI arguments for vLLM server.

    Handles different value types appropriately:
    - None: skipped
    - bool True: adds '--key'
    - bool False: adds '--no-key' for Optional[bool] engine args (whose None
      default resolves to True), otherwise skipped
    - list: expands to '--key item1 item2 ...'
    - empty list: skipped (vLLM uses nargs="+" which requires at least one value)
    - dict: JSON serialized
    - other: string converted

    Args:
        config: Dictionary of configuration key-value pairs

    Returns:
        List of CLI argument strings
    """
    cli_args = []
    for k, v in config.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                cli_args.append(f"--{k}")
            elif k.replace("-", "_") in _optional_bool_vllm_args():
                # Absent flag resolves to True at engine-config time.
                cli_args.append(f"--no-{k}")
        elif isinstance(v, list):
            if not v:
                # Skip empty lists - vLLM uses nargs="+" which requires at least one value
                continue
            # Lists need to be expanded as multiple separate arguments
            # e.g., --cuda-graph-sizes 1 2 4 8 becomes ['--cuda-graph-sizes', '1', '2', '4', '8']
            cli_args.append(f"--{k}")
            cli_args.extend([str(item) for item in v])
        else:
            cli_args.append(f"--{k}")
            # Use json.dumps for dict to ensure valid JSON format
            cli_args.append(json.dumps(v) if isinstance(v, dict) else str(v))
    return cli_args


def build_mtp_speculative_config(
    method: str, num_speculative_tokens: int, engine_speculative_config: Any = None
) -> dict[str, Any]:
    """Build vLLM's MTP speculative config, applying rollout engine overrides."""
    if engine_speculative_config is None:
        engine_speculative_config = {}
    if isinstance(engine_speculative_config, str):
        engine_speculative_config = json.loads(engine_speculative_config)
    if not isinstance(engine_speculative_config, Mapping):
        raise TypeError("rollout.engine_kwargs.vllm.speculative_config must be a mapping when MTP rollout is enabled")

    return {
        "method": method,
        "num_speculative_tokens": num_speculative_tokens,
        **{key: val for key, val in engine_speculative_config.items() if val is not None},
    }


def extract_prompt_logprobs(output: RequestOutput, num_prompt_logprobs: Optional[int], result_dict: dict[str, list]):
    """Extract prompt log probabilities from generation output."""
    if num_prompt_logprobs is None:
        return

    prompt_logprobs_ls, prompt_ids_ls = [], []
    # NOTE: logprob of first prompt token is None.
    for logprobs_dict in output.prompt_logprobs[1:]:
        if num_prompt_logprobs == 0:
            token_id_str = list(logprobs_dict.keys())[0]
            logprob = logprobs_dict[token_id_str].logprob
            prompt_logprobs_ls.append([logprob])
            prompt_ids_ls.append([int(token_id_str)])
        else:
            prompt_ids = [None] * num_prompt_logprobs
            prompt_logprobs = [None] * num_prompt_logprobs
            # We get either top-k logprobs or top-k plus the sampled logprob (if sampled token is not in top-k)
            assert len(logprobs_dict) in [num_prompt_logprobs, num_prompt_logprobs + 1], len(logprobs_dict)
            for token_id_str, token_logprob in logprobs_dict.items():
                rank = token_logprob.rank
                if rank > num_prompt_logprobs:
                    continue  # the sampled token is not in the top-k
                logprob = token_logprob.logprob
                prompt_ids[rank - 1] = int(token_id_str)
                prompt_logprobs[rank - 1] = logprob
            prompt_logprobs_ls.append(prompt_logprobs)
            prompt_ids_ls.append(prompt_ids)

    # NOTE: pad a dummy prompt logprob for last prompt token.
    prompt_logprobs_ls.append([0.0] * max(num_prompt_logprobs, 1))
    prompt_ids_ls.append([0] * max(num_prompt_logprobs, 1))

    result_dict["prompt_ids"] = prompt_ids_ls
    result_dict["prompt_logprobs"] = prompt_logprobs_ls
