# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Patched for single-GPU (A100 40GB) training:
# Original code called vllm_model.cuda() AFTER loading CPU tensors from FSDP,
# which creates a full second copy of model weights on GPU causing OOM.
# Fix: stream each weight to GPU individually before passing to load_weights,
# then skip the final .cuda() call since model is already on GPU.

from typing import Dict

import torch
import torch.nn as nn
from vllm.model_executor.model_loader.utils import set_default_torch_dtype


def update_hf_weight_loader():
    print("no hf weight loader need to be updated")
    return


def load_hf_weights(actor_weights: Dict, vllm_model: nn.Module):
    assert isinstance(actor_weights, Dict)

    with set_default_torch_dtype(next(vllm_model.parameters()).dtype):
        if vllm_model.config.tie_word_embeddings and "lm_head.weight" in actor_weights.keys():
            del actor_weights["lm_head.weight"]

        # ── PATCH: stream each weight to GPU one at a time ──────────────────
        # The original code loaded all weights (CPU tensors from FSDP param_offload)
        # into the model and then called vllm_model.cuda(), which tried to allocate
        # the entire model on GPU simultaneously — causing OOM on 40GB A100.
        # Instead we move each tensor to GPU individually before handing off to
        # load_weights, so peak GPU allocation is one parameter at a time.
        def gpu_weight_iter():
            for name, param in actor_weights.items():
                if isinstance(param, torch.Tensor) and param.device.type == 'cpu':
                    yield name, param.to('cuda', non_blocking=False)
                else:
                    yield name, param

        vllm_model.load_weights(gpu_weight_iter())
        # ── END PATCH ────────────────────────────────────────────────────────

        for _, module in vllm_model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                quant_method.process_weights_after_loading(module)
            if hasattr(module, "process_weights_after_loading"):
                module.process_weights_after_loading()

        # Original: vllm_model = vllm_model.cuda()
        # Removed — weights are already on GPU after the streamed load above.
        # Calling .cuda() here would attempt to allocate the full model again → OOM.
