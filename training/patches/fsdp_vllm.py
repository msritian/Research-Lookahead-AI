# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
#
# Patched for single-GPU (A100 40GB) training.
#
# Root cause of OOM:
#   Original FullStateDictConfig() has offload_to_cpu=False (default).
#   state_dict() therefore materialises all FSDP weights on GPU (+6 GB),
#   then hf_weight_loader also needs GPU space for the vLLM sync → OOM.
#
# Fix:
#   1. Use FullStateDictConfig(offload_to_cpu=True) so the gathered params
#      stay on CPU and only move to GPU one-at-a-time during load_weights.
#   2. Add torch.cuda.empty_cache() before sync to reclaim fragmented memory.

import os
import logging
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import (
    ShardingStrategy, ShardedStateDictConfig,
    StateDictType, FullStateDictConfig,
)
from torch.distributed.device_mesh import DeviceMesh

from verl.third_party.vllm import LLM
from verl.third_party.vllm import parallel_state as vllm_ps
from verl import DataProto
from verl.utils.torch_functional import (broadcast_dict_tensor, allgather_dict_tensors)
from verl.utils.debug import log_gpu_memory_usage

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


class FSDPVLLMShardingManager(BaseShardingManager):

    def __init__(self,
                 module: FSDP,
                 inference_engine: LLM,
                 model_config,
                 full_params: bool = False,
                 device_mesh: DeviceMesh = None):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh

        self.full_params = full_params
        if full_params:
            # ── PATCH: offload_to_cpu=True keeps gathered weights on CPU ──────
            # Original: FullStateDictConfig() → offload_to_cpu=False (default)
            # That materialises all FSDP weights on GPU during state_dict(),
            # consuming ~6 GB right before the vLLM weight sync → OOM on 40 GB.
            # With offload_to_cpu=True the dict holds CPU tensors; hf_weight_loader
            # streams them to GPU one parameter at a time (no spike).
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            )
        else:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.torch_random_states = torch.cuda.get_rng_state()
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh['dp'].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    def __enter__(self):
        log_gpu_memory_usage('Before state_dict() in sharding manager memory', logger=logger)

        # ── PATCH: free fragmented memory before gathering weights ────────────
        torch.cuda.empty_cache()

        params = self.module.state_dict()   # CPU tensors with our patch above
        log_gpu_memory_usage('After state_dict() in sharding manager memory', logger=logger)

        load_format = 'hf' if self.full_params else 'dtensor'
        self.inference_engine.sync_model_weights(params, load_format=load_format)
        log_gpu_memory_usage('After sync model weights in sharding manager', logger=logger)

        del params
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After del state_dict and empty_cache in sharding manager', logger=logger)

        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage('Before vllm offload in sharding manager', logger=logger)
        self.inference_engine.offload_model_weights()
        log_gpu_memory_usage('After vllm offload in sharding manager', logger=logger)

        self.module.train()
        torch.cuda.empty_cache()

        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        data.batch = allgather_dict_tensors(
            data.batch.contiguous(),
            size=vllm_ps.get_tensor_model_parallel_world_size(),
            group=vllm_ps.get_tensor_model_parallel_group(),
            dim=0,
        )
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        broadcast_dict_tensor(
            data.batch,
            src=vllm_ps.get_tensor_model_parallel_src_rank(),
            group=vllm_ps.get_tensor_model_parallel_group(),
        )
        dp_rank = torch.distributed.get_rank()
        tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            local_prompts = data.chunk(chunks=tp_size)
            data = local_prompts[dp_rank % tp_size]
        return data
