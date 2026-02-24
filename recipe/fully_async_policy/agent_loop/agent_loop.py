# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import asyncio
import heapq
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import hydra
import numpy as np
import ray
from cachetools import LRUCache
from omegaconf import DictConfig

from recipe.fully_async_policy.vllm_rollout.vllm_async_server import FullyAsyncvLLMReplica
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorkerBase,
    AsyncLLMServerManager,
    DictConfigWrap,
    _agent_loop_registry,
    get_trajectory_info,
)
from verl.experimental.agent_loop.prometheus_utils import update_prometheus_config
from verl.protocol import DataProto
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup
from verl.utils.rollout_trace import (
    rollout_trace_attr,
    rollout_trace_op,
)

# Configure logger
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FullyAsyncLLMServerManager(AsyncLLMServerManager):
    """
    Old weighted-heap scheduling
    Inflight tracking aligned 100% with FullyAsyncLLMServerManagerBalance
    """

    def __init__(
        self,
        config: Any,
        server_handles: list[Any],
        *,
        max_cache_size: int = 10000,
    ):
        super().__init__(config, server_handles, max_cache_size=max_cache_size)

        # Real inflight tracking (same semantics as Balance)
        self._inflight_requests: list[int] = [0] * len(self.server_handles)

        # O(1) mapping
        self._server_to_index = {actor: idx for idx, actor in enumerate(self.server_handles)}

    # ==========================================================
    # Old heap-based scheduling (UNCHANGED scheduling logic)
    # But inflight++ moved here to match V8.3 structure
    # ==========================================================
    def _choose_server(self, request_id: str):
        if request_id in self.request_id_to_server:
            server = self.request_id_to_server[request_id]
            server_index = self._server_to_index[server]
            self._inflight_requests[server_index] += 1
            return server

        _, _, server = self.weighted_serveres[0]

        # heap weighted logic unchanged
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])

        self.request_id_to_server[request_id] = server

        server_index = self._server_to_index[server]
        self._inflight_requests[server_index] += 1

        return server

    # ==========================================================
    # Generate (same pattern as V8.3)
    # ==========================================================
    @rollout_trace_op
    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> Any:
        server = self._choose_server(request_id)
        server_index = self._server_to_index[server]

        try:
            result = await server.generate.remote(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
            )
            return result

        finally:
            self._inflight_requests[server_index] = max(
                0,
                self._inflight_requests[server_index] - 1,
            )

    # ==========================================================
    # Partial (IDENTICAL inflight semantics to V8.3)
    # ==========================================================
    @rollout_trace_op
    async def generate_for_partial(
        self,
        request_id: str,
        *,
        prompt_ids: list[int] | list[list[int]],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ):
        server = self._choose_server(request_id)
        server_index = self._server_to_index[server]

        try:
            output = await server.generate_for_partial.remote(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
            )
            return output

        finally:
            self._inflight_requests[server_index] = max(
                0,
                self._inflight_requests[server_index] - 1,
            )

    # ==========================================================
    # Monitoring
    # ==========================================================
    def get_server_loads(self) -> list[int]:
        return list(self._inflight_requests)


# ==============================================
# Configuration Constants
# ==============================================

DEFAULT_MAX_CACHE_SIZE = 10000
DEFAULT_EMA_DECAY = 0.1
DEFAULT_RT_WEIGHT = 0.3
DEFAULT_OVERFLOW_RATIO = 1.25
DEFAULT_IDLE_RESET_THRESHOLD = 1  # seconds of continuous idle before EMA decay
DEFAULT_IDLE_DECAY_FACTOR = 0.01  # strong decay when idle reset triggered

LOG_INTERVAL = 128


@dataclass
class ServerState:
    index: int
    actor: Any

    active: int = 0
    ema_rt: float = 1.0
    total_requests: int = 0
    total_time: float = 0.0

    def update_rt(self, response_time: float, decay: float) -> None:
        self.ema_rt = decay * response_time + (1 - decay) * self.ema_rt
        self.ema_rt = max(self.ema_rt, 1e-3)

    def load_score(self, global_avg_rt: float, rt_weight: float) -> float:
        if global_avg_rt <= 0:
            return float(self.active)

        rt_ratio = self.ema_rt / global_avg_rt
        score = (1 - rt_weight) * self.active + rt_weight * self.active * rt_ratio
        return score


# ==============================================
# Main Load Balancer
# ==============================================

class FullyAsyncLLMServerManagerBalance(AsyncLLMServerManager):

    def __init__(
        self,
        config: Any,
        server_handles: list[Any],
        *,
        max_cache_size: int = DEFAULT_MAX_CACHE_SIZE,
        ema_decay: float = DEFAULT_EMA_DECAY,
        rt_weight: float = DEFAULT_RT_WEIGHT,
        overflow_ratio: float = DEFAULT_OVERFLOW_RATIO,
        idle_reset_threshold: float = DEFAULT_IDLE_RESET_THRESHOLD,
        idle_decay_factor: float = DEFAULT_IDLE_DECAY_FACTOR,
    ) -> None:

        super().__init__(config, server_handles, max_cache_size=max_cache_size)

        self.ema_decay = ema_decay
        self.rt_weight = rt_weight
        self.overflow_ratio = overflow_ratio
        self.num_servers = len(server_handles)

        self.idle_reset_threshold = idle_reset_threshold
        self.idle_decay_factor = idle_decay_factor

        self.states: dict[int, ServerState] = {
            idx: ServerState(index=idx, actor=actor)
            for idx, actor in enumerate(self.server_handles)
        }

        self._actor_to_idx = {
            actor: idx for idx, actor in enumerate(self.server_handles)
        }

        self.request_id_to_state: LRUCache = LRUCache(maxsize=max_cache_size)

        self._request_counter = 0

        # idle detection
        self._last_active_timestamp = time.monotonic()
        self._ema_idle_reset_done = False

        print(
            f"[Balance Init] servers={self.num_servers}, "
            f"ema_decay={ema_decay}, rt_weight={rt_weight}, "
            f"overflow_ratio={overflow_ratio}, "
            f"idle_reset_threshold={idle_reset_threshold}, "
            f"idle_decay_factor={idle_decay_factor}"
        )

    # ==============================================
    # Internal Utilities
    # ==============================================

    def _global_avg_rt(self) -> float:
        return sum(s.ema_rt for s in self.states.values()) / max(self.num_servers, 1)

    def _theory_per_server(self) -> float:
        total = sum(s.active for s in self.states.values())
        return total / max(self.num_servers, 1)

    def _total_active(self) -> int:
        return sum(s.active for s in self.states.values())

    # ==============================================
    # Idle Reset Logic
    # ==============================================

    def _decay_ema_on_idle(self) -> None:
        # 问题1修复：直接重置到 1.0，下一个 step 各 Server 完全均等
        # idle_decay_factor=0.01 使得 ema_rt * 0.01 远小于 1.0，max(1.0, ...) 保证下限
        for state in self.states.values():
            state.ema_rt = max(1.0, state.ema_rt * self.idle_decay_factor)

        print("[Balance] Idle detected → EMA decayed")

    def _maybe_idle_reset(self) -> None:
        # 此方法在 active 已经 +1 之后调用，total_active 必然 > 0
        # 因此只需检查距上次归零的时间，不存在 active=0 的竞态窗口
        idle_time = time.monotonic() - self._last_active_timestamp
        if idle_time >= self.idle_reset_threshold and not self._ema_idle_reset_done:
            self._decay_ema_on_idle()
            self._ema_idle_reset_done = True

    # ==============================================
    # Scheduling Core
    # ==============================================

    def _best_state(self) -> ServerState:
        global_avg = self._global_avg_rt()
        theory = self._theory_per_server()
        threshold = theory * self.overflow_ratio

        best: Optional[ServerState] = None
        best_score = float("inf")

        for state in self.states.values():
            score = state.load_score(global_avg, self.rt_weight)
            if score < best_score:
                best_score = score
                best = state

        assert best is not None

        if best.active > threshold and theory > 0:
            best = min(self.states.values(), key=lambda s: s.active)

        return best

    # ==============================================
    # Scheduling Entry
    # ==============================================

    def _choose_server(self, request_id: str, prompt_ids=None) -> ServerState:

        # 问题2修复：粘性会话保留原始方案逻辑，不主动打破，KV Cache 收益优先
        if request_id in self.request_id_to_state:
            state = self.request_id_to_state[request_id]
            state.active += 1
            # 问题3&4修复：active +1 之后再清除 flag，消除 active=0 的竞态窗口
            self._ema_idle_reset_done = False
            self._maybe_idle_reset()
            return state

        state = self._best_state()
        state.active += 1
        state.total_requests += 1
        self.request_id_to_state[request_id] = state

        # 问题3&4修复：active +1 之后再清除 flag 并检测 idle reset
        self._ema_idle_reset_done = False
        self._maybe_idle_reset()

        self._request_counter += 1
        if self._request_counter % LOG_INTERVAL == 0:
            self._log_status()

        return state

    def _release(self, state: ServerState, response_time: float) -> None:
        state.active = max(0, state.active - 1)
        state.total_time += response_time
        state.update_rt(response_time, self.ema_decay)

        # 问题3修复：记录 active 归零的时间点，供 _maybe_idle_reset 计算 idle_time
        # 只在归零时更新，避免被中间的 release 覆盖
        if self._total_active() == 0:
            self._last_active_timestamp = time.monotonic()

    # ==============================================
    # Public APIs
    # ==============================================

    @rollout_trace_op
    async def generate(self, request_id: str, **kwargs):
        state = self._choose_server(request_id)
        start = time.monotonic()

        try:
            return await state.actor.generate.remote(request_id=request_id, **kwargs)
        finally:
            self._release(state, time.monotonic() - start)

    @rollout_trace_op
    async def generate_for_partial(self, request_id: str, **kwargs):
        state = self._choose_server(request_id)
        start = time.monotonic()

        try:
            return await state.actor.generate_for_partial.remote(
                request_id=request_id, **kwargs
            )
        finally:
            self._release(state, time.monotonic() - start)

    # ==============================================
    # Monitoring
    # ==============================================

    def get_server_loads(self) -> list[int]:
        return [self.states[i].active for i in range(self.num_servers)]

    def get_load_metrics(self) -> dict[str, Any]:
        return {
            "active": {i: s.active for i, s in self.states.items()},
            "ema_rt": {i: round(s.ema_rt, 3) for i, s in self.states.items()},
            "global_avg_rt": round(self._global_avg_rt(), 3),
        }

    def _log_status(self) -> None:
        metrics = self.get_load_metrics()
        print(
            f"[Balance] active={metrics['active']} | "
            f"ema_rt={metrics['ema_rt']} | "
            f"global_avg_rt={metrics['global_avg_rt']}"
        )


# ==============================================
# Module Entry Point
# ==============================================


def _should_use_balance_load() -> bool:
    """Check environment variable to determine load balancer selection."""
    try:
        return bool(int(os.environ.get("USE_BALANCE_LOAD", "0")))
    except (ValueError, TypeError):
        logger.warning("Invalid USE_BALANCE_LOAD value, defaulting to base implementation")
        return False


if _should_use_balance_load():
    FullyAsyncLLMServerManager = FullyAsyncLLMServerManagerBalance
    print("Using optimized FullyAsyncLLMServerManagerBalance")
else:
    print("Using base FullyAsyncLLMServerManager")


@ray.remote
class FullyAsyncAgentLoopWorker(AgentLoopWorkerBase):
    def __init__(
        self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], reward_router_address: str = None
    ):
        self.server_manager = FullyAsyncLLMServerManager(config, server_handles)
        super().__init__(config, server_handles, reward_router_address)
        # A shared cancellation event for all agent loops running on this worker.
        self.cancellation_event = asyncio.Event()

    async def generate_sequences_no_post(
        self, batch: DataProto, partial_output_list: Optional[list[AgentLoopOutput]]
    ) -> tuple[list[AgentLoopOutput], bool] | tuple[DataProto, bool]:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.
            partial_output_list: Optional[List[AgentLoopOutput]]: already rollout result.

        Returns:
            list[AgentLoopOutput]: List of agent loop outputs, one per sample in the batch.
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index, batch.meta_info.get("validate", False)
        )

        if not partial_output_list:
            partial_output_list = [None] * len(batch)
        try:
            tasks = []
            for i in range(len(batch)):
                kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
                kwargs["output"] = partial_output_list[i]
                tasks.append(
                    asyncio.create_task(self._partial_run_agent_loop(sampling_params, trajectory_info[i], **kwargs))
                )
            outputs = await asyncio.gather(*tasks)
        except Exception:
            logger.exception("_partial_run_agent_loop failed")
            raise

        is_cancel = any(output.extra_fields.get("is_cancel", False) for output in outputs)
        if not is_cancel:
            output = self._postprocess(outputs)
            output = self._addition_process(output)
            return output, is_cancel
        return outputs, is_cancel

    def _addition_process(self, output: DataProto):
        """collect metirics"""
        metrics = output.meta_info.pop("metrics")  # List[Dict[str, str]]
        processing_times_list = [item["generate_sequences"] for item in metrics]
        tool_calls_times_list = [item["tool_calls"] for item in metrics]
        output.non_tensor_batch["processing_times"] = processing_times_list
        output.non_tensor_batch["tool_calls_times"] = tool_calls_times_list
        return output

    async def _partial_run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> AgentLoopOutput:
        # Completed, return directly
        if kwargs["output"] is not None and not kwargs["output"].extra_fields.get("is_cancel", False):
            logger.info("In _partial_run_agent_loop, already completed, return derictly!")
            return kwargs["output"]
        try:
            with rollout_trace_attr(
                step=trajectory["step"],
                sample_index=trajectory["sample_index"],
                rollout_n=trajectory["rollout_n"],
                validate=trajectory["validate"],
                name="agent_loop",
            ):
                assert agent_name in _agent_loop_registry, (
                    f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
                )

                agent_loop_config = _agent_loop_registry[agent_name]
                agent_loop = hydra.utils.instantiate(
                    config=agent_loop_config,
                    trainer_config=DictConfigWrap(config=self.config),
                    server_manager=self.server_manager,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    dataset_cls=self.dataset_cls,
                    dataset_config=self.config.data,
                )
                output: AgentLoopOutput = await agent_loop.run(
                    sampling_params, cancellation_event=self.cancellation_event, **kwargs
                )
                if not output.extra_fields.get("is_cancel", False):
                    kwargs.pop("output", None)
                    output = await self._agent_loop_postprocess(output, **kwargs)

                return output
        except Exception:
            logger.exception("Agent_loop run failed")
            raise

    async def cancel_agent_loops(self):
        """Set the shared cancellation event to stop all agent loops."""
        self.cancellation_event.set()

    async def resume_agent_loops(self):
        """Clear the shared cancellation event."""
        self.cancellation_event.clear()

    def get_server_loads(self) -> list[int]:
        return self.server_manager.get_server_loads()


class FullyAsyncAgentLoopManager(AgentLoopManager):
    def __init__(
        self, config: DictConfig, worker_group: RayWorkerGroup = None, rm_resource_pool: RayResourcePool = None
    ):
        self.config = config
        self.worker_group = worker_group
        self.reward_model_manager = None
        self.reward_router_address = None
        self.agent_loop_workers_class = FullyAsyncAgentLoopWorker
        self.rollout_replica_class = FullyAsyncvLLMReplica

        self.rm_resource_pool = rm_resource_pool
        self.rollout_replicas = None
        self.server_handles = None
        self.server_addresses = None
        self.agent_loop_workers = None

    @classmethod
    async def create(
        cls, config: DictConfig, worker_group: RayWorkerGroup = None, rm_resource_pool: RayResourcePool = None
    ):
        instance = cls(config, worker_group, rm_resource_pool)
        await instance._async_init()
        return instance

    async def _async_init(self):
        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            from verl.experimental.reward_loop import RewardModelManager

            self.reward_model_manager = RewardModelManager(self.config.reward_model, self.rm_resource_pool)
            self.reward_router_address = self.reward_model_manager.get_router_address()

        await self._initialize_llm_servers_async()
        self._init_agent_loop_workers()

    async def _initialize_llm_servers_async(self):
        rollout_world_size = (
            self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
            * self.config.actor_rollout_ref.rollout.data_parallel_size
            * self.config.actor_rollout_ref.rollout.pipeline_model_parallel_size
        )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        )
        num_replicas = world_size // rollout_world_size

        rollout_config = self.config.actor_rollout_ref.rollout
        model_config = self.config.actor_rollout_ref.model
        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.trainer.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        if self.worker_group:
            await asyncio.gather(*[server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        else:
            await asyncio.gather(*[server.init_standalone() for server in self.rollout_replicas])

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

        print(f"AgentLoopManager: {self.server_addresses}")
        # Update Prometheus configuration with server addresses
        if rollout_config.prometheus.enable:
            if rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            await asyncio.to_thread(update_prometheus_config, rollout_config.prometheus, self.server_addresses)

    async def generate_single_sample_async(
        self,
        sample: DataProto,
        partial_output_list: Optional[list[AgentLoopOutput]],
    ) -> tuple[list[AgentLoopOutput], bool] | tuple[DataProto, bool]:
        """
        Asynchronously process a single sample

        Args:
            sample: Single sample data
            partial_output_list: Optional[List[AgentLoopOutput]]: already rollout result.

        Returns:
            list[AgentLoopOutput]: Processing results
        """
        worker = self._select_best_worker()
        output_future = worker.generate_sequences_no_post.remote(sample, partial_output_list)
        return await asyncio.wrap_future(output_future.future())

    def _select_best_worker(self):
        """Select the best worker, simple round-robin load balancing"""
        if not hasattr(self, "_worker_index"):
            self._worker_index = 0

        worker = self.agent_loop_workers[self._worker_index]
        self._worker_index = (self._worker_index + 1) % len(self.agent_loop_workers)
        return worker

    async def cancel(self):
        worker_cancel_tasks = [worker.cancel_agent_loops.remote() for worker in self.agent_loop_workers]
        rollout_cancel_tasks = [replica.cancel() for replica in self.rollout_replicas]
        await asyncio.gather(*rollout_cancel_tasks, *worker_cancel_tasks)

    async def resume(self):
        rollout_resume_tasks = [replica.resume() for replica in self.rollout_replicas]
        worker_resume_tasks = [worker.resume_agent_loops.remote() for worker in self.agent_loop_workers]
        await asyncio.gather(*rollout_resume_tasks, *worker_resume_tasks)

    async def wake_up(self):
        await asyncio.gather(*[replica.wake_up() for replica in self.rollout_replicas])

    async def sleep(self):
        await asyncio.gather(*[replica.sleep() for replica in self.rollout_replicas])

    async def clear_kv_cache(self):
        await asyncio.gather(*[replica.clear_kv_cache() for replica in self.rollout_replicas])

    async def get_server_loads(self) -> list[int]:
        worker_refs = [worker.get_server_loads.remote() for worker in self.agent_loop_workers]

        worker_loads_list = await asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in worker_refs])

        # 聚合
        aggregated = [0] * len(self.server_handles)

        for loads in worker_loads_list:
            for i, val in enumerate(loads):
                aggregated[i] += val

        return aggregated
