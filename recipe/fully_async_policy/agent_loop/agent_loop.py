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
DEFAULT_EMA_DECAY = 0.1  # Higher = more sensitive to recent samples
DEFAULT_RT_WEIGHT = 0.3  # Response time weight in composite score
DEFAULT_OVERFLOW_RATIO = 1.25  # Trigger protection when active > theory * ratio
LOG_INTERVAL = 128  # Log load status every N requests


# ==============================================
# Server State
# ==============================================


@dataclass
class ServerState:
    """
    Runtime state for a single vLLM server.

    Only two effective signals are maintained:
    - active: Real-time concurrent request count (decremented on completion)
    - ema_rt: Exponential moving average of response time (reflects server speed)
    """

    index: int
    actor: Any

    active: int = 0  # Real-time inflight request count
    ema_rt: float = 1.0  # Response time EMA (seconds), initial 1.0 = no prior
    total_requests: int = 0  # Cumulative request count (for monitoring)
    total_time: float = 0.0  # Cumulative time (for monitoring)

    def update_rt(self, response_time: float, decay: float) -> None:
        """Update EMA response time after request completion."""
        self.ema_rt = decay * response_time + (1 - decay) * self.ema_rt
        self.ema_rt = max(self.ema_rt, 1e-3)  # Prevent approaching zero

    def load_score(self, global_avg_rt: float, rt_weight: float) -> float:
        """
        Calculate composite load score (lower is better).

        Formula:
            score = (1 - rt_weight) * active + rt_weight * active * (ema_rt / global_avg_rt)

        Derivation:
        - Base term is active (real-time concurrent count)
        - Multiplied by rt_weight * (ema_rt / global_avg_rt) as penalty for slow servers
        - When all servers have same ema_rt, rt_ratio=1, score degenerates to pure active sorting
        - When a server is persistently slow, rt_ratio > 1, its score is amplified
        - rt_weight=0 completely degenerates to real-time active sorting

        Args:
            global_avg_rt: Global average response time for normalization.
            rt_weight: Weight for response time penalty (0.0 ~ 0.5).

        Returns:
            Composite load score (lower is better).
        """
        if global_avg_rt <= 0:
            return float(self.active)

        rt_ratio = self.ema_rt / global_avg_rt
        score = (1 - rt_weight) * self.active + rt_weight * self.active * rt_ratio

        # active=0 servers have score=0, naturally prioritized
        return score


# ==============================================
# Main Load Balancer Class
# ==============================================


class FullyAsyncLLMServerManagerBalance(AsyncLLMServerManager):
    """
    Optimized Load Balancer: Real-time Active + EMA RT + Threshold Protection.

    Improvements over Original Scheme:
    1. Corrected counting semantics: active inflight, decrement on completion
       (Original: monotonic increment, degenerates to round-robin)
    2. EMA response time: Identifies persistently slow servers (effective in service deployment)
    3. Threshold protection: Fallback to pure active sorting under global overload
    4. Stable sticky sessions: Only break when server is unavailable (KV cache priority)
    5. Behaves identically to original round-robin under balanced load (no additional risk)

    Abandoned V10 Designs:
    - effective_load (exponential decay progress estimation, incompatible with LLM autoregressive generation)
    - Variance-driven strategy switching (based on distorted effective_load, unreliable)
    - Aggressive sticky session breaking (KV cache benefit > minor load imbalance cost)
    """

    def __init__(
        self,
        config: Any,
        server_handles: list[Any],
        *,
        max_cache_size: int = DEFAULT_MAX_CACHE_SIZE,
        ema_decay: float = DEFAULT_EMA_DECAY,
        rt_weight: float = DEFAULT_RT_WEIGHT,
        overflow_ratio: float = DEFAULT_OVERFLOW_RATIO,
    ) -> None:
        """
        Initialize the optimized load balancer.

        Args:
            config: Global configuration object (OmegaConf DictConfig format).
            server_handles: List of Ray Actor handles for vLLM server replicas.
            max_cache_size: Maximum size for sticky session LRU cache.
            ema_decay: EMA decay factor for response time updates (0.05 ~ 0.3).
            rt_weight: Response time weight in composite score (0.0 ~ 0.5).
            overflow_ratio: Threshold ratio for overload protection (1.0 ~ 2.0).
        """
        super().__init__(config, server_handles, max_cache_size=max_cache_size)

        self.ema_decay = ema_decay
        self.rt_weight = rt_weight
        self.overflow_ratio = overflow_ratio
        self.num_servers = len(server_handles)

        # Server state table
        self.states: dict[int, ServerState] = {
            idx: ServerState(index=idx, actor=actor) for idx, actor in enumerate(self.server_handles)
        }

        # Actor handle -> index, O(1) reverse lookup
        self._actor_to_idx: dict[Any, int] = {actor: idx for idx, actor in enumerate(self.server_handles)}

        # Sticky session cache: request_id -> ServerState
        # Reuse parent class LRU cache key space, but store ServerState instead of actor handle
        self.request_id_to_state: LRUCache = LRUCache(maxsize=max_cache_size)

        # Monitoring
        self._request_counter = 0

        logger.info(
            f"[Balance Init] servers={self.num_servers}, "
            f"ema_decay={ema_decay}, rt_weight={rt_weight}, "
            f"overflow_ratio={overflow_ratio}"
        )

    # ==============================================
    # Internal Utilities
    # ==============================================

    def _global_avg_rt(self) -> float:
        """Calculate global average EMA RT across all servers for rt_ratio normalization."""
        return sum(s.ema_rt for s in self.states.values()) / max(self.num_servers, 1)

    def _theory_per_server(self) -> float:
        """
        Calculate theoretical uniform concurrent count per server.

        Formula: total_active / num_servers

        This is dynamically calculated without external configuration dependencies.
        Corresponds to document formula: total_sample/num_workers*n/num_servers

        Returns:
            Theoretical average active requests per server.
        """
        total = sum(s.active for s in self.states.values())
        return total / max(self.num_servers, 1)

    def _get_max_active(self) -> int:
        """
        Get maximum allowed active requests per server.

        Currently uses a simple heuristic: theory * overflow_ratio.
        Can be extended with configuration-based limits if needed.

        Returns:
            Maximum active requests per server.
        """
        theory = self._theory_per_server()
        return max(1, int(theory * self.overflow_ratio))

    def _best_state(self) -> ServerState:
        """
        Core scheduling logic: select server with lowest composite score.

        Process:
        1. Calculate load_score for each server
        2. Select candidate with lowest score (best)
        3. Threshold check: if best.active > theory * overflow_ratio,
           all servers are under high load, fallback to pure active minimum (safeguard)

        Returns:
            Selected ServerState instance.
        """
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

        assert best is not None, "No server available"

        # Threshold protection: if best is still overloaded, global high load detected
        # Fallback to pure active minimum sorting (rt_weight penalty may not be optimal here)
        if best.active > threshold and theory > 0:
            best = min(self.states.values(), key=lambda s: s.active)

        return best

    # ==============================================
    # Scheduling Entry Points
    # ==============================================

    def _choose_server(
        self,
        request_id: str,
        prompt_ids: Optional[list[int]] = None,
    ) -> ServerState:
        """
        Select server and update active count (+1).

        Priority:
        1. Sticky session: request_id already bound, and corresponding server active > 0
           Note: Do not actively break sticky sessions, KV cache benefit priority
        2. Composite score scheduling: server with lowest load_score

        Args:
            request_id: Unique request identifier for sticky session lookup.
            prompt_ids: Optional prompt token IDs (for future enhancements).

        Returns:
            Selected ServerState instance.
        """
        # 1. Sticky session
        if request_id in self.request_id_to_state:
            state = self.request_id_to_state[request_id]
            # Check if sticky server is still available (not overloaded)
            if state.active < self._get_max_active():
                state.active += 1
                return state
            else:
                # Sticky server overloaded, remove sticky and reselect
                self.request_id_to_state.pop(request_id, None)

        # 2. Composite score scheduling
        state = self._best_state()
        state.active += 1
        state.total_requests += 1
        self.request_id_to_state[request_id] = state

        # Monitoring log
        self._request_counter += 1
        if self._request_counter % LOG_INTERVAL == 0:
            self._log_status()

        return state

    def _release(self, state: ServerState, response_time: float) -> None:
        """
        Request completion callback: decrement active and update EMA RT.

        Called in finally block to ensure count leakage prevention under exceptions.

        Args:
            state: ServerState for the server that handled the request.
            response_time: Measured response time in seconds.
        """
        state.active = max(0, state.active - 1)
        state.total_time += response_time
        state.update_rt(response_time, self.ema_decay)

    # ==============================================
    # Public APIs
    # ==============================================

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
        """
        Generate response tokens for the given prompt.

        Args:
            request_id: Unique identifier for request tracking and sticky session.
            prompt_ids: Token IDs of the input prompt.
            sampling_params: Generation parameters (temperature, top_p, etc.).
            image_data: Optional image data for multimodal models.
            video_data: Optional video data for multimodal models.

        Returns:
            TokenOutput containing generated tokens and metadata.
        """
        state = self._choose_server(request_id, prompt_ids=prompt_ids)
        start = time.monotonic()  # Use monotonic to avoid system clock jumps

        try:
            result = await state.actor.generate.remote(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
            )
            return result
        finally:
            self._release(state, time.monotonic() - start)

    @rollout_trace_op
    async def generate_for_partial(
        self,
        request_id: str,
        *,
        prompt_ids: list[int] | list[list[int]],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> Any:
        """
        Generate partial responses for rollout continuation.

        Args:
            request_id: Unique identifier for request tracking.
            prompt_ids: Token IDs of the input prompt(s).
            sampling_params: Generation parameters.
            image_data: Optional image data for multimodal models.
            video_data: Optional video data for multimodal models.

        Returns:
            Tuple containing output tokens, log probabilities, and completion status.
        """
        state = self._choose_server(request_id, prompt_ids=prompt_ids)
        start = time.monotonic()

        try:
            output = await state.actor.generate_for_partial.remote(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
            )
            return output
        finally:
            self._release(state, time.monotonic() - start)

    # ==============================================
    # Monitoring
    # ==============================================

    def get_server_loads(self) -> list[int]:
        """
        Compatible with original interface, return real-time active count per server.

        Returns:
            List of active request counts for each server.
        """
        return [self.states[i].active for i in range(self.num_servers)]

    def get_load_metrics(self) -> dict[str, Any]:
        """
        Detailed monitoring metrics for external metrics system collection.

        Returns:
            Dictionary containing:
            - active: Per-server active request counts
            - ema_rt: Per-server EMA response times
            - total_requests: Per-server cumulative request counts
            - global_avg_rt: Global average response time
            - rt_weight: Current response time weight configuration
            - overflow_ratio: Current overflow ratio configuration
        """
        return {
            "active": {i: s.active for i, s in self.states.items()},
            "ema_rt": {i: round(s.ema_rt, 3) for i, s in self.states.items()},
            "total_requests": {i: s.total_requests for i, s in self.states.items()},
            "global_avg_rt": round(self._global_avg_rt(), 3),
            "rt_weight": self.rt_weight,
            "overflow_ratio": self.overflow_ratio,
        }

    def _log_status(self) -> None:
        """Log current load distribution status (called every LOG_INTERVAL requests)."""
        metrics = self.get_load_metrics()
        logger.info(
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
    logger.info("Using optimized FullyAsyncLLMServerManagerBalance")
else:
    logger.info("Using base FullyAsyncLLMServerManager")


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
