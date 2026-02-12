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
import logging
import math
import os
import random
import time
from typing import Any, Optional, Sequence

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
    """Base implementation of fully async LLM server manager"""

    @rollout_trace_op
    async def generate_for_partial(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> tuple[list[Any], list[Any], Any] | tuple[Sequence[int], list[float], bool]:
        """Generate tokens from prompt ids, used for async partial.

        Args:
            request_id (str): Request ID for sticky session.
            prompt_ids (List[int]): List of prompt token IDs.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.
            image_data (Optional[List[Any]]): Optional image data for multimodal generation.
            video_data (Optional[List[Any]]): Optional video data for multimodal generation.

        Returns:
            Tuple: Generation output containing:
                - Element 0 (Sequence[int]): Generated response token IDs.
                - Element 1 (List[float]): Log probabilities for the response token IDs.
                - Element 2 (bool): Flag indicating cancellation status.
        """
        server = self._choose_server(request_id)
        output = await server.generate_for_partial.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
        )
        return output


# ==============================================
# Per-actor runtime load entry (V7.1 Helper)
# ==============================================
class _ActorLoadEntry:
    """Runtime load tracker for individual actor instances

    Tracks inflight requests, running weight, latency metrics, and concurrency
    limits for each actor/server instance to enable intelligent load balancing.
    """

    __slots__ = (
        "actor",
        "index",
        "inflight_requests",
        "running_weight",
        "max_concurrency",
        "avg_latency",
        "finished_count",
    )

    # Grace factor to tolerate minor latency fluctuations without penalty
    LATENCY_GRACE_FACTOR = 1.5

    def __init__(self, actor, index: int, max_concurrency: int):
        """Initialize a load tracking entry for an actor

        Args:
            actor: The actor/server instance to track.
            index: Unique identifier/index for the actor.
            max_concurrency: Maximum concurrent requests allowed for this actor.
        """
        self.actor = actor
        self.index = index
        self.inflight_requests: int = 0
        self.running_weight: float = 0.0
        self.max_concurrency = max_concurrency
        self.avg_latency: float = 0.0
        self.finished_count: int = 0

    def effective_load(self, baseline_latency: float = 1.0) -> float:
        """Calculate effective load with latency-based penalty

        The effective load accounts for both current inflight requests and
        a dynamic penalty based on how much the actor's latency exceeds
        the baseline latency across all available actors.

        Args:
            baseline_latency: Minimum latency across all available actors.

        Returns:
            float: Calculated effective load value.
        """
        base_load = self.inflight_requests + self.running_weight

        # Return base load if latency metrics are not available
        if baseline_latency <= 0 or self.avg_latency <= 0:
            return base_load

        # Apply latency penalty only when exceeding grace threshold
        grace_threshold = baseline_latency * self.LATENCY_GRACE_FACTOR
        if self.avg_latency <= grace_threshold:
            return base_load

        # Scale load by latency ratio when above threshold
        return base_load * (self.avg_latency / baseline_latency)

    def is_available(self) -> bool:
        """Check if the actor can accept more requests

        Returns:
            bool: True if inflight requests are below max concurrency limit.
        """
        return self.inflight_requests < self.max_concurrency

    def update_latency(self, latency: float):
        """Update rolling average latency with exponential moving average

        Uses 90% historical average + 10% new value to smooth out fluctuations.

        Args:
            latency: Latency of the most recently completed request (seconds).
        """
        if latency < 0:
            return

        self.finished_count += 1
        self.avg_latency = 0.9 * self.avg_latency + 0.1 * latency


# ==============================================
# Fully Async LLM Server Manager (Optimized)
# ==============================================
class FullyAsyncLLMServerManagerBalance(AsyncLLMServerManager):
    """V7.1 Latency-Aware Optimized Async LLM Server Manager

    Enhanced version with:
    1. Dynamic latency awareness and penalty calculation
    2. Token-based request weight estimation
    3. Optimized logging and monitoring capabilities
    4. Sticky session support with load-aware fallback
    """

    def __init__(
        self,
        config: Any,
        server_handles: list[Any],
        *,
        max_concurrency_per_server: int = 32,
        weight_alpha: float = 0.2,
        max_cache_size: int = 10000,
    ):
        """Initialize the enhanced async LLM server manager

        Args:
            config: Server configuration object.
            server_handles: List of actor/server handles to manage.
            max_concurrency_per_server: Max concurrent requests per server.
            weight_alpha: Scaling factor for token weight calculation.
            max_cache_size: Maximum size for request ID cache.
        """
        # Initialize parent class with base configuration
        super().__init__(config, server_handles, max_cache_size=max_cache_size)

        # Initialize subclass-specific properties
        self.weight_alpha = weight_alpha
        self._lock = asyncio.Lock()  # Protect async state modifications

        # Create load tracking entries for each server/actor
        self._entries: list[_ActorLoadEntry] = [
            _ActorLoadEntry(actor, idx, max_concurrency_per_server) for idx, actor in enumerate(self.server_handles)
        ]

        # Reset sticky session mapping (parent class cache lacks entry info)
        self.request_id_to_entry: LRUCache[str, _ActorLoadEntry] = LRUCache(maxsize=max_cache_size)

        # Request counter for periodic status logging
        self._request_counter = 0

    def _estimate_prompt_weight(self, prompt_ids) -> float:
        """Estimate request weight based on token count using log scaling

        Calculates a weighted value based on the number of tokens in the prompt,
        using log2 scaling to prevent excessive weighting of very long prompts.

        Args:
            prompt_ids: List of token IDs or list of lists of token IDs.

        Returns:
            float: Calculated weight value (0.0 if input is invalid/empty).
        """
        if not prompt_ids:
            return 0.0

        try:
            # Handle both flat list and list-of-lists token formats
            if isinstance(prompt_ids[0], int):
                total_tokens = len(prompt_ids)
            else:
                total_tokens = sum(len(p) for p in prompt_ids)
        except (TypeError, IndexError):
            # Return 0 weight for malformed input
            return 0.0

        if total_tokens <= 0:
            return 0.0

        # Calculate weighted value using log2 scaling
        return self.weight_alpha * math.log2(total_tokens + 1.0)

    async def _choose_server(self, request_id: str, *, prompt_ids) -> tuple[_ActorLoadEntry, float]:
        """Select optimal server using latency-aware load balancing (V7.1)

        Enhanced selection logic with:
        1. Sticky session support with load-aware fallback
        2. Latency-based effective load calculation
        3. Dynamic baseline latency adjustment
        4. Random selection among equally optimal servers

        Note: Extended override with additional prompt_ids parameter.

        Args:
            request_id: Unique identifier for the request.
            prompt_ids: Token IDs for weight calculation.

        Returns:
            Tuple[_ActorLoadEntry, float]: Selected entry and calculated weight.
        """
        # Calculate request weight based on token count
        request_weight = self._estimate_prompt_weight(prompt_ids)

        async with self._lock:
            # Periodic status logging (every 32 requests)
            self._request_counter += 1
            if self._request_counter % 32 == 0:
                self._log_status_unsafe()

            # 1. Check for existing sticky session
            sticky_entry = self.request_id_to_entry.get(request_id)
            if sticky_entry is not None:
                if sticky_entry.is_available():
                    # Reuse sticky server if it has capacity
                    sticky_entry.inflight_requests += 1
                    sticky_entry.running_weight += request_weight
                    print(
                        f"[Sticky] req={request_id[:6]} -> Server {sticky_entry.index} "
                        f"(inflight={sticky_entry.inflight_requests})"
                    )
                    return sticky_entry, request_weight
                else:
                    # Remove sticky mapping if server is overloaded
                    self.request_id_to_entry.pop(request_id)

            # 2. Select server using latency-aware load balancing
            # Filter to available servers first (fallback to all if none available)
            available_entries = [e for e in self._entries if e.is_available()]
            if not available_entries:
                available_entries = self._entries

            # Calculate dynamic baseline latency from valid measurements
            valid_latencies = [e.avg_latency for e in available_entries if e.avg_latency > 0]
            baseline_latency = min(valid_latencies) if valid_latencies else 1.0

            # Find servers with minimum effective load
            min_effective_load = min(e.effective_load(baseline_latency) for e in available_entries)
            optimal_entries = [e for e in available_entries if e.effective_load(baseline_latency) == min_effective_load]

            # Randomly select one of the optimal servers for load distribution
            chosen_entry = random.choice(optimal_entries)

            # Update load tracking for the chosen server
            chosen_entry.inflight_requests += 1
            chosen_entry.running_weight += request_weight
            self.request_id_to_entry[request_id] = chosen_entry

            # Log dispatch decision with load metrics
            print(
                f"[Dispatch] req={request_id[:6]} -> Server {chosen_entry.index} "
                f"(load={chosen_entry.effective_load(baseline_latency):.1f}, "
                f"inflight_requests={chosen_entry.inflight_requests:.1f}, "
                f"lat={chosen_entry.avg_latency:.1f}s)"
            )

            return chosen_entry, request_weight

    # ======================================================
    # Public API (overrides parent class methods)
    # ======================================================
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
        """Generate completions using V7.1 latency-aware scheduling

        Overrides parent class method to use enhanced server selection logic
        with latency awareness and token-based weighting.

        Args:
            request_id: Unique request identifier.
            prompt_ids: List of token IDs for the prompt.
            sampling_params: Generation parameters (temperature, top_p, etc.).
            image_data: Optional image data for multimodal generation.
            video_data: Optional video data for multimodal generation.

        Returns:
            Any: Generation result from the selected server.
        """
        # Select optimal server using enhanced logic
        entry, weight = await self._choose_server(request_id, prompt_ids=prompt_ids)
        start_time = time.time()

        try:
            # Execute generation on the selected actor
            result = await entry.actor.generate.remote(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
            )
            return result
        finally:
            # Update metrics even if request fails
            latency = time.time() - start_time
            async with self._lock:
                # Ensure non-negative values after decrement
                entry.inflight_requests = max(0, entry.inflight_requests - 1)
                entry.running_weight = max(0.0, entry.running_weight - weight)
                entry.update_latency(latency)

    @rollout_trace_op
    async def generate_for_partial(
        self,
        request_id: str,
        *,
        prompt_ids: list[int] | list[list[int]],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> tuple[Any, Any, Any] | tuple[Sequence[int], list[float], bool]:
        """Generate partial completions with V7.1 scheduling

        Specialized generation method for partial token streaming with
        the same latency-aware scheduling as the main generate method.

        Args:
            request_id: Unique request identifier.
            prompt_ids: Token IDs (flat list or list of lists).
            sampling_params: Generation parameters.
            image_data: Optional image data for multimodal generation.
            video_data: Optional video data for multimodal generation.

        Returns:
            Tuple: Partial generation result from the selected server.
        """
        # Select optimal server using enhanced logic
        entry, weight = await self._choose_server(request_id, prompt_ids=prompt_ids)
        start_time = time.time()

        try:
            # Execute partial generation on the selected actor
            output = await entry.actor.generate_for_partial.remote(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
            )
            return output
        finally:
            # Update metrics even if request fails
            latency = time.time() - start_time
            async with self._lock:
                # Ensure non-negative values after decrement
                entry.inflight_requests = max(0, entry.inflight_requests - 1)
                entry.running_weight = max(0.0, entry.running_weight - weight)
                entry.update_latency(latency)

    def _log_status_unsafe(self):
        """Log server status snapshot (NOT thread-safe, must hold lock)

        Generates a compact status summary showing inflight requests,
        average latency, and effective load for each server instance.
        """
        # Calculate baseline latency for load normalization
        valid_latencies = [e.avg_latency for e in self._entries if e.avg_latency > 0]
        baseline_latency = min(valid_latencies) if valid_latencies else 1.0

        # Build status string for each server
        server_status_parts = []
        for entry in self._entries:
            adjusted_load = entry.effective_load(baseline_latency)
            server_status_parts.append(
                f"S{entry.index}({entry.inflight_requests}|{entry.avg_latency:5.1f}s|{adjusted_load:4.1f})"
            )

        # Calculate total system load
        total_inflight = sum(e.inflight_requests for e in self._entries)

        # Log consolidated status
        logger.debug(
            f"[Snapshot] Total={total_inflight} Base={baseline_latency:.2f}s | {' '.join(server_status_parts)}"
        )


# ==============================================
# Main Entry Point (Environment Variable Control)
# ==============================================
# Dynamically select implementation based on environment variable
if int(os.environ.get("USE_BALANCE_LOAD")):
    FullyAsyncLLMServerManager = FullyAsyncLLMServerManagerBalance
    print("Initialized FullyAsyncLLMServerManager with FullyAsyncLLMServerManagerBalance")
else:
    print("Initialized FullyAsyncLLMServerManager with FullyAsyncLLMServerManager")


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
