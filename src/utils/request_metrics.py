# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
"""Utilities for the AI.Generative Triton vLLM engines."""
import json
from dataclasses import asdict, dataclass
from typing import List, Optional

from vllm import RequestOutput
from vllm.config import ParallelConfig
from vllm.sequence import RequestMetrics


@dataclass
class TritonVLLMRequestMetrics(RequestMetrics):
    # The following fields are not part of the original RequestMetrics class
    # but are added for Triton-specific metrics.
    t2_first_token_ms: Optional[int] = None
    t2_last_token_ms: Optional[int] = None
    t2_decode_ms: Optional[int] = None
    t2_encode_ms: Optional[int] = None
    mean_itl_ms: Optional[int] = None
    tokens_in_len: Optional[List[int]] = None
    tokens_in_total: Optional[int] = None
    tokens_out_len: Optional[list[int]] = None
    gpu_total_time_ms: Optional[int] = None
    total_gpus: Optional[int] = None


    @classmethod
    def for_vllm_request_output(cls, ro: RequestOutput, parallel_config: ParallelConfig) -> "TritonVLLMRequestMetrics":
        """
        Creates a TritonRequestMetrics instance from a RequestOutput object.
        Calculates:
        - t2_first_token_ms: Time from arrival to first token in milliseconds.
        - t2_last_token_ms: Time from arrival to last token in milliseconds.
        - t2_decode_ms: Time from first token to finished time in milliseconds.
        - mean_itl_ms: Average inter-token latency in milliseconds.
        - prompt_token_count: Number of tokens in the prompt.
        - output_token_count: List of token counts for each output.
        - total_gpu_time_ms: Total GPU compute time, as elapsed time (from first scheduled or arrival to last token)
                       multiplied by pipeline_parallel_size * tensor_parallel_size.
        Args:
            ro: A RequestOutput from the vLLM Engine.
            vllm_conf: The VllmConfig which contains parallel config. Expected to have a 'parallel' attribute that is a dict
                       with keys 'pipeline_parallel_size' and 'tensor_parallel_size'. Defaults to 1 if not present.
        Returns:
            A TritonRequestMetrics object with the calculated metrics.
        """
        base_metrics = ro.metrics

        completion_tokens = [len(output.token_ids) for output in ro.outputs]
        num_tokens = sum(completion_tokens) if completion_tokens else None

        # In vLLM 0.11.x, metrics field names changed from *_time to *_ts
        # Get values using new names first, fall back to old names
        def get_timestamp(obj, name):
            """Get timestamp field, trying _ts suffix first, then _time"""
            ts_name = name.replace('_time', '_ts')
            return getattr(obj, ts_name, getattr(obj, name, None))
        
        arrival_time = get_timestamp(base_metrics, 'arrival_time')
        first_token_time = get_timestamp(base_metrics, 'first_token_time')
        last_token_time = get_timestamp(base_metrics, 'last_token_time')
        finished_time = get_timestamp(base_metrics, 'finished_time')
        first_scheduled_time = get_timestamp(base_metrics, 'first_scheduled_time')

        # Times in seconds.
        ttf_s = (
            first_token_time - arrival_time
            if first_token_time is not None and arrival_time is not None
            else None
        )
        ttl_s = (
            last_token_time - arrival_time
            if last_token_time is not None and arrival_time is not None
            else None
        )
        decoding_s = (
            finished_time - first_token_time
            if (finished_time is not None and first_token_time is not None)
            else None
        )

        avg_latency_s = None
        if (
            first_token_time is not None
            and last_token_time is not None
            and num_tokens is not None
            and num_tokens > 1
        ):
            avg_latency_s = (last_token_time - first_token_time) / (num_tokens - 1)

        # In vLLM 0.11.x, metrics use monotonic time (time.monotonic()) not Unix timestamps.
        # We can use the relative time differences for ms calculations, but not the absolute timestamps.
        # Only calculate ms metrics if we have valid relative times.
        
        # Convert seconds to milliseconds for relative durations
        t2_first_token_ms = int(ttf_s * 1000) if ttf_s is not None and ttf_s > 0 else None
        t2_last_token_ms = int(ttl_s * 1000) if ttl_s is not None and ttl_s > 0 else None
        t2_decode_ms = int(decoding_s * 1000) if decoding_s is not None and decoding_s > 0 else None
        mean_itl_ms = int(avg_latency_s * 1000) if avg_latency_s is not None and avg_latency_s > 0 else None

        # Calculate total GPU time
        # We use (last_token_time - first_scheduled_time) as the active time on GPU
        # If first_scheduled_time is not available, use arrival_time
        start_time = first_scheduled_time if first_scheduled_time is not None else arrival_time
        end_time = last_token_time if last_token_time is not None else finished_time
        
        gpu_total_time_ms = None
        total_gpus = 1
        if parallel_config:
            total_gpus = parallel_config.pipeline_parallel_size * parallel_config.tensor_parallel_size
            
        if start_time is not None and end_time is not None:
            duration_s = end_time - start_time
            if duration_s > 0:
                gpu_total_time_ms = int(duration_s * 1000 * total_gpus)

        # Create new instance with all fields from base_metrics plus our new ones
        # We need to be careful not to pass extra args if RequestMetrics doesn't accept them in __init__
        # But we are subclassing it.
        # Actually, RequestMetrics is a dataclass.
        # We can create a new instance of TritonVLLMRequestMetrics.
        
        # Copy fields from base_metrics
        kwargs = asdict(base_metrics)
        
        # Add our new fields
        kwargs.update({
            "t2_first_token_ms": t2_first_token_ms,
            "t2_last_token_ms": t2_last_token_ms,
            "t2_decode_ms": t2_decode_ms,
            "mean_itl_ms": mean_itl_ms,
            "tokens_in_len": [len(ro.prompt_token_ids)],
            "tokens_in_total": len(ro.prompt_token_ids),
            "tokens_out_len": completion_tokens,
            "gpu_total_time_ms": gpu_total_time_ms,
            "total_gpus": total_gpus
        })
        
        return cls(**kwargs)

    def to_json(self):
        return json.dumps(asdict(self))
