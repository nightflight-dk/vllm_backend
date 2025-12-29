# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import asyncio
import json
import random
from typing import Dict, Optional

from vllm.engine.protocol import EngineClient
from vllm.inputs import TextPrompt
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.utils import random_uuid


class TritonSamplingParams(SamplingParams):
    """
    Extended sampling parameters for text generation via
    Triton Inference Server and vLLM backend.

    Attributes:
        lora_name (Optional[str]): The name of the LoRA (Low-Rank Adaptation)
        to use for inference.
    """

    lora_name: Optional[str] = None

    def __repr__(self) -> str:
        """
        Returns a string representation of the `TritonSamplingParams` object.

        This method overrides the `__repr__` method of the parent class
        to include additional attributes in the string representation.

        Returns:
            A string representation of the object.
        """
        base = super().__repr__()
        return f"{base}, lora_name={self.lora_name}"

    @staticmethod
    def from_dict(
        params_dict_str: str, logger: "pb_utils.Logger"
    ) -> "TritonSamplingParams":
        """
        Creates a `TritonSamplingParams` object from a dictionary string.

        This method parses a JSON string containing sampling parameters,
        converts the values to appropriate types, and creates a
        `TritonSamplingParams` object.

        Args:
            params_dict (str): A JSON string containing sampling parameters.
            logger (pb_utils.Logger): Triton Inference Server logger object.

        Returns:
            TritonSamplingParams: An instance of TritonSamplingParams.
        """
        try:
            params_dict = json.loads(params_dict_str)
            vllm_params_dict = SamplingParams.__annotations__
            type_mapping = {
                int: int,
                float: float,
                bool: bool,
                str: str,
                Optional[int]: int,
            }
            for key, value in params_dict.items():
                if key == "structured_outputs":
                    params_dict[key] = StructuredOutputsParams(**json.loads(value))
                elif key in vllm_params_dict:
                    vllm_type = vllm_params_dict[key]
                    if vllm_type in type_mapping:
                        params_dict[key] = type_mapping[vllm_type](params_dict[key])

            return TritonSamplingParams(**params_dict)

        except Exception as e:
            logger.log_error(
                f"[vllm] Was trying to create `TritonSamplingParams`, but got exception: {e}"
            )
            return None

async def engine_warmup(engine: EngineClient, model_task: str, lora_repo: Optional[Dict[str, str]]) -> None:
    """
    Warm up the vLLM engine by running a dummy task.
    """
    lora_requests = [None]
    if lora_repo:
        lora_requests = [
            LoRARequest(lora_name=k, lora_int_id=random.randint(0, 10**6), lora_path=v)
            for k, v in lora_repo.items()]

    test_prompt = TextPrompt(prompt="Hello, world?")

    async def fetch_output(async_generator):
        last_output = None
        async for output in async_generator:
            last_output = output
        return last_output

    warmup_tasks = []

    # Note: Using string literals for tasks as SupportedVLLMTask enum is not yet available in this file
    if model_task in ["embed", "score", "classify", "reward"]:
         for lora_request in lora_requests:
            warmup_tasks.append(asyncio.create_task(
                fetch_output(engine.encode(test_prompt, PoolingParams(task=model_task, skip_reading_prefix_cache=False),
                              request_id=random_uuid(), lora_request=lora_request))
            ))
    elif model_task in ["generate", None]:
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1
        )
        for lora_request in lora_requests:
            warmup_tasks.append(asyncio.create_task(
                fetch_output(engine.generate(test_prompt, sampling_params=sampling_params,
                                request_id=random_uuid(), lora_request=lora_request))
            ))

    if warmup_tasks:
        # Use return_exceptions=False so first exception will propagate
        await asyncio.gather(*warmup_tasks, return_exceptions=False)
        await engine.check_health()
