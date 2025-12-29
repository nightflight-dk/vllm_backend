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

import base64
import json
from abc import abstractmethod
from io import BytesIO
from typing import Callable, Dict, List, Optional

import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image
from vllm.inputs.data import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.outputs import (
    EmbeddingOutput,
    EmbeddingRequestOutput,
    PoolingOutput,
    PoolingRequestOutput,
    RequestOutput,
)
from vllm.pooling_params import PoolingParams
from vllm.utils import random_uuid

from utils.vllm_backend_utils import TritonSamplingParams


class RewardEmbeddingOutput:
    """Hidden state output of the model for the reward task."""
    embedding: List[float]

    @staticmethod
    def from_base(pooling_output: PoolingOutput):
        if pooling_output.data[-1] is None:
            raise ValueError("Pooling output is None")
        pooled_data = pooling_output.data[-1]
        return EmbeddingOutput(pooled_data.tolist())

    @property
    def hidden_size(self) -> int:
        return len(self.embedding)


class RewardRequestOutput(EmbeddingRequestOutput):
    """Request output for the reward task."""
    @staticmethod
    def from_base(request_output: PoolingRequestOutput):
        return RewardRequestOutput(
            request_id=request_output.request_id,
            outputs=RewardEmbeddingOutput.from_base(request_output.outputs),
            prompt_token_ids=request_output.prompt_token_ids,
            finished=request_output.finished,
        )


class RequestBase:
    def __init__(
        self,
        request,
        executor_callback: Callable,
        output_dtype: np.dtype,
        logger,
        tokenizer=None,
        truncation_strategy=None,
        max_model_len=None,
        correlation_id=None,
    ):
        self.triton_request = request
        self.executor_callback = executor_callback
        self.output_dtype = output_dtype
        self.logger = logger
        self.tokenizer = tokenizer
        self.truncation_strategy = truncation_strategy
        self.max_model_len = max_model_len
        self.id = random_uuid()
        self.correlation_id = correlation_id
        self.stream = False
        self.prepend_input = False

    def _truncate_prompt(self, prompt: str):
        if not self.tokenizer or not self.truncation_strategy or not self.max_model_len:
            return prompt

        # Tokenize
        # We assume tokenizer is a PreTrainedTokenizer compatible object
        tokens = self.tokenizer.encode(prompt)

        if len(tokens) <= self.max_model_len:
            return prompt

        # Truncate
        if self.truncation_strategy == "left":
            tokens = tokens[-self.max_model_len :]
        else:
            # Default to right truncation (keep start)
            tokens = tokens[: self.max_model_len]

        return TokensPrompt(prompt_token_ids=tokens)

    @abstractmethod
    def _get_input_tensors(self):
        raise NotImplementedError

    @abstractmethod
    def execute(self):
        raise NotImplementedError

    @abstractmethod
    def create_response(self, request_output, *args, **kwargs):
        raise NotImplementedError


class GenerateRequest(RequestBase):
    def __init__(
        self,
        request,
        executor_callback: Callable,
        output_dtype: np.dtype,
        logger,
        lora_repository: Optional[Dict[str, str]] = None,
        supported_loras: Optional[List[str]] = None,
        tokenizer=None,
        truncation_strategy=None,
        max_model_len=None,
        correlation_id=None,
    ):
        super().__init__(
            request,
            executor_callback,
            output_dtype,
            logger,
            tokenizer,
            truncation_strategy,
            max_model_len,
            correlation_id,
        )
        # Attributes for generate requests
        if lora_repository is not None:
            self.lora_repository = lora_repository
        if supported_loras is not None:
            self.supported_loras = supported_loras

    def _get_input_tensors(self):
        # prompt
        prompt = pb_utils.get_input_tensor_by_name(
            self.triton_request, "text_input"
        ).as_numpy()[0]
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")

        prompt = self._truncate_prompt(prompt)

        # image
        images = pb_utils.get_input_tensor_by_name(self.triton_request, "image")
        if images:
            images_vllm = []
            for image_np in images.as_numpy():
                image_b = base64.b64decode(image_np.decode("utf-8"))
                image_rgb = Image.open(BytesIO(image_b)).convert("RGB")
                images_vllm.append(image_rgb)
            if len(images_vllm) > 0:
                prompt = {
                    "prompt": prompt,
                    "multi_modal_data": {"image": images_vllm},
                }

        # stream
        stream = pb_utils.get_input_tensor_by_name(self.triton_request, "stream")
        if stream:
            stream = stream.as_numpy()[0]
        else:
            stream = False

        # prepend_input / exclude_input_in_output
        prepend_input = pb_utils.get_input_tensor_by_name(
            self.triton_request, "exclude_input_in_output"
        )
        if prepend_input:
            # When `exclude_input_in_output` is False, we want to prepend input prompt
            # to output, thus prepend_input should be True, and vice versa.
            prepend_input = not prepend_input.as_numpy()[0]
        elif prepend_input is None and stream:
            prepend_input = False
        else:
            prepend_input = True
        if prepend_input and stream:
            raise ValueError(
                "When streaming, `exclude_input_in_output` = False is not allowed."
            )

        # parameters / sampling_parameters
        # An alternative mechanism to receive serialized parameters as an input
        # tensor, because request parameters are not yet supported via BLS.
        sampling_parameters = pb_utils.get_input_tensor_by_name(
            self.triton_request, "sampling_parameters"
        )
        if sampling_parameters:
            parameters = sampling_parameters.as_numpy()[0].decode("utf-8")
        else:
            parameters = self.triton_request.parameters()

        # additional outputs
        additional_outputs = {
            "return_finish_reason": None,
            "return_cumulative_logprob": None,
            "return_logprobs": None,
            "return_num_input_tokens": None,
            "return_num_output_tokens": None,
        }
        for tensor_name in additional_outputs.keys():
            tensor = pb_utils.get_input_tensor_by_name(self.triton_request, tensor_name)
            if tensor:
                tensor = bool(tensor.as_numpy()[0])
            else:
                tensor = False
            additional_outputs[tensor_name] = tensor

        return prompt, stream, prepend_input, parameters, additional_outputs

    async def execute(self):
        (
            prompt,
            self.stream,
            self.prepend_input,
            parameters,
            self.additional_outputs,
        ) = self._get_input_tensors()

        sampling_params = TritonSamplingParams.from_dict(parameters, self.logger)
        lora_name = sampling_params.lora_name
        lora_request = None
        if lora_name is not None:
            lora_id = str(self.supported_loras.index(lora_name) + 1)
            lora_int_id = int(lora_id)
            lora_local_path = self.lora_repository[lora_name]
            lora_request = LoRARequest(lora_id, lora_int_id, lora_local_path)

        response_iterator = self.executor_callback(
            prompt, sampling_params, self.id, lora_request=lora_request
        )

        async for response in response_iterator:
            yield response

    def create_response(
        self,
        request_output: RequestOutput,
        request_output_state: dict,
        prepend_input: bool,
    ):
        output_tensors = []

        # text_output
        prepend_prompt = ""
        if "prev_lens_text_output" not in request_output_state:
            # this is the first response
            if prepend_input:
                prepend_prompt = request_output.prompt
            request_output_state["prev_lens_text_output"] = [0] * len(
                request_output.outputs
            )
        prev_lens = request_output_state["prev_lens_text_output"]
        text_output = [
            (prepend_prompt + output.text[prev_len:]).encode("utf-8")
            for output, prev_len in zip(request_output.outputs, prev_lens)
        ]
        request_output_state["prev_lens_text_output"] = [
            len(output.text) for output in request_output.outputs
        ]
        output_tensors.append(
            pb_utils.Tensor(
                "text_output", np.asarray(text_output, dtype=self.output_dtype)
            )
        )

        # finish_reason
        if self.additional_outputs["return_finish_reason"]:
            finish_reason = [
                str(output.finish_reason) for output in request_output.outputs
            ]
            output_tensors.append(
                pb_utils.Tensor(
                    "finish_reason", np.asarray(finish_reason, dtype=np.object_)
                )
            )

        # cumulative_logprob
        if self.additional_outputs["return_cumulative_logprob"]:
            cumulative_logprob = [
                output.cumulative_logprob for output in request_output.outputs
            ]
            output_tensors.append(
                pb_utils.Tensor(
                    "cumulative_logprob",
                    np.asarray(cumulative_logprob, dtype=np.float32),
                )
            )

        # logprobs
        # https://github.com/vllm-project/vllm/blob/v0.6.3.post1/vllm/sequence.py#L37-L58
        if self.additional_outputs["return_logprobs"]:
            if "prev_lens_logprobs" not in request_output_state:
                request_output_state["prev_lens_logprobs"] = [0] * len(
                    request_output.outputs
                )
            logprobs = []
            for i in range(len(request_output.outputs)):
                output = request_output.outputs[i]
                if output.logprobs is None:
                    logprobs.append("null".encode("utf-8"))
                    continue
                prev_len = request_output_state["prev_lens_logprobs"][i]
                request_output_state["prev_lens_logprobs"][i] = len(output.logprobs)
                logprobs_py = []
                for logprob_d_vllm in output.logprobs[prev_len:]:
                    logprob_d_py = {}
                    for token_id, logprob_vllm in logprob_d_vllm.items():
                        logprob_d_py[token_id] = {
                            "logprob": logprob_vllm.logprob,
                            "rank": logprob_vllm.rank,
                            "decoded_token": logprob_vllm.decoded_token,
                        }
                    logprobs_py.append(logprob_d_py)
                logprobs.append(json.dumps(logprobs_py).encode("utf-8"))
            output_tensors.append(
                pb_utils.Tensor("logprobs", np.asarray(logprobs, dtype=np.object_))
            )

        # num_input_tokens
        if self.additional_outputs["return_num_input_tokens"]:
            num_input_tokens = len(request_output.prompt_token_ids)
            output_tensors.append(
                pb_utils.Tensor(
                    "num_input_tokens", np.asarray(num_input_tokens, dtype=np.uint32)
                )
            )

        # num_output_tokens
        if self.additional_outputs["return_num_output_tokens"]:
            if "prev_lens_num_output_tokens" not in request_output_state:
                request_output_state["prev_lens_num_output_tokens"] = [0] * len(
                    request_output.outputs
                )
            prev_lens = request_output_state["prev_lens_num_output_tokens"]
            num_output_tokens = [
                (len(output.token_ids) - prev_len)
                for output, prev_len in zip(request_output.outputs, prev_lens)
            ]
            request_output_state["prev_lens_num_output_tokens"] = [
                len(output.token_ids) for output in request_output.outputs
            ]
            output_tensors.append(
                pb_utils.Tensor(
                    "num_output_tokens", np.asarray(num_output_tokens, dtype=np.uint32)
                )
            )

        return pb_utils.InferenceResponse(output_tensors=output_tensors)


class EmbedRequest(RequestBase):
    def __init__(
        self,
        request,
        executor_callback: Callable,
        output_dtype: np.dtype,
        logger,
        tokenizer=None,
        truncation_strategy=None,
        max_model_len=None,
        correlation_id=None,
    ):
        super().__init__(
            request,
            executor_callback,
            output_dtype,
            logger,
            tokenizer,
            truncation_strategy,
            max_model_len,
            correlation_id,
        )

    def _get_input_tensors(self):
        embedding_request = pb_utils.get_input_tensor_by_name(
            self.triton_request, "embedding_request"
        ).as_numpy()[0]
        embedding_request = json.loads(embedding_request.decode("utf-8"))
        # prompt
        prompt = embedding_request["input"]
        if isinstance(prompt, str):
            prompt = self._truncate_prompt(prompt)
        elif (
            isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], int)
        ):
            # Single list of token IDs
            # We can truncate this too if needed, but usually input IDs are already processed?
            # If we want to enforce truncation:
            if (
                self.truncation_strategy
                and self.max_model_len
                and len(prompt) > self.max_model_len
            ):
                if self.truncation_strategy == "left":
                    prompt = prompt[-self.max_model_len :]
                else:
                    prompt = prompt[: self.max_model_len]
            prompt = TokensPrompt(prompt_token_ids=prompt)

        # pooling_params
        pooling_params = self._to_pooling_params(embedding_request)

        # additional outputs
        additional_outputs = {
            "return_num_input_tokens": None,
            "return_num_output_tokens": None,
        }
        for tensor_name in additional_outputs.keys():
            tensor = pb_utils.get_input_tensor_by_name(self.triton_request, tensor_name)
            if tensor:
                tensor = bool(tensor.as_numpy()[0])
            else:
                tensor = False
            additional_outputs[tensor_name] = tensor

        return prompt, pooling_params, additional_outputs

    async def execute(self):
        (
            prompt,
            pooling_params,
            self.additional_outputs,
        ) = self._get_input_tensors()

        # Create PoolingParams for embeddings
        response_iterator = self.executor_callback(prompt, pooling_params, self.id)

        # Yield each response from the async iterator
        async for response in response_iterator:
            yield response

    def _to_pooling_params(self, embedding_request: dict):
        pooling_params_dict = embedding_request.get("pooling_params", {})

        pooling_params = PoolingParams(task="embed")
        dims = None
        if "dimensions" in pooling_params_dict:
            dims = pooling_params_dict["dimensions"][0]
            pooling_params = PoolingParams(dimensions=dims, task="embed")
        return pooling_params

    def create_response(self, request_output: PoolingRequestOutput[EmbeddingOutput]):
        output_tensors = []
        request_output = EmbeddingRequestOutput.from_base(request_output)

        # Extract embedding list from output
        embedding: list[float] = request_output.outputs.embedding
        output_tensors.append(
            pb_utils.Tensor(
                "text_output",
                np.asarray([json.dumps(embedding)], dtype=self.output_dtype),
            )
        )

        # num_input_tokens
        if self.additional_outputs["return_num_input_tokens"]:
            num_input_tokens = len(request_output.prompt_token_ids)
            output_tensors.append(
                pb_utils.Tensor(
                    "num_input_tokens", np.asarray(num_input_tokens, dtype=np.uint32)
                )
            )

        # For embeddings, num_output_tokens is 0 (no generation happened)
        if self.additional_outputs["return_num_output_tokens"]:
            output_tensors.append(
                pb_utils.Tensor("num_output_tokens", np.asarray(0, dtype=np.uint32))
            )

        return pb_utils.InferenceResponse(output_tensors=output_tensors)

class ScoreRequest(RequestBase):
    def __init__(
        self,
        request,
        executor_callback: Callable,
        output_dtype: np.dtype,
        logger,
        tokenizer=None,
        truncation_strategy=None,
        max_model_len=None,
        correlation_id=None,
    ):
        super().__init__(
            request,
            executor_callback,
            output_dtype,
            logger,
            tokenizer,
            truncation_strategy,
            max_model_len,
            correlation_id,
        )

    def _get_input_tensors(self):
        text_input = pb_utils.get_input_tensor_by_name(
            self.triton_request, "text_input"
        )
        if text_input:
            text_input = [t.decode("utf-8") for t in text_input.as_numpy()]
        
        query_input = pb_utils.get_input_tensor_by_name(
            self.triton_request, "query_input"
        )
        if query_input:
            query_input = [t.decode("utf-8") for t in query_input.as_numpy()]

        # additional outputs
        additional_outputs = {
            "return_num_input_tokens": None,
            "return_num_output_tokens": None,
        }
        for tensor_name in additional_outputs.keys():
            tensor = pb_utils.get_input_tensor_by_name(self.triton_request, tensor_name)
            if tensor:
                tensor = bool(tensor.as_numpy()[0])
            else:
                tensor = False
            additional_outputs[tensor_name] = tensor

        return text_input, query_input, additional_outputs

    async def execute(self):
        text_input, query_input, self.additional_outputs = self._get_input_tensors()
        
        # Determine task and construct prompts
        if query_input:
            task = "score"
            # Broadcast query if needed
            if len(query_input) == 1 and len(text_input) > 1:
                query_input = query_input * len(text_input)
            
            if len(query_input) != len(text_input):
                 raise ValueError(f"Query length {len(query_input)} does not match Text length {len(text_input)}")
            
            prompts = [f"{q} {t}" for q, t in zip(query_input, text_input)]
        else:
            task = "classify"
            prompts = text_input

        # Apply truncation
        prompts = [self._truncate_prompt(p) for p in prompts]

        pooling_params = PoolingParams(task=task)
        
        response_iterator = self.executor_callback(prompts, pooling_params, self.id)
        
        async for response in response_iterator:
            yield response

    def create_response(self, request_output):
        output_tensors = []
        
        # Extract data from PoolingRequestOutput
        # outputs is a list of PoolingOutput
        # We expect one output per request usually, but vLLM might batch?
        # No, request_output corresponds to one request ID.
        # But wait, we passed a list of prompts to executor_callback?
        # If we passed a list of prompts, does it return one RequestOutput with multiple outputs?
        # Or does it return multiple RequestOutputs?
        # engine.encode with a list of prompts returns a RequestOutput object?
        # No, engine.encode returns an AsyncIterator[RequestOutput].
        # If we sent multiple prompts in one call, vLLM treats them as a batch?
        # Actually, `engine.encode` signature is `prompts: List[str]`.
        # It returns an iterator.
        # The `RequestOutput` contains `outputs` which is `List[PoolingOutput]`.
        # If we have multiple prompts, do we get multiple outputs in the list?
        # Yes, `outputs` has length equal to number of prompts if we did a batch?
        # No, usually `RequestOutput` is for a single sequence group.
        # If we pass multiple prompts to `encode`, it might create multiple request objects internally?
        # Wait, `AsyncLLMEngine.encode` takes `request_id`. One ID.
        # So it treats the list of prompts as... what?
        # If I pass multiple prompts, it might be for beam search? No, that's generate.
        # For encode, it might be multiple sequences in one group?
        
        # Let's assume 1:1 mapping for now.
        # If we have multiple text inputs, we might need to make multiple requests or handle the batching.
        # But `TritonPythonModel` handles batching by receiving a batch of requests?
        # No, `execute` receives a list of `requests`.
        # `ScoreRequest` wraps ONE Triton request.
        # One Triton request can contain a BATCH of inputs (text_input has dims [batch_size]).
        # So `text_input` is a list of strings.
        # We pass this list to `engine.encode`.
        # Does `engine.encode` handle a list of prompts for a SINGLE request ID?
        # Yes, it seems so.
        
        data = []
        for output in request_output.outputs:
            # output.data is the embedding/score
            # It can be a list (embedding) or float (score)
            if hasattr(output, 'data'):
                 data.append(output.data)
            else:
                 data.append(None) # Should not happen

        # Serialize
        # If it's a list of arrays/lists, we can serialize to JSON
        # or if it's a simple array, we can return it directly?
        # The output type is TYPE_STRING (JSON) for text_output.
        # So we dump to JSON.
        
        # We need to handle numpy arrays if data contains them
        def default_serializer(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json_str = json.dumps(data, default=default_serializer)
        
        output_tensors.append(
            pb_utils.Tensor(
                "text_output",
                np.asarray([json_str], dtype=self.output_dtype),
            )
        )
        
        # num_input_tokens
        if self.additional_outputs["return_num_input_tokens"]:
            num_input_tokens = len(request_output.prompt_token_ids)
            output_tensors.append(
                pb_utils.Tensor(
                    "num_input_tokens", np.asarray(num_input_tokens, dtype=np.uint32)
                )
            )

        # num_output_tokens -> 0
        if self.additional_outputs["return_num_output_tokens"]:
            output_tensors.append(
                pb_utils.Tensor("num_output_tokens", np.asarray(0, dtype=np.uint32))
            )

        return pb_utils.InferenceResponse(output_tensors=output_tensors)
