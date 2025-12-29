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

import json
from typing import Any, Dict, List, Optional, Union, cast

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
    apply_hf_chat_template,
    apply_mistral_chat_template,
    parse_chat_messages,
    resolve_chat_template_content_format,
)
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.transformers_utils.tokenizer_base import TokenizerBase
from vllm.transformers_utils.tokenizers import MistralTokenizer

AnyTokenizer = Union[
    PreTrainedTokenizer, PreTrainedTokenizerFast, TokenizerBase, MistralTokenizer
]


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

def get_conversation_prompt(
    messages: list,
    tokenizer: AnyTokenizer,
    model_config: ModelConfig,
    chat_template: str = None,
    chat_template_content_format: ChatTemplateContentFormatOption = "auto",
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
    tools: Optional[list[dict[str, Any]]] = None,
    multi_modal_data: Optional[dict[str, Any]] = None,
    mm_processor_kwargs: Optional[dict[str, Any]] = None,
) -> TokensPrompt:
    """
    This function generates a vLLM conversation prompt based on the input messages.
    """
    
    msgs = cast(List[ChatCompletionMessageParam], messages)

    resolved_content_format = resolve_chat_template_content_format(
        chat_template,
        tools,
        chat_template_content_format,
        tokenizer,
        model_config=model_config
    )

    # Handle multimodal content in messages
    conversation, mm_data, _ = parse_chat_messages(
        msgs,
        model_config,
        tokenizer,
        content_format=resolved_content_format,
    )
    
    # If external multi_modal_data is provided, merge it with parsed data
    if mm_data is None:
        mm_data = multi_modal_data
    elif multi_modal_data is not None:
        mm_data.update(multi_modal_data)

    if isinstance(tokenizer, MistralTokenizer):
        prompt_token_ids = apply_mistral_chat_template(
            tokenizer,
            messages=msgs,
            chat_template=chat_template,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
        )
    else:
        prompt_str = apply_hf_chat_template(
            tokenizer,
            trust_remote_code=model_config.trust_remote_code,
            conversation=conversation,
            chat_template=chat_template,
            tools=tools,
            model_config=model_config,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
        )
        # Special tokens are already included in chat templates so
        # should not be added by the tokenizer in this case.
        prompt_token_ids = tokenizer.encode(prompt_str,
                                          add_special_tokens=False)

    prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

    if mm_data is not None:
        prompt["multi_modal_data"] = mm_data

    if mm_processor_kwargs is not None:
        prompt["mm_processor_kwargs"] = mm_processor_kwargs

    return prompt
