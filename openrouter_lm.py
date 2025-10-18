import logging
import os.path
from typing import Any, List

import httpx
import openai
import dotenv
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception, before_sleep_log
from ismcore.processor.base_processor_lm import BaseProcessorLM
from ismcore.processor.monitored_processor_state import MonitoredUsage
from ismcore.utils.general_utils import parse_response
from ismcore.utils.ism_logger import ism_logger
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, APIStatusError

dotenv.load_dotenv()

OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', None)
openai.api_key = OPENROUTER_API_KEY

logger = ism_logger(__name__)
logger.info(f'**** OPENROUTER API KEY (last 4 chars): {OPENROUTER_API_KEY[-4:]} ****')



def _is_retryable(e: BaseException) -> bool:
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code if e.response else None
        return status in (429, 500, 502, 503, 504)
    return isinstance(e, (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout))


class OpenRouterChatCompletionProcessor(BaseProcessorLM, MonitoredUsage):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        MonitoredUsage.__init__(self, **kwargs)

    async def _stream(self, input_data: Any, template: str):

        if not template:
            template = str(input_data)

        # rendered message we want to submit to the model
        message_list = self.derive_messages_with_session_data_if_any(template=template, input_data=input_data)
        # TODO FLAG: OFF history flag injected here
        # TODO FEATURE: CONFIG PARAMETERS -> EMBEDDINGS

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

        # Processor Runtime Properties
        properties = self.properties

        # Create a streaming completion
        stream = client.chat.completions.create(
            model=self.provider.version,
            messages=message_list,
            max_tokens=properties.max_tokens,
            temperature=properties.temperature,
            top_p=properties.top_p,
            frequency_penalty=properties.frequencyPenalty,
            presence_penalty=properties.presencePenalty,
            stream=True,
            stream_options={"include_usage": True}  # Standard OpenAI format for usage in streams
        )

        # Iterate over the streamed responses and yield the content
        output_data = []
        input_token_count = 0
        output_token_count = 0

        for chunk in stream:
            # Check for usage data (usually in the last chunk)
            if hasattr(chunk, 'usage') and chunk.usage:
                input_token_count = chunk.usage.prompt_tokens
                output_token_count = chunk.usage.completion_tokens

            # Process content
            if chunk.choices and len(chunk.choices) > 0:
                content = chunk.choices[0].delta.content
                if content:
                    output_data.append(content)
                    yield content

        # add both the user and assistant generated data to the session
        self.update_session_data(
            input_data=input_data,
            input_template=template,
            output_data="".join(output_data))

        await self.send_usage_input_tokens(input_token_count)
        await self.send_usage_output_tokens(output_token_count)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _execute(self, user_prompt: str, system_prompt: str, values: dict):
        messages_dict = []

        if user_prompt:
            user_prompt = user_prompt.strip()
            messages_dict.append({
                "role": "user",
                "content": f"{user_prompt}"
            })

        if system_prompt:
            system_prompt = system_prompt.strip()
            messages_dict.append({
                "role": "system",
                "content": system_prompt
            })

        if not messages_dict:
            raise Exception(f'no prompts specified for values {values}')

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

        # Processor Runtime Properties
        properties = self.properties

        # Create a streaming completion
        stream = client.chat.completions.create(
            model=self.provider.version,
            messages=messages_dict,
            max_tokens=properties.maxTokens,
            temperature=properties.temperature,
            top_p=properties.topP,
            frequency_penalty=properties.frequencyPenalty,
            presence_penalty=properties.presencePenalty,
            stream=False,
        )

        await self.send_usage_input_tokens(stream.usage.prompt_tokens)
        await self.send_usage_output_tokens(stream.usage.completion_tokens)

        # final raw response, without stripping or splitting
        raw_response = stream.choices[0].message.content
        return parse_response(raw_response=raw_response)
