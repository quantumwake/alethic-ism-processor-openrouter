import logging
import os.path
from typing import List

import httpx
import openai
import dotenv
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception, before_sleep_log
from ismcore.processor.base_processor_lm import BaseProcessorLM
from ismcore.processor.monitored_processor_state import MonitoredUsage
from ismcore.utils.general_utils import parse_response
from ismcore.utils.ism_logger import ism_logger
from openai import AsyncOpenAI

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

    def     __init__(self, **kwargs):
        super().__init__(**kwargs)
        MonitoredUsage.__init__(self, **kwargs)

    async def stream_llm(self, user_prompt: str, system_prompt: str, values: dict | List[dict]):
        message_list = self.derive_messages_with_session_data_if_any(
            user_prompt=user_prompt, system_prompt=system_prompt, input_data=values
        )

        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

        properties = self.properties

        stream = await client.chat.completions.create(
            model=self.provider.version,
            messages=message_list,
            max_tokens=properties.maxTokens,
            temperature=properties.temperature,
            top_p=properties.topP,
            frequency_penalty=properties.frequencyPenalty,
            presence_penalty=properties.presencePenalty,
            stream=True,
            stream_options={"include_usage": True}
        )

        output_data = []
        input_token_count = 0
        output_token_count = 0

        async for chunk in stream:
            if hasattr(chunk, 'usage') and chunk.usage:
                input_token_count = chunk.usage.prompt_tokens
                output_token_count = chunk.usage.completion_tokens

            if chunk.choices and len(chunk.choices) > 0:
                content = chunk.choices[0].delta.content
                if content:
                    output_data.append(content)
                    yield content

        self.update_session_data(
            input_data=values,
            input_template=user_prompt,
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
    async def execute_llm(self, user_prompt: str, system_prompt: str, values: dict | list[dict])\
            -> tuple[dict | list[dict] | None, any]:

        messages_dict = self.derive_messages_with_session_data_if_any(
            user_prompt=user_prompt, system_prompt=system_prompt, input_data=values
        )

        if not messages_dict:
            raise Exception(f'no prompts specified for values {values}')

        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

        properties = self.properties

        response = await client.chat.completions.create(
            model=self.provider.version,
            messages=messages_dict,
            max_tokens=properties.maxTokens,
            temperature=properties.temperature,
            top_p=properties.topP,
            frequency_penalty=properties.frequencyPenalty,
            presence_penalty=properties.presencePenalty,
            stream=False,
        )

        extra = {
            "upstream_generation_id": response.id,
            "upstream_model_provider": response.model_extra["provider"]
        }

        await self.send_usage_input_tokens(response.usage.prompt_tokens, metadata=extra)
        await self.send_usage_output_tokens(response.usage.completion_tokens, metadata=extra)

        raw_response = response.choices[0].message.content
        return parse_response(raw_response=raw_response)
