from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM as ChainLLM
from langchain.chat_models.base import SimpleChatModel, BaseChatModel

from vllm import LLM, SamplingParams


class OpenSourceChatLLM(BaseChatModel):
    pass

class OpenSourceLLM(ChainLLM):
    llm: LLM
    # llm: str
    temperature: float = 0.1
    top_p: float = 0.7
    max_tokens: int = 200
    stop: Optional[List[str]] = None

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            stops = stop
        else:
            stops = self.stop
        #     raise ValueError("stop kwargs are not permitted.")
        sampling_param = SamplingParams(
            temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens, stop=stops
        )
        output = self.llm.generate(prompt, sampling_param)
        if output[0].outputs[0].finish_reason == "length":
            output[0].outputs[0].text += "..."
        return output[0].outputs[0].text
        # return prompt[:10]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"llm": self.llm, "temperature": self.temperature, "top_p": self.top_p, "max_tokens": self.max_tokens}

