from typing import Any, Dict, List
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_classic.schema import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):
    """Callback handler for capturing agent intermediate steps."""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running"""
        print(f"***Prompt to LLM was: ***\n{prompts[0]}")
        print("********")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running"""
        print(f"***LLM Response: ***\n{response.generations[0][0].text}")
        print("********")
