"""A scripted chat model: the tests decide exactly what the "LLM" says, and can inspect what it was sent."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aac.config import ROOT, Settings

Reply = str | AIMessage | Exception | Callable[[list[BaseMessage]], "str | AIMessage"]


class ScriptedLLM(BaseChatModel):
    """Replies are taken in order. A callable reply gets the messages; an Exception is raised."""

    replies: list = []
    calls: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(list(messages))
        if not self.replies:
            raise AssertionError(f"ScriptedLLM ran out of replies (call {len(self.calls)})")
        reply = self.replies.pop(0)
        if callable(reply) and not isinstance(reply, AIMessage):
            reply = reply(messages)
        if isinstance(reply, Exception):
            raise reply
        msg = reply if isinstance(reply, AIMessage) else AIMessage(content=reply)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def scripted(*replies: Reply) -> ScriptedLLM:
    return ScriptedLLM(replies=list(replies), calls=[])


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings()
    s.kb_dir = ROOT / "data" / "kb"
    s.trace_path = tmp_path / "trace.jsonl"
    s.llm_retries = 0
    return s
