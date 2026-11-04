"""One chat-model factory for every part, plus the two call helpers the graphs use.

`call` adds retry with backoff and records latency/token usage on the tracer.
`call_json` asks for a JSON object, validates it against a Pydantic model and, if the model
returns something invalid, re-prompts once with the validation error. If it still fails it
raises LLMError, and the callers fail closed (Part 1 routes the email to a human).
"""

from __future__ import annotations

import json
import re
import time
from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from .config import Settings
from .obs import Tracer

T = TypeVar("T", bound=BaseModel)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class LLMError(RuntimeError):
    """The model could not be reached, or never produced a valid answer."""


def get_chat_model(settings: Settings, max_tokens: int | None = None) -> BaseChatModel:
    provider = settings.llm_provider.lower()
    model = settings.model_name
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=0,
            reasoning=False,
            num_predict=max_tokens,
            client_kwargs={"timeout": settings.llm_timeout_s},
        )
    if provider in ("openai", "gemini"):
        import os

        from langchain_openai import ChatOpenAI

        kwargs = {}
        if provider == "gemini":  # Gemini exposes an OpenAI-compatible endpoint
            kwargs = {
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "api_key": os.environ.get("GEMINI_API_KEY"),
            }
        return ChatOpenAI(
            model=model, temperature=0, max_tokens=max_tokens, timeout=settings.llm_timeout_s, max_retries=0, **kwargs
        )
    raise ValueError(f"Unknown AAC_LLM_PROVIDER: {settings.llm_provider!r} (use ollama, openai or gemini)")


def text_of(message) -> str:
    """Plain text from a chat response, with any <think> block removed."""
    content = message.content if hasattr(message, "content") else message
    if isinstance(content, list):
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return _THINK.sub("", str(content)).strip()


def call(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    *,
    tracer: Tracer | None = None,
    step: str = "llm",
    retries: int = 2,
    backoff_s: float = 1.0,
) -> AIMessage:
    """Invoke the model with retry on transport errors. Every attempt is traced."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        start = time.perf_counter()
        try:
            reply = llm.invoke(messages)
        except Exception as exc:  # provider SDKs raise many types; all are transient from our point of view
            last = exc
            if tracer:
                tracer.event(step, status="error", attempt=attempt, error=f"{type(exc).__name__}: {exc}"[:300])
            if attempt < retries:
                time.sleep(backoff_s * 2**attempt)
            continue
        usage = getattr(reply, "usage_metadata", None) or {}
        if tracer:
            tracer.event(
                step,
                status="ok",
                attempt=attempt,
                ms=round((time.perf_counter() - start) * 1000),
                in_tokens=usage.get("input_tokens"),
                out_tokens=usage.get("output_tokens"),
            )
        return reply
    raise LLMError(f"{step}: model call failed after {retries + 1} attempts: {last}") from last


def parse_json(text: str, model_cls: type[T]) -> T:
    match = _JSON_OBJECT.search(text)
    if not match:
        raise ValueError("no JSON object in the reply")
    return model_cls.model_validate(json.loads(match.group(0)))


def call_json(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    model_cls: type[T],
    *,
    tracer: Tracer | None = None,
    step: str = "llm_json",
    retries: int = 2,
) -> T:
    """Call the model and parse its reply into `model_cls`; one repair round on invalid output."""
    reply = call(llm, messages, tracer=tracer, step=step, retries=retries)
    try:
        return parse_json(text_of(reply), model_cls)
    except (ValueError, ValidationError) as exc:
        if tracer:
            tracer.event(step, status="invalid_json", error=str(exc)[:300])
        repair = messages + [
            reply,
            HumanMessage(
                f"Your reply was not valid: {str(exc)[:500]}\nReply again with ONLY the corrected JSON object."
            ),
        ]
        reply = call(llm, repair, tracer=tracer, step=f"{step}_repair", retries=retries)
        try:
            return parse_json(text_of(reply), model_cls)
        except (ValueError, ValidationError) as exc2:
            raise LLMError(f"{step}: invalid JSON after repair: {exc2}") from exc2
