"""A ReAct-style document agent built as a small LangGraph graph.

    START ─► remember ─► agent ──(tool call, budget left)──► tools ─┐
                           ▲                                        │
                           ├────────────────────────────────────────┘
                           │
                           └──(plain answer)──► check ──(numbers grounded, or retry used)──► END
                           ▲                      │
                           └──(1 retry, feedback)─┘

* The single sample document is short (~600 tokens), so it is placed in the system prompt rather
  than retrieved: no retrieval step that can miss. A long document would swap that for a search tool.
* The model decides when to call `calculator`; the prompt tells it to use the tool for arithmetic
  and never for reading facts. `python -m aac eval-agent` measures how often it gets that right.
* Memory has two layers. The LangGraph checkpointer stores the whole conversation per `thread_id`,
  and only the last N messages are sent to the model. Facts that must survive that window (the
  user's name) are pulled out by `remember` into state and written into every system prompt.
* A per-turn tool budget stops a model that keeps calling tools; once spent, the model is called
  without tools and must answer.
* `check` verifies the final answer the same way Part 1 verifies drafts: every number in it must
  come from the document, the user's own messages, or a calculator result. A live qwen3.5:9b run
  called the calculator for 4*12*0.8 and then did the subtraction in its head ("save USD 6.40",
  correct is 9.60). Prompting alone did not stop that, so an ungrounded answer is dropped and the
  model retries once with the offending numbers named. If it still fails, the answer is kept but
  flagged in `warnings` and in the trace.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from ..llm import call, text_of
from ..obs import Tracer
from .tools import TOOLS

SYSTEM_PROMPT = """You are SkyVault's assistant. Answer questions using ONLY the document below.
- If the document does not contain the answer, say you don't know from the document. Do not guess.
- When an answer needs arithmetic (totals, discounts, differences, conversions), call the `calculator` tool
  with the expression and use its result. Do not do any arithmetic in your head, not even the last step:
  put the whole calculation in one expression (e.g. "4*12 - 4*12*0.8") and state its result first.
- Do NOT call any tool for questions you can answer by reading the document or the conversation.
- You may use what the user told you earlier in this conversation. Keep answers short.
{profile}
<document>
{document}
</document>"""

_NAME = re.compile(r"(?i:\bmy name is|\bi am|\bi'm|\bcall me|\bthis is)\s+([A-Z][a-zA-Z'\-]{1,30})")
_NOT_NAMES = frozenset(
    "A An The Not Just Here Looking Interested Planning Trying Going Using Asking Wondering Sure Fine Good Glad "
    "Happy New Still Also Currently On In With From".split()
)


_NUMBER = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?")

CHECK_FEEDBACK = """[automatic check] Your answer states numbers that are not in the document, not in the user's
messages and not the result of a calculator call: {numbers}. Call the calculator for every number you compute,
with the whole calculation in one expression, and state only numbers you read or computed with the tool.
Answer the user's last question again."""


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_name: str | None
    tool_calls: int
    checks: int
    feedback: str | None
    warnings: list[str]


def numbers_in(text: str) -> set[float]:
    return {float(m.group(0).replace(",", "")) for m in _NUMBER.finditer(text)}


def ungrounded_numbers(answer: str, document: str, history: list[AnyMessage]) -> list[str]:
    """Numbers in `answer` that no source supports. Sources: the document, anything the user wrote,
    and every calculator output in the conversation. 0-3 are allowed for list numbering and wording."""
    allowed = numbers_in(document) | {0.0, 1.0, 2.0, 3.0}
    for m in history:
        if isinstance(m, HumanMessage | ToolMessage):
            allowed |= numbers_in(text_of(m))
    bad = []
    for m in _NUMBER.finditer(answer):
        if float(m.group(0).replace(",", "")) not in allowed and m.group(0) not in bad:
            bad.append(m.group(0))
    return bad


def extract_name(text: str) -> str | None:
    for m in _NAME.finditer(text):
        if m.group(1) not in _NOT_NAMES:
            return m.group(1)
    return None


def recent_window(messages: list[AnyMessage], n: int) -> list[AnyMessage]:
    """Last `n` messages, moved forward to start at a user turn so no tool result loses its call."""
    window = messages[-n:]
    for i, msg in enumerate(window):
        if isinstance(msg, HumanMessage):
            return window[i:]
    return window


class DocAgent:
    def __init__(
        self,
        llm: BaseChatModel,
        document: str,
        *,
        history_messages: int = 20,
        max_tool_calls: int = 4,
        tracer: Tracer | None = None,
        checkpointer=None,
    ):
        self.llm = llm
        self.llm_with_tools = llm.bind_tools(TOOLS)
        self.tools = {t.name: t for t in TOOLS}
        self.document = document
        self.history_messages, self.max_tool_calls = history_messages, max_tool_calls
        self.tracer = tracer or Tracer()
        self.graph = self._build(checkpointer or MemorySaver())

    @classmethod
    def from_file(cls, llm: BaseChatModel, path: Path, **kwargs) -> DocAgent:
        return cls(llm, path.read_text(encoding="utf-8"), **kwargs)

    # ---- nodes -------------------------------------------------------------------------------

    def remember(self, state: AgentState) -> AgentState:
        last = state["messages"][-1]
        name = extract_name(text_of(last)) if isinstance(last, HumanMessage) else None
        update: AgentState = {"tool_calls": 0, "checks": 0, "feedback": None, "warnings": []}  # per user turn
        if name and name != state.get("user_name"):
            update["user_name"] = name
            self.tracer.event("memory", status="ok", fact="user_name")
        return update

    def agent(self, state: AgentState) -> AgentState:
        profile = f"The user's name is {state['user_name']}." if state.get("user_name") else ""
        system = SystemMessage(SYSTEM_PROMPT.format(profile=profile, document=self.document))
        messages = [system, *recent_window(state["messages"], self.history_messages)]
        if state.get("feedback"):  # sent once, never stored in the conversation history
            messages.append(HumanMessage(state["feedback"]))
        budget_left = state.get("tool_calls", 0) < self.max_tool_calls
        model = self.llm_with_tools if budget_left else self.llm
        reply = call(model, messages, tracer=self.tracer, step="agent" if budget_left else "agent_no_tools")
        if reply.tool_calls:
            self.tracer.event("decision", action="tool", tools=[c["name"] for c in reply.tool_calls])
        else:
            self.tracer.event("decision", action="answer")
        return {"messages": [reply], "feedback": None}

    def check(self, state: AgentState) -> AgentState:
        last = state["messages"][-1]
        bad = ungrounded_numbers(text_of(last), self.document, state["messages"][:-1])
        if not bad:
            return {}
        if state.get("checks", 0) < 1:
            self.tracer.event("check", status="retry", ungrounded=bad, guardrails=["ungrounded_number"])
            return {
                "messages": [RemoveMessage(id=last.id)],  # the wrong answer never enters memory
                "feedback": CHECK_FEEDBACK.format(numbers=", ".join(bad)),
                "checks": state.get("checks", 0) + 1,
            }
        self.tracer.event("check", status="flagged", ungrounded=bad, guardrails=["ungrounded_number_kept"])
        return {"warnings": [f"unverified numbers in the answer: {', '.join(bad)}"]}

    def run_tools(self, state: AgentState) -> AgentState:
        last: AIMessage = state["messages"][-1]
        results = []
        for tc in last.tool_calls:
            tool = self.tools.get(tc["name"])
            if tool is None:
                output = f"error: unknown tool {tc['name']!r}; available: {list(self.tools)}"
            else:
                output = str(tool.invoke(tc["args"]))
            status = "error" if output.startswith("error") else "ok"
            self.tracer.event("tool", name=tc["name"], args=tc["args"], output=output[:200], status=status)
            results.append(ToolMessage(content=output, tool_call_id=tc["id"], name=tc["name"]))
        return {"messages": results, "tool_calls": state.get("tool_calls", 0) + len(last.tool_calls)}

    @staticmethod
    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "check"

    @staticmethod
    def after_check(state: AgentState) -> str:
        return "agent" if state.get("feedback") else END

    def _build(self, checkpointer):
        g = StateGraph(AgentState)
        g.add_node("remember", self.remember)
        g.add_node("agent", self.agent)
        g.add_node("tools", self.run_tools)
        g.add_node("check", self.check)
        g.add_edge(START, "remember")
        g.add_edge("remember", "agent")
        g.add_conditional_edges("agent", self.route, {"tools": "tools", "check": "check"})
        g.add_edge("tools", "agent")
        g.add_conditional_edges("check", self.after_check, {"agent": "agent", END: END})
        return g.compile(checkpointer=checkpointer)

    # ---- entry point -------------------------------------------------------------------------

    def ask(self, question: str, thread_id: str = "default") -> dict:
        """One user turn. Returns the answer plus which tools were called during this turn."""
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 8 + 2 * self.max_tool_calls}
        before = len(self.graph.get_state(config).values.get("messages", []))
        final = self.graph.invoke({"messages": [HumanMessage(question)]}, config)
        turn = final["messages"][before:]
        tool_calls = [tc for m in turn if isinstance(m, AIMessage) for tc in m.tool_calls]
        return {
            "answer": text_of(final["messages"][-1]),
            "tool_calls": [{"name": tc["name"], "args": tc["args"]} for tc in tool_calls],
            "user_name": final.get("user_name"),
            "warnings": final.get("warnings", []),
        }
