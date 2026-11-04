import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from aac.config import ROOT
from aac.docagent.agent import DocAgent, extract_name, recent_window
from aac.docagent.tools import CalculatorError, calculate

from .conftest import scripted

DOC = (ROOT / "data" / "docagent" / "skyvault_guide.md").read_text(encoding="utf-8")


def tool_call(expr, id="c1"):
    return AIMessage(content="", tool_calls=[{"name": "calculator", "args": {"expression": expr}, "id": id}])


@pytest.mark.parametrize(
    "expr, value",
    [
        ("18 * 12 * 12 * 0.8", "2073.6"),
        ("3*10 + 50", "80"),
        ("43,200 * 0.001", "43.2"),
        ("2^10", "1024"),
        ("round(10/3, 2)", "3.33"),
    ],
)
def test_calculator(expr, value):
    assert calculate(expr) == value


@pytest.mark.parametrize("expr", ["__import__('os').system('dir')", "9**9**9", "open('x')", "1/0", "a + 1", ""])
def test_calculator_rejects_unsafe_or_invalid(expr):
    with pytest.raises(CalculatorError):
        calculate(expr)


def test_name_extraction():
    assert extract_name("Hi, my name is Lawrance.") == "Lawrance"
    assert extract_name("I'm Priya and I have a question") == "Priya"
    assert extract_name("I am looking for the price") is None
    assert extract_name("I'm Interested in Business") is None


def test_window_never_starts_with_an_orphan_tool_result():
    msgs = [
        HumanMessage("q1"),
        tool_call("1+1"),
        ToolMessage("2", tool_call_id="c1"),
        AIMessage("2"),
        HumanMessage("q2"),
    ]
    assert recent_window(msgs, 3) == [HumanMessage("q2")]
    assert recent_window(msgs, 5)[0] == msgs[0]


def test_answers_from_document_without_tool():
    llm = scripted("The Business plan costs USD 18 per user per month.")
    out = DocAgent(llm, DOC).ask("How much is Business per user?")
    assert out["tool_calls"] == [] and "18" in out["answer"]
    assert "<document>" in llm.calls[0][0].content  # the document is in the system prompt


def test_calls_calculator_then_answers_with_its_result():
    def final(messages):
        assert isinstance(messages[-1], ToolMessage) and messages[-1].content == "2073.6"
        return "A team of 12 would pay USD 2,073.60 per year."

    llm = scripted(tool_call("18 * 12 * 12 * 0.8"), final)
    out = DocAgent(llm, DOC).ask("Team of 12 on Business, annual billing - yearly cost?")
    assert out["tool_calls"] == [{"name": "calculator", "args": {"expression": "18 * 12 * 12 * 0.8"}}]
    assert "2,073.60" in out["answer"]


def test_tool_error_is_returned_to_the_model_not_raised():
    def final(messages):
        assert messages[-1].content.startswith("error:")
        return "I could not compute that."

    out = DocAgent(scripted(tool_call("1/0"), final), DOC).ask("divide by zero")
    assert out["answer"] == "I could not compute that."


def test_tool_budget_forces_an_answer():
    llm = scripted(tool_call("1+1", "a"), tool_call("2+2", "b"), "Done.")
    out = DocAgent(llm, DOC, max_tool_calls=2).ask("loop forever")
    assert out["answer"] == "Done." and len(out["tool_calls"]) == 2


def test_memory_remembers_name_and_context_across_turns():
    llm = scripted("Nice to meet you, Lawrance.", "We are a team of 8 - noted.", "Your name is Lawrance.")
    agent = DocAgent(llm, DOC)
    agent.ask("Hi, my name is Lawrance.", "t1")
    agent.ask("We are a team of 8 on Business.", "t1")
    out = agent.ask("What's my name?", "t1")
    assert out["user_name"] == "Lawrance"
    last_call = llm.calls[-1]
    assert "The user's name is Lawrance." in last_call[0].content
    assert any("team of 8" in m.content for m in last_call[1:])  # earlier turns are in the prompt


def test_name_survives_when_history_window_drops_it():
    llm = scripted(*["ok"] * 6)
    agent = DocAgent(llm, DOC, history_messages=2)
    agent.ask("my name is Lawrance", "t")
    agent.ask("question two", "t")
    agent.ask("question three", "t")
    sent = llm.calls[-1]
    assert not any("my name is" in m.content for m in sent[1:])  # pushed out of the window...
    assert "The user's name is Lawrance." in sent[0].content  # ...but kept as a fact


def test_ungrounded_number_is_dropped_and_retried():
    # Replays the live failure: tool for 4*12*0.8, then a head-computed (wrong) saving.
    def retry(messages):
        assert "[automatic check]" in messages[-1].content and "6.40" in messages[-1].content
        return tool_call("4*12 - 4*12*0.8", "c2")

    llm = scripted(tool_call("4*12*0.8"), "You would save USD 6.40 a year.", retry, "You would save USD 9.60 a year.")
    agent = DocAgent(llm, DOC)
    out = agent.ask("How much do I save per year on Personal by paying annually?", "t")
    assert out["answer"] == "You would save USD 9.60 a year." and out["warnings"] == []
    history = agent.graph.get_state({"configurable": {"thread_id": "t"}}).values["messages"]
    assert not any("6.40" in m.content for m in history)  # the wrong answer never entered memory
    assert not any("[automatic check]" in m.content for m in history)


def test_second_ungrounded_answer_is_kept_but_flagged():
    llm = scripted("It costs USD 77.", "It costs USD 77.")
    out = DocAgent(llm, DOC).ask("price?")
    assert out["answer"] == "It costs USD 77." and "77" in out["warnings"][0]


def test_numbers_from_document_user_and_tool_are_grounded():
    from aac.docagent.agent import ungrounded_numbers

    history = [HumanMessage("we are 8 people"), ToolMessage("2,073.6", tool_call_id="x")]
    assert ungrounded_numbers("8 people pay USD 18 each, USD 2,073.60 a year, 43,200 minutes", DOC, history) == []
    assert ungrounded_numbers("USD 7.60", DOC, history) == ["7.60"]


def test_eval_scores_the_headline_not_the_working():
    from aac.docagent.evaluate import headline

    answer = "You would save USD 6.40 in a year.\n\nCalculation:\n- 48 x 0.20 = USD 9.60"
    assert headline(answer) == "You would save USD 6.40 in a year."
    assert "9.60" not in headline(answer)


def test_threads_are_isolated():
    llm = scripted("hi", "I don't know your name.")
    agent = DocAgent(llm, DOC)
    agent.ask("my name is Lawrance", "a")
    out = agent.ask("what is my name?", "b")
    assert out["user_name"] is None and "Lawrance" not in llm.calls[-1][0].content
