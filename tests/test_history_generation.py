"""Generation sees earlier turns between the system prompt and the latest question."""
import sys
import types
from unittest.mock import MagicMock

# `openai`/`dotenv` are only needed to construct a real client; the test injects a fake one.
for name in ("openai", "dotenv"):
    if name not in sys.modules:
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = types.SimpleNamespace(OpenAI=object, load_dotenv=lambda *a, **k: None)

from src.openai_client import OpenAIClient

CHUNKS = [{"chunk_id": "c1", "section_title": "s", "score": 0.9, "content": "context text"}]
HISTORY = [
    {"role": "user", "content": "first question"},
    {"role": "assistant", "content": "first answer"},
]


def make_client():
    client = OpenAIClient.__new__(OpenAIClient)
    client.model = "fake-model"
    client.temperature = 0.0
    client.max_tokens = 50
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = "an answer"
    completion.usage.total_tokens = 3
    completion.usage.prompt_tokens = 2
    completion.usage.completion_tokens = 1
    client.client = MagicMock()
    client.client.chat.completions.create.return_value = completion
    return client


def sent_messages(client):
    return client.client.chat.completions.create.call_args.kwargs["messages"]


def test_history_sits_between_system_prompt_and_latest_question():
    client = make_client()
    client.generate_response("second question", CHUNKS, history=HISTORY)
    messages = sent_messages(client)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1:3] == HISTORY
    assert "second question" in messages[3]["content"]
    assert "context text" in messages[3]["content"]


def test_without_history_the_prompt_is_unchanged():
    client = make_client()
    client.generate_response("only question", CHUNKS)
    assert [m["role"] for m in sent_messages(client)] == ["system", "user"]
