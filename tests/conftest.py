import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "agents"))

# Vocabulary the fake embedding model counts, so retrieval stays deterministic
EMBEDDING_VOCAB = ["overtime", "salary", "age", "sales", "attrition", "remote", "training", "manager"]


class FakeSentenceTransformer:
    """Offline stand-in for SentenceTransformer: embeds text as keyword counts."""

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts):
        return np.array(
            [[text.lower().count(word) for word in EMBEDDING_VOCAB] for text in texts],
            dtype="float32"
        )


# rag_agent builds its embedding model at import time, which would download it.
# Replace the library before any test imports the agent.
_fake_module = types.ModuleType("sentence_transformers")
_fake_module.SentenceTransformer = FakeSentenceTransformer
sys.modules["sentence_transformers"] = _fake_module

FAKE_INSIGHTS = "KEY FINDINGS\n- Attrition is 37.5%\nROOT CAUSES\nOvertime.\nRECOMMENDATIONS\n- Reduce overtime"


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Age":             [30, 40, 28, 45, 50, 35, 26, 38],
        "Attrition":       ["Yes", "No", "Yes", "No", "No", "No", "Yes", "No"],
        "Department":      ["Sales", "Sales",
                            "Research & Development", "Research & Development",
                            "Research & Development", "Research & Development",
                            "Human Resources", "Human Resources"],
        "MonthlyIncome":   [3000, 5000, 2500, 7000, 8000, 6000, 2000, 4000],
        "OverTime":        ["Yes", "No", "Yes", "No", "No", "Yes", "No", "No"],
        "JobSatisfaction": [1, 3, 2, 4, 4, 3, 1, 2],
    })


@pytest.fixture
def sample_csv(sample_df, tmp_path):
    path = tmp_path / "hr.csv"
    sample_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run in a temp directory so charts, reports and rag_docs never touch the project."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def mock_llm(monkeypatch):
    """Replace the OpenAI/OpenRouter client; returns the mocked `create` call."""
    import insight_agent

    client_class = MagicMock(name="OpenAI")
    create = client_class.return_value.chat.completions.create
    create.return_value.choices = [MagicMock()]
    create.return_value.choices[0].message.content = FAKE_INSIGHTS
    monkeypatch.setattr(insight_agent, "OpenAI", client_class)
    return create


# The pieces the fake LLM streams; joined together they equal FAKE_INSIGHTS
FAKE_STREAM_PIECES = ["KEY FINDINGS\n- Attrition ", "is 37.5%\nROOT CAUSES\nOvertime.\n", "RECOMMENDATIONS\n- Reduce overtime"]


def _stream_chunk(text):
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = text
    return chunk


async def fake_llm_stream(pieces, error=None):
    """Stand-in for the SDK's streaming response: yields one chunk per piece, then optionally fails."""
    # Real streams also contain chunks without text; the agent has to skip them
    yield _stream_chunk(None)
    for piece in pieces:
        yield _stream_chunk(piece)
    usage_only = MagicMock()
    usage_only.choices = []
    yield usage_only
    if error:
        raise error


@pytest.fixture
def mock_async_llm(monkeypatch):
    """Replace the AsyncOpenAI client used by the API; returns the mocked `create` call.

    `create` answers with a full completion, or with a chunk stream when called with stream=True.
    """
    import insight_agent

    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = FAKE_INSIGHTS

    async def fake_create(**kwargs):
        return fake_llm_stream(FAKE_STREAM_PIECES) if kwargs.get("stream") else completion

    client_class = MagicMock(name="AsyncOpenAI")
    client = client_class.return_value
    client.__aenter__.return_value = client
    client.chat.completions.create = AsyncMock(side_effect=fake_create)
    monkeypatch.setattr(insight_agent, "AsyncOpenAI", client_class)
    return client.chat.completions.create
