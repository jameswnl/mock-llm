import json
import os

import pytest
from httpx import ASGITransport, AsyncClient

# Set env vars before importing the app module
os.environ["MOCK_LLM_STREAM_DELAY_MS"] = "0"
os.environ["MOCK_LLM_LATENCY_MS"] = "0"

from mock_openai_server import _parse_models, app


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# ---------- /v1/models ----------

async def test_list_models(client):
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 3
    ids = [m["id"] for m in body["data"]]
    assert "mock-gpt-4o" in ids


async def test_list_models_shape(client):
    resp = await client.get("/v1/models")
    for model in resp.json()["data"]:
        assert "id" in model
        assert model["object"] == "model"
        assert "created" in model
        assert "owned_by" in model


# ---------- /v1/chat/completions ----------

async def test_chat_completions_non_streaming(client):
    resp = await client.post("/v1/chat/completions", json={
        "model": "mock-gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "mock-gpt-4o"
    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert choice["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] > 0
    assert body["usage"]["total_tokens"] > 0


async def test_chat_completions_default_model(client):
    resp = await client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
    })
    body = resp.json()
    assert body["model"] == "mock-gpt-4o"


async def test_chat_completions_streaming(client):
    resp = await client.post("/v1/chat/completions", json={
        "model": "mock-gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    })
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    chunks = _parse_sse(resp.text)
    assert len(chunks) >= 3  # role + at least one content + finish

    # first chunk has role
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"

    # last data chunk has finish_reason
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    # all chunks share the same id
    ids = {c["id"] for c in chunks}
    assert len(ids) == 1


async def test_chat_completions_streaming_include_usage(client):
    resp = await client.post("/v1/chat/completions", json={
        "model": "mock-gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    chunks = _parse_sse(resp.text)
    last = chunks[-1]
    assert "usage" in last
    assert last["usage"]["prompt_tokens"] > 0
    assert last["usage"]["total_tokens"] > 0


async def test_chat_completions_streaming_no_usage_by_default(client):
    resp = await client.post("/v1/chat/completions", json={
        "model": "mock-gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    })
    chunks = _parse_sse(resp.text)
    last = chunks[-1]
    assert "usage" not in last


async def test_chat_completions_streaming_content_reassembles(client):
    resp = await client.post("/v1/chat/completions", json={
        "model": "mock-gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    })
    chunks = _parse_sse(resp.text)
    content = ""
    for c in chunks:
        delta = c["choices"][0]["delta"]
        if "content" in delta:
            content += delta["content"]
    assert content == "This is a mock response from the mock OpenAI server."


# ---------- /v1/completions ----------

async def test_completions_non_streaming(client):
    resp = await client.post("/v1/completions", json={
        "model": "mock-gpt-4o",
        "prompt": "Once upon a time",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "text_completion"
    assert body["model"] == "mock-gpt-4o"
    assert len(body["choices"]) == 1
    assert isinstance(body["choices"][0]["text"], str)
    assert body["choices"][0]["finish_reason"] == "stop"


async def test_completions_streaming(client):
    resp = await client.post("/v1/completions", json={
        "model": "mock-gpt-4o",
        "prompt": "Once upon a time",
        "stream": True,
    })
    assert resp.status_code == 200
    chunks = _parse_sse(resp.text)
    assert len(chunks) >= 2

    # last chunk has finish_reason
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    # reassemble text
    text = "".join(c["choices"][0]["text"] for c in chunks)
    assert "mock response" in text


# ---------- /v1/embeddings ----------

async def test_embeddings_single_input(client):
    resp = await client.post("/v1/embeddings", json={
        "model": "mock-gpt-4o",
        "input": "hello world",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    emb = body["data"][0]
    assert emb["object"] == "embedding"
    assert emb["index"] == 0
    assert isinstance(emb["embedding"], list)
    assert len(emb["embedding"]) == 8
    assert body["usage"]["prompt_tokens"] > 0


async def test_embeddings_multiple_inputs(client):
    resp = await client.post("/v1/embeddings", json={
        "model": "mock-gpt-4o",
        "input": ["hello", "world", "foo"],
    })
    body = resp.json()
    assert len(body["data"]) == 3
    for i, emb in enumerate(body["data"]):
        assert emb["index"] == i


async def test_embeddings_deterministic(client):
    resp1 = await client.post("/v1/embeddings", json={"input": "hello"})
    resp2 = await client.post("/v1/embeddings", json={"input": "hello"})
    assert resp1.json()["data"][0]["embedding"] == resp2.json()["data"][0]["embedding"]


# ---------- _parse_models ----------

def test_parse_models_default(monkeypatch):
    monkeypatch.delenv("MOCK_LLM_MODELS", raising=False)
    models = _parse_models()
    assert len(models) == 3
    assert models[0]["id"] == "mock-gpt-4o"


def test_parse_models_custom(monkeypatch):
    monkeypatch.setenv("MOCK_LLM_MODELS", "my-model-1,my-model-2")
    models = _parse_models()
    assert len(models) == 2
    assert models[0]["id"] == "my-model-1"
    assert models[1]["id"] == "my-model-2"
    assert models[0]["object"] == "model"


def test_parse_models_whitespace(monkeypatch):
    monkeypatch.setenv("MOCK_LLM_MODELS", " foo , bar , ")
    models = _parse_models()
    assert [m["id"] for m in models] == ["foo", "bar"]


# ---------- helpers ----------

def _parse_sse(text: str) -> list[dict]:
    chunks = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[6:]))
    return chunks
