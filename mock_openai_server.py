#!/usr/bin/env python3
"""
Minimal mock OpenAI-compatible server for performance testing.

Returns canned responses with near-zero latency so you can benchmark
routing, middleware, and other non-LLM components in isolation.

All configuration via environment variables:
    MOCK_LLM_PORT             Port to listen on (default: 8765)
    MOCK_LLM_WORKERS          Number of uvicorn workers (default: 4)
    MOCK_LLM_LATENCY_MS       Artificial per-request latency in ms (default: 0)
    MOCK_LLM_STREAM_DELAY_MS  Delay between SSE chunks in ms (default: 10)
    MOCK_LLM_CANNED_REPLY     Canned response text
    MOCK_LLM_MODELS           Comma-separated model IDs (default: mock-gpt-4o,mock-gpt-4o-mini,mock-llama-3)
    MOCK_LLM_LOG_LEVEL        Log level: debug, info, warning, error (default: info)
"""

import asyncio
import json
import logging
import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# Configure logging early so workers inherit it via env var
_log_level = os.environ.get("MOCK_LLM_LOG_LEVEL", "info").upper()
logging.basicConfig(
    level=getattr(logging, _log_level),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    force=True,
)
logger = logging.getLogger("mock-llm")
logger.setLevel(_log_level)

app = FastAPI(title="Mock OpenAI Server")

# All config from env vars — change via deployment env and restart
_latency_ms = float(os.environ.get("MOCK_LLM_LATENCY_MS", "0"))
_stream_delay_ms = float(os.environ.get("MOCK_LLM_STREAM_DELAY_MS", "10"))
_canned_reply = os.environ.get("MOCK_LLM_CANNED_REPLY", "This is a mock response from the mock OpenAI server.")

def _parse_models() -> list[dict]:
    raw = os.environ.get("MOCK_LLM_MODELS", "")
    if not raw:
        return [
            {"id": "mock-gpt-4o", "object": "model", "created": 1700000000, "owned_by": "mock"},
            {"id": "mock-gpt-4o-mini", "object": "model", "created": 1700000000, "owned_by": "mock"},
            {"id": "mock-llama-3", "object": "model", "created": 1700000000, "owned_by": "mock"},
        ]
    return [
        {"id": m.strip(), "object": "model", "created": 1700000000, "owned_by": "mock"}
        for m in raw.split(",") if m.strip()
    ]

_models = _parse_models()


def _completion_id() -> str:
    return f"chatcmpl-mock-{uuid.uuid4().hex[:12]}"


def _ts() -> int:
    return int(time.time())


# ---------- /v1/models ----------

@app.get("/v1/models")
async def list_models():
    await _maybe_sleep()
    model_ids = [m["id"] for m in _models]
    logger.debug("GET /v1/models -> %s", model_ids)
    return JSONResponse({"object": "list", "data": _models})


# ---------- /v1/chat/completions ----------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "mock-gpt-4o")
    stream = body.get("stream", False)
    n_messages = len(body.get("messages", []))
    logger.debug("POST /v1/chat/completions model=%s stream=%s messages=%d", model, stream, n_messages)
    include_usage = False
    if stream:
        so = body.get("stream_options") or {}
        include_usage = so.get("include_usage", False)

    await _maybe_sleep()

    if stream:
        return StreamingResponse(
            _stream_chat_chunks(model, include_usage),
            media_type="text/event-stream",
        )

    return JSONResponse({
        "id": _completion_id(),
        "object": "chat.completion",
        "created": _ts(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": _canned_reply,
                    "refusal": None,
                },
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        },
    })


async def _stream_chat_chunks(model: str, include_usage: bool):
    cid = _completion_id()
    ts = _ts()
    words = _canned_reply.split()

    # role chunk
    chunk = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": ts,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(chunk)}\n\n"

    # content chunks
    for i, word in enumerate(words):
        token = word if i == 0 else f" {word}"
        chunk = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": ts,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        if _stream_delay_ms > 0:
            await asyncio.sleep(_stream_delay_ms / 1000)

    # finish chunk
    finish = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": ts,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    if include_usage:
        finish["usage"] = {
            "prompt_tokens": 10,
            "completion_tokens": len(words),
            "total_tokens": 10 + len(words),
        }
    yield f"data: {json.dumps(finish)}\n\n"
    yield "data: [DONE]\n\n"


# ---------- /v1/completions ----------

@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    model = body.get("model", "mock-gpt-4o")
    stream = body.get("stream", False)
    logger.debug("POST /v1/completions model=%s stream=%s", model, stream)

    await _maybe_sleep()

    if stream:
        return StreamingResponse(
            _stream_completion_chunks(model),
            media_type="text/event-stream",
        )

    return JSONResponse({
        "id": f"cmpl-mock-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": _ts(),
        "model": model,
        "choices": [
            {
                "text": _canned_reply,
                "index": 0,
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        },
    })


async def _stream_completion_chunks(model: str):
    cid = f"cmpl-mock-{uuid.uuid4().hex[:12]}"
    ts = _ts()
    words = _canned_reply.split()

    for i, word in enumerate(words):
        token = word if i == 0 else f" {word}"
        chunk = {
            "id": cid,
            "object": "text_completion",
            "created": ts,
            "model": model,
            "choices": [{"text": token, "index": 0, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        if _stream_delay_ms > 0:
            await asyncio.sleep(_stream_delay_ms / 1000)

    finish = {
        "id": cid,
        "object": "text_completion",
        "created": ts,
        "model": model,
        "choices": [{"text": "", "index": 0, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(finish)}\n\n"
    yield "data: [DONE]\n\n"


# ---------- /v1/embeddings ----------

@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    model = body.get("model", "mock-gpt-4o")

    # accept string or list of strings
    input_text = body.get("input", "")
    if isinstance(input_text, str):
        input_text = [input_text]
    logger.debug("POST /v1/embeddings model=%s inputs=%d", model, len(input_text))

    await _maybe_sleep()

    data = []
    for i, _ in enumerate(input_text):
        # 8-dim deterministic fake embedding
        data.append({
            "object": "embedding",
            "embedding": [0.1 * (i + 1)] * 8,
            "index": i,
        })

    return JSONResponse({
        "object": "list",
        "data": data,
        "model": model,
        "usage": {
            "prompt_tokens": len(input_text) * 4,
            "total_tokens": len(input_text) * 4,
        },
    })


# ---------- helpers ----------

async def _maybe_sleep():
    if _latency_ms > 0:
        await asyncio.sleep(_latency_ms / 1000)


def main():
    port = int(os.environ.get("MOCK_LLM_PORT", "8765"))
    workers = int(os.environ.get("MOCK_LLM_WORKERS", "4"))
    log_level = _log_level.lower()

    logger.info("Starting mock OpenAI server on port %d", port)
    logger.info("  workers: %d", workers)
    logger.info("  log level: %s", log_level)
    logger.info("  latency: %sms per request", _latency_ms)
    logger.info("  stream delay: %sms between chunks", _stream_delay_ms)
    logger.info("  base URL: http://localhost:%d/v1", port)
    logger.info("  models: %s", [m["id"] for m in _models])

    uvicorn.run("mock_openai_server:app", host="0.0.0.0", port=port, workers=workers, log_level=log_level)


if __name__ == "__main__":
    main()
