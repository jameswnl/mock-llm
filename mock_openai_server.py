#!/usr/bin/env python3
"""
Minimal mock OpenAI-compatible server for performance testing.

Returns canned responses with near-zero latency so you can benchmark
routing, middleware, and other non-LLM components in isolation.

Usage:
    uv run mock_openai_server.py [--port 8765] [--latency-ms 0]

Then point your OpenAI-compatible client at http://localhost:8765/v1.

Options:
    --port              Port to listen on (default: 8765)
    --latency-ms        Artificial per-request latency in milliseconds (default: 0)
    --stream-delay-ms   Delay between SSE chunks in milliseconds (default: 10)
"""

import argparse
import asyncio
import json
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

app = FastAPI(title="Mock OpenAI Server")

# -- runtime-configurable state --
_config = {
    "latency_ms": 0.0,
    "stream_delay_ms": 10.0,
    "canned_reply": "This is a mock response from the mock OpenAI server.",
    "models": [
        {"id": "mock-gpt-4o", "object": "model", "created": 1700000000, "owned_by": "mock"},
        {"id": "mock-gpt-4o-mini", "object": "model", "created": 1700000000, "owned_by": "mock"},
        {"id": "mock-llama-3", "object": "model", "created": 1700000000, "owned_by": "mock"},
    ],
}


def _completion_id() -> str:
    return f"chatcmpl-mock-{uuid.uuid4().hex[:12]}"


def _ts() -> int:
    return int(time.time())


# ---------- /admin/config ----------

@app.get("/admin/config")
async def get_config():
    return JSONResponse(_config)


@app.patch("/admin/config")
async def update_config(request: Request):
    body = await request.json()
    for key in ("latency_ms", "stream_delay_ms", "canned_reply"):
        if key in body:
            _config[key] = body[key]
    if "models" in body:
        models = []
        for m in body["models"]:
            if isinstance(m, str):
                m = {"id": m, "object": "model", "created": int(time.time()), "owned_by": "mock"}
            models.append(m)
        _config["models"] = models
    return JSONResponse(_config)


# ---------- /v1/models ----------

@app.get("/v1/models")
async def list_models():
    await _maybe_sleep()
    return JSONResponse({"object": "list", "data": _config["models"]})


# ---------- /v1/chat/completions ----------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "mock-gpt-4o")
    stream = body.get("stream", False)
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
                    "content": _config["canned_reply"],
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
    words = _config["canned_reply"].split()

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
        if _config["stream_delay_ms"] > 0:
            await asyncio.sleep(_config["stream_delay_ms"] / 1000)

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
                "text": _config["canned_reply"],
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
    words = _config["canned_reply"].split()

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
        if _config["stream_delay_ms"] > 0:
            await asyncio.sleep(_config["stream_delay_ms"] / 1000)

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
    if _config["latency_ms"] > 0:
        await asyncio.sleep(_config["latency_ms"] / 1000)


def main():
    parser = argparse.ArgumentParser(description="Mock OpenAI-compatible server for perf testing")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument("--latency-ms", type=float, default=0, help="Artificial per-request latency in ms (default: 0)")
    parser.add_argument("--stream-delay-ms", type=float, default=10, help="Delay between SSE chunks in ms (default: 10)")
    parser.add_argument("--workers", type=int, default=4, help="Number of uvicorn workers (default: 4)")
    args = parser.parse_args()

    _config["latency_ms"] = args.latency_ms
    _config["stream_delay_ms"] = args.stream_delay_ms

    print(f"Starting mock OpenAI server on port {args.port}")
    print(f"  workers: {args.workers}")
    print(f"  latency: {_config['latency_ms']}ms per request")
    print(f"  stream delay: {_config['stream_delay_ms']}ms between chunks")
    print(f"  base URL: http://localhost:{args.port}/v1")
    print(f"  models: {[m['id'] for m in _config['models']]}")

    uvicorn.run(app, host="0.0.0.0", port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
