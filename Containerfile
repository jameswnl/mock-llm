FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir fastapi uvicorn

COPY mock_openai_server.py .

ENV MOCK_LLM_PORT=8765
ENV MOCK_LLM_WORKERS=4
ENV MOCK_LLM_LOG_LEVEL=info

EXPOSE 8765

CMD ["python", "mock_openai_server.py"]
