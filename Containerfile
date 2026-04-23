FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir fastapi uvicorn

COPY mock_openai_server.py .

EXPOSE 8765

ENTRYPOINT ["python", "mock_openai_server.py"]
CMD ["--port", "8765", "--workers", "4"]
