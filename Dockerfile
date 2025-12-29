FROM astral/uv:python3.9-bookworm-slim

ENV UV_NO_DEV=1
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock .
COPY bin/ bin/
COPY src/ src/

RUN uv sync --locked --no-cache

CMD ["uv", "run", "src/main.py"]
