FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache .
EXPOSE 1234
CMD ["lms-passthrough", "serve", "--config", "/app/config.yaml"]
