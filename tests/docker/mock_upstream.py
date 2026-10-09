"""Fake LM Studio upstream for Docker smoke tests.

Standalone stdlib HTTP server — no third-party dependencies.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

MODELS = {
    "models": [
        {
            "modelId": "test-model",
            "object": "model",
            "type": "llm",
            "publisher": "test",
            "key": "test-model",
            "display_name": "Test Model",
            "quantization": None,
            "size_bytes": 0,
            "params_string": "7B",
            "max_context_length": 4096,
            "format": "gguf",
            "loaded_instances": [],
        }
    ]
}

HEALTH = {"status": "ok"}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self: Handler) -> None:
        if self.path in ("/api/v1/models", "/v1/models"):
            self._json(200, MODELS)
        elif self.path in ("/", "/health/live", "/health/ready"):
            # "/" is the readiness probe target (ProviderConfig.health_path
            # defaults to it), so answer it rather than letting the proxy
            # count a 404 as "reachable".
            self._json(200, HEALTH)
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self: Handler) -> None:
        self._json(
            200,
            {
                "model": "test-model",
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1},
            },
        )

    def _json(self: Handler, status: int, body: dict[str, object]) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self: Handler, format: str, *args: object) -> None:  # noqa: A002
        pass


def main() -> None:
    server = HTTPServer(("0.0.0.0", 9876), Handler)  # noqa: S104
    server.serve_forever()


if __name__ == "__main__":
    main()
