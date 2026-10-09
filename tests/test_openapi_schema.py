"""Test to ensure OpenAPI schema has not regressed.

This test generates the live OpenAPI schema and compares it against
the committed fixture. Any regression to POST endpoint request bodies
will fail this test, surfacing schema drift in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

from lms_passthrough.app import create_app


def test_openapi_schema_matches_fixture() -> None:
    """Verify the generated OpenAPI schema matches the committed fixture.

    This catches regressions to POST endpoint schemas that affect /docs.
    """
    # Generate the live schema
    app = create_app()
    live_schema = app.openapi()

    # Load the committed fixture
    fixture_path = Path(__file__).parent / "fixtures" / "openapi.json"
    with open(fixture_path) as f:
        fixture_schema = json.load(f)

    # Compare
    assert live_schema == fixture_schema, (
        "OpenAPI schema has drifted from the committed fixture. "
        "Run `uv run python3 -c "
        '"import json; from lms_passthrough.app import create_app; '
        'app = create_app(); print(json.dumps(app.openapi(), indent=2))" '
        '> tests/fixtures/openapi.json` to update the fixture.'
    )


def test_openapi_response_schema_refs_resolve() -> None:
    """Every ``$ref`` in the generated OpenAPI document must resolve.

    Guards against the Swagger UI resolver errors that appeared when
    response content schemas embedded ``#/$defs/...`` pointers without a
    document-root ``$defs`` object.
    """
    schema = create_app().openapi()

    refs: list[str] = []

    def collect(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "$ref" and isinstance(value, str):
                    refs.append(value)
                else:
                    collect(value)
        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(schema)

    local = [ref for ref in refs if ref.startswith("#/")]
    assert local, "expected at least one local $ref in the OpenAPI document"
    assert not any("$defs" in ref for ref in local), (
        "document embeds #/$defs refs that cannot resolve; use "
        "#/components/schemas/... instead"
    )

    def resolve(pointer: str) -> object | None:
        node: object = schema
        for raw_part in pointer.split("/")[1:]:
            part = raw_part.replace("~1", "/").replace("~0", "~")
            if not isinstance(node, dict):
                return None
            node = node.get(part)
            if node is None:
                return None
        return node

    unresolved = [ref for ref in local if resolve(ref) is None]
    assert not unresolved, f"unresolved $refs in OpenAPI document: {unresolved}"


def test_post_endpoints_have_request_bodies() -> None:
    """Verify all POST endpoints have request body schemas.

    These four endpoints are critical for the typed HTTP boundary.
    """
    app = create_app()
    schema = app.openapi()
    paths = schema.get("paths", {})

    required_endpoints = [
        "/api/v1/chat",
        "/v1/chat/completions",
        "/v1/responses",
        "/v1/embeddings",
    ]

    for endpoint in required_endpoints:
        assert endpoint in paths, f"Endpoint {endpoint} not found in OpenAPI schema"
        post_op = paths[endpoint].get("post")
        assert post_op is not None, f"No POST operation found for {endpoint}"

        # Check that requestBody is present and populated
        assert (
            "requestBody" in post_op
        ), f"POST {endpoint} missing requestBody in OpenAPI schema"
        request_body = post_op["requestBody"]
        assert "content" in request_body, (
            f"POST {endpoint} requestBody missing content"
        )
        assert "application/json" in request_body["content"], (
            f"POST {endpoint} requestBody missing application/json"
        )

        json_content = request_body["content"]["application/json"]
        assert "schema" in json_content, (
            f"POST {endpoint} application/json missing schema"
        )

        # Schema should be a reference to a component
        schema_def = json_content["schema"]
        assert "$ref" in schema_def or "type" in schema_def, (
            f"POST {endpoint} schema is neither a reference nor inline typed"
        )
