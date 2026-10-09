"""Command line interface."""

from __future__ import annotations

from pathlib import Path

import typer
import uvicorn

from lms_passthrough import __version__
from lms_passthrough.config import load_config

app = typer.Typer(help="LMS Passthrough CLI")
CONFIG_OPTION = typer.Option(Path("./config.yaml"), "--config")


@app.command()
def version() -> None:
    """Show version."""
    typer.echo(__version__)


@app.command()
def check_config(config: Path = CONFIG_OPTION) -> None:
    """Validate configuration."""
    load_config(config)
    typer.echo("Configuration is valid")


@app.command()
def serve(
    config: Path = CONFIG_OPTION,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """Run the service."""
    from lms_passthrough.app import create_app  # noqa: PLC0415

    cfg = load_config(config)
    uvicorn.run(
        create_app(config),
        host=host or cfg.host,
        port=port or cfg.port,
    )
