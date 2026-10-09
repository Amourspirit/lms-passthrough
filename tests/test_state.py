from datetime import UTC, datetime, timedelta
from pathlib import Path

from lms_passthrough.state.store import StateStore, StoredResponse


def test_state_store_round_trip(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    record = StoredResponse(
        response_id="resp_1",
        provider_name="lm-studio",
        model="m",
        created_at=datetime.now(tz=UTC),
        payload={"a": 1},
    )
    store.save_response(record)
    loaded = store.get_response("resp_1")
    assert loaded is not None
    assert loaded.payload["a"] == 1


def test_state_store_expiration(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    old = StoredResponse(
        response_id="resp_old",
        provider_name="lm-studio",
        model="m",
        created_at=datetime.now(tz=UTC) - timedelta(days=40),
        payload={},
    )
    store.save_response(old)
    deleted = store.delete_expired(30)
    assert deleted == 1
    assert store.get_response("resp_old") is None
