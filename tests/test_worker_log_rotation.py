"""Exercise worker log publication against real files in a disposable directory."""
from types import SimpleNamespace
from uuid import uuid4

import pytest


def test_archive_error_preserves_active_attempt_and_recovers(tmp_path, monkeypatch):
    from mlx_serve import process_manager as pm
    monkeypatch.setattr(pm, "_LOG_DIR", tmp_path)
    first = SimpleNamespace(model="synthetic", id=str(uuid4()))
    next_attempt = SimpleNamespace(model=first.model, id=str(uuid4()))
    with pm._open_attempt_log(first) as log:
        log.write("prior diagnostic evidence\n")
    current = tmp_path / "synthetic.log"
    before = current.read_bytes()
    archive = tmp_path / "_previous.synthetic.log"
    archive.mkdir()
    with pytest.raises(OSError):
        pm._open_attempt_log(next_attempt)
    assert current.read_bytes() == before
    assert set(tmp_path.iterdir()) == {current, archive}
    archive.rmdir()
    with pm._open_attempt_log(next_attempt) as log:
        log.write("next attempt\n")
    assert archive.read_bytes() == before
    assert next_attempt.id in current.read_text() and first.id not in current.read_text()
