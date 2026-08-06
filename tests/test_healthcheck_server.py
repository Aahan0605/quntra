"""_start_healthcheck_server — the cloud-deploy equivalent of watchdog.py's
heartbeat check. Railway's restart policy only detects a process EXIT, not
a hang (there's no watchdog process in a cloud deploy to catch that), so
this closes the exact gap that let the scheduler sit silently hung for 5
days earlier in this project's history.
"""

import time
import urllib.error
import urllib.request

import pytest

import scripts.scheduler as sch


@pytest.fixture(autouse=True)
def _clean_port_env(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)


def test_no_op_when_port_unset():
    """Local Mac runs never set PORT — must stay completely inert."""
    sch._start_healthcheck_server()  # must not raise, must not bind anything


def test_serves_200_when_heartbeat_is_fresh(monkeypatch, tmp_path):
    port = "18081"
    monkeypatch.setenv("PORT", port)
    monkeypatch.setattr(sch, "HEARTBEAT_FILE", tmp_path / "hb")
    sch.HEARTBEAT_FILE.write_text(str(int(time.time())))

    sch._start_healthcheck_server()
    time.sleep(0.3)
    resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/health")
    assert resp.status == 200


def test_serves_503_when_heartbeat_is_stale(monkeypatch, tmp_path):
    port = "18082"
    monkeypatch.setenv("PORT", port)
    monkeypatch.setattr(sch, "HEARTBEAT_FILE", tmp_path / "hb")
    sch.HEARTBEAT_FILE.write_text(str(int(time.time()) - 999))

    sch._start_healthcheck_server(max_stale_seconds=300)
    time.sleep(0.3)
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health")
    assert exc_info.value.code == 503


def test_serves_503_when_heartbeat_file_missing(monkeypatch, tmp_path):
    port = "18083"
    monkeypatch.setenv("PORT", port)
    monkeypatch.setattr(sch, "HEARTBEAT_FILE", tmp_path / "does-not-exist")

    sch._start_healthcheck_server()
    time.sleep(0.3)
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health")
    assert exc_info.value.code == 503


# --- /status ---------------------------------------------------------------
# The Render deploy had no way to report a single trade: its Postgres is
# only reachable via the platform-injected connection string.

def test_status_404s_without_a_token(monkeypatch, tmp_path):
    port = "18085"
    monkeypatch.setenv("PORT", port)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret-bot-token")
    monkeypatch.setattr(sch, "HEARTBEAT_FILE", tmp_path / "hb")
    sch.HEARTBEAT_FILE.write_text(str(int(time.time())))
    sch._start_healthcheck_server()
    time.sleep(0.3)
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/status")
    assert e.value.code == 404


def test_status_token_matches_hash_of_bot_token(monkeypatch):
    import hashlib
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret-bot-token")
    good = hashlib.sha256(b"secret-bot-token").hexdigest()
    assert sch._status_token_ok(good)
    assert not sch._status_token_ok("wrong")
    assert not sch._status_token_ok("")


def test_status_endpoint_closed_when_no_bot_token(monkeypatch):
    """Fail closed: an unconfigured secret must not mean an open endpoint."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    import hashlib
    assert not sch._status_token_ok(hashlib.sha256(b"").hexdigest())
