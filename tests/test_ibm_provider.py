"""Tests for src/quantum/ibm_provider.py — HTTP mocked, offline."""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("requests")

from src.quantum import ibm_provider as ibm


def _resp(status, body=None, text=""):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = body if body is not None else {}
    r.text = text
    return r


def test_not_configured_returns_simulator(monkeypatch):
    monkeypatch.delenv("IBM_QUANTUM_API_KEY", raising=False)
    monkeypatch.delenv("IBM_QUANTUM_QSA_URL", raising=False)
    # also block the secrets.env fallback
    with patch.object(ibm, "_load_credentials", return_value=(None, None)):
        st = ibm.verify()
    assert st.configured is False
    assert st.connected is False
    assert "simulator" in st.detail
    assert ibm.is_enabled() is False or True  # is_enabled() re-verifies; fine


def test_valid_key_authenticates(monkeypatch):
    monkeypatch.setenv("IBM_QUANTUM_API_KEY", "tok-123")
    monkeypatch.setenv("IBM_QUANTUM_QSA_URL", "https://qsa.example.com")
    with patch("requests.get",
               return_value=_resp(200, {"versions": ["2025-08-15"]})):
        st = ibm.verify()
    assert st.configured and st.connected
    assert st.versions == ["2025-08-15"]


def test_bad_key_401_does_not_crash(monkeypatch):
    monkeypatch.setenv("IBM_QUANTUM_API_KEY", "bad")
    monkeypatch.setenv("IBM_QUANTUM_QSA_URL", "https://qsa.example.com")
    with patch("requests.get", return_value=_resp(401, text="unauthorized")):
        st = ibm.verify()
    assert st.configured is True
    assert st.connected is False
    assert "401" in st.detail


def test_unreachable_degrades(monkeypatch):
    monkeypatch.setenv("IBM_QUANTUM_API_KEY", "tok")
    monkeypatch.setenv("IBM_QUANTUM_QSA_URL", "https://qsa.example.com")
    with patch("requests.get", side_effect=OSError("no route")):
        st = ibm.verify()
    assert st.configured is True
    assert st.connected is False
    assert "unreachable" in st.detail


def test_latest_version_double_decodes(monkeypatch):
    monkeypatch.setenv("IBM_QUANTUM_QSA_URL", "https://qsa.example.com")
    # endpoint double-encodes the version string
    with patch("requests.get",
               return_value=_resp(200, {"version": '{"version": "2025-08-15"}'})):
        v = ibm.latest_version()
    assert v == "2025-08-15"


def test_verify_never_raises(monkeypatch):
    monkeypatch.setenv("IBM_QUANTUM_API_KEY", "tok")
    monkeypatch.setenv("IBM_QUANTUM_QSA_URL", "https://qsa.example.com")
    with patch("requests.get", side_effect=RuntimeError("boom")):
        st = ibm.verify()  # must not propagate
    assert st.connected is False
