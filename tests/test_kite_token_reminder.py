"""Kite token expiry reminder — mocked, offline."""

import sys
import types
from unittest.mock import MagicMock

import pytest

import src.integrations.kite_session as ks
from src.governor.brain import QuNtraBrain
from src.governor.hermes import HermesCoordinator


class SpyTelegram:
    def __init__(self):
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)


def _install_fake_kite(monkeypatch, *, raises=None):
    """Install a fake kiteconnect module whose profile() raises `raises`."""
    mod = types.ModuleType("kiteconnect")
    exc_mod = types.ModuleType("kiteconnect.exceptions")

    class TokenException(Exception):
        pass

    class PermissionException(Exception):
        pass

    exc_mod.TokenException = TokenException
    exc_mod.PermissionException = PermissionException

    client = MagicMock()
    if raises == "token":
        client.profile.side_effect = TokenException("expired")
    elif raises == "permission":
        client.profile.side_effect = PermissionException("no data sub")
    else:
        client.profile.return_value = {"user_id": "AB1234"}
    mod.KiteConnect = MagicMock(return_value=client)
    mod.exceptions = exc_mod
    monkeypatch.setitem(sys.modules, "kiteconnect", mod)
    monkeypatch.setitem(sys.modules, "kiteconnect.exceptions", exc_mod)


def test_status_not_configured(monkeypatch):
    monkeypatch.setattr(ks, "token_status",
                        ks.token_status)  # keep real fn
    monkeypatch.delenv("KITE_API_KEY", raising=False)
    monkeypatch.delenv("KITE_ACCESS_TOKEN", raising=False)
    # force the dotenv load to a nonexistent path so nothing is populated
    monkeypatch.setattr(ks, "_ROOT", ks._ROOT / "nonexistent")
    assert ks.token_status() == "not_configured"


def test_status_valid(monkeypatch):
    monkeypatch.setenv("KITE_API_KEY", "k")
    monkeypatch.setenv("KITE_ACCESS_TOKEN", "t")
    _install_fake_kite(monkeypatch, raises=None)
    assert ks.token_status() == "valid"


def test_status_expired(monkeypatch):
    monkeypatch.setenv("KITE_API_KEY", "k")
    monkeypatch.setenv("KITE_ACCESS_TOKEN", "t")
    _install_fake_kite(monkeypatch, raises="token")
    assert ks.token_status() == "expired"


def test_permission_error_counts_as_valid(monkeypatch):
    # account without market-data sub: token is valid, endpoint isn't
    monkeypatch.setenv("KITE_API_KEY", "k")
    monkeypatch.setenv("KITE_ACCESS_TOKEN", "t")
    _install_fake_kite(monkeypatch, raises="permission")
    assert ks.token_status() == "valid"


def test_hermes_alerts_on_expired(monkeypatch):
    spy = SpyTelegram()
    h = HermesCoordinator(brain=QuNtraBrain(), trader=None, telegram=spy,
                          research_team={})
    monkeypatch.setattr("src.integrations.kite_session.token_status",
                        lambda: "expired")
    assert h.check_kite_token() == "expired"
    assert len(spy.sent) == 1
    assert "EXPIRED" in spy.sent[0]
    assert "kite_login.py" in spy.sent[0]


def test_hermes_silent_when_valid(monkeypatch):
    spy = SpyTelegram()
    h = HermesCoordinator(brain=QuNtraBrain(), trader=None, telegram=spy,
                          research_team={})
    monkeypatch.setattr("src.integrations.kite_session.token_status",
                        lambda: "valid")
    assert h.check_kite_token() == "valid"
    assert spy.sent == []  # no nagging when the token is fine
