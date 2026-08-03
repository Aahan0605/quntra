"""ICICI Breeze session expiry reminder — mocked, offline.

Unlike Kite, breeze_connect doesn't raise on a bad session — it returns
{"Status": 5, "Error": "..."} as an ordinary response dict, so validity has
to be read out of that dict rather than caught as an exception.

breeze_connect also downloads a security-master zip at *import* time (bare
module-level urlopen) — patching BreezeConnect after import is too late,
so a fake module has to go into sys.modules before the lazy import runs.
"""

import sys
import types

import src.integrations.breeze_session as bs
from src.governor.brain import QuNtraBrain
from src.governor.hermes import HermesCoordinator


class SpyTelegram:
    def __init__(self):
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)


def _install_fake_breeze(monkeypatch, *, error=None, raises=None):
    mod = types.ModuleType("breeze_connect")

    class FakeBreeze:
        def __init__(self, api_key):
            pass

        def generate_session(self, api_secret, session_token):
            if raises:
                raise raises

        def get_quotes(self, stock_code, exchange_code, product_type):
            return ({"Status": 5, "Error": error} if error
                   else {"Status": 200, "Error": None, "Success": []})

    mod.BreezeConnect = FakeBreeze
    monkeypatch.setitem(sys.modules, "breeze_connect", mod)


def test_status_not_configured(monkeypatch):
    monkeypatch.delenv("ICICI_BREEZE_API_KEY", raising=False)
    monkeypatch.delenv("ICICI_BREEZE_API_SECRET", raising=False)
    monkeypatch.delenv("ICICI_BREEZE_SESSION_TOKEN", raising=False)
    monkeypatch.setattr(bs, "_ROOT", bs._ROOT / "nonexistent")
    assert bs.token_status() == "not_configured"


def test_status_valid(monkeypatch):
    monkeypatch.setenv("ICICI_BREEZE_API_KEY", "k")
    monkeypatch.setenv("ICICI_BREEZE_API_SECRET", "s")
    monkeypatch.setenv("ICICI_BREEZE_SESSION_TOKEN", "t")
    _install_fake_breeze(monkeypatch)
    assert bs.token_status() == "valid"


def test_status_expired(monkeypatch):
    monkeypatch.setenv("ICICI_BREEZE_API_KEY", "k")
    monkeypatch.setenv("ICICI_BREEZE_API_SECRET", "s")
    monkeypatch.setenv("ICICI_BREEZE_SESSION_TOKEN", "t")
    _install_fake_breeze(monkeypatch,
                        error="Authentication Fail :: Invalid Checksum.")
    assert bs.token_status() == "expired"


def test_status_network_error_is_reported_not_crashed(monkeypatch):
    monkeypatch.setenv("ICICI_BREEZE_API_KEY", "k")
    monkeypatch.setenv("ICICI_BREEZE_API_SECRET", "s")
    monkeypatch.setenv("ICICI_BREEZE_SESSION_TOKEN", "t")
    _install_fake_breeze(monkeypatch, raises=ConnectionError("reset"))
    assert bs.token_status().startswith("error")


def test_login_url_quotes_special_characters():
    url = bs.login_url("!FAKE_KEY_1234567890XYZ#abcd")
    assert url.startswith("https://api.icicidirect.com/apiuser/login?api_key=")
    assert "!" not in url and "#" not in url


def test_hermes_alerts_on_expired(monkeypatch):
    spy = SpyTelegram()
    h = HermesCoordinator(brain=QuNtraBrain(), trader=None, telegram=spy,
                          research_team={})
    monkeypatch.setattr("src.integrations.breeze_session.token_status",
                        lambda: "expired")
    assert h.check_breeze_token() == "expired"
    assert len(spy.sent) == 1
    assert "EXPIRED" in spy.sent[0]
    assert "/breeze_token" in spy.sent[0]


def test_hermes_silent_when_valid(monkeypatch):
    spy = SpyTelegram()
    h = HermesCoordinator(brain=QuNtraBrain(), trader=None, telegram=spy,
                          research_team={})
    monkeypatch.setattr("src.integrations.breeze_session.token_status",
                        lambda: "valid")
    assert h.check_breeze_token() == "valid"
    assert spy.sent == []  # no nagging when the token is fine
