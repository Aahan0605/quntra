"""Tests for the consecutive-loss kill switch (Task 2-3 / 11)."""
from src.risk.consecutive_loss_guard import ConsecutiveLossGuard


class StubOMS:
    def __init__(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True


class StubTelegram:
    def __init__(self):
        self.messages = []

    def send(self, msg):
        self.messages.append(msg)


class StubBrain:
    def __init__(self):
        self.lessons = []
        self.research = []

    def store_lesson(self, lesson, context):
        self.lessons.append((lesson, context))

    def remember_research(self, note):
        self.research.append(note)


def make_guard():
    return ConsecutiveLossGuard(
        brain=StubBrain(), telegram=StubTelegram(), oms=StubOMS(), threshold=3
    )


def test_counter_increments_and_resets():
    g = make_guard()
    g.record_trade_outcome(-100)
    g.record_trade_outcome(-50)
    assert g.counter == 2
    g.record_trade_outcome(+10)
    assert g.counter == 0


def test_halt_on_third_loss():
    g = make_guard()
    for pnl in (-1, -2, -3):
        g.record_trade_outcome(pnl, trade={"pnl": pnl, "regime": "BEAR",
                                           "signal_score": 8})
    assert g.halted
    assert not g.oms.enabled
    assert any("KILL SWITCH" in m for m in g.telegram.messages)
    assert len(g.brain.lessons) == 1
    assert len(g.brain.research) == 1  # mistake report stored


def test_mistake_report_observations():
    g = make_guard()
    for pnl in (-1, -2, -3):
        g.record_trade_outcome(pnl, trade={"pnl": pnl, "regime": "BEAR",
                                           "signal_score": 7})
    report = g.brain.lessons[0][1]["report"]
    assert report["n_losses"] == 3
    assert any("regime" in o for o in report["observations"])
    assert any("MIN_SIGNAL_SCORE" in o for o in report["observations"])


def test_daily_reset_keeps_halt():
    g = make_guard()
    for pnl in (-1, -2, -3):
        g.record_trade_outcome(pnl)
    g.reset()
    assert g.counter == 0
    assert g.halted  # only /resume clears the halt


def test_resume_reenables_oms():
    g = make_guard()
    for pnl in (-1, -2, -3):
        g.record_trade_outcome(pnl)
    g.resume()
    assert not g.halted
    assert g.oms.enabled


def test_zero_pnl_counts_as_non_loss():
    g = make_guard()
    g.record_trade_outcome(-1)
    g.record_trade_outcome(0.0)
    assert g.counter == 0
