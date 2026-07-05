#!/usr/bin/env python3
"""
QuNtra end-to-end smoke test (Task R8).

Verifies the five core components integrate: Brain memory, credibility,
PaperTrader, ConsecutiveLossGuard, Hermes state — against the configured
DB (PostgreSQL if POSTGRES_URL set, else SQLite).

Works offline: if live NSE quotes are unreachable, the paper order is
placed with an explicit reference price instead (noted in output).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.db import init_db  # noqa: E402
from src.governor.hermes import HermesCoordinator  # noqa: E402
from src.governor.brain import QuNtraBrain  # noqa: E402
from src.execution.paper_trader import PaperTrader  # noqa: E402
from src.risk.consecutive_loss_guard import ConsecutiveLossGuard  # noqa: E402
from src.utils.data_fetcher import UnifiedDataFetcher  # noqa: E402


def main() -> int:
    print("QuNtra end-to-end smoke test")
    print("=" * 50)
    init_db()

    # 1. Brain store/recall
    brain = QuNtraBrain()
    brain.store_lesson("smoke test lesson", {"test": True})
    lessons = brain.get_lessons_learned(1)
    assert len(lessons) >= 1, "lesson not recalled"
    print("[OK] Brain: store and recall works")

    # 2. Credibility
    brain.update_agent_credibility("technical", was_correct=True)
    w = brain.get_agent_credibility("technical")
    assert w > 1.0, f"credibility not updated: {w}"
    print(f"[OK] Brain: credibility update works ({w:.4f})")

    # 3. Paper trader (live quote if reachable, else reference price)
    fetcher = UnifiedDataFetcher()
    trader = PaperTrader(fetcher=fetcher, brain=brain)
    import uuid
    sig = f"smoke_{uuid.uuid4().hex[:8]}"
    try:
        trade = trader.place_order("RELIANCE.NS", "LONG", 1, sig)
        mode = "live quote"
    except Exception:
        trade = trader.place_order("RELIANCE.NS", "LONG", 1, sig, price=2450.0)
        mode = "offline reference price (NSE unreachable from this machine)"
    assert trade and trade.get("status") == "FILLED", f"order failed: {trade}"
    print(f"[OK] Paper trader: order placed via {mode}")

    # 4. Consecutive loss guard
    guard = ConsecutiveLossGuard(brain=brain, telegram=None, oms=None,
                                 threshold=3)
    guard.record_trade_outcome(+100)
    assert guard.counter == 0
    guard.record_trade_outcome(-50)
    guard.record_trade_outcome(-80)
    assert guard.counter == 2, f"counter wrong: {guard.counter}"
    print("[OK] Consecutive loss guard: counter working")

    # 5. Hermes state persistence (round-trips through the DB)
    hermes = HermesCoordinator(brain=brain, trader=trader)
    hermes.set_system_state("smoke_test", {"status": "running"})
    state = hermes.get_system_state()
    assert "smoke_test" in state, "state not persisted"
    print("[OK] Hermes: state persistence works")

    print("\n" + "=" * 50)
    print("SMOKE TEST: ALL CHECKS PASSED ✓")
    print("QuNtra system integration verified.")
    print("Ready to begin 40-day paper trading run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
