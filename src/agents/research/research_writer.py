"""
ResearchWriter — synthesizes all research agent outputs into the daily
pre-market intelligence report.

Not an analyst: it formats and combines what the specialists produced,
renders a trade / no-trade recommendation from their signals, stores the
report, and hands it to Telegram.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from src.agents.research.base import BaseResearchAgent, ResearchOutput

IST = ZoneInfo("Asia/Kolkata")

# Above this geopolitical score the writer recommends standing down
GEO_RISK_NO_TRADE = 7.5


class ResearchWriter(BaseResearchAgent):
    name = "research_writer"
    description = "synthesizes agent outputs into readable reports"
    note_type = "daily_report"

    def compose(self, outputs: dict[str, ResearchOutput],
                context: dict | None = None) -> str:
        """Build the structured daily report from agent outputs.

        outputs: {agent_name: ResearchOutput} from the research team run.
        """
        context = context or {}
        regime = context.get("regime", "UNKNOWN")

        macro = self._payload(outputs, "macro_agent")
        news = self._payload(outputs, "news_agent")
        sector = self._payload(outputs, "sector_agent")
        corp = self._payload(outputs, "company_analysis_agent")
        geo = self._payload(outputs, "geopolitical_agent")

        macro_bias = macro.get("macro_bias", "UNKNOWN")
        macro_reason = self._summary(outputs, "macro_agent")
        geo_score = geo.get("geopolitical_risk_score", 0.0)
        top_geo = geo.get("top_events") or []
        leaders = sector.get("leaders") or []
        blackout = corp.get("earnings_blackout") or []

        risks = self._top_risks(macro, news, geo, corp)
        trade, why = self._recommendation(macro_bias, geo_score, outputs)

        lines = [
            f"QUNTRA PRE-MARKET INTELLIGENCE — "
            f"{datetime.now(IST).strftime('%Y-%m-%d %H:%M IST')}",
            "=" * 48,
            f"MARKET REGIME: {regime}",
            f"MACRO BIAS: {macro_bias} — {macro_reason}",
            "TOP RISKS:",
            *[f"  • {r}" for r in (risks or ["none identified"])],
            f"SECTOR LEADERS: {', '.join(leaders) or 'n/a'}",
            f"CORPORATE EVENTS: "
            f"{'earnings blackout: ' + ', '.join(blackout) if blackout else 'no flags today'}",
            f"GEOPOLITICAL: {geo_score}/10"
            + (f" — {top_geo[0]['title'][:80]}" if top_geo else ""),
            f"RECOMMENDATION: {trade} — {why}",
            "=" * 48,
        ]
        report = "\n".join(lines)
        self.store(ResearchOutput(
            agent=self.name,
            summary=report,
            confidence=0.7,
            payload={"trade": trade, "macro_bias": macro_bias,
                     "geo_risk": geo_score},
        ))
        return report

    def run(self, context: dict) -> ResearchOutput:
        """BaseResearchAgent interface: compose from context['outputs']."""
        report = self.compose(context.get("outputs") or {}, context)
        return ResearchOutput(agent=self.name, summary=report, confidence=0.7)

    def answer_question(self, question: str, context: dict | None = None) -> str:
        """Freeform Q&A: organizational memory + recent research + state.

        Deterministic synthesis (no LLM): recalls matching knowledge items
        and research notes, then frames them with the current context.
        """
        context = context or {}
        parts: list[str] = []

        from src.knowledge import KnowledgeManager
        hits = KnowledgeManager(self.db_url).recall(question, limit=4)
        if hits:
            parts.append("From QuNtra's memory:")
            parts += [f"• [{h['knowledge_type']}] {h['content'][:160]}"
                      for h in hits]

        try:
            from sqlalchemy import or_, select
            from src.db import ResearchNote, get_session
            terms = [t for t in question.lower().split() if len(t) >= 4][:5]
            if terms:
                with get_session(self.db_url) as s:
                    rows = s.execute(
                        select(ResearchNote)
                        .where(or_(*[ResearchNote.summary.ilike(f"%{t}%")
                                     for t in terms]))
                        .order_by(ResearchNote.created_at.desc())
                        .limit(3)
                    ).scalars().all()
                if rows:
                    parts.append("Recent research:")
                    parts += [f"• [{r.source}] {(r.summary or '')[:160]}"
                              for r in rows]
        except Exception:  # noqa: BLE001
            pass

        regime = (context.get("regime") or {})
        regime_state = (regime.get("state") or regime.get("current")
                        if isinstance(regime, dict) else regime) or "UNKNOWN"
        parts.append(f"Current regime: {regime_state}.")
        if not hits:
            parts.append("Nothing specific in memory yet — the knowledge "
                         "base grows with every trading day. Try /research "
                         "or /context for the current picture.")
        return "\n".join(parts)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _payload(outputs: dict, name: str) -> dict:
        out = outputs.get(name)
        return out.payload if out is not None and out.ok else {}

    @staticmethod
    def _summary(outputs: dict, name: str) -> str:
        out = outputs.get(name)
        return out.summary if out is not None else "agent unavailable"

    @staticmethod
    def _top_risks(macro: dict, news: dict, geo: dict, corp: dict) -> list[str]:
        risks = []
        moves = macro.get("moves") or {}
        oil = moves.get("crude_oil")
        if oil is not None and oil > 1.5:
            risks.append(f"crude oil spiking {oil:+.1f}% (import-cost headwind)")
        inr = moves.get("usdinr")
        if inr is not None and inr > 0.3:
            risks.append(f"rupee weakening {inr:+.2f}%")
        if (news.get("avg_sentiment") or 0) < -0.3:
            risks.append(f"news sentiment negative "
                         f"({news['avg_sentiment']:+.2f})")
        geo_score = geo.get("geopolitical_risk_score", 0)
        if geo_score >= 5:
            risks.append(f"geopolitical risk elevated ({geo_score}/10)")
        if corp.get("earnings_blackout"):
            risks.append(f"{len(corp['earnings_blackout'])} tickers in "
                         f"earnings blackout")
        return risks[:3]

    @staticmethod
    def _recommendation(macro_bias: str, geo_score: float,
                        outputs: dict) -> tuple[str, str]:
        dead_agents = [n for n, o in outputs.items() if o is not None and not o.ok]
        if geo_score >= GEO_RISK_NO_TRADE:
            return "NO TRADE", (f"geopolitical risk {geo_score}/10 above "
                                f"{GEO_RISK_NO_TRADE} threshold")
        if len(dead_agents) >= 3:
            return "NO TRADE", (f"research coverage degraded "
                                f"({len(dead_agents)} agents down: "
                                f"{', '.join(dead_agents)})")
        if macro_bias == "NEGATIVE":
            return "TRADE (reduced size)", "macro bias negative — halve risk"
        return "TRADE", f"macro bias {macro_bias.lower()}, risks manageable"
