"""
GeopoliticalAgent — global risk events relevant to Indian markets.

Monitors public RSS (Reuters World via Google News, Al Jazeera) for:
India-Pakistan, Russia-Ukraine, Israel-Hamas, US-China trade, oil supply
disruptions, sanctions affecting Indian imports. Output: risk score 0-10
with the top 3 relevant events.
"""

from __future__ import annotations

from src.agents.research.base import BaseResearchAgent, ResearchOutput, fetch_rss

FEEDS = {
    "aljazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "google_news_world": "https://news.google.com/rss/search?q=geopolitics+OR+sanctions+OR+conflict&hl=en-IN&gl=IN&ceid=IN:en",
}

# theme -> (keywords, weight 0-3): India-adjacent themes weigh more
THEMES = {
    "india_pakistan": ({"india", "pakistan", "kashmir", "loc "}, 3.0),
    "oil_supply": ({"oil supply", "opec", "strait of hormuz", "crude",
                    "oil price"}, 2.5),
    "sanctions": ({"sanction", "sanctions", "embargo"}, 2.0),
    "us_china": ({"tariff", "us-china", "trade war", "export controls"}, 2.0),
    "russia_ukraine": ({"russia", "ukraine", "kyiv", "moscow"}, 1.5),
    "middle_east": ({"israel", "hamas", "gaza", "hezbollah", "iran"}, 1.5),
}

ESCALATION_WORDS = {"strike", "attack", "invasion", "missile", "war",
                    "escalat", "retaliat", "blockade", "closes", "disrupt"}


class GeopoliticalAgent(BaseResearchAgent):
    name = "geopolitical_agent"
    description = "wars, sanctions, elections, trade policy risk"
    note_type = "geopolitical"

    def __init__(self, db_url: str | None = None, feeds: dict | None = None):
        super().__init__(db_url)
        self.feeds = feeds or FEEDS

    def run(self, context: dict) -> ResearchOutput:
        events, sources_used = [], []
        for source, url in self.feeds.items():
            entries = fetch_rss(url, limit=40)
            if entries:
                sources_used.append(source)
            for e in entries:
                text = f"{e['title']} {e['summary']}".lower()
                theme, weight = self._match_theme(text)
                if theme is None:
                    continue
                escalating = any(w in text for w in ESCALATION_WORDS)
                events.append({
                    "title": e["title"],
                    "theme": theme,
                    "weight": weight * (1.5 if escalating else 1.0),
                    "escalating": escalating,
                    "source": source,
                    "link": e["link"],
                })

        events.sort(key=lambda x: -x["weight"])
        top3 = events[:3]
        risk_score = self._risk_score(events)
        summary = (f"Geopolitical risk {risk_score}/10; top: "
                   + "; ".join(e["title"][:70] for e in top3)
                   if top3 else
                   f"Geopolitical risk {risk_score}/10 — no notable events")
        return ResearchOutput(
            agent=self.name,
            summary=summary,
            findings=events[:15],
            confidence=0.6 if sources_used else 0.0,
            sources=sources_used,
            reasoning="theme-weighted keyword scan; escalation words x1.5",
            payload={"geopolitical_risk_score": risk_score,
                     "top_events": top3},
        )

    @staticmethod
    def _match_theme(text: str) -> tuple[str | None, float]:
        best, best_w = None, 0.0
        for theme, (keywords, weight) in THEMES.items():
            if any(k in text for k in keywords) and weight > best_w:
                best, best_w = theme, weight
        return best, best_w

    @staticmethod
    def _risk_score(events: list[dict]) -> float:
        """0-10: saturating sum of event weights (12 pts of weight = 10)."""
        total = sum(e["weight"] for e in events)
        return round(min(10.0, total / 1.2), 1)
