"""
NewsAgent — scans trusted Indian financial news for market-moving items.

Sources (all public RSS): Economic Times Markets, Moneycontrol,
Business Standard Markets. Sentiment: keyword scoring (FinBERT optional,
Phase 3). Only items with relevance > 0.5 are stored.
"""

from __future__ import annotations

from src.agents.research.base import BaseResearchAgent, ResearchOutput, fetch_rss
from src.utils.universe import UNIVERSE, nse_symbol

FEEDS = {
    "economic_times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "moneycontrol": "https://www.moneycontrol.com/rss/marketsnews.xml",
    "business_standard": "https://www.business-standard.com/rss/markets-106.rss",
    "livemint": "https://www.livemint.com/rss/markets",
}

MAX_AGE_HOURS = 18   # stale headlines must not steer today's bias

POSITIVE_WORDS = {
    "surge", "rally", "gain", "jump", "record", "upgrade", "beat", "profit",
    "growth", "strong", "bullish", "buy", "outperform", "expansion", "wins",
}
NEGATIVE_WORDS = {
    "fall", "drop", "crash", "plunge", "downgrade", "miss", "loss", "weak",
    "bearish", "sell", "underperform", "fraud", "probe", "default", "cuts",
}
# Phrases carry more signal than single words — weighted 2x
POSITIVE_PHRASES = ["strong earnings", "beat estimates", "record revenue",
                    "buy rating", "buyback", "new contract", "profit growth"]
NEGATIVE_PHRASES = ["miss estimates", "sell rating", "regulatory action",
                    "debt default", "under investigation", "layoffs",
                    "guided lower"]

# Company name fragments -> universe ticker (headline matching)
COMPANY_KEYWORDS = {
    "reliance": "RELIANCE.NS", "tcs": "TCS.NS", "tata consultancy": "TCS.NS",
    "hdfc bank": "HDFCBANK.NS", "icici": "ICICIBANK.NS", "infosys": "INFY.NS",
    "airtel": "BHARTIARTL.NS", "itc": "ITC.NS", "larsen": "LT.NS",
    "l&t": "LT.NS", "sbi": "SBIN.NS", "state bank": "SBIN.NS",
    "axis bank": "AXISBANK.NS", "kotak": "KOTAKBANK.NS",
    "hindustan unilever": "HINDUNILVR.NS", "bajaj finance": "BAJFINANCE.NS",
    "maruti": "MARUTI.NS", "mahindra": "M&M.NS", "sun pharma": "SUNPHARMA.NS",
    "titan": "TITAN.NS", "ultratech": "ULTRACEMCO.NS", "ntpc": "NTPC.NS",
    "power grid": "POWERGRID.NS", "tata steel": "TATASTEEL.NS",
    "tata motors": "TATAMOTORS.NS", "asian paints": "ASIANPAINT.NS",
    "hcl": "HCLTECH.NS", "wipro": "WIPRO.NS",
}

MARKET_KEYWORDS = {"nifty", "sensex", "nse", "bse", "fii", "dii", "rbi",
                   "sebi", "rupee", "market"}


def score_sentiment(text: str) -> float:
    """Keyword + phrase sentiment in [-1, 1]. Phrases weigh double."""
    low = text.lower()
    words = set(low.split())
    pos = len(words & POSITIVE_WORDS) \
        + 2 * sum(1 for p in POSITIVE_PHRASES if p in low)
    neg = len(words & NEGATIVE_WORDS) \
        + 2 * sum(1 for p in NEGATIVE_PHRASES if p in low)
    if pos + neg == 0:
        return 0.0
    return round((pos - neg) / (pos + neg), 2)


def score_relevance(text: str, watchlist: list[str] | None = None) -> tuple[float, list[str]]:
    """Relevance in [0, 1] + matched universe tickers."""
    low = text.lower()
    tickers = sorted({tck for kw, tck in COMPANY_KEYWORDS.items() if kw in low})
    relevance = 0.0
    if tickers:
        relevance = 0.7
        if watchlist and any(t in watchlist for t in tickers):
            relevance = 0.9
    elif any(kw in low for kw in MARKET_KEYWORDS):
        relevance = 0.55
    return relevance, tickers


class NewsAgent(BaseResearchAgent):
    name = "news_agent"
    description = "scans trusted Indian financial news RSS feeds"
    note_type = "news"

    def __init__(self, db_url: str | None = None, feeds: dict | None = None):
        super().__init__(db_url)
        self.feeds = feeds or FEEDS

    def run(self, context: dict) -> ResearchOutput:
        import hashlib
        from datetime import datetime, timedelta, timezone

        watchlist = context.get("watchlist") or []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=MAX_AGE_HOURS)
        items, sources_used = [], []
        seen_titles: set[str] = set()
        for source, url in self.feeds.items():
            entries = fetch_rss(url, limit=25)
            if entries:
                sources_used.append(source)
            for e in entries:
                # Freshness: skip anything older than 18h (undated kept —
                # most feeds date entries; dropping undated loses too much)
                pub = e.get("published_dt")
                if pub is not None and pub < cutoff:
                    continue
                # Cross-source dedup on normalized title
                key = hashlib.md5(
                    e["title"].lower().strip().encode()).hexdigest()
                if key in seen_titles:
                    continue
                seen_titles.add(key)

                text = f"{e['title']} {e['summary']}"
                relevance, tickers = score_relevance(text, watchlist)
                if relevance <= 0.5:
                    continue
                items.append({
                    "title": e["title"],
                    "source": source,
                    "link": e["link"],
                    "sentiment": score_sentiment(text),
                    "relevance": relevance,
                    "tickers": tickers,
                })

        items.sort(key=lambda x: (-x["relevance"], -abs(x["sentiment"])))
        items = items[:20]
        avg_sent = (round(sum(i["sentiment"] for i in items) / len(items), 2)
                    if items else 0.0)
        # Per-ticker sentiment: average across items mentioning each ticker,
        # so the signal council can tilt each stock's score by its own news.
        _by_ticker: dict[str, list[float]] = {}
        for it in items:
            for tk in it.get("tickers", []):
                _by_ticker.setdefault(tk, []).append(it["sentiment"])
        ticker_sentiment = {tk: round(sum(v) / len(v), 3)
                            for tk, v in _by_ticker.items()}
        summary = (f"{len(items)} relevant news items from "
                   f"{len(sources_used)}/{len(self.feeds)} feeds; "
                   f"average sentiment {avg_sent:+.2f}"
                   if sources_used else
                   "No news feeds reachable — flying without news today")
        return ResearchOutput(
            agent=self.name,
            summary=summary,
            findings=items,
            confidence=0.7 if sources_used else 0.0,
            sources=sources_used,
            reasoning="keyword sentiment over trusted RSS; relevance gate 0.5",
            payload={"avg_sentiment": avg_sent,
                     "n_items": len(items),
                     "ticker_sentiment": ticker_sentiment,
                     "tickers_in_news": sorted({t for i in items
                                                for t in i["tickers"]})},
        )
