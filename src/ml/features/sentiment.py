"""
Sentiment Feature Engineering — Multi-source NLP with FinBERT.
================================================================

Analyzes financial news sentiment using FinBERT (ProsusAI/finbert),
a BERT model fine-tuned on financial text that outperforms generic
sentiment models on financial language.

Sources (in priority order):
  1. Finnhub news API (global + Indian stocks, requires API key)
  2. Google News RSS feed (no API key needed, baseline source)

Features computed:
  - 24h/7d mean sentiment scores
  - Sentiment momentum (recent vs historical)
  - High-impact negative/positive news detection
  - Named entity risk extraction (regulatory, fraud, earnings)
"""

import logging
import warnings
import hashlib
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("transformers/torch not installed. Sentiment will use keyword fallback.")

try:
    import finnhub
    HAS_FINNHUB = True
except ImportError:
    HAS_FINNHUB = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Keyword-based fallback sentiment (when FinBERT unavailable)
# ─────────────────────────────────────────────

POSITIVE_KEYWORDS = {
    'surge', 'jump', 'rally', 'gain', 'soar', 'beat', 'outperform',
    'upgrade', 'bullish', 'profit', 'growth', 'record', 'high',
    'strong', 'optimistic', 'boost', 'recover', 'breakout', 'buy',
    'dividend', 'expansion', 'innovation', 'approval', 'positive'
}

NEGATIVE_KEYWORDS = {
    'crash', 'plunge', 'sink', 'fall', 'drop', 'miss', 'downgrade',
    'bearish', 'loss', 'decline', 'weak', 'fear', 'panic', 'sell',
    'fraud', 'scam', 'probe', 'raid', 'default', 'bankruptcy',
    'warning', 'cut', 'layoff', 'shutdown', 'investigation', 'negative',
    'slump', 'tumble', 'recession', 'inflation', 'sanctions'
}


class SentimentFeatures:
    """
    Multi-source sentiment analysis using FinBERT.

    Parameters
    ----------
    ticker : str
        Stock ticker for news search.
    company_name : str
        Full company name for broader news search.
    finnhub_api_key : str, optional
        Finnhub API key. If None, uses Google News RSS only.
    """

    _model = None
    _tokenizer = None
    _model_loaded = False

    def __init__(self, ticker: str, company_name: str = '',
                 finnhub_api_key: Optional[str] = None):
        self.ticker = ticker.upper().strip()
        self.company_name = company_name or self.ticker
        self.finnhub_api_key = finnhub_api_key
        self._news_cache = None

    @classmethod
    def _load_model(cls):
        """Load FinBERT model (cached at class level for reuse)."""
        if cls._model_loaded:
            return

        if not HAS_TRANSFORMERS:
            logger.warning("FinBERT unavailable. Using keyword-based sentiment.")
            cls._model_loaded = True
            return

        try:
            model_name = 'ProsusAI/finbert'
            logger.info(f"Loading FinBERT model: {model_name}")
            cls._tokenizer = AutoTokenizer.from_pretrained(model_name)
            cls._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            cls._model.eval()  # Inference mode
            cls._model_loaded = True
            logger.info("FinBERT loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load FinBERT: {e}. Using keyword fallback.")
            cls._model_loaded = True

    def fetch_news(self, days_back: int = 7, max_articles: int = 50) -> list:
        """
        Fetch news from all available sources.
        Each article: {title, source, published_at, url}
        Deduplicate by title similarity.
        Return list sorted by recency.
        """
        if self._news_cache is not None:
            return self._news_cache

        articles = []

        # Source 1: Finnhub
        if self.finnhub_api_key and HAS_FINNHUB:
            articles.extend(self._fetch_finnhub(days_back, max_articles))

        # Source 2: Google News RSS
        if HAS_REQUESTS and HAS_BS4:
            articles.extend(self._fetch_google_news_rss(days_back, max_articles))

        # Deduplicate by title hash
        seen = set()
        unique = []
        for art in articles:
            title_hash = hashlib.md5(
                art['title'].lower().strip()[:50].encode()
            ).hexdigest()
            if title_hash not in seen:
                seen.add(title_hash)
                unique.append(art)

        # Sort by recency
        unique.sort(key=lambda x: x.get('published_at', ''), reverse=True)
        self._news_cache = unique[:max_articles]
        return self._news_cache

    def _fetch_finnhub(self, days_back: int, max_articles: int) -> list:
        """Fetch news from Finnhub API."""
        articles = []
        try:
            client = finnhub.Client(api_key=self.finnhub_api_key)
            end = datetime.now()
            start = end - timedelta(days=days_back)

            news = client.company_news(
                self.ticker,
                _from=start.strftime('%Y-%m-%d'),
                to=end.strftime('%Y-%m-%d')
            )

            for item in (news or [])[:max_articles]:
                articles.append({
                    'title': item.get('headline', ''),
                    'source': item.get('source', 'Finnhub'),
                    'published_at': datetime.fromtimestamp(
                        item.get('datetime', 0)
                    ).isoformat(),
                    'url': item.get('url', ''),
                })
        except Exception as e:
            logger.warning(f"Finnhub fetch failed: {e}")

        return articles

    def _fetch_google_news_rss(self, days_back: int, max_articles: int) -> list:
        """Fetch news from Google News RSS (no API key needed)."""
        articles = []
        try:
            # Search for ticker + company name
            queries = [
                f"{self.ticker} stock",
                f"{self.company_name} shares" if self.company_name != self.ticker else None
            ]

            for query in filter(None, queries):
                url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
                resp = requests.get(url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; QuantraBot/1.0)'
                })

                if resp.status_code != 200:
                    continue

                soup = BeautifulSoup(resp.content, 'xml')
                items = soup.find_all('item')

                for item in items[:max_articles // 2]:
                    title = item.find('title')
                    pub_date = item.find('pubDate')
                    source = item.find('source')
                    link = item.find('link')

                    if title:
                        articles.append({
                            'title': title.text.strip(),
                            'source': source.text.strip() if source else 'Google News',
                            'published_at': pub_date.text.strip() if pub_date else '',
                            'url': link.text.strip() if link else '',
                        })

        except Exception as e:
            logger.warning(f"Google News RSS fetch failed: {e}")

        return articles

    def analyze_sentiment(self, texts: list) -> list:
        """
        Run FinBERT (or keyword fallback) on each text.
        Returns [{text, sentiment, positive_score, negative_score,
                  neutral_score, compound_score}]
        """
        self._load_model()
        results = []

        for text in texts:
            if not text or not text.strip():
                results.append({
                    'text': text or '',
                    'sentiment': 'neutral',
                    'positive_score': 0.0,
                    'negative_score': 0.0,
                    'neutral_score': 1.0,
                    'compound_score': 0.0,
                })
                continue

            if self._model is not None and self._tokenizer is not None:
                # FinBERT inference
                try:
                    inputs = self._tokenizer(
                        text[:512], return_tensors='pt',
                        truncation=True, max_length=512, padding=True
                    )
                    with torch.no_grad():
                        outputs = self._model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

                    # FinBERT labels: positive(0), negative(1), neutral(2)
                    pos = probs[0][0].item()
                    neg = probs[0][1].item()
                    neu = probs[0][2].item()
                    compound = pos - neg

                    sentiment = 'positive' if pos > max(neg, neu) else \
                                'negative' if neg > max(pos, neu) else 'neutral'

                    results.append({
                        'text': text[:200],
                        'sentiment': sentiment,
                        'positive_score': round(pos, 4),
                        'negative_score': round(neg, 4),
                        'neutral_score': round(neu, 4),
                        'compound_score': round(compound, 4),
                    })
                except Exception as e:
                    logger.warning(f"FinBERT inference failed: {e}")
                    results.append(self._keyword_sentiment(text))
            else:
                # Keyword fallback
                results.append(self._keyword_sentiment(text))

        return results

    @staticmethod
    def _keyword_sentiment(text: str) -> dict:
        """Keyword-based sentiment as fallback when FinBERT unavailable."""
        words = set(text.lower().split())
        pos_count = len(words & POSITIVE_KEYWORDS)
        neg_count = len(words & NEGATIVE_KEYWORDS)
        total = pos_count + neg_count + 1

        pos_score = pos_count / total
        neg_score = neg_count / total
        neu_score = 1.0 - pos_score - neg_score
        compound = pos_score - neg_score

        sentiment = 'positive' if compound > 0.1 else \
                    'negative' if compound < -0.1 else 'neutral'

        return {
            'text': text[:200],
            'sentiment': sentiment,
            'positive_score': round(pos_score, 4),
            'negative_score': round(neg_score, 4),
            'neutral_score': round(neu_score, 4),
            'compound_score': round(compound, 4),
        }

    def compute_sentiment_features(self) -> dict:
        """
        Aggregate sentiment into ML features.

        Returns dict with:
          news_sentiment_24h, news_sentiment_7d, news_count_24h/7d,
          sentiment_momentum, negative_news_ratio,
          high_impact_negative, high_impact_positive,
          mentions_regulatory, mentions_fraud, mentions_promoter_sell,
          mentions_earnings_beat, mentions_earnings_miss,
          mentions_geopolitical
        """
        articles = self.fetch_news(days_back=7, max_articles=50)
        titles = [a['title'] for a in articles]

        if not titles:
            return self._empty_sentiment_features()

        # Run sentiment analysis on all titles
        sentiments = self.analyze_sentiment(titles)
        compounds = [s['compound_score'] for s in sentiments]

        # Time-based filtering
        now = datetime.now()
        compounds_24h = []
        compounds_7d = compounds  # all are within 7 days

        for i, art in enumerate(articles):
            try:
                pub = pd.Timestamp(art.get('published_at', '')).to_pydatetime()
                if (now - pub).total_seconds() < 86400:
                    compounds_24h.append(compounds[i])
            except Exception:
                # If parsing fails, include in 24h (assume recent)
                compounds_24h.append(compounds[i])

        if not compounds_24h:
            compounds_24h = compounds[:5]  # Use first 5 as fallback

        features = {
            'sent_news_sentiment_24h': np.mean(compounds_24h) if compounds_24h else 0,
            'sent_news_sentiment_7d': np.mean(compounds_7d) if compounds_7d else 0,
            'sent_news_count_24h': len(compounds_24h),
            'sent_news_count_7d': len(compounds_7d),
            'sent_sentiment_momentum': (
                (np.mean(compounds_24h) if compounds_24h else 0) -
                (np.mean(compounds_7d) if compounds_7d else 0)
            ),
            'sent_negative_news_ratio': sum(1 for c in compounds if c < -0.1) / max(len(compounds), 1),
            'sent_high_impact_negative': int(any(c < -0.7 for c in compounds)),
            'sent_high_impact_positive': int(any(c > 0.7 for c in compounds)),
            'sent_sentiment_std': np.std(compounds) if compounds else 0,
            'sent_source_diversity': len(set(a.get('source', '') for a in articles)),
        }

        # Named entity risk extraction from headlines
        all_text = ' '.join(titles).lower()

        features['sent_mentions_regulatory'] = int(
            any(w in all_text for w in ['sebi', 'rbi', 'sec', 'fed', 'regulatory', 'compliance'])
        )
        features['sent_mentions_fraud'] = int(
            any(w in all_text for w in ['fraud', 'scam', 'probe', 'raid', 'investigation', 'arrest'])
        )
        features['sent_mentions_promoter_sell'] = int(
            any(w in all_text for w in ['promoter sell', 'insider sell', 'stake sale', 'block deal'])
        )
        features['sent_mentions_earnings_beat'] = int(
            any(w in all_text for w in ['beat', 'outperform', 'above estimate', 'surprise'])
        )
        features['sent_mentions_earnings_miss'] = int(
            any(w in all_text for w in ['miss', 'disappoint', 'below estimate', 'guidance cut'])
        )
        features['sent_mentions_geopolitical'] = int(
            any(w in all_text for w in ['war', 'sanctions', 'tariff', 'geopolitical', 'conflict'])
        )

        return features

    @staticmethod
    def _empty_sentiment_features() -> dict:
        """Return zeroed sentiment features when no data available."""
        return {
            'sent_news_sentiment_24h': 0, 'sent_news_sentiment_7d': 0,
            'sent_news_count_24h': 0, 'sent_news_count_7d': 0,
            'sent_sentiment_momentum': 0, 'sent_negative_news_ratio': 0,
            'sent_high_impact_negative': 0, 'sent_high_impact_positive': 0,
            'sent_sentiment_std': 0, 'sent_source_diversity': 0,
            'sent_mentions_regulatory': 0, 'sent_mentions_fraud': 0,
            'sent_mentions_promoter_sell': 0, 'sent_mentions_earnings_beat': 0,
            'sent_mentions_earnings_miss': 0, 'sent_mentions_geopolitical': 0,
        }

    def get_news_summary(self, n: int = 5) -> list:
        """
        Return top N most impactful headlines (highest abs compound score).
        Formatted as: "[POSITIVE/NEGATIVE/NEUTRAL] Source: Headline"
        Used for terminal analysis display.
        """
        articles = self.fetch_news(days_back=7, max_articles=30)
        titles = [a['title'] for a in articles]

        if not titles:
            return ["No recent news available."]

        sentiments = self.analyze_sentiment(titles)

        # Sort by absolute compound score (most impactful first)
        paired = list(zip(articles, sentiments))
        paired.sort(key=lambda x: abs(x[1]['compound_score']), reverse=True)

        summaries = []
        for art, sent in paired[:n]:
            label = sent['sentiment'].upper()
            source = art.get('source', 'Unknown')
            headline = art['title'][:100]
            summaries.append(f"[{label}] {source}: {headline}")

        return summaries
