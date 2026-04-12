"""
Report Generator — Formats analysis results into terminal-ready text.
=====================================================================

Produces two output formats:
  1. Terminal format: Bloomberg-style colored text for the HTML terminal
  2. JSON format: structured data for API responses
"""

from datetime import datetime
from typing import Dict, Any


class ReportGenerator:
    """Generate formatted analysis reports for terminal display."""

    def generate_summary(self, analysis: Dict) -> str:
        """
        Generate a Bloomberg-style text summary from analysis results.
        This is what gets displayed in the ANALYZE panel.
        """
        ticker = analysis.get('ticker', '???')
        overview = analysis.get('overview', {})
        tech = analysis.get('technical', {})
        fund = analysis.get('fundamental', {})
        sent = analysis.get('sentiment', {})
        ml = analysis.get('ml_signals', {})
        plan = analysis.get('trade_plan', {})
        risk = analysis.get('risk_assessment', {})

        lines = []
        lines.append(f"{'═' * 60}")
        lines.append(f"  QUANTRA ANALYSIS: {ticker} ({analysis.get('exchange', '')})")
        lines.append(f"  {overview.get('name', ticker)} | {overview.get('sector', 'N/A')}")
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"{'═' * 60}")
        lines.append("")

        # Market data
        close = tech.get('close', 0)
        lines.append(f"  LAST PRICE: ₹{close:,.2f}" if analysis.get('exchange') == 'NSE'
                     else f"  LAST PRICE: ${close:,.2f}")
        lines.append("")

        # Verdict
        verdict = plan.get('verdict', 'HOLD')
        comp = plan.get('composite_score', 0)
        lines.append(f"  ┌─── VERDICT: {verdict} (Score: {comp:+.2f}) ───┐")
        lines.append(f"  │  Risk Level: {risk.get('level', 'MEDIUM'):>10}         │")
        lines.append(f"  └─────────────────────────────────────────────┘")
        lines.append("")

        # Technical summary
        lines.append("  ── TECHNICAL ANALYSIS ──")
        lines.append(f"  Structure: {tech.get('structure_trend', 'N/A')}")
        for sig in tech.get('trend_signals', [])[:3]:
            lines.append(f"    • {sig}")
        for sig in tech.get('momentum_signals', [])[:2]:
            lines.append(f"    • {sig}")

        vol = tech.get('volatility', {})
        lines.append(f"  ATR: {vol.get('atr', 0):.2f} ({vol.get('atr_pct', 0):.1f}%)")
        if vol.get('bb_squeeze'):
            lines.append("    ⚡ BOLLINGER SQUEEZE — BREAKOUT IMMINENT")
        if tech.get('active_patterns'):
            lines.append(f"  Patterns: {', '.join(tech['active_patterns'][:3])}")
        lines.append("")

        # Fundamental summary
        lines.append("  ── FUNDAMENTAL ANALYSIS ──")
        fund_grade = fund.get('grade', 'N/A')
        fund_score = fund.get('score', 0)
        lines.append(f"  Grade: {fund_grade} ({fund_score}/{fund.get('max_score', 7)})")
        for reason in fund.get('reasoning', [])[:4]:
            lines.append(f"    • {reason}")
        lines.append("")

        # Sentiment
        lines.append("  ── SENTIMENT ANALYSIS ──")
        lines.append(f"  Overall Mood: {sent.get('overall_mood', 'N/A')}")
        for headline in sent.get('top_headlines', [])[:3]:
            lines.append(f"    {headline[:65]}")
        risk_flags = [f for f in sent.get('risk_flags', []) if f]
        if risk_flags:
            lines.append(f"  ⚠ RISKS: {', '.join(risk_flags)}")
        lines.append("")

        # ML signals
        lines.append("  ── ML/DL SIGNALS ──")
        if ml.get('available'):
            lines.append(f"  Ensemble: {ml.get('signal', 'N/A')} "
                         f"(Confidence: {ml.get('confidence', 0):.0%})")
            lines.append(f"  Agreement: {ml.get('agreement', 'N/A')}")
            model_sigs = ml.get('model_signals', {})
            for model, sig in model_sigs.items():
                if isinstance(sig, dict):
                    lines.append(f"    {model.upper()}: {sig.get('signal', 'N/A')} "
                                 f"({sig.get('confidence', 0):.0%})")
        else:
            lines.append("  [Models not trained — run TRAIN command]")
        lines.append("")

        # Trade plan
        lines.append("  ── TRADE PLAN ──")
        if plan.get('stop_loss'):
            lines.append(f"  Entry:    {plan.get('entry_price', '?')}")
            lines.append(f"  Stop:     {plan.get('stop_loss', '?')}")
            lines.append(f"  Target 1: {plan.get('target_1', '?')}")
            lines.append(f"  Target 2: {plan.get('target_2', '?')}")
            lines.append(f"  R:R:      {plan.get('risk_reward', '?')}")
        else:
            lines.append("  No actionable trade at this time.")
        lines.append("")

        # Risk assessment
        if risk.get('flags'):
            lines.append("  ── RISK FLAGS ──")
            for flag in risk['flags']:
                lines.append(f"  ⚠ {flag}")
            lines.append("")

        lines.append(f"{'═' * 60}")
        lines.append("  Quantra Intelligence Engine v4.0 | Not financial advice")
        lines.append(f"{'═' * 60}")

        return '\n'.join(lines)

    def generate_json(self, analysis: Dict) -> Dict:
        """Return analysis as structured JSON for API responses."""
        return {
            'ticker': analysis.get('ticker'),
            'exchange': analysis.get('exchange'),
            'timestamp': analysis.get('analyzed_at'),
            'price': analysis.get('technical', {}).get('close', 0),
            'verdict': analysis.get('trade_plan', {}).get('verdict', 'HOLD'),
            'composite_score': analysis.get('trade_plan', {}).get('composite_score', 0),
            'risk_level': analysis.get('risk_assessment', {}).get('level', 'MEDIUM'),
            'trade_plan': analysis.get('trade_plan', {}),
            'technical': analysis.get('technical', {}),
            'fundamental_grade': analysis.get('fundamental', {}).get('grade', 'N/A'),
            'sentiment_mood': analysis.get('sentiment', {}).get('overall_mood', 'NEUTRAL'),
            'ml_signal': analysis.get('ml_signals', {}).get('signal', None),
            'ml_confidence': analysis.get('ml_signals', {}).get('confidence', 0),
        }
