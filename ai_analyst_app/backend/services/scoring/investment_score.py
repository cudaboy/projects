"""Lightweight deterministic scoring and risk layer for the AI analyst graph."""

from __future__ import annotations

import json
import re
from typing import Any, Dict

ALGORITHM_VERSION = "v2-technical-risk-2026-07-31"


def clamp_score(value: float) -> int:
    """Clamp score to 0-100 integer range."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 50.0
    return int(max(0, min(100, round(numeric))))


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def score_technical_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize technical report JSON into scoring dimensions."""
    if not summary or summary.get("status") != "success":
        return {
            "technical_score": 50,
            "technical_signal": "neutral",
            "technical_risk": 50,
            "notes": ["기술적 지표 계산 결과가 없어 중립 점수를 적용했습니다."],
        }

    technical_score = clamp_score(summary.get("technical_score", 50))
    volatility = float(summary.get("volatility_20d_pct") or 0)
    rsi = float(summary.get("rsi_14") or 50)
    macd_signal = str(summary.get("macd_signal") or "neutral")
    trend = str(summary.get("trend") or "sideways")

    risk = 35
    notes = []
    if volatility > 6:
        risk += 25
        notes.append("20일 변동성이 높아 단기 리스크가 큽니다.")
    elif volatility > 4:
        risk += 12
        notes.append("20일 변동성이 다소 높습니다.")

    if rsi >= 75:
        risk += 15
        notes.append("RSI 과열 구간으로 추격 매수 위험이 있습니다.")
    elif rsi <= 30:
        risk += 10
        notes.append("RSI 과매도 구간으로 변동성 확대 가능성이 있습니다.")

    if trend == "downtrend":
        risk += 15
        notes.append("가격이 주요 이동평균 아래에 있어 하락 추세 리스크가 있습니다.")
    elif trend == "uptrend":
        notes.append("가격이 주요 이동평균 위에 있어 추세는 우호적입니다.")

    if macd_signal in {"bearish", "bearish_cross"}:
        risk += 8
        notes.append("MACD가 약세 신호를 보입니다.")
    elif macd_signal in {"bullish", "bullish_cross"}:
        notes.append("MACD가 강세 신호를 보입니다.")

    if not notes:
        notes.append("기술적 위험은 중립 범위입니다.")

    if technical_score >= 70:
        signal = "positive"
    elif technical_score <= 40:
        signal = "negative"
    else:
        signal = "neutral"

    return {
        "technical_score": technical_score,
        "technical_signal": signal,
        "technical_risk": clamp_score(risk),
        "notes": notes,
    }


def derive_rating(final_score: int, risk_score: int) -> str:
    """Derive conservative Buy/Hold/Sell rating from score and risk."""
    final_score = clamp_score(final_score)
    risk_score = clamp_score(risk_score)

    if final_score >= 75 and risk_score <= 55:
        return "Buy"
    if final_score <= 40 or risk_score >= 80:
        return "Sell"
    return "Hold"


def build_risk_summary(
    finance_text: str,
    news_text: str,
    technical_text: str,
    valuation_text: str = "",
) -> Dict[str, Any]:
    """Build deterministic score/risk context from agent outputs.

    This does not replace LLM judgment.  It constrains the final Fund Manager
    prompt with explicit score/risk context and algorithm metadata.
    """
    technical_json = _extract_json(technical_text)
    technical = score_technical_summary(technical_json)

    valuation_json = _extract_json(valuation_text)
    combined = f"{finance_text}\n{news_text}\n{valuation_text}".lower()
    risk_flags = []
    risk_score = technical["technical_risk"]

    negative_keywords = [
        "적자",
        "손실",
        "부채",
        "감소",
        "하락",
        "리스크",
        "규제",
        "소송",
        "불확실",
        "악재",
        "우려",
    ]
    positive_keywords = ["성장", "증가", "개선", "흑자", "수주", "신사업", "호재", "견조", "상승"]

    negative_hits = sum(1 for word in negative_keywords if word in combined)
    positive_hits = sum(1 for word in positive_keywords if word in combined)

    if negative_hits >= 4:
        risk_score += 15
        risk_flags.append("재무/뉴스 텍스트에서 부정 키워드가 다수 탐지되었습니다.")
    elif negative_hits >= 2:
        risk_score += 8
        risk_flags.append("재무/뉴스 텍스트에서 일부 부정 키워드가 탐지되었습니다.")

    if positive_hits >= 4:
        risk_score -= 8
        risk_flags.append("성장/개선 관련 긍정 키워드가 다수 탐지되었습니다.")

    if len(finance_text or "") < 80:
        risk_score += 10
        risk_flags.append("재무 분석 텍스트가 짧아 근거 신뢰도를 낮게 봅니다.")
    if len(news_text or "") < 80:
        risk_score += 6
        risk_flags.append("뉴스 분석 텍스트가 짧아 모멘텀 신뢰도를 낮게 봅니다.")

    risk_score = clamp_score(risk_score)
    valuation_score = valuation_json.get("valuation_score")
    try:
        valuation_score = float(valuation_score) if valuation_score is not None else 55.0
    except (TypeError, ValueError):
        valuation_score = 55.0

    if valuation_json.get("confidence") == "low":
        risk_score = clamp_score(risk_score + 5)
        risk_flags.append("밸류에이션 근거 신뢰도가 낮아 보수적으로 리스크를 가산했습니다.")

    final_score = clamp_score(
        (technical["technical_score"] * 0.40)
        + ((100 - risk_score) * 0.30)
        + (valuation_score * 0.30)
    )
    rating = derive_rating(final_score, risk_score)

    if not risk_flags:
        risk_flags.append("결정론적 리스크 필터에서 특이 위험 신호는 제한적입니다.")

    return {
        "algorithm_version": ALGORITHM_VERSION,
        "final_score": final_score,
        "risk_score": risk_score,
        "rating_hint": rating,
        "technical": technical,
        "valuation": valuation_json,
        "risk_flags": risk_flags,
        "disclaimer": "본 점수는 교육/연구 목적의 보조 지표이며 투자 자문이 아닙니다.",
    }
