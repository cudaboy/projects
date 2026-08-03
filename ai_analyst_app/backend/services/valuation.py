"""Lightweight valuation helpers for the AI analyst workflow.

This module intentionally starts with a conservative DCF-lite/relative-value
layer.  Korean web financial statements are not always stable enough for a full
spreadsheet-style DCF, so the helper extracts available signals and emits a
traceable valuation context that the Fund Manager can cite or downgrade.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _numbers_from_text(text: str) -> list[float]:
    values: list[float] = []
    for raw in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text or ""):
        try:
            values.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return values


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _technical_current_price(technical_text: str) -> float | None:
    payload = _extract_json(technical_text)
    value = payload.get("current_close")
    try:
        if value is not None and not math.isnan(float(value)):
            return float(value)
    except Exception:
        return None
    return None


def build_valuation_summary(finance_text: str, technical_text: str) -> Dict[str, Any]:
    """Build a conservative valuation context from available agent outputs.

    The function prefers explicit technical current price if present and uses
    keyword/numeric heuristics only as a confidence-weighted valuation signal.
    It is not a replacement for audited DCF data.
    """
    current_price = _technical_current_price(technical_text)
    lower_text = (finance_text or "").lower()
    numbers = _numbers_from_text(finance_text)

    positive_hits = sum(1 for word in ["roe", "영업이익", "순이익", "성장", "증가", "흑자", "현금"] if word in lower_text)
    negative_hits = sum(1 for word in ["적자", "감소", "부채", "손실", "마이너스", "악화"] if word in lower_text)

    valuation_score = 50 + positive_hits * 6 - negative_hits * 8
    # If the analysis includes many numeric references, raise confidence; if it
    # is almost pure prose, keep valuation conservative.
    confidence = "low"
    if len(numbers) >= 10:
        confidence = "medium"
        valuation_score += 5
    if len(numbers) >= 25:
        confidence = "high"
        valuation_score += 5

    valuation_score = int(_clamp(round(valuation_score), 0, 100))

    if current_price:
        # Conservative range: score moves upside/downside, capped to avoid fake precision.
        midpoint_multiplier = 1 + ((valuation_score - 50) / 100) * 0.35
        target_mid = current_price * midpoint_multiplier
        target_low = target_mid * 0.9
        target_high = target_mid * 1.1
        upside_pct = ((target_mid - current_price) / current_price) * 100
    else:
        target_low = target_mid = target_high = upside_pct = None

    return {
        "status": "success",
        "method": "relative_dcf_lite_v1",
        "valuation_score": valuation_score,
        "confidence": confidence,
        "current_price": round(current_price, 2) if current_price else None,
        "target_price_low": round(target_low, 2) if target_low else None,
        "target_price_mid": round(target_mid, 2) if target_mid else None,
        "target_price_high": round(target_high, 2) if target_high else None,
        "upside_pct_mid": round(upside_pct, 2) if upside_pct is not None else None,
        "positive_signal_count": positive_hits,
        "negative_signal_count": negative_hits,
        "numeric_evidence_count": len(numbers),
        "caveat": "FnGuide/LLM 텍스트 기반 DCF-lite 보조 지표입니다. 감사 재무모델이 아니므로 신뢰도와 함께 해석해야 합니다.",
    }
