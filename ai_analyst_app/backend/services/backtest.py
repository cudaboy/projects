"""Algorithm-version and lightweight return backtest helpers."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List

import httpx

from backend.services.indicators.technical import parse_naver_chart_response

FOLLOWUP_HORIZONS = (5, 20, 60)


def parse_score_json(value: str | None) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _rating_direction(rating: str | None) -> int:
    text = (rating or "").lower()
    if "buy" in text or "매수" in text:
        return 1
    if "sell" in text or "매도" in text:
        return -1
    return 0


def _fetch_price_frame(company_code: str, start: datetime, end: datetime):
    """Fetch Naver daily OHLCV frame for a code and date window."""
    if not company_code:
        return None
    url = (
        "https://m.stock.naver.com/front-api/external/chart/domestic/info"
        f"?symbol={company_code}&requestType=1"
        f"&startTime={start:%Y%m%d}&endTime={end:%Y%m%d}&timeframe=day"
    )
    try:
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
        frame = parse_naver_chart_response(response.text)
        if frame.empty:
            return None
        return frame
    except Exception:
        return None


def price_snapshot(company_code: str, as_of: datetime | None = None, lookback_days: int = 30) -> Dict[str, Any]:
    """Return latest close price snapshot at or before `as_of`."""
    as_of = as_of or datetime.now()
    frame = _fetch_price_frame(company_code, as_of - timedelta(days=lookback_days), as_of)
    if frame is None or frame.empty:
        return {"status": "error", "message": "price_not_available", "stock_code": company_code}
    row = frame.iloc[-1]
    return {
        "status": "success",
        "stock_code": company_code,
        "price": float(row["close"]),
        "price_date": row["date"].strftime("%Y-%m-%d"),
    }


def _price_on_or_after(company_code: str, target_date: datetime, lookahead_days: int = 10) -> Dict[str, Any]:
    frame = _fetch_price_frame(company_code, target_date, target_date + timedelta(days=lookahead_days))
    if frame is None or frame.empty:
        return {"status": "missing_price"}
    row = frame.iloc[0]
    return {"status": "success", "price": float(row["close"]), "price_date": row["date"].strftime("%Y-%m-%d")}


def build_followup_returns(history: Any, now: datetime | None = None) -> List[Dict[str, Any]]:
    """Calculate 5/20/60-day follow-up returns for one stored analysis.

    Returns are calculated against the stored analysis close price.  If the
    rating was Sell, `directional_return_pct` flips the sign so a price decline
    is counted as a correct directional call.  Hold is scored as 0 directional
    exposure while still reporting raw return.
    """
    now = now or datetime.now()
    created_at = _parse_datetime(getattr(history, "created_at", None))
    code = getattr(history, "stock_code", None)
    base_price = _safe_float(getattr(history, "analysis_price", None))
    rating = getattr(history, "rating", None) or parse_score_json(getattr(history, "investment_score", None)).get("rating_hint")
    direction = _rating_direction(rating)

    rows: List[Dict[str, Any]] = []
    for horizon in FOLLOWUP_HORIZONS:
        base = {"horizon_days": horizon, "rating": rating or "Unknown"}
        if not created_at or not code or not base_price:
            rows.append({**base, "status": "pending_metadata", "return_pct": None, "directional_return_pct": None})
            continue
        target_date = created_at + timedelta(days=horizon)
        if target_date > now:
            rows.append({**base, "status": "pending_until", "target_date": target_date.date().isoformat(), "return_pct": None, "directional_return_pct": None})
            continue
        followup = _price_on_or_after(code, target_date)
        if followup.get("status") != "success":
            rows.append({**base, "status": "missing_followup_price", "target_date": target_date.date().isoformat(), "return_pct": None, "directional_return_pct": None})
            continue
        raw_return = (float(followup["price"]) - base_price) / base_price * 100
        rows.append(
            {
                **base,
                "status": "success",
                "target_date": target_date.date().isoformat(),
                "followup_price_date": followup["price_date"],
                "base_price": round(base_price, 2),
                "followup_price": round(float(followup["price"]), 2),
                "return_pct": round(raw_return, 2),
                "directional_return_pct": round(raw_return * direction, 2) if direction else 0.0,
            }
        )
    return rows


def extract_score_record(history: Any, include_followups: bool = True) -> Dict[str, Any]:
    score = parse_score_json(getattr(history, "investment_score", None))
    valuation = parse_score_json(getattr(history, "valuation_summary", None))
    followups = build_followup_returns(history) if include_followups else []
    return {
        "id": getattr(history, "id", None),
        "company_name": getattr(history, "company_name", None),
        "stock_code": getattr(history, "stock_code", None),
        "analysis_price": getattr(history, "analysis_price", None),
        "analysis_price_date": getattr(history, "analysis_price_date", None),
        "created_at": getattr(history, "created_at", None).isoformat() if getattr(history, "created_at", None) else None,
        "algorithm_version": score.get("algorithm_version") or getattr(history, "algorithm_version", None) or "legacy",
        "rating": score.get("rating_hint") or getattr(history, "rating", None) or "Unknown",
        "final_score": score.get("final_score"),
        "risk_score": score.get("risk_score"),
        "valuation_score": valuation.get("valuation_score"),
        "target_price_mid": valuation.get("target_price_mid"),
        "upside_pct_mid": valuation.get("upside_pct_mid"),
        "followup_returns": followups,
    }


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def summarize_algorithm_performance(records: Iterable[Any]) -> Dict[str, Any]:
    """Summarize stored analysis records by algorithm version and returns."""
    rows = [extract_score_record(record) for record in records]
    by_version: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_version[row["algorithm_version"]].append(row)

    versions = []
    for version, items in sorted(by_version.items()):
        ratings = Counter(item["rating"] for item in items)
        final_scores = [float(item["final_score"]) for item in items if item.get("final_score") is not None]
        risk_scores = [float(item["risk_score"]) for item in items if item.get("risk_score") is not None]
        horizon_summary: Dict[str, Any] = {}
        for horizon in FOLLOWUP_HORIZONS:
            successful = [
                f for item in items for f in item.get("followup_returns", [])
                if f.get("horizon_days") == horizon and f.get("status") == "success"
            ]
            horizon_summary[f"{horizon}d"] = {
                "completed_count": len(successful),
                "avg_return_pct": _avg([float(f["return_pct"]) for f in successful if f.get("return_pct") is not None]),
                "avg_directional_return_pct": _avg([float(f["directional_return_pct"]) for f in successful if f.get("directional_return_pct") is not None]),
                "win_rate_pct": round(
                    sum(1 for f in successful if (f.get("directional_return_pct") or 0) > 0) / len(successful) * 100,
                    2,
                ) if successful else None,
            }
        versions.append(
            {
                "algorithm_version": version,
                "analysis_count": len(items),
                "rating_distribution": dict(ratings),
                "avg_final_score": _avg(final_scores),
                "avg_risk_score": _avg(risk_scores),
                "return_backtest": horizon_summary,
            }
        )

    return {
        "status": "success",
        "total_records": len(rows),
        "versions": versions,
        "records": rows[-50:],
        "note": "Phase 5 완료: 신규 분석부터 종목코드/분석시점 가격을 저장하고 5/20/60일 후속 수익률을 계산합니다. 기존 legacy 기록은 metadata pending 상태일 수 있습니다.",
    }
