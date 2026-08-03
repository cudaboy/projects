"""Technical indicator calculation engine.

The Trader agent should receive deterministic indicators instead of asking an
LLM to infer RSI/MACD/Bollinger values from raw OHLCV text.  These helpers are
pure pandas transformations so they can be tested without network or LLM calls.
"""

from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, Dict, Iterable, List

import pandas as pd

OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def _safe_float(value: Any) -> float:
    """Convert API string/number values to float, tolerating commas/blanks."""
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def _extract_rows(raw: str) -> List[Iterable[Any]]:
    """Extract list-like rows from Naver chart API text.

    Naver mobile chart responses are commonly JSON arrays, but historical
    variants include JavaScript-like arrays.  This parser keeps the accepted
    surface narrow: a top-level list of rows or a dict containing a list under a
    common key.
    """
    if not raw or not raw.strip():
        return []

    text = raw.strip()

    # Remove possible anti-JSON prefix/suffix noise while preserving arrays.
    start_candidates = [i for i in [text.find("["), text.find("{")] if i != -1]
    if start_candidates:
        text = text[min(start_candidates) :]

    parsed: Any = None
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
            break
        except Exception:
            continue

    if parsed is None:
        # Last resort: parse bracket rows from strings like [["20240101",...],...]
        rows = re.findall(r"\[([^\[\]]+)\]", text)
        extracted: List[List[str]] = []
        for row in rows:
            parts = [p.strip().strip("'\"") for p in row.split(",")]
            if len(parts) >= 6 and re.match(r"^\d{8}$", parts[0]):
                extracted.append(parts)
        return extracted

    if isinstance(parsed, dict):
        for key in ("data", "result", "items", "chartData"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value
        return []

    if isinstance(parsed, list):
        return parsed

    return []


def parse_naver_chart_response(raw: str) -> pd.DataFrame:
    """Parse Naver domestic chart API response into a normalized OHLCV frame.

    Returns columns: date, open, high, low, close, volume.  Invalid rows are
    skipped.  The result is sorted by date ascending.
    """
    rows = _extract_rows(raw)
    normalized: List[Dict[str, Any]] = []

    for row in rows:
        if isinstance(row, dict):
            date_value = row.get("date") or row.get("localDate") or row.get("x")
            values = [
                date_value,
                row.get("openPrice") or row.get("open"),
                row.get("highPrice") or row.get("high"),
                row.get("lowPrice") or row.get("low"),
                row.get("closePrice") or row.get("close"),
                row.get("accumulatedTradingVolume") or row.get("volume"),
            ]
        else:
            values = list(row) if isinstance(row, (list, tuple)) else []

        if len(values) < 6:
            continue

        date_text = str(values[0]).strip().strip("'\"")
        if not re.match(r"^\d{8}$", date_text):
            continue

        normalized.append(
            {
                "date": pd.to_datetime(date_text, format="%Y%m%d", errors="coerce"),
                "open": _safe_float(values[1]),
                "high": _safe_float(values[2]),
                "low": _safe_float(values[3]),
                "close": _safe_float(values[4]),
                "volume": _safe_float(values[5]),
            }
        )

    df = pd.DataFrame(normalized, columns=OHLCV_COLUMNS)
    if df.empty:
        return df

    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return df


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index using Wilder-style rolling averages."""
    close = pd.to_numeric(close, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)
    return rsi.fillna(50).clip(0, 100)


def calculate_macd(close: pd.Series) -> pd.DataFrame:
    """Calculate MACD, signal line, and histogram."""
    close = pd.to_numeric(close, errors="coerce")
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return pd.DataFrame({"macd": macd, "macd_signal": signal, "macd_histogram": histogram})


def calculate_bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Calculate Bollinger moving average, upper band, and lower band."""
    close = pd.to_numeric(close, errors="coerce")
    middle = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return pd.DataFrame({"bb_middle": middle, "bb_upper": upper, "bb_lower": lower})


def _latest_number(series: pd.Series, default: float = 0.0) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return default
    return float(valid.iloc[-1])


def _round_or_none(value: float, digits: int = 2) -> float | None:
    if value is None or pd.isna(value) or math.isinf(float(value)):
        return None
    return round(float(value), digits)


def _score_from_summary(rsi: float, price_vs_ma20_pct: float, macd_hist: float, volume_z: float, volatility_20d_pct: float) -> int:
    score = 50.0

    # Trend/momentum contribution.
    score += max(min(price_vs_ma20_pct, 10), -10) * 1.2
    score += 8 if macd_hist > 0 else -8 if macd_hist < 0 else 0

    # RSI: strongest near 55-65, penalize overbought/oversold extremes.
    if 50 <= rsi <= 70:
        score += 10
    elif 40 <= rsi < 50:
        score += 2
    elif rsi > 75:
        score -= 10
    elif rsi < 30:
        score -= 8

    # Volume confirmation, capped to avoid runaway scores.
    if volume_z > 1.5:
        score += 5
    elif volume_z < -1.0:
        score -= 3

    # High volatility is a risk penalty for this conservative PB use case.
    if volatility_20d_pct > 6:
        score -= 8
    elif volatility_20d_pct > 4:
        score -= 4

    return int(max(0, min(100, round(score))))


def calculate_technical_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Return deterministic technical-analysis summary from normalized OHLCV.

    The output is intentionally JSON-serializable so it can be passed directly
    to an LLM tool response, API response, or DB text field.
    """
    required = {"date", "open", "high", "low", "close", "volume"}
    if df is None or df.empty or not required.issubset(df.columns):
        return {"status": "error", "message": "OHLCV 데이터가 비어 있거나 필수 컬럼이 없습니다."}

    data = df.copy().sort_values("date").reset_index(drop=True)
    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["close"])

    if len(data) < 20:
        return {"status": "error", "message": "기술적 지표 계산에는 최소 20개 이상의 일봉 데이터가 필요합니다."}

    close = data["close"]
    volume = data["volume"].fillna(0)

    data["ma5"] = close.rolling(5, min_periods=5).mean()
    data["ma20"] = close.rolling(20, min_periods=20).mean()
    data["ma60"] = close.rolling(60, min_periods=20).mean()
    data["rsi_14"] = calculate_rsi(close, 14)
    macd_df = calculate_macd(close)
    bb_df = calculate_bollinger_bands(close)
    data = pd.concat([data, macd_df, bb_df], axis=1)

    current_close = _latest_number(data["close"])
    previous_close = float(data["close"].iloc[-2]) if len(data) >= 2 else current_close
    ma20 = _latest_number(data["ma20"], current_close)
    ma60 = _latest_number(data["ma60"], ma20)
    rsi = _latest_number(data["rsi_14"], 50)
    macd_hist = _latest_number(data["macd_histogram"], 0)
    prev_macd_hist = float(data["macd_histogram"].dropna().iloc[-2]) if len(data["macd_histogram"].dropna()) >= 2 else 0.0

    returns = close.pct_change()
    volatility_20d_pct = float(returns.tail(20).std(ddof=0) * 100) if len(returns.dropna()) else 0.0
    volume_mean_20 = volume.rolling(20, min_periods=20).mean()
    volume_std_20 = volume.rolling(20, min_periods=20).std(ddof=0).replace(0, pd.NA)
    volume_z = ((volume - volume_mean_20) / volume_std_20).fillna(0)
    latest_volume_z = _latest_number(volume_z, 0)

    recent_20 = data.tail(20)
    support_levels = sorted(recent_20["low"].dropna().nsmallest(3).unique().tolist())[:3]
    resistance_levels = sorted(recent_20["high"].dropna().nlargest(3).unique().tolist(), reverse=True)[:3]

    price_vs_ma20_pct = ((current_close - ma20) / ma20 * 100) if ma20 else 0.0
    price_change_1d_pct = ((current_close - previous_close) / previous_close * 100) if previous_close else 0.0

    if current_close > ma20 > ma60:
        trend = "uptrend"
    elif current_close < ma20 < ma60:
        trend = "downtrend"
    else:
        trend = "sideways"

    if prev_macd_hist <= 0 < macd_hist:
        macd_signal = "bullish_cross"
    elif prev_macd_hist >= 0 > macd_hist:
        macd_signal = "bearish_cross"
    elif macd_hist > 0:
        macd_signal = "bullish"
    elif macd_hist < 0:
        macd_signal = "bearish"
    else:
        macd_signal = "neutral"

    technical_score = _score_from_summary(rsi, price_vs_ma20_pct, macd_hist, latest_volume_z, volatility_20d_pct)

    return {
        "status": "success",
        "data_points": int(len(data)),
        "start_date": data["date"].iloc[0].strftime("%Y-%m-%d"),
        "end_date": data["date"].iloc[-1].strftime("%Y-%m-%d"),
        "current_close": _round_or_none(current_close, 2),
        "price_change_1d_pct": _round_or_none(price_change_1d_pct, 2),
        "trend": trend,
        "ma5": _round_or_none(_latest_number(data["ma5"], current_close), 2),
        "ma20": _round_or_none(ma20, 2),
        "ma60": _round_or_none(ma60, 2),
        "price_vs_ma20_pct": _round_or_none(price_vs_ma20_pct, 2),
        "rsi_14": _round_or_none(rsi, 2),
        "macd": _round_or_none(_latest_number(data["macd"], 0), 4),
        "macd_signal_value": _round_or_none(_latest_number(data["macd_signal"], 0), 4),
        "macd_histogram": _round_or_none(macd_hist, 4),
        "macd_signal": macd_signal,
        "bb_upper": _round_or_none(_latest_number(data["bb_upper"], current_close), 2),
        "bb_middle": _round_or_none(_latest_number(data["bb_middle"], current_close), 2),
        "bb_lower": _round_or_none(_latest_number(data["bb_lower"], current_close), 2),
        "volatility_20d_pct": _round_or_none(volatility_20d_pct, 2),
        "volume_zscore_20d": _round_or_none(latest_volume_z, 2),
        "volume_spike": bool(latest_volume_z > 1.5),
        "support_levels": [_round_or_none(v, 2) for v in support_levels],
        "resistance_levels": [_round_or_none(v, 2) for v in resistance_levels],
        "technical_score": technical_score,
    }
