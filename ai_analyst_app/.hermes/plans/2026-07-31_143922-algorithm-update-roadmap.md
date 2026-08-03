# AI Analyst App Algorithm Update Implementation Plan

> **For Hermes:** Use software-development-workflows to implement this plan task-by-task, verifying after each phase.

**Goal:** `ai_analyst_app`의 투자 분석 알고리즘을 단순 3-Agent 취합 구조에서 기술지표·점수화·리스크 검토 기반 구조로 단계적으로 개선한다.

**Architecture:** 기존 LangGraph 병렬 분석 구조는 유지하되, 원시 주가 데이터 위에 결정론적 기술지표 계산 계층을 추가하고, LLM 최종 판단 전에 점수화/리스크 요약을 삽입한다. UI/DB 호환성을 위해 기존 응답 필드(`company_finance`, `company_news`, `company_stock`, `final_report`)는 유지하고 신규 필드는 선택적으로 확장한다.

**Tech Stack:** FastAPI, LangGraph, LangChain, Pydantic, SQLAlchemy, Streamlit, Pandas, FinanceDataReader, Naver/FnGuide data sources.

---

## Current Context

현재 핵심 파일:

- `backend/services/graph.py`: `finance_node`, `news_node`, `stock_node`, `summarize_node` LangGraph 구성
- `backend/services/tools.py`: `get_code`, `finance_report`, `get_news`, `get_data` 도구
- `backend/core/prompts.py`: CFO/뉴스/Trader/Fund Manager 프롬프트
- `backend/core/schemas.py`: API 응답 스키마
- `backend/main.py`: 분석 API와 DB 저장
- `backend/models.py`: 분석 히스토리 테이블

현재 한계:

1. 기술적 분석 프롬프트는 MACD/RSI/볼린저밴드 등을 요구하지만 실제 코드는 원시 OHLCV 데이터만 제공한다.
2. Buy/Hold/Sell 최종 판단이 LLM 자연어에 의존하며 점수/신뢰도/리스크 제약이 약하다.
3. Risk Manager 또는 Portfolio Manager 역할이 Fund Manager 프롬프트 안에 섞여 있다.
4. 알고리즘 버전별 비교와 백테스트 기반 검증을 위한 구조가 아직 없다.

---

## Priority Roadmap

### Phase 1 — Algorithm Boundary 정리 및 문서화

**Objective:** 기존 앱 동작을 유지하면서 알고리즘 업데이트 경계를 명확히 한다.

**Files:**
- Create: `.hermes/plans/2026-07-31_143922-algorithm-update-roadmap.md`
- Later modify: `README.md` 또는 `docs/algorithm_update.md` if documentation is requested.

**Steps:**
1. 현재 LangGraph 및 도구 구조를 문서화한다.
2. 신규 알고리즘 구성요소를 `indicators`, `scoring`, `risk` 계층으로 분리한다.
3. 기존 API 응답과 프론트엔드 렌더링은 깨지지 않도록 backward-compatible 확장만 수행한다.

**Verification:**
- Plan file exists.
- `git diff -- .hermes/plans/...`로 문서 변경 확인.

---

### Phase 2 — Deterministic Technical Indicator Engine

**Objective:** Trader Agent가 LLM 추정이 아닌 코드로 계산한 기술지표를 사용하게 한다.

**Files:**
- Create: `backend/services/indicators/__init__.py`
- Create: `backend/services/indicators/technical.py`
- Modify: `backend/services/tools.py`
- Optional Test: `tests/test_technical_indicators.py`

**Implementation details:**

1. `technical.py`에 다음 함수 추가:
   - `parse_naver_chart_response(raw: str) -> pd.DataFrame`
   - `calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series`
   - `calculate_macd(close: pd.Series) -> pd.DataFrame`
   - `calculate_bollinger_bands(close: pd.Series) -> pd.DataFrame`
   - `calculate_technical_summary(df: pd.DataFrame) -> dict`

2. `tools.py`에 신규 LangChain tool 추가:
   - `technical_report(company_code: str, sdate: str, edate: str) -> str`

3. `stock_node`의 tools를 `[get_data, get_code, technical_report]`로 확장한다.

**Verification:**
- `python -m py_compile backend/services/indicators/technical.py backend/services/tools.py backend/services/graph.py`
- 샘플 Naver chart 응답으로 RSI/MACD/Bollinger 계산 테스트.

---

### Phase 3 — Structured Score and Risk Layer

**Objective:** 최종 판단 전 재무/뉴스/기술분석 결과를 기반으로 점수와 리스크 요약을 만든다.

**Files:**
- Create: `backend/services/scoring/__init__.py`
- Create: `backend/services/scoring/investment_score.py`
- Modify: `backend/services/graph.py`
- Modify: `backend/core/prompts.py`
- Modify: `backend/core/schemas.py`
- Modify: `backend/main.py`
- Modify: `backend/models.py`

**Implementation details:**

1. `investment_score.py`에 다음 함수 추가:
   - `clamp_score(value: float) -> int`
   - `score_technical_summary(summary: dict) -> dict`
   - `build_risk_summary(finance_text: str, news_text: str, technical_text: str) -> dict`
   - `derive_rating(final_score: int, risk_score: int) -> str`

2. `CompanyState` 확장:
   - `company_risk: str`
   - `investment_score: str`

3. `risk_node` 추가:
   - `company_finance`, `company_news`, `company_stock`를 받아 risk/score JSON 생성

4. Graph 연결:

```text
finance_node ┐
news_node    ├─> risk_node -> summarize_node -> END
stock_node   ┘
```

5. Fund Manager 프롬프트에 score/risk context 추가.

**Verification:**
- API 응답이 기존 필드를 계속 반환하는지 확인.
- 신규 score/risk 필드가 optional로 포함되는지 확인.
- `python -m py_compile` 통과.

---

### Phase 4 — Valuation / DCF-lite Agent

**Objective:** Buy/Hold/Sell뿐 아니라 목표주가 range와 상승여력 근거를 제공한다.

**Files:**
- Create: `backend/services/valuation.py`
- Modify: `backend/services/graph.py`
- Modify: `backend/core/prompts.py`
- Modify: frontend rendering if needed.

**Implementation details:**

1. FnGuide financial highlight에서 PER/PBR/ROE/매출/영업이익 계열 값을 추출한다.
2. 상대가치 기반 valuation score를 먼저 구현한다.
3. DCF-lite는 데이터 안정성이 확인된 후 도입한다.
4. 목표가 range와 confidence를 산출한다.

**Verification:**
- 삼성전자 등 대표 종목으로 valuation output 생성.
- 데이터 부족 시 graceful fallback.

---

### Phase 5 — Backtest and Algorithm Versioning

**Objective:** 알고리즘 업데이트 효과를 정량 검증할 수 있게 한다.

**Files:**
- Modify: `backend/models.py`
- Modify: `backend/main.py`
- Create: `backend/services/backtest.py`
- Modify: `frontend/pages/1_📊_History.py`

**Implementation details:**

1. `StockAnalysisHistory`에 optional 컬럼 추가:
   - `algorithm_version`
   - `investment_score`
   - `risk_summary`
   - `rating`
   - `confidence`
2. 과거 추천 기준 5/20/60영업일 수익률 계산 함수를 추가한다.
3. 히스토리 페이지에 알고리즘 버전별 성과 비교 UI를 추가한다.

**Verification:**
- 기존 SQLite DB migration 이슈 확인.
- 신규 DB로 테이블 생성 성공 확인.
- History API 응답 확인.

---

## Immediate Execution Choice

이번 세션에서는 리스크가 낮고 효과가 즉시 보이는 아래 작업부터 진행한다.

1. Phase 1: 이 MD 로드맵 작성 완료
2. Phase 2: 기술적 지표 엔진 구현
3. Phase 3 일부: score/risk summary를 API backward-compatible 방식으로 추가

Phase 4~5는 데이터 안정성/DB migration 범위가 커서 Phase 2~3 검증 후 진행한다.

---

## Validation Commands

```bash
cd /home/muckja999/projects/cudaboy-projects/ai_analyst_app
python -m py_compile backend/services/indicators/technical.py backend/services/tools.py backend/services/graph.py backend/core/schemas.py backend/main.py backend/models.py
python - <<'PY'
from backend.services.indicators.technical import calculate_technical_summary
import pandas as pd

df = pd.DataFrame({
    'date': pd.date_range('2026-01-01', periods=40),
    'open': range(100, 140),
    'high': range(101, 141),
    'low': range(99, 139),
    'close': range(100, 140),
    'volume': [1000 + i * 10 for i in range(40)],
})
print(calculate_technical_summary(df))
PY
```

Expected:
- py_compile exits with code 0.
- technical summary returns keys such as `rsi_14`, `macd_signal`, `technical_score`, `support_levels`, `resistance_levels`.
