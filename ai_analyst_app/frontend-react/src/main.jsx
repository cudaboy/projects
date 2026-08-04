import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { marked } from 'marked';
import { Activity, BarChart3, Brain, CheckCircle2, Database, LineChart, Loader2, ShieldAlert, TrendingUp } from 'lucide-react';
import { BILLING_MODES, PROVIDERS, supportsThinking } from './modelCatalog';
import './styles.css';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api/v1';

function safeJson(text) {
  if (!text) return null;
  try { return JSON.parse(text); } catch { return null; }
}

function markdown(text) {
  return { __html: marked.parse(text || '내용 없음') };
}

function scoreTone(value) {
  if (value == null) return 'neutral';
  if (value >= 70) return 'good';
  if (value <= 40) return 'bad';
  return 'neutral';
}

function MetricCard({ label, value, caption, icon: Icon, tone = 'neutral' }) {
  return <div className={`metric-card ${tone}`}>
    <div className="metric-icon">{Icon ? <Icon size={18}/> : null}</div>
    <div>
      <p>{label}</p>
      <strong>{value ?? '-'}</strong>
      {caption ? <span>{caption}</span> : null}
    </div>
  </div>;
}

function SettingsPanel({ settings, setSettings }) {
  const update = (key, value) => setSettings(prev => ({ ...prev, [key]: value }));
  const providerSpec = PROVIDERS[settings.provider] || PROVIDERS.OpenAI;
  const thinkingAvailable = supportsThinking(settings.provider, settings.model_name);
  const changeProvider = (provider) => {
    const spec = PROVIDERS[provider] || PROVIDERS.OpenAI;
    setSettings(prev => ({
      ...prev,
      provider,
      model_name: spec.defaultModel,
      base_url: spec.defaultBaseUrl || '',
      enable_thinking: supportsThinking(provider, spec.defaultModel) ? prev.enable_thinking : false,
    }));
  };
  const changeModel = (modelName) => {
    setSettings(prev => ({
      ...prev,
      model_name: modelName,
      enable_thinking: supportsThinking(prev.provider, modelName) ? prev.enable_thinking : false,
    }));
  };

  return <aside className="settings-card">
    <div className="eyebrow">Agent Runtime</div>
    <h2>분석 설정</h2>
    <label>Model Provider
      <select value={settings.provider} onChange={e => changeProvider(e.target.value)}>
        {Object.keys(PROVIDERS).map(provider => <option key={provider}>{provider}</option>)}
      </select>
    </label>
    <label>Model
      <input list="model-presets" value={settings.model_name} onChange={e => changeModel(e.target.value)} placeholder={providerSpec.defaultModel} />
      <datalist id="model-presets">{providerSpec.models.map(model => <option key={model} value={model}/>)}</datalist>
    </label>
    <label>Temperature
      <input type="number" min="0" max="1" step="0.1" value={settings.temperature} onChange={e => update('temperature', Number(e.target.value))} />
    </label>
    <label>{providerSpec.apiLabel}
      <input type="password" value={settings.custom_api_key} onChange={e => update('custom_api_key', e.target.value)} placeholder="서버 .env 사용 시 비워두기" />
    </label>
    {providerSpec.defaultBaseUrl ? <label>Base URL
      <input value={settings.base_url} onChange={e => update('base_url', e.target.value)} placeholder={providerSpec.defaultBaseUrl || 'https://.../v1'} />
    </label> : null}
    <div className="billing-box">
      <p className="eyebrow">Billing</p>
      <label>요금제 방식
        <select value={settings.billing_mode} onChange={e => update('billing_mode', e.target.value)}>
          {Object.entries(BILLING_MODES).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select>
      </label>
      {settings.billing_mode !== 'token_metered' ? <div className="billing-grid">
        <label>월 예산/정액
          <input type="number" min="0" value={settings.monthly_budget} onChange={e => update('monthly_budget', e.target.value)} placeholder="예: 20" />
        </label>
        <label>월 정량 한도
          <input type="number" min="0" value={settings.monthly_quota} onChange={e => update('monthly_quota', e.target.value)} placeholder="예: 1000" />
        </label>
        <label>단위
          <input value={settings.quota_unit} onChange={e => update('quota_unit', e.target.value)} placeholder="requests / credits" />
        </label>
      </div> : null}
    </div>
    <label className={`checkbox-row ${thinkingAvailable ? '' : 'disabled'}`} title={thinkingAvailable ? '선택한 모델에서 thinking/reasoning 옵션을 전달합니다.' : '선택한 provider/model은 명시적 thinking 옵션 대상이 아닙니다.'}>
      <input type="checkbox" disabled={!thinkingAvailable} checked={Boolean(settings.enable_thinking && thinkingAvailable)} onChange={e => update('enable_thinking', e.target.checked)} />
      Thinking / Reasoning 사용
    </label>
    {thinkingAvailable ? <label>Reasoning Effort
      <select value={settings.reasoning_effort} onChange={e => update('reasoning_effort', e.target.value)} disabled={!settings.enable_thinking}>
        <option value="low">low</option>
        <option value="medium">medium</option>
        <option value="high">high</option>
      </select>
    </label> : null}
    <label>Naver Client ID
      <input value={settings.naver_client_id} onChange={e => update('naver_client_id', e.target.value)} placeholder="뉴스 품질 향상용" />
    </label>
    <label>Naver Client Secret
      <input type="password" value={settings.naver_client_secret} onChange={e => update('naver_client_secret', e.target.value)} placeholder="선택 입력" />
    </label>
    <label className="checkbox-row">
      <input type="checkbox" checked={settings.use_langsmith} onChange={e => update('use_langsmith', e.target.checked)} />
      LangSmith tracing
    </label>
    <label>LangSmith API Key
      <input type="password" value={settings.langsmith_api_key} onChange={e => update('langsmith_api_key', e.target.value)} placeholder="선택 입력" />
    </label>
  </aside>;
}

function ResultPanel({ result }) {
  const [active, setActive] = useState(0);
  if (!result) return <div className="empty-state">
    <div className="empty-orb"><Brain size={42}/></div>
    <h2>분석할 종목을 입력하세요</h2>
    <p>재무·뉴스·기술지표·밸류에이션·리스크 점수를 통합해 PB 스타일 리포트를 생성합니다.</p>
  </div>;

  if (result.status === 'error') return <div className="error-card"><ShieldAlert/><h3>분석 실패</h3><p>{result.error_message}</p></div>;

  const data = result.data || {};
  const risk = safeJson(data.investment_score);
  const valuation = safeJson(data.company_valuation) || risk?.valuation;
  const tabs = [
    ['최종 리포트', data.final_report],
    ['재무 CFO', data.company_finance],
    ['뉴스 Analyst', data.company_news],
    ['기술 Trader', data.company_stock],
    ['Valuation', data.company_valuation],
    ['Risk JSON', data.company_risk],
  ];

  return <section className="result-panel">
    <div className="result-header">
      <div><span className="success-chip"><CheckCircle2 size={15}/> 분석 완료</span><h2>{result.company_name} 종합 리포트</h2></div>
      <div className="rating-pill">{risk?.rating_hint || 'Report'}</div>
    </div>

    <div className="metric-grid">
      <MetricCard label="Final Score" value={risk?.final_score} caption={risk?.algorithm_version} icon={Activity} tone={scoreTone(risk?.final_score)} />
      <MetricCard label="Risk Score" value={risk?.risk_score} caption="낮을수록 안정적" icon={ShieldAlert} tone={risk?.risk_score >= 70 ? 'bad' : 'neutral'} />
      <MetricCard label="Technical" value={risk?.technical?.technical_score} caption={risk?.technical?.technical_signal} icon={LineChart} tone={scoreTone(risk?.technical?.technical_score)} />
      <MetricCard label="Target Mid" value={valuation?.target_price_mid ? `${valuation.target_price_mid.toLocaleString()}원` : '-'} caption={valuation?.upside_pct_mid != null ? `Upside ${valuation.upside_pct_mid}%` : valuation?.confidence} icon={TrendingUp} tone={valuation?.upside_pct_mid > 0 ? 'good' : 'neutral'} />
    </div>

    {risk?.risk_flags?.length ? <div className="risk-strip">{risk.risk_flags.map((x, i) => <span key={i}>{x}</span>)}</div> : null}

    <div className="tabs">{tabs.map(([name], idx) => <button className={active === idx ? 'active' : ''} onClick={() => setActive(idx)} key={name}>{name}</button>)}</div>
    <article className="markdown-card" dangerouslySetInnerHTML={markdown(tabs[active][1])} />
  </section>;
}

function HistoryPanel() {
  const [history, setHistory] = useState([]);
  const [performance, setPerformance] = useState(null);
  useEffect(() => {
    fetch(`${API_BASE}/history`).then(r => r.json()).then(j => setHistory(j.data || [])).catch(() => setHistory([]));
    fetch(`${API_BASE}/performance`).then(r => r.json()).then(setPerformance).catch(() => setPerformance(null));
  }, []);
  const latestVersion = performance?.versions?.[performance.versions.length - 1];
  const backtestRows = latestVersion ? Object.entries(latestVersion.return_backtest || {}) : [];
  return <section className="history-section">
    <div className="section-heading"><BarChart3/><div><p className="eyebrow">History & Versioning</p><h2>알고리즘 성과 추적</h2></div></div>
    <div className="history-grid">
      <MetricCard label="분석 건수" value={history.length} icon={Database}/>
      <MetricCard label="버전 수" value={performance?.versions?.length || 0} icon={Activity}/>
      <MetricCard label="최근 종목" value={history[history.length - 1]?.company_name || '-'} icon={TrendingUp}/>
      <MetricCard label="Phase 5" value="5/20/60D" caption="후속 수익률 추적" icon={LineChart}/>
    </div>
    {latestVersion ? <div className="backtest-card">
      <div><p className="eyebrow">Return Backtest</p><h3>{latestVersion.algorithm_version}</h3></div>
      <div className="backtest-grid">{backtestRows.map(([horizon, item]) => <div key={horizon} className="backtest-item"><span>{horizon}</span><strong>{item.avg_directional_return_pct ?? '-'}%</strong><small>Win {item.win_rate_pct ?? '-'}% · N {item.completed_count}</small></div>)}</div>
    </div> : null}
    <div className="table-card">
      <table><thead><tr><th>시간</th><th>종목</th><th>코드</th><th>기준가</th><th>Rating</th><th>Version</th></tr></thead><tbody>
        {history.slice(-8).reverse().map(row => <tr key={row.id}><td>{row.created_at?.slice(0, 16)}</td><td>{row.company_name}</td><td>{row.stock_code || '-'}</td><td>{row.analysis_price ? Number(row.analysis_price).toLocaleString() : '-'}</td><td><span className="mini-chip">{row.rating || '-'}</span></td><td>{row.algorithm_version || 'legacy'}</td></tr>)}
      </tbody></table>
    </div>
  </section>;
}

function App() {
  const [company, setCompany] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [settings, setSettings] = useState({ provider: 'OpenAI', model_name: 'gpt-5.6', temperature: 0.2, enable_thinking: false, reasoning_effort: 'medium', base_url: '', billing_mode: 'token_metered', monthly_budget: '', monthly_quota: '', quota_unit: 'requests', custom_api_key: '', openai_api_key: '', use_langsmith: false, langsmith_api_key: '', naver_client_id: '', naver_client_secret: '' });

  const canSubmit = useMemo(() => company.trim() && !loading, [company, loading]);
  const analyze = async (e) => {
    e.preventDefault();
    if (!canSubmit) return;
    setLoading(true); setResult(null);
    try {
      const res = await fetch(`${API_BASE}/analyze`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ company_name: company.trim(), model_settings: settings }) });
      setResult(await res.json());
    } catch (err) { setResult({ status: 'error', error_message: String(err) }); }
    finally { setLoading(false); }
  };

  return <main>
    <section className="hero">
      <nav><div className="brand-mark">AI</div><span>AI Analyst PB</span><a href="#history">History</a></nav>
      <div className="hero-grid">
        <div className="hero-copy"><p className="eyebrow">Toss-like Blue Fintech AI Report</p><h1>주식 분석을 PB 리포트처럼, 더 정량적으로.</h1><p>LangGraph 멀티 에이전트가 재무·뉴스·기술지표·DCF-lite·리스크 점수를 결합해 투자 판단 보조 리포트를 생성합니다.</p></div>
        <form className="search-card" onSubmit={analyze}>
          <label>분석 종목명</label>
          <div className="search-row"><input value={company} onChange={e => setCompany(e.target.value)} placeholder="예: 삼성전자, 현대차, SK하이닉스"/><button disabled={!canSubmit}>{loading ? <Loader2 className="spin"/> : '심층 분석'}</button></div>
          <div className="pipeline"><span>CFO</span><span>News</span><span>Trader</span><span>Valuation</span><span>Risk</span></div>
        </form>
      </div>
    </section>
    <section className="workspace"><SettingsPanel settings={settings} setSettings={setSettings}/><ResultPanel result={result}/></section>
    <div id="history"><HistoryPanel/></div>
  </main>;
}

createRoot(document.getElementById('root')).render(<App />);
