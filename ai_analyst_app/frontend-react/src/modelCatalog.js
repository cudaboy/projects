// Model preset catalog for the app sidebar.
// Updated for the 2026-08 provider landscape. Users can still type any model id manually.
export const PROVIDERS = {
  OpenAI: {
    defaultModel: 'gpt-5.6',
    models: [
      'gpt-5.6', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-5.6-luna',
      'gpt-5.4', 'gpt-5', 'gpt-5-mini', 'gpt-5-nano',
      'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4o', 'gpt-4o-mini',
      'o4-mini', 'o3-pro', 'o3', 'o3-mini',
    ],
    apiLabel: 'OpenAI API Key',
    thinking: /^(o1|o3|o4)|gpt-5/i,
  },
  Anthropic: {
    defaultModel: 'claude-opus-5',
    models: [
      'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5', 'claude-mythos-5', 'claude-mythos-preview',
      'claude-opus-4-8', 'claude-opus-4-7', 'claude-opus-4-6', 'claude-sonnet-4-6',
      'claude-opus-4-5', 'claude-sonnet-4-5', 'claude-haiku-4-5',
      'claude-opus-4-1', 'claude-opus-4-0', 'claude-sonnet-4-0',
      'claude-3-7-sonnet-latest', 'claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest',
    ],
    apiLabel: 'Anthropic API Key',
    thinking: /(claude-3-7|claude.*4|claude.*5|sonnet-4|opus-4|sonnet-5|opus-5|mythos|fable)/i,
  },
  'Google Gemini': {
    defaultModel: 'gemini-3.6-flash',
    models: [
      'gemini-3.6-flash', 'gemini-3.5-flash', 'gemini-3.5-flash-lite',
      'gemini-3.1-pro-preview', 'gemini-3.1-flash-lite', 'gemini-3.1-flash-live-preview',
      'gemini-3-pro-preview', 'gemini-3-flash-preview',
      'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite',
      'gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-1.5-pro', 'gemini-1.5-flash',
    ],
    apiLabel: 'Google API Key',
    thinking: /gemini-(2\.5|3)/i,
  },
  Grok: {
    defaultModel: 'grok-4.5',
    models: [
      'grok-4.5', 'grok-4.5-latest',
      'grok-4.3', 'grok-4.3-latest', 'grok-latest',
      'grok-4.20', 'grok-4.20-reasoning', 'grok-4.20-reasoning-latest',
      'grok-4.20-non-reasoning', 'grok-4.20-non-reasoning-latest',
      'grok-4.20-multi-agent', 'grok-4.20-multi-agent-latest',
      'grok-4', 'grok-3', 'grok-3-mini', 'grok-code-fast-1', 'grok-code-fast',
    ],
    apiLabel: 'xAI / Grok API Key',
    defaultBaseUrl: 'https://api.x.ai/v1',
    thinking: /(grok-4\.5|grok-4\.3|grok-4\.20|grok-4|grok-3-mini|reasoning|multi-agent)/i,
  },
  Ollama: {
    defaultModel: 'llama3.3',
    models: [
      'llama3.3', 'llama3.2', 'llama3.1',
      'qwen3', 'qwen2.5', 'deepseek-r1', 'gpt-oss:120b', 'gpt-oss:20b',
      'gemma3', 'mistral', 'mixtral', 'phi4', 'qwq', 'codellama', 'nomic-embed-text',
    ],
    apiLabel: 'Ollama API Key (보통 비움)',
    defaultBaseUrl: 'http://localhost:11434/v1',
    thinking: /(think|reason|r1|qwq|qwen3|gpt-oss)/i,
  },
  OpenRouter: {
    defaultModel: 'openai/gpt-5.6',
    models: [
      'openai/gpt-5.6', 'openai/gpt-5.4', 'openai/gpt-5', 'openai/o4-mini', 'openai/o3-pro',
      'anthropic/claude-opus-5', 'anthropic/claude-sonnet-5', 'anthropic/claude-sonnet-4.5', 'anthropic/claude-opus-4.1',
      'google/gemini-3.6-flash', 'google/gemini-3-pro-preview', 'google/gemini-2.5-pro',
      'x-ai/grok-4.5', 'x-ai/grok-4.20', 'x-ai/grok-4',
      'deepseek/deepseek-r1', 'deepseek/deepseek-chat',
      'meta-llama/llama-3.3-70b-instruct', 'mistralai/mistral-large', 'cohere/command-a-03-2025',
    ],
    apiLabel: 'OpenRouter API Key',
    defaultBaseUrl: 'https://openrouter.ai/api/v1',
    thinking: /(gpt-5|o3|o4|claude.*4|claude.*5|grok-4|deepseek-r1|reason)/i,
  },
  DeepSeek: {
    defaultModel: 'deepseek-reasoner',
    models: ['deepseek-reasoner', 'deepseek-r1', 'deepseek-chat', 'deepseek-v3'],
    apiLabel: 'DeepSeek API Key',
    defaultBaseUrl: 'https://api.deepseek.com/v1',
    thinking: /(reasoner|r1|reason)/i,
  },
  Mistral: {
    defaultModel: 'mistral-large-latest',
    models: ['mistral-large-latest', 'mistral-medium-latest', 'mistral-small-latest', 'magistral-medium-latest', 'magistral-small-latest', 'ministral-8b-latest', 'ministral-3b-latest', 'pixtral-large-latest', 'codestral-latest'],
    apiLabel: 'Mistral API Key',
    defaultBaseUrl: 'https://api.mistral.ai/v1',
    thinking: /(magistral|reason)/i,
  },
  Cohere: {
    defaultModel: 'command-a-03-2025',
    models: ['command-a-03-2025', 'command-r-plus', 'command-r', 'command-r7b-12-2024'],
    apiLabel: 'Cohere API Key',
    defaultBaseUrl: 'https://api.cohere.com/compatibility/v1',
    thinking: /reason/i,
  },
};

export const BILLING_MODES = {
  token_metered: '토큰 종량제',
  quota_metered: '정량제 / 크레딧 차감',
  flat_rate: '월정액 / 구독형',
  local_free: '로컬 / 별도 과금 없음',
};

export function supportsThinking(provider, modelName) {
  const spec = PROVIDERS[provider];
  return Boolean(spec?.thinking?.test(modelName || ''));
}
