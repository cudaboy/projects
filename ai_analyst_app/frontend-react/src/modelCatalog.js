export const PROVIDERS = {
  OpenAI: {
    defaultModel: 'gpt-5',
    models: ['gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4o', 'gpt-4o-mini', 'o3', 'o3-mini', 'o3-pro', 'o4-mini'],
    apiLabel: 'OpenAI API Key',
    thinking: /^(o1|o3|o4)|gpt-5/i,
  },
  Anthropic: {
    defaultModel: 'claude-sonnet-4-5',
    models: ['claude-sonnet-4-5', 'claude-opus-4-1', 'claude-opus-4-0', 'claude-sonnet-4-0', 'claude-3-7-sonnet-latest', 'claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest'],
    apiLabel: 'Anthropic API Key',
    thinking: /(claude-3-7|claude.*4|sonnet-4|opus-4)/i,
  },
  'Google Gemini': {
    defaultModel: 'gemini-2.5-pro',
    models: ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-1.5-pro', 'gemini-1.5-flash'],
    apiLabel: 'Google API Key',
    thinking: /gemini-2\.5/i,
  },
  Grok: {
    defaultModel: 'grok-4',
    models: ['grok-4', 'grok-3', 'grok-3-mini', 'grok-2-vision-latest', 'grok-2-image-latest'],
    apiLabel: 'xAI / Grok API Key',
    defaultBaseUrl: 'https://api.x.ai/v1',
    thinking: /(grok-4|grok-3-mini|reasoning)/i,
  },
  Ollama: {
    defaultModel: 'llama3.1',
    models: ['llama3.1', 'llama3.2', 'llama3.3', 'qwen3', 'qwen2.5', 'deepseek-r1', 'gpt-oss:20b', 'gpt-oss:120b', 'gemma3', 'mistral', 'mixtral', 'phi4', 'qwq'],
    apiLabel: 'Ollama API Key (보통 비움)',
    defaultBaseUrl: 'http://localhost:11434/v1',
    thinking: /(think|reason|r1|qwq|qwen3|gpt-oss)/i,
  },
  OpenRouter: {
    defaultModel: 'openai/gpt-5',
    models: ['openai/gpt-5', 'openai/gpt-4o', 'anthropic/claude-sonnet-4.5', 'anthropic/claude-opus-4.1', 'google/gemini-2.5-pro', 'x-ai/grok-4', 'deepseek/deepseek-r1', 'meta-llama/llama-3.3-70b-instruct', 'mistralai/mistral-large'],
    apiLabel: 'OpenRouter API Key',
    defaultBaseUrl: 'https://openrouter.ai/api/v1',
    thinking: /(gpt-5|o3|o4|claude.*4|grok-4|deepseek-r1|reason)/i,
  },
  DeepSeek: {
    defaultModel: 'deepseek-chat',
    models: ['deepseek-chat', 'deepseek-reasoner', 'deepseek-r1', 'deepseek-v3'],
    apiLabel: 'DeepSeek API Key',
    defaultBaseUrl: 'https://api.deepseek.com/v1',
    thinking: /(reasoner|r1|reason)/i,
  },
  Mistral: {
    defaultModel: 'mistral-large-latest',
    models: ['mistral-large-latest', 'mistral-medium-latest', 'mistral-small-latest', 'ministral-8b-latest', 'ministral-3b-latest', 'pixtral-large-latest', 'codestral-latest'],
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
