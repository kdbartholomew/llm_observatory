# LLM Observatory SDK

Lightweight Python SDK for tracking LLM API usage, costs, and latency.

## Installation

```bash
pip install llm-observatory
```

## Quick Start

```python
import llm_observatory
from openai import OpenAI

# Configure once at startup
llm_observatory.configure(
    endpoint="https://your-api.vercel.app",
    api_key="your-api-key"
)

# Wrap your LLM calls with @observe
@llm_observatory.observe
def ask_gpt(prompt: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Metrics are automatically tracked!
result = ask_gpt("What is the capital of France?")
```

## Features

- **Zero-friction**: One decorator, automatic tracking
- **Non-blocking**: Async batch sending doesn't slow your app
- **Multi-provider**: Works with OpenAI, Anthropic, Google Gemini
- **Cost tracking**: Automatic cost calculation from token counts
- **Endpoint tagging**: Track costs per feature with `@observe(endpoint="feature")`

