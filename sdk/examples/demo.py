"""
Demo showing how to use LLM Observatory.

Run this after setting up the backend:
    pip install openai
    export OPENAI_API_KEY="your-key"
    python demo.py
"""

import os
from openai import OpenAI

# Import and configure the observatory
import llm_observatory

llm_observatory.configure(
    endpoint="http://localhost:8000",  # Your backend URL
    api_key=os.getenv("OBSERVATORY_API_KEY", ""),  # Your API key
)


# Wrap your LLM calls with @observe
@llm_observatory.observe
def ask_gpt(prompt: str) -> str:
    """Simple GPT call with automatic tracking."""
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=512,
    )
    return response.choices[0].message.content


# Use endpoint tagging for per-feature tracking
@llm_observatory.observe(endpoint="summarization")
def summarize(text: str) -> str:
    """Summarize text - tracked under 'summarization' endpoint."""
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-5-mini",
        max_completion_tokens=512,
        messages=[
            {"role": "system", "content": "Summarize the following text concisely."},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Make some API calls - metrics are automatically tracked
    print("Making tracked API calls...\n")
    
    result = ask_gpt("What is the capital of France?")
    print(f"GPT response: {result}\n")
    
    summary = summarize(
        "The quick brown fox jumps over the lazy dog. "
        "This sentence contains every letter of the alphabet. "
        "It has been used for typing practice since the late 1800s."
    )
    print(f"Summary: {summary}\n")
    
    print("✓ Metrics sent to Observatory!")
    print("  Check your dashboard at http://localhost:3000")

