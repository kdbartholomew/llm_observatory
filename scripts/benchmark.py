#!/usr/bin/env python3
"""
LLM Benchmark Script

Runs the same tasks across multiple models/providers to generate
real comparison data for the Observatory dashboard.

Usage:
    # Set API keys
    export OPENAI_API_KEY="sk-..."
    export ANTHROPIC_API_KEY="sk-ant-..."
    export GOOGLE_API_KEY="..."
    
    # Run benchmark
    python benchmark.py

Cost estimate: ~$2-5 for a full run across all models
"""

import os
import sys
import time
from pathlib import Path

# Add SDK to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))

import llm_observatory
from code_validator import check_code_accuracy

# --- Configuration ---

# Configure Observatory (update endpoint if deployed)
llm_observatory.configure(
    endpoint=os.getenv("OBSERVATORY_URL", "http://localhost:8000"),
    api_key=os.getenv("OBSERVATORY_API_KEY", "demo-api-key-12345"),
    project="benchmark",  # Project name for tracking
    batch_size=5,
    flush_interval=2.0,
)

# Models to benchmark (updated for 2026 availability)
# Add/remove models as needed for your comparison
MODELS = {
    "openai": [
        "gpt-5-mini",           # Fast, cost-efficient
        "gpt-5",                # Current flagship
        "gpt-5.2-2025-12-11",
        "gpt-4.1-2025-04-14",
    ],
    "anthropic": [
        "claude-haiku-4-5-20251001",   # Latest high-speed variant  
        "claude-sonnet-4-5-20250929",  # Recommended high-intelligence model
    ],
    "google": [
        "gemini-2.5-flash-lite",   # Low latency, high volume
        "gemini-2.5-flash",        # Balanced speed and capability
        "gemini-2.5-pro",          # Advanced reasoning (min thinking_budget: 128)
    ],
}


# Number of times to run each task per model (for latency variance)
RUNS_PER_TASK = 2  # Reduced to minimize rate limiting

# Provider-specific delays (seconds) to avoid rate limits
PROVIDER_DELAYS = {
    "openai": 0.5,
    "anthropic": 0.5,
    "google": 0.5,  # Stable models have much higher RPM limits than preview
}


# --- Task Definitions with Expected Answers ---

TASKS = {
    "short_qa": {
        "description": "Quick factual questions",
        "system": "Answer concisely in 1-2 sentences.",
        "prompts": [
            ("What is the capital of Japan?", ["tokyo"]),
            ("Who painted the Mona Lisa?", ["leonardo", "da vinci", "davinci"]),
            ("What year did World War II end?", ["1945"]),
            ("What is the chemical symbol for gold?", ["au"]),
            ("Who wrote '1984'?", ["george orwell", "orwell"]),
        ],
    },
    "reasoning": {
        "description": "Logic and math problems",
        "system": "Solve the problem step by step, then give the final answer.",
        "prompts": [
            ("If a train travels 120 miles in 2 hours, what is its average speed in mph?", ["60"]),
            ("A store sells apples for $0.50 each. If you have $5, how many apples can you buy and how much change will you have?", ["10", "no change", "$0", "0 change"]),
            ("If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?", ["no", "cannot", "can't", "not necessarily"]),
            ("What comes next in the sequence: 2, 6, 12, 20, 30, ?", ["42"]),
            ("A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?", ["0.05", "5 cents", "$0.05", "five cents"]),
        ],
    },
    "summarization": {
        "description": "Condense text into key points",
        "system": "Summarize the following text in 2-3 bullet points.",
        "prompts": [
            ("""The Internet of Things (IoT) refers to the network of physical devices, vehicles, 
            appliances, and other objects embedded with sensors, software, and connectivity that 
            enables them to collect and exchange data. This technology has transformed industries 
            from healthcare to manufacturing, enabling real-time monitoring, predictive maintenance, 
            and automated responses. However, IoT also raises significant security and privacy 
            concerns, as each connected device represents a potential entry point for cyber attacks.""",
            ["iot", "device", "sensor", "security", "privacy"]),  # Key concepts that should appear
            
            ("""Machine learning is a subset of artificial intelligence that enables systems to learn 
            and improve from experience without being explicitly programmed. It focuses on developing 
            algorithms that can access data and use it to learn for themselves. The process begins 
            with observations or data, such as examples, direct experience, or instruction, to look 
            for patterns in data and make better decisions in the future based on the examples provided.""",
            ["machine learning", "ai", "artificial intelligence", "algorithm", "pattern", "data"]),
            
            ("""Climate change refers to long-term shifts in global temperatures and weather patterns. 
            While natural factors have caused climate changes in the past, human activities have been 
            the main driver since the 1800s, primarily due to burning fossil fuels like coal, oil, 
            and gas. This releases greenhouse gases that trap heat in the atmosphere, leading to 
            rising temperatures, melting ice caps, and more extreme weather events worldwide.""",
            ["climate", "temperature", "fossil fuel", "greenhouse", "human"]),
        ],
    },
    "code_generation": {
        "description": "Generate simple code snippets",
        "system": "Write clean, working code. Include brief comments. Return only the code, no explanations.",
        "prompts": [
            ("Write a Python function that checks if a string is a palindrome.", 
             ["Write a Python function that checks if a string is a palindrome."]),
            ("Write a JavaScript function that removes duplicates from an array.", 
             ["Write a JavaScript function that removes duplicates from an array."]),
            ("Write a Python function that finds the nth Fibonacci number using recursion with memoization.", 
             ["Write a Python function that finds the nth Fibonacci number using recursion with memoization."]),
            ("Write a SQL query to find the top 5 customers by total order value from tables 'customers' and 'orders'.", 
             ["Write a SQL query to find the top 5 customers by total order value from tables 'customers' and 'orders'."]),
        ],
    },
}


def check_accuracy(response_text: str, expected_keywords: list[str], task_type: str) -> bool:
    """
    Check if response contains expected answer/keywords.
    
    For short_qa/reasoning: at least one keyword must match
    For summarization: multiple keywords should appear
    For code_generation: execute and test the code
    """
    if not response_text:
        return False
    
    # Code generation uses executable validation
    if task_type == "code_generation":
        # expected_keywords[0] contains the original prompt for routing
        prompt = expected_keywords[0] if expected_keywords else ""
        return check_code_accuracy(response_text, prompt)
    
    response_lower = response_text.lower()
    
    if task_type in ["short_qa", "reasoning"]:
        # For factual/reasoning, any matching keyword counts as correct
        return any(kw.lower() in response_lower for kw in expected_keywords)
    else:
        # For summarization, require at least 2 key concepts
        matches = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
        return matches >= 2


def extract_response_text(result, provider: str) -> str:
    """
    Extract text response, ensuring we skip internal thinking parts for Google 2026.
    
    When thinking is enabled, Gemini responses have multiple parts:
    - Part 0: Often contains the thought (reasoning)
    - Part 1: Contains the actual text (answer)
    
    The 2026 SDK's .text attribute automatically filters to exclude thoughts.
    """
    try:
        if provider == "openai":
            return result.choices[0].message.content or ""
        elif provider == "anthropic":
            return result.content[0].text if result.content else ""
        elif provider == "google":
            # 1. Best practice: Use the top-level .text attribute 
            # In 2026 SDK, this is filtered to exclude thoughts
            if hasattr(result, 'text') and result.text:
                return result.text.strip()
            
            # 2. Fallback: Manually filter parts to exclude thoughts
            if hasattr(result, 'candidates'):
                content = result.candidates[0].content
                # Join all parts that are NOT 'thought' parts
                text_parts = []
                for part in content.parts:
                    # Skip parts that have thought attribute set to True
                    if not (hasattr(part, 'thought') and part.thought):
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                return "".join(text_parts).strip()
            return ""
    except (AttributeError, IndexError, ValueError):
        return ""
    return ""


# --- Provider Clients ---

def get_openai_client():
    """Initialize OpenAI client."""
    try:
        from openai import OpenAI
        return OpenAI()
    except ImportError:
        print("  ⚠ OpenAI not installed: pip install openai")
        return None
    except Exception as e:
        print(f"  ⚠ OpenAI init failed: {e}")
        return None


def get_anthropic_client():
    """Initialize Anthropic client."""
    try:
        from anthropic import Anthropic
        return Anthropic()
    except ImportError:
        print("  ⚠ Anthropic not installed: pip install anthropic")
        return None
    except Exception as e:
        print(f"  ⚠ Anthropic init failed: {e}")
        return None


def get_google_client():
    """Initialize Google Gemini client (2026 google-genai package)."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("  ⚠ GOOGLE_API_KEY not set")
        return None
    
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        return client
    except ImportError:
        print("  ⚠ Google GenAI not installed: pip install google-genai")
        return None
    except Exception as e:
        print(f"  ⚠ Google init failed: {e}")
        return None


# --- Gemini Response Wrappers (2026 google-genai package) ---

class GeminiResponseWrapper:
    """
    Wrapper to normalize Gemini response for metric extraction (2026 google-genai package).
    
    response.text contains only the final answer (thinking is excluded).
    Usage metadata includes separate counts for answer tokens and thinking tokens.
    """
    def __init__(self, response, model: str):
        self.model = model
        self.usage = GeminiUsage(response)
        # response.text automatically excludes thinking content, contains only the answer
        try:
            self.text = response.text or ""
        except (AttributeError, ValueError):
            try:
                # Fallback to candidates structure if text attribute unavailable
                self.text = response.candidates[0].content.parts[0].text
            except (AttributeError, IndexError):
                self.text = ""


class GeminiUsage:
    """
    Extract usage from google-genai response (2026 API).
    
    Handles thinking tokens separately:
    - prompt_token_count: Input tokens
    - candidates_token_count: Answer/output tokens (what user sees)
    - thoughts_token_count: Internal reasoning tokens (for thinking models)
    
    Total completion tokens = candidates + thoughts (for cost calculation)
    """
    def __init__(self, response):
        # Use getattr with default 0 to prevent NoneType errors
        m = getattr(response, 'usage_metadata', None)
        
        self.prompt_tokens = getattr(m, 'prompt_token_count', 0) or 0
        
        # Answer tokens (the actual response content)
        ans_tokens = getattr(m, 'candidates_token_count', 0) or 0
        
        # Thinking tokens (internal reasoning, only for thinking-enabled models)
        tht_tokens = getattr(m, 'thoughts_token_count', 0) or 0
        
        # Total completion tokens = answer + thinking (for cost calculation)
        self.completion_tokens = ans_tokens + tht_tokens


# --- Provider-specific call functions (decorated at module level for proper observability) ---

@llm_observatory.observe
def call_openai_api(client, model: str, system: str, prompt: str):
    """Make OpenAI API call (2026 API uses max_completion_tokens)."""
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=500,
    )


@llm_observatory.observe
def call_anthropic_api(client, model: str, system: str, prompt: str):
    """Make Anthropic API call."""
    return client.messages.create(
        model=model,
        max_tokens=500,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )


@llm_observatory.observe  
def call_google_api(client, model: str, system: str, prompt: str):
    """
    Make Google Gemini API call (2026 google-genai package).
    
    Thinking configuration (for this benchmark):
    - gemini-2.5-flash-lite: thinking disabled (thinking_budget=0)
    - gemini-2.5-flash: thinking disabled for apples-to-apples comparison
    - gemini-2.5-pro: thinking enabled (minimum thinking_budget=128, cannot be disabled)
    
    Note: response.text automatically excludes thinking content.
    """
    from google.genai import types
    
    # Configure thinking based on model
    thinking_config = None
    if "flash-lite" in model:
        # Disable thinking for flash-lite (low latency)
        thinking_config = types.ThinkingConfig(thinking_budget=0)
    elif "flash" in model:
        # Disable thinking for regular flash (makes benchmark apples-to-apples with flash-lite)
        # This ensures answers are in parts[0] instead of parts[1] when thinking is enabled
        thinking_config = types.ThinkingConfig(thinking_budget=0)
    elif "pro" in model:
        # Pro models require minimum thinking_budget of 128
        thinking_config = types.ThinkingConfig(thinking_budget=128)
    
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=500,
    )
    
    if thinking_config:
        config.thinking_config = thinking_config
    
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    
    return GeminiResponseWrapper(response, model)


def make_call(provider: str, client, model: str, task_name: str, system: str, prompt: str):
    """Route to correct provider API call."""
    if provider == "openai":
        return call_openai_api(client, model, system, prompt)
    
    elif provider == "anthropic":
        return call_anthropic_api(client, model, system, prompt)
    
    elif provider == "google":
        return call_google_api(client, model, system, prompt)
    
    raise ValueError(f"Unknown provider: {provider}")


# --- Main Benchmark ---

def run_benchmark():
    """Run the full benchmark suite."""
    print("=" * 60)
    print("🔭 LLM Observatory Benchmark")
    print("=" * 60)
    
    # Initialize clients
    print("\n📡 Initializing API clients...")
    clients = {
        "openai": get_openai_client(),
        "anthropic": get_anthropic_client(),
        "google": get_google_client(),
    }
    
    # Filter to available clients
    available = {k: v for k, v in clients.items() if v is not None}
    if not available:
        print("\n❌ No API clients available. Set API keys and install packages.")
        return
    
    print(f"\n✓ Available providers: {', '.join(available.keys())}")
    
    # Count total calls
    total_calls = 0
    for provider in available:
        for model in MODELS.get(provider, []):
            for task_name, task in TASKS.items():
                total_calls += len(task["prompts"]) * RUNS_PER_TASK
    
    print(f"📊 Planning {total_calls} API calls across {len(available)} providers")
    print("\n" + "-" * 60)
    
    call_count = 0
    errors = 0
    error_details = []  # Track unique errors
    
    # Track accuracy per model
    accuracy_tracker: dict[str, dict[str, list[bool]]] = {}  # model -> task -> [correct, correct, ...]
    
    # Run benchmark
    for provider, client in available.items():
        models = MODELS.get(provider, [])
        if not models:
            continue
            
        print(f"\n🏢 {provider.upper()}")
        
        for model in models:
            print(f"\n  📦 {model}")
            model_first_error_shown = False
            
            if model not in accuracy_tracker:
                accuracy_tracker[model] = {}
            
            for task_name, task in TASKS.items():
                print(f"    📝 {task_name}: ", end="", flush=True)
                
                if task_name not in accuracy_tracker[model]:
                    accuracy_tracker[model][task_name] = []
                
                for prompt_data in task["prompts"]:
                    prompt, expected = prompt_data  # Unpack (prompt, expected_keywords)
                    
                    for run in range(RUNS_PER_TASK):
                        try:
                            result = make_call(
                                provider=provider,
                                client=client,
                                model=model,
                                task_name=task_name,
                                system=task["system"],
                                prompt=prompt,
                            )
                            
                            # Check accuracy
                            response_text = extract_response_text(result, provider)
                            is_correct = check_accuracy(response_text, expected, task_name)
                            accuracy_tracker[model][task_name].append(is_correct)
                            
                            # Show . for correct, ○ for incorrect
                            print("✓" if is_correct else "○", end="", flush=True)
                            call_count += 1
                            
                            # Provider-specific delay to avoid rate limits
                            delay = PROVIDER_DELAYS.get(provider, 1.0)
                            time.sleep(delay)
                            
                        except Exception as e:
                            print("✗", end="", flush=True)
                            errors += 1
                            accuracy_tracker[model][task_name].append(False)
                            
                            # Show first error for each model
                            error_msg = str(e)
                            if not model_first_error_shown:
                                print(f"\n      ⚠ {error_msg[:100]}{'...' if len(error_msg) > 100 else ''}")
                                model_first_error_shown = True
                                error_details.append(f"{provider}/{model}: {error_msg[:150]}")
                            
                            if "rate" in error_msg.lower():
                                time.sleep(5)  # Back off on rate limits
                
                print()  # Newline after task
    
    # Final flush - give time for async sends
    print("\n" + "-" * 60)
    print("⏳ Flushing metrics to backend...")
    time.sleep(3)
    
    print(f"\n✅ Benchmark complete!")
    print(f"   Successful calls: {call_count}")
    print(f"   Errors: {errors}")
    
    if error_details:
        print(f"\n📋 Error Summary:")
        for err in error_details:
            print(f"   • {err}")
    
    # Print accuracy report
    print(f"\n📊 Accuracy Report:")
    print("-" * 70)
    
    # Header
    task_names = list(TASKS.keys())
    header = f"{'Model':<30}" + "".join(f"{t:<12}" for t in task_names) + "Overall"
    print(header)
    print("-" * 70)
    
    for model, tasks in accuracy_tracker.items():
        short_model = model[:28] + ".." if len(model) > 30 else model
        row = f"{short_model:<30}"
        
        total_correct = 0
        total_attempts = 0
        
        for task_name in task_names:
            results = tasks.get(task_name, [])
            if results:
                correct = sum(results)
                total = len(results)
                pct = (correct / total) * 100
                row += f"{pct:>5.0f}%{correct:>2}/{total:<3} "
                total_correct += correct
                total_attempts += total
            else:
                row += f"{'N/A':<12}"
        
        # Overall accuracy
        if total_attempts > 0:
            overall_pct = (total_correct / total_attempts) * 100
            row += f"  {overall_pct:>5.1f}%"
        else:
            row += "  N/A"
        
        print(row)
    
    print("-" * 70)
    print("\nLegend: ✓ = correct, ○ = incorrect, ✗ = error")
    print(f"\n🔭 View cost/latency results at your Observatory dashboard!")


if __name__ == "__main__":
    run_benchmark()

