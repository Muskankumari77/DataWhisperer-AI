"""
Safety validation utilities for DataWhisperer.

This module inspects LLM-generated Python code BEFORE execution and blocks
anything that could be dangerous (file access, OS commands, network calls,
arbitrary eval/exec, etc.). This is the core of the "Safe Python Sandbox"
requirement.
"""

from __future__ import annotations

import ast
import re

# Explicitly blocked module names (cannot be imported at all)
BLOCKED_MODULES: set[str] = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "pathlib",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "smtplib",
    "ctypes",
    "multiprocessing",
    "threading",
    "importlib",
    "pickle",
    "shelve",
    "sqlite3",
}

# Explicitly blocked function / builtin names (cannot be called at all)
BLOCKED_CALLS: set[str] = {
    "eval",
    "exec",
    "open",
    "compile",
    "__import__",
    "input",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "exit",
    "quit",
}

# Dangerous substrings as a fast first-pass regex filter (belt and suspenders,
# in addition to the AST-based check below).
_DANGEROUS_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(BLOCKED_MODULES | BLOCKED_CALLS)) + r")\b"
)


class UnsafeCodeError(Exception):
    """Raised when generated code fails the safety validation."""


def quick_pattern_check(code: str) -> list[str]:
    """
    Fast regex-based scan for obviously dangerous keywords.
    Returns a list of matched dangerous keywords (empty list = looks clean).
    This is only a pre-filter; the AST check below is the authoritative one.
    """
    return sorted(set(_DANGEROUS_PATTERN.findall(code)))


def validate_code_ast(code: str) -> None:
    """
    Parse the code into an AST and walk every node, raising UnsafeCodeError
    if anything dangerous is found. This is much more reliable than plain
    string/regex matching because it understands actual Python structure.

    Raises:
        UnsafeCodeError: if the code contains a blocked import, call, or
                         dangerous attribute/dunder access.
        SyntaxError: if the code is not valid Python at all.
    """
    tree = ast.parse(code, mode="exec")

    for node in ast.walk(tree):
        # Block "import x" / "import x as y"
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split(".")[0]
                if root_module in BLOCKED_MODULES:
                    raise UnsafeCodeError(f"Import of module '{alias.name}' is not allowed.")

        # Block "from x import y"
        if isinstance(node, ast.ImportFrom):
            root_module = (node.module or "").split(".")[0]
            if root_module in BLOCKED_MODULES:
                raise UnsafeCodeError(f"Import from module '{node.module}' is not allowed.")

        # Block calls to dangerous builtins/functions, e.g. eval(...), open(...)
        if isinstance(node, ast.Call):
            func = node.func
            func_name = None
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr

            if func_name in BLOCKED_CALLS:
                raise UnsafeCodeError(f"Call to '{func_name}(...)' is not allowed.")

        # Block dunder attribute access used for sandbox escapes,
        # e.g. ().__class__.__bases__ tricks
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                raise UnsafeCodeError(f"Access to dunder attribute '{node.attr}' is not allowed.")

        # Block direct Name references to blocked modules
        # (covers cases where a module was somehow already bound)
        if isinstance(node, ast.Name) and node.id in BLOCKED_MODULES:
            raise UnsafeCodeError(f"Reference to '{node.id}' is not allowed.")


def is_code_safe(code: str) -> tuple[bool, str]:
    """
    Convenience wrapper combining the quick pattern check and the AST check.

    Returns:
        (True, "") if the code is safe.
        (False, reason) if the code is unsafe or invalid.
    """
    try:
        validate_code_ast(code)
        return True, ""
    except UnsafeCodeError as exc:
        return False, str(exc)
    except SyntaxError as exc:
        return False, f"Generated code has a syntax error: {exc}"


def clean_llm_code_output(raw_text: str) -> str:
    """
    Strip markdown code fences (```python ... ``` or ``` ... ```) that LLMs
    sometimes add even when told not to, and strip leading/trailing whitespace.
    """
    text = raw_text.strip()
    text = re.sub(r"^```(?:python)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def column_exists(column_name: str, columns: list[str]) -> bool:
    """Case-sensitive check whether a column exists in the dataset."""
    return column_name in columns
