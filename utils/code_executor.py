"""
Safe execution engine for LLM-generated pandas code.

The generated code is validated (see utils/validators.py) and then executed
inside a restricted namespace that only exposes pandas, numpy, matplotlib,
and the user's DataFrame. Builtins are stripped down to a safe minimal set.
"""

from __future__ import annotations

import contextlib
import io
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # headless backend, required for Streamlit + threads
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.validators import clean_llm_code_output, is_code_safe

# A minimal, safe subset of Python builtins. Anything not listed here is
# unavailable to the executed code (no open, no eval, no __import__, etc.)
_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}


@dataclass
class ExecutionResult:
    """Container for everything produced by running one snippet of code."""
    success: bool
    result: Any = None
    figure: Optional[Any] = None
    error_message: str = ""
    stdout: str = ""
    result_type: str = "none"  # one of: "text", "table", "chart", "none", "error"


def _classify_result(result: Any, figure: Optional[Any]) -> str:
    """Decide how the result should be rendered in the Streamlit UI."""
    if figure is not None:
        return "chart"
    if isinstance(result, (pd.DataFrame, pd.Series)):
        return "table"
    if result is None:
        return "none"
    return "text"


def execute_pandas_code(raw_code: str, df: pd.DataFrame) -> ExecutionResult:
    """
    Validate and safely execute LLM-generated pandas code against `df`.

    Args:
        raw_code: The raw code string returned by the LLM (may include markdown fences).
        df: The user's uploaded DataFrame.

    Returns:
        ExecutionResult with the outcome, ready to be rendered by the UI layer.
    """
    code = clean_llm_code_output(raw_code)

    # Step 1: safety validation (blocks os/subprocess/eval/exec/etc.)
    is_safe, reason = is_code_safe(code)
    if not is_safe:
        return ExecutionResult(
            success=False,
            error_message=f"This code was blocked by the safety sandbox: {reason}",
            result_type="error",
        )

    # Step 2: build a restricted execution namespace.
    # Only pandas, numpy, matplotlib and the dataframe are exposed.
    plt.close("all")  # make sure we start with a clean figure state
    local_namespace: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "df": df.copy(),  # never let generated code mutate the original df
    }
    global_namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}

    stdout_capture = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, global_namespace, local_namespace)  # noqa: S102 - sandboxed on purpose

        result = local_namespace.get("result", None)

        # Handle the "column not found" convention from the system prompt.
        if isinstance(result, str) and result.startswith("COLUMN_NOT_FOUND:"):
            missing_col = result.split(":", 1)[1].strip()
            return ExecutionResult(
                success=False,
                error_message=(
                    f"The uploaded dataset does not contain a '{missing_col}' column. "
                    f"Please rephrase your question using one of the available columns."
                ),
                result_type="error",
            )

        figure = None
        if plt.get_fignums():
            figure = plt.gcf()

        result_type = _classify_result(result, figure)

        return ExecutionResult(
            success=True,
            result=result,
            figure=figure,
            stdout=stdout_capture.getvalue(),
            result_type=result_type,
        )

    except Exception as exc:  # noqa: BLE001 - we want to catch anything at runtime
        tb_last_line = traceback.format_exception_only(type(exc), exc)[-1].strip()
        return ExecutionResult(
            success=False,
            error_message=f"The generated code raised an error while running: {tb_last_line}",
            result_type="error",
        )
