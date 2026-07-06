"""
Chart helper utilities for DataWhisperer.

Most charts are actually produced by the LLM-generated pandas/matplotlib
code itself (see utils/code_executor.py). This module provides supporting
utilities: saving a matplotlib figure to PNG bytes for download, and a
simple heuristic chart-type suggester that can be surfaced in the UI or
used as a hint in the prompt.
"""

from __future__ import annotations

import io
from typing import Optional

import matplotlib.figure
import pandas as pd


def figure_to_png_bytes(fig: matplotlib.figure.Figure) -> bytes:
    """
    Convert a matplotlib Figure into PNG bytes suitable for a Streamlit
    st.download_button.
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def dataframe_to_csv_bytes(data: "pd.DataFrame | pd.Series") -> bytes:
    """
    Convert a DataFrame or Series result into CSV bytes suitable for a
    Streamlit st.download_button.
    """
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.to_csv(index=True).encode("utf-8")


def suggest_chart_type(question: str) -> Optional[str]:
    """
    Very lightweight keyword-based heuristic to suggest a chart type based
    on the user's question wording. This is only used as an optional hint;
    the LLM makes the final decision about what chart (if any) to draw.

    Returns one of: "bar", "line", "pie", "histogram", or None.
    """
    q = question.lower()

    if any(word in q for word in ["trend", "over time", "monthly", "yearly", "daily", "growth"]):
        return "line"
    if any(word in q for word in ["distribution", "spread", "histogram", "frequency"]):
        return "histogram"
    if any(word in q for word in ["proportion", "percentage", "share", "pie", "split"]):
        return "pie"
    if any(word in q for word in ["compare", "top", "highest", "lowest", "rank", "by category"]):
        return "bar"
    return None
