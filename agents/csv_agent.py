"""
CSV Agent for DataWhisperer.

This module wraps the LLM (Google Gemini by default, OpenAI as a fallback)
and is responsible for:
  1. Turning a natural-language question + DataFrame schema into pandas code.
  2. Explaining that generated code in plain English.

The actual SAFE EXECUTION of the code happens separately in
utils/code_executor.py — this module never runs the code itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from prompts.system_prompt import build_explanation_prompt, build_system_prompt
from utils.validators import clean_llm_code_output


@dataclass
class AgentResponse:
    """The raw output of one round-trip to the LLM for code generation."""
    code: str
    raw_llm_output: str


class CSVAgent:
    """
    A thin orchestration layer around a LangChain chat model that turns
    natural-language questions about a DataFrame into pandas code.
    """

    def __init__(self, provider: Optional[str] = None) -> None:
        self.provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
        self.llm = self._build_llm()

    def _build_llm(self):
        """Instantiate the correct LangChain chat model based on provider."""
        if self.provider == "grok":
            # xAI's Grok API is OpenAI-compatible, so we reuse langchain_openai
            # and simply point it at xAI's base URL. No extra package needed.
            from langchain_openai import ChatOpenAI

            api_key = os.getenv("GROK_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "GROK_API_KEY is not set. Please add it to your .env file."
                )
            model_name = os.getenv("GROK_MODEL", "grok-4.3")
            return ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url="https://api.x.ai/v1",
                temperature=0,
            )

        if self.provider == "groq":
            # Groq's API is OpenAI-compatible too (different company from xAI's Grok).
            from langchain_openai import ChatOpenAI

            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "GROQ_API_KEY is not set. Please add it to your .env file."
                )
            model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            return ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                temperature=0,
            )

        if self.provider == "openai":
            from langchain_openai import ChatOpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY is not set. Please add it to your .env file."
                )
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            return ChatOpenAI(model=model_name, api_key=api_key, temperature=0)

        # Default: Gemini
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set. Please add it to your .env file."
            )
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0)

    @staticmethod
    def _dataframe_context(df: pd.DataFrame) -> tuple[list[str], dict[str, str], str]:
        """Extract the schema + a small sample preview from the DataFrame."""
        columns = list(df.columns)
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        sample_rows = df.head(5).to_markdown(index=False)
        return columns, dtypes, sample_rows

    def generate_code(self, question: str, df: pd.DataFrame) -> AgentResponse:
        """
        Ask the LLM to produce pandas code that answers `question` about `df`.

        Returns:
            AgentResponse containing the cleaned code and the raw LLM text.
        """
        columns, dtypes, sample_rows = self._dataframe_context(df)
        system_prompt = build_system_prompt(
            df_name="df", columns=columns, dtypes=dtypes, sample_rows=sample_rows
        )

        messages = [
            ("system", system_prompt),
            ("human", question),
        ]

        response = self.llm.invoke(messages)
        raw_output = response.content if hasattr(response, "content") else str(response)
        cleaned_code = clean_llm_code_output(raw_output)

        return AgentResponse(code=cleaned_code, raw_llm_output=raw_output)

    def explain_code(self, question: str, code: str) -> str:
        """
        Ask the LLM to explain the previously generated code in simple English.
        """
        prompt = build_explanation_prompt(question=question, code=code)
        response = self.llm.invoke([("human", prompt)])
        explanation = response.content if hasattr(response, "content") else str(response)
        return explanation.strip()
