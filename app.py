"""
DataWhisperer: Talk to Your CSV
--------------------------------
Main Streamlit application entry point.

Upload a CSV, ask questions in plain English, and get back a text answer,
a table, or a chart — powered by an LLM that writes pandas code, which is
then executed inside a safety-validated sandbox.
"""

from __future__ import annotations

import traceback

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agents.csv_agent import CSVAgent
from utils.chart_generator import dataframe_to_csv_bytes, figure_to_png_bytes
from utils.code_executor import execute_pandas_code

load_dotenv()

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="DataWhisperer | Talk to Your CSV",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Custom theme (CSS injection) — gradient header, colorful cards & buttons
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
:root {
    --dw-purple: #7C3AED;
    --dw-purple-dark: #5B21B6;
    --dw-pink: #EC4899;
    --dw-blue: #3B82F6;
    --dw-teal: #14B8A6;
    --dw-amber: #F59E0B;
    --dw-bg-card: #ffffff;
}

/* Overall app background */
.stApp {
    background: linear-gradient(180deg, #F5F3FF 0%, #FDF4FF 45%, #F0FDFA 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #4C1D95 0%, #6D28D9 55%, #7C3AED 100%);
}
section[data-testid="stSidebar"] * {
    color: #F5F3FF !important;
}
section[data-testid="stSidebar"] .stFileUploader section {
    background: rgba(255,255,255,0.08);
    border: 1.5px dashed rgba(255,255,255,0.4);
    border-radius: 14px;
}
section[data-testid="stSidebar"] div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 10px 14px;
    margin-bottom: 8px;
    border: 1px solid rgba(255,255,255,0.18);
}
section[data-testid="stSidebar"] button {
    background: linear-gradient(90deg, #EC4899, #F97316) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* Hero banner */
.dw-hero {
    background: linear-gradient(120deg, #7C3AED 0%, #EC4899 55%, #F59E0B 100%);
    padding: 34px 38px;
    border-radius: 20px;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(124, 58, 237, 0.25);
}
.dw-hero h1 {
    color: white !important;
    font-size: 2.1rem;
    margin: 0 0 6px 0;
    font-weight: 800;
}
.dw-hero p {
    color: rgba(255,255,255,0.92) !important;
    font-size: 1.02rem;
    margin: 0;
}

/* Metric cards in main area */
div[data-testid="stMetric"] {
    background: var(--dw-bg-card);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 3px 14px rgba(124, 58, 237, 0.10);
    border: 1px solid #EDE9FE;
}
div[data-testid="stMetricLabel"] { color: #7C3AED !important; font-weight: 600; }
div[data-testid="stMetricValue"] { color: #4C1D95 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: #EDE9FE;
    border-radius: 10px 10px 0 0;
    padding: 8px 18px;
    font-weight: 600;
    color: #6D28D9;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #7C3AED, #EC4899) !important;
    color: white !important;
}

/* Suggested question buttons in main content */
div[data-testid="stVerticalBlock"] .stButton button {
    background: white;
    border: 1.6px solid #DDD6FE;
    color: #5B21B6;
    border-radius: 12px;
    font-weight: 600;
    padding: 10px 6px;
    transition: all 0.15s ease-in-out;
}
div[data-testid="stVerticalBlock"] .stButton button:hover {
    background: linear-gradient(90deg, #7C3AED, #EC4899);
    color: white;
    border-color: transparent;
    transform: translateY(-2px);
}

/* Chat input */
.stChatInput textarea {
    border-radius: 14px !important;
}

/* Chat bubbles */
div[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 6px 4px;
    margin-bottom: 4px;
}

/* Section header pill */
.dw-section-title {
    display: inline-block;
    background: linear-gradient(90deg, #EDE9FE, #FCE7F3);
    color: #6D28D9;
    padding: 6px 16px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 14px;
}

/* Download buttons */
.stDownloadButton button {
    background: linear-gradient(90deg, #14B8A6, #3B82F6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* Expander headers */
.streamlit-expanderHeader {
    background: #F5F3FF;
    border-radius: 10px;
    font-weight: 600;
    color: #5B21B6;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# icon, question, accent color (used for potential future styling per-card)
SUGGESTED_QUESTIONS = [
    ("👀", "Show first 10 rows"),
    ("🏆", "What is the highest value column-wise?"),
    ("📐", "Show the average of all numeric columns"),
    ("🕳️", "Are there any missing values?"),
    ("🔗", "Show correlation between numeric columns"),
    ("⭐", "Show top 5 categories by count"),
]


# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------
def init_session_state() -> None:
    defaults = {
        "df": None,
        "file_name": None,
        "chat_history": [],  # list of dicts: {question, code, explanation, exec_result}
        "agent": None,
        "agent_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


def get_agent() -> CSVAgent | None:
    """Lazily build the CSVAgent once and cache it in session state."""
    if st.session_state.agent is not None:
        return st.session_state.agent
    try:
        st.session_state.agent = CSVAgent()
        st.session_state.agent_error = None
        return st.session_state.agent
    except Exception as exc:  # noqa: BLE001
        st.session_state.agent_error = str(exc)
        return None


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🗣️ DataWhisperer")
    st.caption("✨ Talk to your CSV in plain English.")
    st.divider()

    uploaded_file = st.file_uploader("📤 Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.file_name = uploaded_file.name
            st.success(f"✅ Loaded **{uploaded_file.name}**")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read this CSV file: {exc}")

    st.divider()

    if st.session_state.df is not None:
        st.markdown("### 📊 Dataset Info")
        df = st.session_state.df
        c1, c2 = st.columns(2)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3, c4 = st.columns(2)
        c3.metric("Duplicates", int(df.duplicated().sum()))
        c4.metric("Missing", int(df.isna().sum().sum()))

        with st.expander("🧬 Column names & types"):
            dtype_df = pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values})
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    st.divider()
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.caption("Built with 💜 using Streamlit, LangChain & AI.")


# --------------------------------------------------------------------------
# Main area — Hero banner
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="dw-hero">
        <h1>🗣️ DataWhisperer</h1>
        <p>Upload a CSV, ask a question in plain English, and get a text answer, table, or chart — instantly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.df is None:
    st.info("👈 Upload a CSV file from the sidebar to get started.")
    st.stop()

df = st.session_state.df

# ---- Dataset preview & info panel ----
st.markdown('<span class="dw-section-title">📄 Your Dataset</span>', unsafe_allow_html=True)
tab_preview, tab_info = st.tabs(["📄 Dataset Preview", "ℹ️ Dataset Information"])

with tab_preview:
    st.dataframe(df.head(10), use_container_width=True)

with tab_info:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Duplicate rows", int(df.duplicated().sum()))
    col4.metric("Missing values", int(df.isna().sum().sum()))

    st.markdown("**Columns & Data Types**")
    dtype_df = pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values})
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    st.markdown("**Missing Values per Column**")
    missing_df = df.isna().sum().reset_index()
    missing_df.columns = ["Column", "Missing Values"]
    st.dataframe(missing_df, use_container_width=True, hide_index=True)

st.write("")

# ---- Suggested questions ----
st.markdown('<span class="dw-section-title">💡 Try one of these</span>', unsafe_allow_html=True)
suggestion_cols = st.columns(3)
clicked_suggestion = None
for idx, (icon, question) in enumerate(SUGGESTED_QUESTIONS):
    if suggestion_cols[idx % 3].button(f"{icon}  {question}", use_container_width=True, key=f"suggestion_{idx}"):
        clicked_suggestion = question

st.write("")

# ---- Chat interface ----
st.markdown('<span class="dw-section-title">💬 Ask a question about your data</span>', unsafe_allow_html=True)

user_question = st.chat_input("e.g. What is the average of the Sales column?")
question_to_process = clicked_suggestion or user_question

if question_to_process:
    agent = get_agent()

    if agent is None:
        st.error(
            "⚠️ Could not initialize the AI agent. "
            f"Reason: {st.session_state.agent_error}\n\n"
            "Please make sure your API key is set correctly in your .env file."
        )
    else:
        with st.spinner("🤔 DataWhisperer is thinking..."):
            try:
                agent_response = agent.generate_code(question_to_process, df)
                exec_result = execute_pandas_code(agent_response.code, df)

                explanation = ""
                if exec_result.success:
                    explanation = agent.explain_code(question_to_process, agent_response.code)

                st.session_state.chat_history.append(
                    {
                        "question": question_to_process,
                        "code": agent_response.code,
                        "explanation": explanation,
                        "exec_result": exec_result,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                st.session_state.chat_history.append(
                    {
                        "question": question_to_process,
                        "code": "",
                        "explanation": "",
                        "error": f"Something went wrong: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )

# ---- Render chat history (most recent first) ----
for turn_index, turn in enumerate(reversed(st.session_state.chat_history)):
    with st.chat_message("user", avatar="🙋"):
        st.markdown(turn["question"])

    with st.chat_message("assistant", avatar="🗣️"):
        if turn.get("error"):
            st.error(turn["error"])
            continue

        exec_result = turn["exec_result"]

        if not exec_result.success:
            st.warning(exec_result.error_message)
        else:
            if exec_result.result_type == "chart" and exec_result.figure is not None:
                st.pyplot(exec_result.figure, use_container_width=True)
                png_bytes = figure_to_png_bytes(exec_result.figure)
                st.download_button(
                    "⬇️ Download chart as PNG",
                    data=png_bytes,
                    file_name="datawhisperer_chart.png",
                    mime="image/png",
                    key=f"chart_dl_{turn_index}",
                )
                if exec_result.result is not None:
                    st.markdown("**Underlying data:**")
                    st.dataframe(exec_result.result, use_container_width=True)

            elif exec_result.result_type == "table":
                st.dataframe(exec_result.result, use_container_width=True)
                csv_bytes = dataframe_to_csv_bytes(exec_result.result)
                st.download_button(
                    "⬇️ Download table as CSV",
                    data=csv_bytes,
                    file_name="datawhisperer_result.csv",
                    mime="text/csv",
                    key=f"table_dl_{turn_index}",
                )

            elif exec_result.result_type == "text":
                st.success(f"**Answer:** {exec_result.result}")

            else:
                st.info("The code ran successfully but did not produce a displayable result.")

            with st.expander("🧾 View generated Pandas code"):
                st.code(turn["code"], language="python")

            if turn.get("explanation"):
                with st.expander("💡 Explanation in simple English"):
                    st.write(turn["explanation"])
