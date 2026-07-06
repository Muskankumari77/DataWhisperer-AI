# 🗣️ DataWhisperer: Talk to Your CSV

Upload a CSV and ask questions about it in plain English. An AI agent
translates your question into Pandas code, runs it inside a safety-validated
sandbox, and returns the answer as text, a table, or a chart.

---

## 📌 Project Overview

DataWhisperer lets anyone — technical or not — explore a CSV dataset by
simply chatting with it. Behind the scenes, an LLM (Google Gemini by
default) is prompted with the dataset's schema and a sample of its rows,
and asked to generate a short pandas snippet that computes the answer.
That code is then:

1. Statically checked with Python's `ast` module to block dangerous
   operations (file access, `os`/`subprocess`, `eval`/`exec`, etc.).
2. Executed inside a restricted namespace that only exposes `pandas`,
   `numpy`, `matplotlib`, and the user's DataFrame.
3. Rendered back to the user as text, a table, or a matplotlib chart —
   along with the generated code and a plain-English explanation of it.

---

## ✨ Features

**Core**
- Upload a CSV file and preview it instantly
- Automatic dataset info panel (shape, columns, dtypes, missing values, duplicates)
- Ask questions in natural language
- AI-generated pandas code, executed safely
- Answers returned as text, table, or chart

**Professional / Placement-Ready Additions**
- Clean multi-tab Streamlit UI with sidebar, metrics cards, and chat interface
- One-click suggested questions
- Multiple chart types (bar, line, pie, histogram) — auto-selected by the LLM
- Expandable "View generated code" and "Explanation" panels
- AST-based safe Python sandbox (blocks `os`, `subprocess`, `eval`, `exec`, `open`, etc.)
- Persistent chat history during the session (`st.session_state`)
- Download results as CSV and charts as PNG
- Friendly, non-crashing error handling (e.g. missing column names)

---

## 🏗️ Architecture

```
┌──────────────┐      question       ┌───────────────┐
│  Streamlit    │ ───────────────────▶ │   CSVAgent    │
│  UI (app.py)  │                      │ (LangChain +  │
│               │◀─────────────────────│  Gemini/OpenAI)│
└──────┬────────┘   generated code    └───────┬───────┘
       │                                       │
       │                              pandas code (untrusted)
       ▼                                       ▼
┌──────────────┐                      ┌───────────────┐
│  Validators   │ ───── validated ────▶│ Code Executor │
│ (AST safety)  │                      │ (safe sandbox)│
└──────────────┘                      └───────┬───────┘
                                               │
                                   text / table / chart
                                               ▼
                                        Rendered in UI
```

---

## 🧰 Tech Stack

| Layer            | Technology                              |
|-------------------|------------------------------------------|
| Frontend           | Streamlit                                |
| Backend            | Python                                   |
| LLM Framework      | LangChain                                |
| LLM                | Google Gemini API (OpenAI supported too) |
| Data Processing    | Pandas / NumPy                           |
| Code Execution     | Restricted `exec()` sandbox + `ast` checks |
| Visualization      | Matplotlib                               |
| Env Management     | python-dotenv                            |
| Version Control    | Git / GitHub                             |

---

## 🚀 Installation

### 1. Clone / open the project
```bash
cd DataWhisperer
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your API key
```bash
# Copy the example env file
cp .env.example .env      # Windows: copy .env.example .env
```
Open `.env` and paste your Gemini API key (get one free at
https://aistudio.google.com/app/apikey):
```
GOOGLE_API_KEY=your_actual_key_here
LLM_PROVIDER=gemini
```

### 5. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. Upload `data/sample_sales.csv`
(included) or your own CSV to try it out.

---

## 📁 Folder Structure

```
DataWhisperer/
│
├── app.py                     # Main Streamlit application
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── agents/
│   ├── __init__.py
│   └── csv_agent.py            # LangChain LLM wrapper (code generation + explanation)
│
├── utils/
│   ├── __init__.py
│   ├── chart_generator.py      # PNG/CSV export helpers + chart-type heuristic
│   ├── code_executor.py        # Safe sandboxed execution engine
│   └── validators.py           # AST-based safety validation
│
├── prompts/
│   ├── __init__.py
│   └── system_prompt.py        # Prompt templates for code gen + explanation
│
├── data/
│   └── sample_sales.csv        # Sample dataset to try the app immediately
│
├── assets/                     # Logos / static images for the UI
│
└── screenshots/                # App screenshots for this README
```

---

## 📸 Screenshots

> Add your own screenshots here after running the app locally.

| Dataset Preview | Chat + Chart | Generated Code |
|---|---|---|
| _screenshot 1_ | _screenshot 2_ | _screenshot 3_ |

---

## 🔮 Future Improvements

- Support multi-file / multi-table joins
- Add voice input for questions
- Cache repeated questions to reduce LLM calls
- Add unit tests for the sandbox validator (adversarial prompt injection cases)
- Deploy to Streamlit Community Cloud / Docker
- Support Excel (.xlsx) uploads in addition to CSV

---

## ⚠️ Safety Note

This project executes LLM-generated code. While it is protected by an
AST-based sandbox that blocks imports like `os`/`subprocess` and calls like
`eval`/`exec`/`open`, no sandbox is 100% bulletproof against a
sufficiently adversarial LLM output. Do not deploy this publicly with
write access to sensitive systems without additional hardening (e.g.
running in a fully isolated container/process with no network access).
