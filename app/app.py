# app.py
import os
import json
import re
import requests
import psycopg2
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from io import StringIO
from datetime import datetime

# --- Load env ---
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "142003")
DB_NAME = os.getenv("DB_NAME", "buoy_measurements")
DB_PORT = os.getenv("DB_PORT", "5432")

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    # We still render the UI but block generation
    st.warning("GOOGLE_API_KEY not set in environment. Gemini calls will not work until you set it.")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
HEADERS = {"Content-Type": "application/json"}

TABLE_NAME = "buoy_measurements"
COLUMNS = ["id", "buoy_id", "parameter", "timestamp", "value"]

FORBIDDEN_KEYWORDS = [
    "insert", "update", "delete", "drop", "create", "alter", "truncate", "grant", "revoke",
    "--", "/*", "*/", "exec", "call", "prepare", "execute"
]

SQL_KEYWORDS = {
    "select","from","where","and","or","group","by","order","limit","asc","desc",
    "as","on","join","inner","left","right","full","having","distinct","count",
    "avg","min","max","sum","between","in","is","null","not","like"
}

# --- Helper functions ---
def generate_sql(nl_question: str) -> str:
    columns_list = ", ".join(COLUMNS)
    prompt = f"""
You are an assistant that converts natural language questions about measurement data
into a single SQL SELECT statement for a PostgreSQL table named "{TABLE_NAME}" with columns:
{columns_list}

Rules:
- ONLY Show relevant columns
- ONLY output a single SELECT statement and end it with a single semicolon.
- Do NOT generate any UPDATE/DELETE/INSERT/CREATE/DROP/ALTER statements or multiple statements.
- Use only the columns listed above; do not invent columns or tables.
- If the user asks for aggregations, use valid SQL aggregation functions (e.g., avg(value)).
- Use ISO-like date literals for date filters (e.g., '2023-03-01'::date).

User question: "{nl_question}"
"""
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    resp = requests.post(GEMINI_URL, headers=HEADERS, json=data, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"LLM request failed: {resp.status_code} {resp.text}")
    body = resp.json()

    # robust extraction for text
    try:
        sql_text = body["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        def find_text(obj):
            if isinstance(obj, dict):
                for k,v in obj.items():
                    if k == "text" and isinstance(v, str):
                        return v
                    res = find_text(v)
                    if res:
                        return res
            elif isinstance(obj, list):
                for item in obj:
                    res = find_text(item)
                    if res:
                        return res
            return None
        sql_text = find_text(body) or json.dumps(body)
    return sql_text

def extract_and_validate_sql(raw_text: str) -> str:
    idx = raw_text.find(";")
    candidate = raw_text[: idx+1].strip() if idx != -1 else raw_text.strip()
    lowered = candidate.lower()

    if not lowered.lstrip().startswith("select"):
        raise ValueError("Generated SQL does not start with SELECT. Aborting for safety.")

    for kw in FORBIDDEN_KEYWORDS:
        if kw in lowered:
            raise ValueError(f"Forbidden token found in SQL: '{kw}'")

    if TABLE_NAME.lower() not in lowered:
        raise ValueError(f"SQL must reference the table '{TABLE_NAME}'.")

    # rewrite SELECT * -> explicit columns
    if re.search(r"(?i)select\s+\*", lowered):
        explicit = "SELECT " + ", ".join(COLUMNS)
        candidate = re.sub(r"(?i)select\s+\*", explicit, candidate, count=1)
        lowered = candidate.lower()

    # token checks (conservative)
    tokens = [tok.strip(" ,();") for tok in lowered.replace("\n"," ").split()]
    ident_pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    allowed_idents = set([c.lower() for c in COLUMNS] + [TABLE_NAME.lower()]) | SQL_KEYWORDS
    for tok in tokens:
        if ident_pattern.match(tok) and tok not in allowed_idents:
            # allow table.column tokens (contain '.')
            if "." in tok:
                continue
            raise ValueError(f"Unknown identifier in SQL: '{tok}'. Allowed columns: {', '.join(COLUMNS)}")

    return candidate

def execute_query(pg_sql: str, limit_rows: int = 10000) -> pd.DataFrame:
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASS, dbname=DB_NAME, port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute(pg_sql)
        rows = cur.fetchmany(limit_rows)
        cols = [desc[0] for desc in cur.description] if cur.description else []
        df = pd.DataFrame(rows, columns=cols)
        return df
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

# --- Streamlit UI ---
st.set_page_config(page_title="Buoy Measurements — NL → SQL", layout="wide")
st.title("FloatChat — ARGO Ocean Data Discovery")

with st.sidebar:
    st.header("Settings")
    st.write("Database & LLM settings are read from environment variables (.env).")
    st.text_input("DB_HOST", DB_HOST, disabled=True)
    st.text_input("DB_NAME", DB_NAME, disabled=True)
    st.text_input("TABLE", TABLE_NAME, disabled=True)
    st.markdown("---")
    st.header("Example queries")
    examples = [
        "Show air_pressure values for buoy BD11",
        "Give average value for parameter air_pressure for buoy BD11 in June 2011",
        "List all measurements for buoy BD11 where parameter = 'air_pressure' and value > 1010",
        "Show the latest 20 measurements for buoy BD11"
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state["nl_query"] = ex

st.subheader("Enter a natural language question")
nl_query = st.text_area("Natural language question", value=st.session_state.get("nl_query", ""), height=120)

col1, col2 = st.columns([3,1])
with col2:
    gen_btn = st.button("Generate & Run", type="primary")
    show_raw = st.checkbox("Show raw LLM output", value=False)
    show_sql = st.checkbox("Show validated SQL", value=True)
    download_btn_placeholder = st.empty()

# placeholders
status_ph = st.empty()
raw_ph = st.empty()
sql_ph = st.empty()
result_ph = st.empty()
csv_ph = st.empty()

if gen_btn:
    if not nl_query.strip():
        st.error("Please enter a natural language query.")
    elif not API_KEY:
        st.error("GOOGLE_API_KEY not set. Set it in your .env before generating SQL.")
    else:
        try:
            status_ph.info("Calling Gemini to generate SQL...")
            raw = generate_sql(nl_query)
            idx = raw.find(";")
            if (idx != -1):
                raw = raw[7:idx+1]
            print(raw)
            status_ph.success("LLM returned output.")
            if show_raw:
                raw_ph.code(raw, language="text")
            safe_sql = extract_and_validate_sql(raw)
            if show_sql:
                sql_ph.code(safe_sql, language="sql")

            status_ph.info("Executing SQL against Postgres...")
            df = execute_query(safe_sql)
            if df.empty:
                result_ph.info("Query executed successfully but returned no rows.")
            else:
                result_ph.write(f"Results ({len(df)} rows):")
                result_ph.dataframe(df, use_container_width=True)
                # CSV download
                csv = df.to_csv(index=False)
                b64 = csv.encode("utf-8")
                download_btn = download_btn_placeholder.download_button(
                    label="Download CSV",
                    data=b64,
                    file_name=f"query_results_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            status_ph.error(f"Error: {e}")

st.markdown("---")
st.caption("Note: This app performs conservative validation of LLM-generated SQL. For production use add a full SQL parser/whitelist and parameterization to avoid injection risk.")
