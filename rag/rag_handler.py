#!/usr/bin/env python3
import os
import sys
import json
import requests
from dotenv import load_dotenv
from tabulate import tabulate
import psycopg2

# -----------------------
# Input / Env
# -----------------------
if len(sys.argv) < 2:
    print("Usage: python parse_postgres_measurements.py \"<natural language question>\"")
    sys.exit(1)

user_input = sys.argv[1]

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("Missing GOOGLE_API_KEY in environment.")
    sys.exit(1)

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
HEADERS = {"Content-Type": "application/json"}

# Table & columns (from your spec)
TABLE_NAME = "buoy_measurements"
COLUMNS = ["id", "buoy_id", "parameter", "timestamp", "value"]

# -----------------------
# LLM prompt & call
# -----------------------
def generate_sql(nl_question: str) -> str:
    """
    Ask Gemini to produce a SELECT statement for the buoy_measurements table.
    The prompt constraints help reduce hallucinated column/table names.
    """
    columns_list = ", ".join(COLUMNS)
    prompt = f"""
You are an assistant that converts natural language questions about measurement data
into a single SQL SELECT statement for a PostgreSQL table named "{TABLE_NAME}" with columns:
{columns_list}

Rules:
- ONLY output a single SELECT statement and end it with a single semicolon.
- Do NOT generate any UPDATE/DELETE/INSERT/CREATE/DROP/ALTER statements or multiple statements.
- Use only the columns listed above; do not invent columns or tables.
- Prefer explicit column names (no SELECT *).
- If the user asks for aggregations, use valid SQL aggregation functions (e.g., avg(value)).
- Use ISO-like date literals for date filters (e.g., '2023-03-01'::date).

User question: "{nl_question}"
"""

    data = {"contents": [{"parts": [{"text": prompt}]}]}
    resp = requests.post(GEMINI_URL, headers=HEADERS, json=data, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"LLM request failed: {resp.status_code} {resp.text}")
    body = resp.json()

    # Extract textual output (robust to structure)
    try:
        sql_text = body["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        # fallback: search for first "text" in json
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

# -----------------------
# SQL extraction & validation
# -----------------------
FORBIDDEN_KEYWORDS = [
    "insert", "update", "delete", "drop", "create", "alter", "truncate", "grant", "revoke",
    "--", "/*", "*/", "exec", "call", "prepare", "execute"
]

def extract_and_validate_sql(raw_text: str) -> str:
    """
    Extract first statement up to semicolon and validate:
    - starts with SELECT
    - does not contain forbidden keywords
    - uses only allowed column names and the allowed table name
    """
    # Extract up to first semicolon (inclusive)
    idx = raw_text.find(";")
    candidate = raw_text[: idx+1].strip() if idx != -1 else raw_text.strip()

    lowered = candidate.lower()
    if not lowered.lstrip().startswith("select"):
        raise ValueError("Generated SQL does not start with SELECT. Aborting for safety.")

    # Basic forbidden keyword check
    for kw in FORBIDDEN_KEYWORDS:
        if kw in lowered:
            raise ValueError(f"Forbidden token found in SQL: '{kw}'")

    # Ensure only the allowed table name is referenced
    if TABLE_NAME.lower() not in lowered:
        raise ValueError(f"SQL must reference the table '{TABLE_NAME}'.")

    # Ensure no unknown columns are referenced: simple check by tokens
    # (this is a conservative heuristic, not a full SQL parser)
    tokens = [tok.strip(" ,()") for tok in lowered.replace("\n"," ").split()]
    referenced_cols = set()
    for col in COLUMNS:
        if col.lower() in lowered:
            referenced_cols.add(col.lower())

    # If SELECT * is used, rewrite to explicit column list
    if "select *" in lowered:
        explicit = "SELECT " + ", ".join(COLUMNS)
        # replace the first occurrence of "select *" ignoring case
        import re
        candidate = re.sub(r"(?i)select\s+\*", explicit, candidate, count=1)

    # Final check for unknown identifier patterns (very conservative)
    # Find words that look like column identifiers and reject if not in allowed set or SQL keywords
    sql_keywords = {
        "select","from","where","and","or","group","by","order","limit","asc","desc",
        "as","on","join","inner","left","right","full","having","distinct","count",
        "avg","min","max","sum","between","in","is","null","not","like"
    }
    # tokens that look like identifiers: letters + underscores and not in keywords
    import re
    ident_pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    for tok in tokens:
        if ident_pattern.match(tok) and tok not in sql_keywords:
            # if it's not a known column and not the table name, warn (but allow function names also pass through)
            if tok not in [c.lower() for c in COLUMNS] and tok != TABLE_NAME.lower():
                # allow common function names (count, avg...) handled by sql_keywords; otherwise flag
                # But don't be overly strict: allow 'date' usage as cast: we won't try to catch everything
                # If suspicious, raise error (conservative)
                # We'll allow tokens containing '.' as table.column which may include table alias; skip in that case
                if "." in tok:
                    continue
                raise ValueError(f"Unknown identifier in SQL: '{tok}'. Allowed columns: {', '.join(COLUMNS)}")

    # Finally, return candidate
    return candidate

# -----------------------
# DB execution
# -----------------------
def execute_query(pg_sql: str):
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASS, dbname=DB_NAME
        )
        cur = conn.cursor()
        cur.execute(pg_sql)
        # Fetch at most 10000 rows to avoid runaway outputs
        rows = cur.fetchmany(10000)
        cols = [desc[0] for desc in cur.description] if cur.description else []
        return rows, cols
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

# -----------------------
# Main
# -----------------------
def main():
    try:
        raw = generate_sql(user_input)
        print("=== Raw LLM output ===")
        print(raw)
        print("======================\n")

        safe_sql = extract_and_validate_sql(raw)
        print("=== Executing SQL ===")
        print(safe_sql)
        print("=====================\n")

        rows, cols = execute_query(safe_sql)
        if not rows:
            print("Query executed successfully but returned no rows.")
        else:
            print(tabulate(rows, headers=cols, tablefmt="psql"))

    except Exception as e:
        print("Error:", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
