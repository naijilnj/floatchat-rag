# SIH – Buoy Measurements Explorer (RAG)

Interactive Streamlit app to explore ocean buoy measurements stored in PostgreSQL. The app uses a lightweight RAG-style helper to turn natural language questions into safe SQL against your database, then visualizes the results.

## Features

- Streamlit UI with filters and charts
- PostgreSQL-backed data using SQLAlchemy
- Natural language to SQL (configurable LLM)
- Vector store of DB schema using FAISS + Hugging Face embeddings
- Flexible LLM backends:
	- Google Gemini (cloud)
	- Local Llama via llama.cpp (optional, offline)

## Project structure

Key files in this repo:

- `app.py` – Streamlit UI entrypoint
- `rag_handler.py` – RAG helper: schema retrieval, LLM prompt, FAISS, DB querying
- `pipeline.py`, `data_ingest.py`, `fetch.py`, etc. – data utilities/scripts
- `prompts/` – system and query prompt snippets
- `.env` – environment variables (not committed)

## Prerequisites

- Python 3.10+ recommended
- PostgreSQL 13+ running locally (port 5432 by default)
- Optionally: a Google Generative AI API key (for Gemini)
- Optionally: a local GGUF Llama model (for offline use)

## Setup

You can use either uv (recommended) or pip. If you’re on Windows using WSL, the repo path is typically `/mnt/e/Documents/SIH`.

### Using uv (recommended)

```bash
# In WSL/bash
cd /mnt/e/Documents/SIH

# Create and sync a virtual environment from pyproject/uv.lock
uv sync

# Activate the venv (uv prints the correct command; commonly):
source .venv/bin/activate

# Run the app
streamlit run app.py
```

### Using pip (Windows cmd/PowerShell)

```powershell
cd E:\Documents\SIH

python -m venv .venv
./.venv/Scripts/Activate.ps1  # (or .venv\Scripts\activate for cmd)

pip install -r requirements.txt

streamlit run app.py
```

## Environment variables (.env)

Create a `.env` file in the project root:

```dotenv
# Google (only if using Gemini)
GOOGLE_API_KEY=your_api_key_here

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=argo_db
DB_USER=postgres
DB_PASSWORD=change_me

# Optional: set DB_PASS for legacy code paths
DB_PASS=change_me

# Optional: Move Hugging Face caches off C: (Windows examples)
# PowerShell (temporary for current session):
# $env:HF_HOME = 'E:\hf_cache'
# $env:TRANSFORMERS_CACHE = 'E:\hf_cache\transformers'
# $env:SENTENCE_TRANSFORMERS_HOME = 'E:\hf_cache\sentence-transformers'
# $env:HUGGINGFACE_HUB_CACHE = 'E:\hf_cache\hub'

# WSL/bash (e.g., in ~/.bashrc):
# export HF_HOME=/mnt/e/hf_cache
# export TRANSFORMERS_CACHE=/mnt/e/hf_cache/transformers
# export SENTENCE_TRANSFORMERS_HOME=/mnt/e/hf_cache/sentence-transformers
# export HUGGINGFACE_HUB_CACHE=/mnt/e/hf_cache/hub

# Optional: Local Llama model path (if using llama.cpp backend)
# LLAMA_MODEL_PATH=E:\models\llama-2-7b-chat.Q4_K_M.gguf
```

> Important: Do not commit `.env` to version control.

## Database schema

This app expects a single table named `buoy_measurements`:

```sql
CREATE TABLE IF NOT EXISTS buoy_measurements (
	id SERIAL PRIMARY KEY,
	buoy_id TEXT NOT NULL,
	parameter TEXT NOT NULL,
	timestamp TIMESTAMP NOT NULL,
	value DOUBLE PRECISION NOT NULL,
	created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
```

Example query used by the app:

```sql
SELECT buoy_id, parameter, timestamp, value
FROM buoy_measurements
WHERE parameter = 'temperature'
	AND timestamp BETWEEN '2025-08-24' AND '2025-09-23'
LIMIT 1000;
```

## Running the app

```bash
streamlit run app.py
```

Open the URL printed by Streamlit (e.g., http://localhost:8501).

## Data scraping and ingestion

This project includes two complementary paths for getting data into PostgreSQL:

1) Lightweight web scrapers for moored buoys and HF Radar
2) A robust ARGO/NetCDF pipeline for bulk scientific profiles

Both paths ultimately load measurements into the database. The app reads from `buoy_measurements` for time-series visualizations and NL-to-SQL queries.

### 1) Web scrapers (moored buoys / HF Radar)

- `extract.py` – Scrapes INCOIS moored buoy data pages for selected `parameter` and `buoy` IDs using requests + BeautifulSoup.
- `api.py` – Directs to the same endpoint, useful for debugging/raw dumps of responses.
- `hf_radar.py` – Fetches HF Radar vectors from INCOIS and can save to CSV.
- `argo.py` – Hits the `fetchArgoData.jsp` endpoint and writes a CSV (`argo.csv`).

Run examples (WSL/bash):

```bash
# From repo root
python extract.py
python hf_radar.py
python argo.py
```

These scripts print or write CSV/json-like outputs. To persist into Postgres, normalize them into the canonical columns used by the app:

- `buoy_id` (TEXT)
- `parameter` (TEXT)
- `timestamp` (TIMESTAMP)
- `value` (DOUBLE PRECISION)

Then insert into `buoy_measurements`. Example loader snippet you can adapt:

```python
import os
import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine(
		f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

def load_to_buoy_measurements(df: pd.DataFrame):
		# Ensure required columns
		df = df[["buoy_id", "parameter", "timestamp", "value"]].copy()
		# Coerce dtypes
		df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
		df = df.dropna(subset=["timestamp"])  # drop rows with bad timestamps

		# Optional: deduplicate on (buoy_id, parameter, timestamp)
		df = df.drop_duplicates(subset=["buoy_id", "parameter", "timestamp"]).reset_index(drop=True)

		# Bulk insert (requires table to exist)
		df.to_sql("buoy_measurements", engine, if_exists="append", index=False, method="multi", chunksize=1000)

# Example usage
# df = pd.read_csv("argo.csv")
# load_to_buoy_measurements(df)
```

Recommended DB constraints and indexes:

```sql
-- Avoid duplicates
ALTER TABLE buoy_measurements
	ADD CONSTRAINT uq_buoy_param_time UNIQUE (buoy_id, parameter, timestamp);

-- Helpful indexes
CREATE INDEX idx_buoy_measurements_time ON buoy_measurements (timestamp);
CREATE INDEX idx_buoy_measurements_param ON buoy_measurements (parameter);
```

Notes:
- Respect remote servers (rate limiting, retries, and robots.txt). The provided scripts are for internal/educational use.
- CSVs produced by scrapers can be bulk-loaded with `psql` or pandas.

### 2) ARGO/NetCDF pipeline (bulk ingestion)

File: `pipeline.py`

Key components inside the pipeline:
- `DataFetcher` – Connects to INCOIS FTP/GDAC endpoints to download recent NetCDF files.
- `NetCDFProcessor` – Uses `xarray`/`netCDF4`/`numpy` to parse profiles and standard variables (e.g., TEMP, PSAL, PRES).
- `DatabaseManager` – Persists parsed profiles to Postgres (SQLAlchemy models are defined; adjust mapping if you want to funnel into `buoy_measurements`).
- `ArgoDataPipeline` – Orchestrates fetch → process → store.

Quickstart (uses the built-in `__main__` in `pipeline.py`):

```bash
python pipeline.py
```

By default it runs a 7-day window against INCOIS and attempts to insert processed records. You can adapt the `config` and date range in `if __name__ == "__main__":` to:

```python
config = {
	'database_url': 'postgresql://postgres:yourpass@localhost:5432/argo_db',
	'download_dir': './data/raw/',
	'log_level': 'INFO'
}

from datetime import datetime, timedelta
end_date = datetime.now()
start_date = end_date - timedelta(days=3)
pipeline.run_pipeline(source='incois', date_range=(start_date, end_date))
```

Mapping into `buoy_measurements` (option):
- You may prefer to distill ARGO profile outputs to the same canonical schema used by the app and insert via the `load_to_buoy_measurements` pattern above. This keeps the UI simple and consistent.
- For richer scientific storage, keep the normalized SQLAlchemy models provided in `pipeline.py` (profiles, metadata) and build views/materialized tables to expose simplified time series to the app.

### Scheduling ingestion

- Windows Task Scheduler: run `python extract.py` or `python pipeline.py` on a schedule.
- Linux/WSL cron: `crontab -e` and add entries to run scrapers/pipeline periodically.
- Log outputs to files and consider retry/backoff.

## LLM and embeddings configuration

### Embeddings

- Default: `sentence-transformers/all-mpnet-base-v2`
- If disk space is limited, switch to a smaller model like `sentence-transformers/paraphrase-MiniLM-L6-v2`.
- To move model caches to E:, set `HF_HOME`, `TRANSFORMERS_CACHE`, `SENTENCE_TRANSFORMERS_HOME`, and `HUGGINGFACE_HUB_CACHE` as shown above.

### LLM backends

You can choose one of the following:

1) Google Gemini (cloud)

- Install: `uv add google-generativeai langchain-google-genai` (or `pip install ...`)
- Use a supported model name, e.g. `gemini-1.5-flash`.
- Ensure `GOOGLE_API_KEY` is set in `.env`.

2) Local Llama (offline)

- Install: `uv add llama-cpp-python` (or `pip install llama-cpp-python`)
- Download a GGUF model (e.g., `llama-2-7b-chat.Q4_K_M.gguf`) to `E:\models` and set `LLAMA_MODEL_PATH` in `.env`.
- In `rag_handler.py`, initialize the LLM using llama.cpp and point to the model path. Example pattern:

```python
from langchain_community.llms import LlamaCpp
model_path = os.getenv("LLAMA_MODEL_PATH", "E:/models/llama-2-7b-chat.Q4_K_M.gguf")
llm = LlamaCpp(model_path=model_path, n_ctx=2048, temperature=0.1)
```

## Troubleshooting

- Not enough disk space / OSError: [Errno 28] No space left on device
	- Move HF caches to E: using the env vars in the `.env` section.

- `psycopg2.OperationalError: fe_sendauth: no password supplied`
	- Ensure `.env` has `DB_PASSWORD` (and optionally `DB_PASS`), and that your code uses the right one.

- `relation "measurements" does not exist`
	- Use the correct table name `buoy_measurements` and the `timestamp` column (not `time`).

- `column "time" does not exist`
	- The timestamp column is `timestamp`. Update queries accordingly.

- Pandas warning: "only supports SQLAlchemy connectable" when using `pd.read_sql`
	- Create an SQLAlchemy engine and pass `sqlalchemy.text(query)` + `engine` to `pd.read_sql`.

- LangChain deprecation warnings
	- Import from `langchain_community` and `langchain_huggingface`.
	- Use `chain.invoke(...)` instead of calling the chain directly.

- `ModuleNotFoundError: langchain_huggingface`
	- Install with `uv add langchain-huggingface` or `pip install langchain-huggingface`.

- Torch meta tensor / device errors for embeddings
	- Force CPU or smaller model:
		```python
		from langchain_huggingface import HuggingFaceEmbeddings
		embeddings = HuggingFaceEmbeddings(
			model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
			model_kwargs={"device": "cpu"}
		)
		```

- Gemini model not found (404 `models/gemini-pro`)
	- Use a supported model like `gemini-1.5-flash` and ensure the API is enabled for your key.

## Security notes

- Keep your `.env` out of version control.
- Use least-privilege DB users in production.

## License

This project is provided as-is for the SIH 2025 context.


