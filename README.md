# Cotality Hackathon 2026 — BRD to Insight

Two agents that take a business requirements document all the way to a chart an
analyst can read, without anyone writing SQL.

A business user drops in a requirements document and the raw data files it names.
The first agent reads the document, builds governed BigQuery tables from it, and
verifies what it published. The second agent reads those tables and answers questions
about them in plain English, returning charts.

**The operating principle: the model understands the request, code calculates every
number.** The language model only ever returns a plan as JSON — a table contract, or a
chart specification. It never writes SQL, never executes anything, and never sees a data
row or a computed figure. Every value an analyst sees is produced by pandas or by BigQuery.

## The two agents

| Folder | Agent | Role |
|---|---|---|
| [`brd-to-bigquery-agent/`](brd-to-bigquery-agent) | **Agent 1** | Reads the BRD and builds the governed tables |
| [`brd-to-insight-agent/`](brd-to-insight-agent) | **Agent 2** | Reads those tables and explains them conversationally |

Each folder has its own README with setup and usage. The two are joined only by
environment variables and by the BigQuery tables Agent 1 publishes — neither imports the
other. Agent 2 also runs standalone: upload an Excel file and chart it with no pipeline
at all.

## Architecture

![Architecture](brd-to-insight-agent/ARCHITECTURE_HLD.png)

## Stack

- **Models:** Claude Sonnet 4.5 reads the document, Gemini 3.1 Flash Lite reads the
  question — both through the Cotality LiteLLM gateway.
- **Pipeline:** Python 3.12, Jinja2 SQL templating, sqlglot for SQL validation,
  Pydantic contracts, Apache Airflow for orchestration.
- **App:** Python 3.12, Streamlit, pandas and NumPy for all computation,
  Plotly with Kaleido for rendering and export.
- **Warehouse:** BigQuery in three managed layers — `00_` raw as delivered,
  `01_` curated facts and dimensions, `02_` KPI summaries — plus Cloud Storage for
  landing, authenticated with Google Application Default Credentials.

## Why it is built this way

Every step that could go wrong is a gate that stops the run rather than guessing:
the proposed contract must cite verbatim evidence from the BRD, the source files must
match that contract exactly, the SQL must pass a dry run, the BRD's own worked examples
must still come out right, and the published row counts must reconcile. A human approves
before tables are published and again before any chart is built.

Rates, percentages and identifiers are never summed. Ask for a total of a rate column
and the app refuses and offers the average instead, rather than returning a number that
looks right and is meaningless.
