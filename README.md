# AI CSV Analyzer

A portfolio-ready CSV profiling application inspired by Pandas Profiling, built with FastAPI + Streamlit + Pandas + scikit-learn + optional OpenAI AI insights.

## Features

- CSV upload (up to 100 MB in the starter API)
- Dataset overview and data-quality score
- Data types, missing values, cardinality, descriptive statistics
- IQR-based outlier counts
- Correlation analysis
- Isolation Forest anomaly detection
- K-Means clustering
- Interactive Plotly distribution chart
- Optional AI-generated data-scientist insights
- Docker support

## Quick start

### 1. Create environment

Windows:
```powershell
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Configure AI (optional)

Copy `.env.example` to `.env` and add your OpenAI API key.

### 4. Start FastAPI

```bash
uvicorn backend.main:app --reload --port 8000
```

### 5. Start Streamlit

In a second terminal:
```bash
streamlit run frontend/app.py
```

Open the Streamlit URL shown in the terminal.

## Docker

```bash
docker compose up --build
```

Then open the Streamlit service URL shown by Docker.

## Project architecture

```text
Streamlit UI
    |
    v
FastAPI
    |
    +--> Profiler (Pandas)
    +--> ML Analyzer (scikit-learn)
    +--> AI Analyzer (OpenAI, optional)
```

## Next upgrades

- Parquet/Excel support
- Large-file processing with Polars/DuckDB
- Automated cleaning suggestions and downloadable cleaned CSV
- Feature engineering assistant
- Predictive-model wizard
- Authentication and per-user workspaces
- Background jobs with Celery/RQ
- Database-backed report history
- PDF/HTML report export
