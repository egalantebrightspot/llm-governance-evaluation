# Structured LLM Evaluation Framework

Skeleton project layout for the structured LLM governance and evaluation system.

## Run the API

Install dependencies:

```bash
pip install -r requirements.txt
```

Start from this folder:

```bash
uvicorn api.main:app --reload --port 8000
```

Or start from workspace root:

```bash
uvicorn api.main:app --app-dir structured-llm-eval-framework --reload --port 8000
```

If port 8000 is already in use, run on another port:

```bash
uvicorn api.main:app --reload --port 8001
```

