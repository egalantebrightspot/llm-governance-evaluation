# Structured LLM Evaluation Framework

A schema-driven, Azure OpenAI-powered system for generating, validating, critiquing, and scoring structured LLM outputs.

## Purpose

This framework evaluates the reliability of large language models by enforcing JSON schemas, validating outputs, comparing them to gold-standard examples, and scoring performance across multiple dimensions.

## Core Capabilities

- Azure OpenAI structured output generation
- JSON schema validation and governance checks
- Gold-standard comparison for correctness and completeness
- Hallucination detection
- Multi-agent evaluation loop: Generator -> Validator -> Critic -> Scorer
- FastAPI service for interactive testing

## Project Structure

```text
structured-llm-eval-framework/
├── api/
├── agents/
├── evaluation/
├── schemas/
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure environment variables in `.env`:

```env
AZURE_OPENAI_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

## Run the API

Start from this folder:

```bash
uvicorn api.main:app --reload --port 8000
```

From the workspace root:

```bash
cd structured-llm-eval-framework
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Or start from workspace root:

```bash
uvicorn api.main:app --app-dir structured-llm-eval-framework --reload --port 8000
```

If port 8000 is already in use:

```bash
uvicorn api.main:app --reload --port 8001
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

