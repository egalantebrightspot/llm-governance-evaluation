Structured LLM Evaluation Framework
A schema‑driven, Azure OpenAI–powered system for validating, critiquing, and scoring structured LLM outputs.

Purpose
This framework evaluates the reliability of large language models by enforcing strict JSON schemas, validating outputs, comparing them to gold‑standard examples, and scoring their performance across multiple dimensions. It reflects how modern AI engineering teams build governed, production‑grade LLM systems that require predictable, auditable, and explainable behavior.

The system integrates directly with Azure OpenAI, enabling real structured‑output generation using models such as gpt‑4o, gpt‑4o-mini, and gpt‑4.1.

Core Capabilities
Azure OpenAI structured output generation using enforced JSON responses.

Schema validation using pydantic and jsonschema.

Gold‑standard comparison for correctness and completeness.

Hallucination detection for extra or fabricated fields.

Multi‑agent evaluation loop (Generator → Validator → Critic → Scorer).

FastAPI service exposing the full evaluation pipeline.

Test harness for deterministic, reproducible evaluation.

Architecture Overview
The framework follows a multi‑stage evaluation pipeline:

Generator Agent  
Produces structured JSON using Azure OpenAI with enforced schema constraints.

Validator Agent  
Checks schema compliance, missing fields, type mismatches, and hallucinations.

Critic Agent  
Compares the LLM output to a gold standard and produces a structured critique.

Scoring Agent  
Converts the critique into quantitative metrics.

FastAPI Layer  
Exposes endpoints for generation, validation, evaluation, scoring, and full‑pipeline execution.

                ┌──────────────────────────┐
                │      Input Task          │
                │   + JSON Schema          │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │     Generator Agent      │
                │  (Azure OpenAI JSON)     │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │     Validator Agent      │
                │ (Schema Compliance Check)│
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │       Critic Agent       │
                │ (Gold Standard Compare)  │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │      Scoring Agent       │
                │ (Metrics + Evaluation)   │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │        FastAPI           │
                │ (generate/validate/etc.) │
                └──────────────────────────┘

structured-llm-eval-framework/
│
├── schemas/
│   ├── classification.json
│   ├── extraction.json
│   └── reasoning.json
│
├── agents/
│   ├── generator_azure.py
│   ├── validator.py
│   ├── critic.py
│   └── scorer.py
│
├── evaluation/
│   ├── pipeline.py
│   ├── metrics.py
│   └── golden_set.py
│
├── api/
│   ├── main.py
│   └── routers/
│       ├── generate.py
│       ├── validate.py
│       └── evaluate.py
│
├── tests/
│   ├── test_schema_validation.py
│   ├── test_pipeline.py
│   └── test_metrics.py
│
└── README.md
