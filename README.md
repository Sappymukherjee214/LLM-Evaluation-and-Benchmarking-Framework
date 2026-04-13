# 🚀 LLM Evaluation & Benchmarking Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![React 19](https://img.shields.io/badge/React-19-blue.svg)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

A production-grade, research-quality platform designed to systematically evaluate, benchmark, and audit Large Language Models (LLMs). This framework provides a unified interface to compare model performance across diverse datasets, tasks, and critical metrics such as fairness, bias, and scalability.

## ✨ Key Features

### 📡 Multi-Model Orchestration
*   **Unified Interface**: Seamlessly switch between OpenAI (GPT-4/3.5), HuggingFace (Llama, Mistral), and local inference.
*   **Parallel Execution**: High-throughput evaluation engine utilizing multi-threading for rapid benchmarking of large datasets.

### ⚖️ Advanced Research & Auditing
*   **Fairness Evaluation**: Automated group disparity analysis across demographic attributes (Gender, Domain, Category).
*   **Bias Detection**: Heuristic and statistical detection of biased language patterns and model skew.
*   **Adversarial Testing**: Built-in dataset augmenter to inject noise and distractions, testing model robustness.

### 📊 Professional Analytics Dashboard
*   **Performance Visualization**: Real-time charts for Accuracy vs. Latency trade-offs.
*   **Error Analysis**: Deep-dive into failure cases (hallucinations, incorrect answers) with categorical breakdown.
*   **Experiment History**: Comprehensive logging of reproducible runs with full configuration tracking.

### 🛠️ Developer-First Design
*   **Modular Metric System**: Registry pattern for easy addition of custom evaluation functions.
*   **Pydantic Validation**: Strict schema enforcement for datasets and model configurations.
*   **Full-Stack Ready**: Integrated FastAPI backend and a minimalist React + Tailwind CSS v4 frontend.

## 📁 System Architecture

```text
├── models/             # LLM Provider interfaces (OpenAI, Transformers, Mock)
├── datasets/           # Dataset loaders, MMLU integration, and quality scoring
├── tasks/              # Task types (QA, Summarization, Classification) logic
├── metrics/            # Core metrics (Fairness, NLP scores, Toxicity, Latency)
├── evaluation/         # Central execution engine & parallel orchestration
├── experiments/        # Structured JSON logs of reproducible benchmarks
├── backend/            # FastAPI REST API implementation
├── frontend/           # React v19 + Tailwind v4 + Recharts dashboard
└── run_advanced_research.py  # High-level research benchmarking script
```

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/Sappymukherjee214/LLM-Evaluation-and-Benchmarking-Framework.git
cd LLM-Evaluation-and-Benchmarking-Framework

# Install backend dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
```

### 2. Configure Environment
Create a `.env` file in the root:
```env
OPENAI_API_KEY=your_key_here
```

### 3. Run Your First Benchmark
```bash
# Execute a research-level evaluation
python run_advanced_research.py
```

### 4. Launch the Platform
```bash
# Start Backend
python backend/main.py

# Start Frontend (in a new terminal)
cd frontend
npm run dev
```

## 📈 Supported Metrics

| Category | Metrics |
| :--- | :--- |
| **Productivity** | Accuracy, Response Matching, Throughput |
| **Performance** | Tokens/sec, Latency, Memory Usage |
| **NLP Quality** | ROUGE-L, BLEU, Semantic Similarity |
| **Safety & Trust** | Toxicity Scoring, Bias Index, Group Disparity |

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for more information.

---
*Created with focus on Real-World AI Engineering and MLOps best practices.*
