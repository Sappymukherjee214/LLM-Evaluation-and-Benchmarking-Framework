# 🚀 LLM Evaluation & Benchmarking Framework

A production-ready, modular system for evaluating and benchmarking Large Language Models (LLMs). This framework allows for systematic comparison of models across various datasets, tasks, and performance metrics.

## ✨ Features

- **Multi-Model Support**: Integrated wrappers for OpenAI, HuggingFace (local), and Mock models for testing.
- **Modular Metrics**: Support for Accuracy, BLEU, ROUGE, Latency, Toxicity, and Bias detection.
- **Task Abstraction**: Dedicated logic for QA, Summarization, and Classification tasks.
- **Robust Dataset Pipeline**: JSON-based dataset management with Pydantic validation and versioning.
- **Experiment Tracking**: Automatic logging of every evaluation run with reproducibility in mind.
- **Interactive Dashboard**: Streamlit-based UI to visualize performance trends and side-by-side comparisons.

## 📁 Project Structure

```text
├── models/             # LLM interfaces (OpenAI, HuggingFace, Mock)
├── datasets/           # Dataset loaders and sample data
├── tasks/              # Task-specific prompt formatting and evaluation
├── metrics/            # Independent metric modules
├── evaluation/         # Central evaluation engine logic
├── experiments/        # JSON logs of evaluation runs
├── dashboard/          # Streamlit visualization app
├── configs/            # Configuration files
├── utils/              # Common utilities
├── run_benchmarks.py   # Main entry point for running evaluations
└── requirements.txt    # Dependency list
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_key_here
```

### 3. Run a Benchmark
Execute the sample evaluation script to generate results:
```bash
python run_benchmarks.py
```

### 4. Launch the Dashboard
Visualize the results in your browser:
```bash
streamlit run dashboard/app.py
```

## 🛠️ Extensibility

- **Add a Model**: Create a new class in `models/` inheriting from `BaseModel`.
- **Add a Metric**: Implement `BaseMetric` in the `metrics/` folder.
- **Add a Task**: Extend `BaseTask` in `tasks/` to define new input/output schemas.
