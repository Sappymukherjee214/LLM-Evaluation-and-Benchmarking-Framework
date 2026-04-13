import os
from dotenv import load_dotenv
from models import get_model
from datasets.base_dataset import JSONDataset
from tasks.base_task import QATask
from metrics.standard_metrics import AccuracyMetric, LatencyMetric
from evaluation.engine import EvaluationEngine

# Load environment variables
load_dotenv()

def run_mmlu_evaluation():
    # 1. Initialize MMLU Dataset
    dataset = JSONDataset(name="MMLU (Subsampled)")
    dataset_path = os.path.join("datasets", "mmlu_sample.json")
    dataset.load(dataset_path)

    # 2. Define Task and Metrics
    # For MMLU, accuracy is the most important
    metrics = [
        AccuracyMetric(),
        LatencyMetric()
    ]
    task = QATask(name="MMLU Multiple Choice", metrics=metrics)

    # 3. Setup Models
    engine = EvaluationEngine(experiment_dir="experiments")
    
    # Run with Mock models to demonstrate
    models = [
        get_model("mock", "GPT-4-Turbo-Mock"),
        get_model("mock", "Claude-3-Opus-Mock")
    ]

    for model in models:
        print(f"Benchmarking {model.model_id} on MMLU...")
        engine.run(model, dataset, task)

    print("\nMMLU Evaluation Run Complete!")

if __name__ == "__main__":
    run_mmlu_evaluation()
