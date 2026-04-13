import os
from dotenv import load_dotenv
from models import get_model
from datasets.base_dataset import JSONDataset
from tasks.base_task import QATask
from metrics.standard_metrics import AccuracyMetric, LatencyMetric
from metrics.nlp_metrics import ROUGEMetric, BLEUMetric, ToxicityMetric
from evaluation.engine import EvaluationEngine

# Load environment variables (API keys)
load_dotenv()

def run_sample_evaluation():
    # 1. Initialize Dataset
    dataset = JSONDataset(name="Sample QA")
    dataset_path = os.path.join("datasets", "sample_qa.json")
    dataset.load(dataset_path)

    # 2. Define Task and Metrics
    metrics = [
        AccuracyMetric(),
        LatencyMetric(),
        ROUGEMetric(),
        BLEUMetric(),
        ToxicityMetric()
    ]
    task = QATask(name="Question Answering", metrics=metrics)

    # 3. Setup Models
    # We use Mock models for the demo to ensure it runs without API keys
    model_a = get_model("mock", "GPT-4-Mock")
    model_b = get_model("mock", "Llama-3-Mock")

    # 4. Initialize Evaluation Engine
    engine = EvaluationEngine(experiment_dir="experiments")

    # 5. Run Evaluations
    print("Evaluating Model A...")
    engine.run(model_a, dataset, task)

    print("\nEvaluating Model B...")
    # Optional: Use a specific prompt template for Model B
    custom_template = "Answer this specific question: {{input}}\nShort answer:"
    engine.run(model_b, dataset, task, prompt_template=custom_template)

    print("\nEvaluation Run Complete! Results are in the 'experiments/' folder.")

if __name__ == "__main__":
    run_sample_evaluation()
