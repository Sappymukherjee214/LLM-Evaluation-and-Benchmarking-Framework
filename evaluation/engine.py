import os
import json
import time
import datetime
import concurrent.futures
from typing import List, Dict, Any, Optional
from pydantic import BaseModel as PydanticBaseModel
from tqdm import tqdm

from models.base_model import BaseModel
from datasets.base_dataset import BaseDataset, DatasetItem
from tasks.base_task import BaseTask

class ExperimentConfig(PydanticBaseModel):
    """
    Configuration for an evaluation run to ensure reproducibility.
    """
    experiment_id: str
    model_id: str
    dataset_name: str
    dataset_version: str
    task_name: str
    prompt_template: Optional[str] = None
    parameters: Dict[str, Any] = {}
    timestamp: str = datetime.datetime.now().isoformat()

class EvaluationEngine:
    """
    High-performance evaluation engine with parallel execution support.
    """
    def __init__(self, experiment_dir: str = "experiments", max_workers: int = 4):
        self.experiment_dir = experiment_dir
        self.max_workers = max_workers
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

    def _process_sample(self, item: DatasetItem, model: BaseModel, task: BaseTask, prompt_template: str) -> Dict[str, Any]:
        """Process a single sample: generate and evaluate."""
        try:
            # 1. Generate
            formatted_prompt = task.format_prompt(item.input, prompt_template)
            response_data = model.generate(formatted_prompt)
            prediction = response_data.get("text", "")
            gen_metadata = response_data.get("metadata", {})
            
            # 2. Evaluate
            sample_results = task.evaluate_sample(prediction, item.expected_output, gen_metadata)
            
            # 3. Error Analysis (Simple)
            error_type = None
            if item.expected_output and item.expected_output.lower() not in prediction.lower():
                error_type = "incorrect_answer"
            
            return {
                "sample_id": item.id,
                "input": item.input,
                "prediction": prediction,
                "expected": item.expected_output,
                "metrics": sample_results,
                "metadata": {**gen_metadata, **item.metadata},
                "error_analysis": {"type": error_type}
            }
        except Exception as e:
            return {"sample_id": item.id, "error": str(e), "metrics": {}}

    def run(self, 
            model: BaseModel, 
            dataset: BaseDataset, 
            task: BaseTask, 
            prompt_template: str = None,
            parallel: bool = True) -> Dict[str, Any]:
        
        start_time = time.time()
        config = ExperimentConfig(
            experiment_id=f"exp_{int(time.time())}",
            model_id=model.model_id,
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            task_name=task.name,
            prompt_template=prompt_template
        )

        print(f"Running Experiment: {config.experiment_id} | Model: {model.model_id}")

        results = []
        if parallel and self.max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(self._process_sample, item, model, task, prompt_template) 
                    for item in dataset.items
                ]
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Samples"):
                    results.append(future.result())
        else:
            for item in tqdm(dataset.items, desc="Processing Samples"):
                results.append(self._process_sample(item, model, task, prompt_template))

        # Performance Monitoring
        total_time = time.time() - start_time
        throughput = len(dataset.items) / total_time if total_time > 0 else 0

        # Aggregate Metrics
        avg_metrics = self._aggregate_metrics(results)
        
        # Package Experiment
        experiment_report = {
            "config": config.model_dump(),
            "performance": {
                "total_runtime_sec": total_time,
                "throughput_samples_per_sec": throughput,
                "max_workers": self.max_workers
            },
            "summary_metrics": avg_metrics,
            "results": results
        }

        # Save to disk
        filepath = os.path.join(self.experiment_dir, f"{config.experiment_id}.json")
        with open(filepath, 'w') as f:
            json.dump(experiment_report, f, indent=4)
            
        print(f"Completed. Throughput: {throughput:.2f} samples/s. Saved to {filepath}")
        return experiment_report

    def _aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        total_metrics = {}
        valid_counts = {}
        
        for res in results:
            if "error" in res: continue
            for m_name, m_val in res.get("metrics", {}).items():
                total_metrics[m_name] = total_metrics.get(m_name, 0) + m_val
                valid_counts[m_name] = valid_counts.get(m_name, 0) + 1
        
        return {m_name: total_metrics[m_name] / valid_counts[m_name] for m_name in total_metrics}
