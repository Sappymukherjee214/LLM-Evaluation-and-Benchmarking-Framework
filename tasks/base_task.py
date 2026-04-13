from abc import ABC, abstractmethod
from typing import Dict, Any, List
from metrics.base_metric import BaseMetric

class BaseTask(ABC):
    def __init__(self, name: str, metrics: List[BaseMetric]):
        self.name = name
        self.metrics = metrics

    @abstractmethod
    def format_prompt(self, input_text: str, template: str = None) -> str:
        """Apply task-specific prompt formatting"""
        pass

    def evaluate_sample(self, prediction: str, reference: str, metadata: Dict[str, Any] = None) -> Dict[str, float]:
        """Compute all metrics for a single sample"""
        results = {}
        for metric in self.metrics:
            result_obj = metric.compute(prediction, reference, **(metadata or {}))
            results[metric.name] = result_obj.value
        return results

class QATask(BaseTask):
    def format_prompt(self, input_text: str, template: str = None) -> str:
        if template:
            return template.replace("{{input}}", input_text)
        return f"Question: {input_text}\nAnswer accurately and concisely:"

class SummarizationTask(BaseTask):
    def format_prompt(self, input_text: str, template: str = None) -> str:
        if template:
            return template.replace("{{input}}", input_text)
        return f"Summarize the following text:\n\n{input_text}\n\nSummary:"

class ClassificationTask(BaseTask):
    def format_prompt(self, input_text: str, template: str = None) -> str:
        if template:
            return template.replace("{{input}}", input_text)
        return f"Classify the following text into categories:\n\n{input_text}\n\nCategory:"
