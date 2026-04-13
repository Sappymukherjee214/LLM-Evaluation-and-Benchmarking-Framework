import numpy as np
from typing import Dict, List, Any
from .base_metric import BaseMetric, MetricResult, MetricRegistry

class FairnessEvaluator:
    """
    Evaluates fairness by comparing metric performance across different demographic groups.
    """
    def __init__(self, protected_attribute: str):
        self.protected_attribute = protected_attribute

    def compute_group_metrics(self, results: List[Dict[str, Any]], metric_name: str) -> Dict[str, float]:
        """
        Calculates the average of a specific metric for each group.
        """
        group_data = {}
        for res in results:
            # Metadata should contain the protected attribute info
            group = res.get("metadata", {}).get(self.protected_attribute, "unknown")
            metric_val = res.get("metrics", {}).get(metric_name, 0.0)
            
            if group not in group_data:
                group_data[group] = []
            group_data[group].append(metric_val)
            
        return {group: np.mean(vals) for group, vals in group_data.items()}

    def calculate_disparity(self, group_scores: Dict[str, float]) -> float:
        """
        Measures disparity as the difference between max and min group scores.
        Lower is better (more fair).
        """
        if not group_scores:
            return 0.0
        scores = list(group_scores.values())
        return max(scores) - min(scores)

@MetricRegistry.register
class BiasDetectionMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "bias_index"

    def compute(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        """
        Simple heuristic for biased language patterns.
        In production, this would use a fine-tuned BERT model for bias detection.
        """
        # Example biased markers
        biased_markers = ["always", "never", "obviously", "everyone knows", "typical"]
        prediction_lower = prediction.lower()
        
        matches = [m for m in biased_markers if m in prediction_lower]
        score = len(matches) / len(biased_markers)
        
        return MetricResult(
            name=self.name,
            value=score,
            metadata={"matched_markers": matches}
        )
