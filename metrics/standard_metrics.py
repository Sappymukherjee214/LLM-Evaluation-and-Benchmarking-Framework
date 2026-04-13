from .base_metric import BaseMetric, MetricResult, MetricRegistry

@MetricRegistry.register
class AccuracyMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "accuracy"

    def compute(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        if not reference:
            val = 0.0
        else:
            val = 1.0 if reference.lower() in prediction.lower() else 0.0
        return MetricResult(name=self.name, value=val)

@MetricRegistry.register
class LatencyMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "latency"

    def compute(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        val = kwargs.get("latency", 0.0)
        return MetricResult(name=self.name, value=val)
