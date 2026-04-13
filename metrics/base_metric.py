from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel as PydanticBaseModel

class MetricResult(PydanticBaseModel):
    name: str
    value: float
    metadata: Dict[str, Any] = {}

class BaseMetric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        """
        Returns a MetricResult object containing the computed value and metadata.
        """
        pass

class MetricRegistry:
    _metrics = {}

    @classmethod
    def register(cls, metric_class):
        instance = metric_class()
        cls._metrics[instance.name] = instance
        return metric_class

    @classmethod
    def get(cls, name: str) -> Optional[BaseMetric]:
        return cls._metrics.get(name)

    @classmethod
    def list_available(cls) -> List[str]:
        return list(cls._metrics.keys())
