import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError

class DatasetItem(BaseModel):
    """
    Schema for a single item in a dataset.
    """
    id: str = Field(..., description="Unique identifier for the sample")
    input: str = Field(..., description="The query or prompt to be sent to the model")
    expected_output: Optional[str] = Field(None, description="The ground truth or ideal response")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context info")

class BaseDataset(ABC):
    """
    Abstract Base Class for datasets.
    """
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.items: List[DatasetItem] = []

    @abstractmethod
    def load(self, source: str):
        """Load dataset from source (file, DB, etc.)"""
        pass

    def validate(self) -> bool:
        """Validate all items in the dataset"""
        try:
            for item in self.items:
                DatasetItem(**item.model_dump() if isinstance(item, DatasetItem) else item)
            return True
        except ValidationError as e:
            print(f"Validation error in dataset {self.name}: {e}")
            return False

    def to_json(self, path: str):
        """Save dataset to JSON"""
        data = {
            "name": self.name,
            "version": self.version,
            "items": [item.model_dump() for item in self.items]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

class JSONDataset(BaseDataset):
    """
    Implementation for loading datasets from JSON files.
    """
    def load(self, source: str):
        with open(source, 'r') as f:
            data = json.load(f)
            self.name = data.get("name", self.name)
            self.version = data.get("version", self.version)
            self.items = [DatasetItem(**item) for item in data.get("items", [])]
        print(f"Loaded {len(self.items)} items from {source}")
