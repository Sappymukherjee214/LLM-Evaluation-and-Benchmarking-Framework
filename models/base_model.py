from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseModel(ABC):
    """
    Abstract Base Class for all LLM interfaces.
    Ensures a consistent input/output schema across different providers.
    """
    
    def __init__(self, model_id: str, config: Dict[str, Any] = None):
        self.model_id = model_id
        self.config = config or {}

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response for a given prompt.
        Returns a dictionary containing 'text' and 'metadata'.
        """
        pass

    @abstractmethod
    async def generate_async(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Asynchronous generation for a given prompt.
        """
        pass

    def __repr__(self):
        return f"Model({self.model_id})"
