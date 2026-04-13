from .base_model import BaseModel
from .openai_model import OpenAIModel
from .huggingface_model import HuggingFaceModel
from .mock_model import MockModel

def get_model(provider: str, model_id: str, **kwargs) -> BaseModel:
    if provider.lower() == "openai":
        return OpenAIModel(model_id, **kwargs)
    elif provider.lower() in ["huggingface", "hf"]:
        return HuggingFaceModel(model_id, **kwargs)
    elif provider.lower() == "mock":
        return MockModel(model_id, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
