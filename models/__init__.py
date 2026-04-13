from .base_model import BaseModel
from .huggingface_model import HuggingFaceModel
from .mock_model import MockModel
from .google_model import GoogleModel

def get_model(provider: str, model_id: str, **kwargs) -> BaseModel:
    if provider.lower() in ["huggingface", "hf"]:
        return HuggingFaceModel(model_id, **kwargs)
    elif provider.lower() == "mock":
        return MockModel(model_id, **kwargs)
    elif provider.lower() == "google":
        return GoogleModel(model_id, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
