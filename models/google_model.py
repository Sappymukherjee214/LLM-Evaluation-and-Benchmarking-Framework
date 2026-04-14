import os
import time
from typing import Dict, Any
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

from .base_model import BaseModel

class GoogleModel(BaseModel):
    """
    Wrapper for Google Gemini models using the NEW google-genai SDK.
    """
    
    def __init__(self, model_id: str, config: Dict[str, Any] = None):
        super().__init__(model_id, config)
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if genai and self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        if not genai:
            return {"error": "google-genai not installed", "text": "", "metadata": {}}
        if not self.client:
            return {"error": "GOOGLE_API_KEY not found or Client init failed", "text": "", "metadata": {}}
            
        start_time = time.time()
        try:
            # New SDK parameter structure
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=kwargs.get("temperature", 0.7),
                    max_output_tokens=kwargs.get("max_output_tokens", 1024),
                )
            )
            
            latency = time.time() - start_time
            
            return {
                "text": response.text,
                "metadata": {
                    "latency": latency,
                    "model": self.model_id,
                    "provider": "google"
                }
            }
        except Exception as e:
            return {"error": str(e), "text": "", "metadata": {"latency": time.time() - start_time}}

    async def generate_async(self, prompt: str, **kwargs) -> Dict[str, Any]:
        # The new SDK has a native async client if needed: client.aio.models...
        # For uniformity with our framework, we keep the sync call in this wrapper
        return self.generate(prompt, **kwargs)
