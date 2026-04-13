import os
import time
from typing import Dict, Any
try:
    import google.generativeai as genai
except ImportError:
    genai = None

from .base_model import BaseModel

class GoogleModel(BaseModel):
    """
    Wrapper for Google Gemini models.
    """
    
    def __init__(self, model_id: str, config: Dict[str, Any] = None):
        super().__init__(model_id, config)
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if genai and self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_id)
        else:
            self.model = None

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        if not genai:
            return {"error": "google-generativeai not installed", "text": "", "metadata": {}}
        if not self.api_key:
            return {"error": "GOOGLE_API_KEY not found", "text": "", "metadata": {}}
            
        start_time = time.time()
        try:
            # Setting default generation config if needed
            generation_config = kwargs.get("generation_config", {
                "temperature": 0.7,
                "top_p": 0.95,
                "max_output_tokens": 1024,
            })
            
            response = self.model.generate_content(prompt, generation_config=generation_config)
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
        # Google SDK supports async but for simplicity in this wrapper we use sync
        return self.generate(prompt, **kwargs)
