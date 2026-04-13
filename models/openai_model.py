import os
import time
from typing import Dict, Any
try:
    from openai import OpenAI, AsyncOpenAI
except ImportError:
    OpenAI = None
    AsyncOpenAI = None
from .base_model import BaseModel

class OpenAIModel(BaseModel):
    """
    Wrapper for OpenAI API models.
    """
    
    def __init__(self, model_id: str, config: Dict[str, Any] = None):
        super().__init__(model_id, config)
        if OpenAI is None:
             print(f"Warning: openai package not installed. {model_id} will fail if called.")
             self.client = None
             self.async_client = None
             return

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            # For demonstration purposes, if no API key is found, we might want to log it
            print(f"Warning: OPENAI_API_KEY not found. {model_id} will fail if called.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        if self.client is None:
            return {"error": "openai package not installed", "text": "", "metadata": {}}
        
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            latency = time.time() - start_time
            return {
                "text": response.choices[0].message.content,
                "metadata": {
                    "latency": latency,
                    "usage": response.usage.model_dump(),
                    "model": self.model_id
                }
            }
        except Exception as e:
            return {"error": str(e), "text": "", "metadata": {"latency": time.time() - start_time}}

    async def generate_async(self, prompt: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            latency = time.time() - start_time
            return {
                "text": response.choices[0].message.content,
                "metadata": {
                    "latency": latency,
                    "usage": response.usage.model_dump(),
                    "model": self.model_id
                }
            }
        except Exception as e:
            return {"error": str(e), "text": "", "metadata": {"latency": time.time() - start_time}}
