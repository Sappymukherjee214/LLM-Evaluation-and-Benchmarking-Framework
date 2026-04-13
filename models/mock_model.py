import time
import random
from typing import Dict, Any
from .base_model import BaseModel

class MockModel(BaseModel):
    """
    A mock model for testing and demonstration purposes.
    Returns deterministic or random responses based on the prompt.
    """
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        # Simulate some processing time
        time.sleep(random.uniform(0.1, 0.5))
        
        latency = time.time() - start_time
        
        responses = [
            "This is a simulated response for the given task.",
            "The model suggests that the answer depends on the context provided.",
            "Based on the input, the most likely outcome is positive.",
            "I'm a mock model, but I'm doing my best!",
            "Sample output for: " + prompt[:20] + "..."
        ]
        
        return {
            "text": random.choice(responses),
            "metadata": {
                "latency": latency,
                "model": self.model_id,
                "type": "mock"
            }
        }

    async def generate_async(self, prompt: str, **kwargs) -> Dict[str, Any]:
        return self.generate(prompt, **kwargs)
