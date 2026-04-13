import time
from typing import Dict, Any
try:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    import torch
except ImportError:
    pipeline = None
    torch = None
from .base_model import BaseModel

class HuggingFaceModel(BaseModel):
    """
    Wrapper for HuggingFace models (Transformers).
    Supports local inference if hardware allows.
    """
    
    def __init__(self, model_id: str, config: Dict[str, Any] = None):
        super().__init__(model_id, config)
        self.device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        self.pipe = None
        
    def _initialize_pipeline(self):
        if self.pipe is None and pipeline is not None and torch is not None:
            self.pipe = pipeline(
                "text-generation", 
                model=self.model_id, 
                device=self.device,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        if pipeline is None:
            return {"error": "transformers not installed", "text": "", "metadata": {}}
            
        self._initialize_pipeline()
        start_time = time.time()
        
        try:
            # Setting default max_new_tokens if not provided
            max_new_tokens = kwargs.pop("max_new_tokens", 512)
            
            output = self.pipe(prompt, max_new_tokens=max_new_tokens, **kwargs)
            latency = time.time() - start_time
            
            # Extract generated text (handle pipeline output format)
            gen_text = output[0]['generated_text']
            # Common pattern: remove prompt from output if it's prepended
            if gen_text.startswith(prompt):
                gen_text = gen_text[len(prompt):].strip()

            return {
                "text": gen_text,
                "metadata": {
                    "latency": latency,
                    "model": self.model_id,
                    "device": self.device
                }
            }
        except Exception as e:
            return {"error": str(e), "text": "", "metadata": {"latency": time.time() - start_time}}

    async def generate_async(self, prompt: str, **kwargs) -> Dict[str, Any]:
        # Transformers pipeline is synchronous, so we just run it
        return self.generate(prompt, **kwargs)
