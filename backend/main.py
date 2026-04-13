import os
import sys
import glob
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add root to sys.path to import the framework
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import get_model
from datasets.base_dataset import JSONDataset
from tasks.base_task import QATask, SummarizationTask, ClassificationTask
from metrics.standard_metrics import AccuracyMetric, LatencyMetric
from metrics.nlp_metrics import ToxicityMetric
from evaluation.engine import EvaluationEngine

app = FastAPI(title="LLM Benchmarking API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schemas ---

class EvalRequest(BaseModel):
    model_provider: str
    model_id: str
    dataset_name: str
    dataset_path: str
    task_type: str
    prompt_template: Optional[str] = None

class PromptRequest(BaseModel):
    prompt: str
    models: List[Dict[str, str]] # [{"provider": "mock", "id": "gpt-4"}]

# --- Endpoints ---

@app.get("/models")
def list_models():
    return {
        "providers": ["openai", "huggingface", "mock"],
        "popular": ["gpt-4", "gpt-3.5-turbo", "llama-3", "mistral-7b", "mock-model-a"]
    }

@app.get("/datasets")
def list_datasets():
    dataset_files = glob.glob("../datasets/*.json")
    return [os.path.basename(f) for f in dataset_files]

@app.get("/experiments")
def list_experiments():
    experiment_files = glob.glob("../experiments/*.json")
    experiments = []
    for f in experiment_files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                # Just return summary for listing
                experiments.append({
                    "id": data["config"]["experiment_id"],
                    "model": data["config"]["model_id"],
                    "dataset": data["config"]["dataset_name"],
                    "accuracy": data["summary_metrics"].get("accuracy", 0),
                    "latency": data["summary_metrics"].get("latency", 0),
                    "timestamp": data["config"]["timestamp"]
                })
        except:
            continue
    return sorted(experiments, key=lambda x: x["timestamp"], reverse=True)

@app.post("/evaluate")
def run_evaluation(req: EvalRequest):
    try:
        # 1. Setup Model
        model = get_model(req.model_provider, req.model_id)
        
        # 2. Setup Dataset
        ds = JSONDataset(name=req.dataset_name)
        ds.load(os.path.join("../datasets", req.dataset_path))
        
        # 3. Setup Task
        metrics = [AccuracyMetric(), LatencyMetric(), ToxicityMetric()]
        if req.task_type == "qa":
            task = QATask(name="QA", metrics=metrics)
        elif req.task_type == "summarization":
            task = SummarizationTask(name="Summarization", metrics=metrics)
        else:
            task = ClassificationTask(name="Classification", metrics=metrics)
            
        # 4. Run Engine
        engine = EvaluationEngine(experiment_dir="../experiments")
        result = engine.run(model, ds, task, prompt_template=req.prompt_template)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test-prompt")
def test_prompt(req: PromptRequest):
    results = []
    for mod_info in req.models:
        try:
            model = get_model(mod_info["provider"], mod_info["id"])
            res = model.generate(req.prompt)
            results.append({
                "model": mod_info["id"],
                "output": res.get("text", ""),
                "latency": res.get("metadata", {}).get("latency", 0)
            })
        except Exception as e:
            results.append({"model": mod_info["id"], "output": f"Error: {str(e)}", "latency": 0})
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
