import os
from dotenv import load_dotenv
from models import get_model

load_dotenv()

def test_gemini():
    # Attempting to use the requested id, falling back to 1.5-flash for safety
    model_id = "gemini-1.5-flash" # Change to "gemini-2.5-flash" if you have access
    
    print(f"Testing model: {model_id}...")
    model = get_model("google", model_id)
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in environment.")
        return

    res = model.generate("Hello Gemini! Tell me a fun fact about AI evaluation.")
    
    if "error" in res:
        print(f"Run failed: {res['error']}")
    else:
        print("\nGemini Response:")
        print(res["text"])
        print(f"\nLatency: {res['metadata']['latency']:.2f}s")

if __name__ == "__main__":
    test_gemini()
