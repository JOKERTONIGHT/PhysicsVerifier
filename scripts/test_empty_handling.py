
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pure_llm_verifier import PureLLMVerifier

def main():
    # Load test data
    data_path = Path("PhysicsVerifier/data/test_data_for_empty.json")
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return

    with open(data_path, "r") as f:
        samples = json.load(f)

    # Initialize verifier with the requested model
    # Fallback to gemini-1.5-pro if 2.0/3.0 is unavailable
    model_name = "gemini-3-pro-preview" 
    print(f"Initializing verifier with model: {model_name}")
    
    verifier = PureLLMVerifier(model_name=model_name)

    # Run analysis
    print(f"Starting analysis on {len(samples)} samples...")
    results = verifier.analyze_batch(samples)

    # Save results
    output_dir = Path("PhysicsVerifier/results")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "test_empty_results.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
