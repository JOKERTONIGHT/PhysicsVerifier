import json
import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path to import pure_llm_verifier
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pure_llm_verifier import PureLLMVerifier

def main():
    parser = argparse.ArgumentParser(description="Run Pure LLM Verifier on a dataset.")
    parser.add_argument("--input", "-i", type=str, default="data/evaluation_sample_300.json", help="Input JSON file path")
    parser.add_argument("--output", "-o", type=str, default="results/pure_llm_eval_results_300.json", help="Output JSON file path")
    parser.add_argument("--model", "-m", type=str, default="gpt-5", help="Model name to use (e.g., gemini-1.5-pro, gpt-4o)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    try:
        with input_path.open("r", encoding="utf-8") as f:
            samples = json.load(f)
            
        if not isinstance(samples, list):
            print("Error: Input JSON must contain a list of items.")
            return
            
        print(f"Loaded {len(samples)} samples from {input_path}")
        
        verifier = PureLLMVerifier(model_name=args.model)
        
        results = verifier.analyze_batch(samples)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"Analysis complete. Results saved to {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
