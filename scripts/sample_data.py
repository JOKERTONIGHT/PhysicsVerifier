import json
import random
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Randomly sample items from a JSON file.")
    parser.add_argument("--input", "-i", type=str, default="data/evaluation_input.json", help="Input JSON file path")
    parser.add_argument("--output", "-o", type=str, default="data/evaluation_sample_300.json", help="Output JSON file path")
    parser.add_argument("--count", "-n", type=int, default=300, help="Number of items to sample")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    try:
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            print("Error: Input JSON must contain a list of items.")
            return
            
        total_items = len(data)
        sample_count = min(args.count, total_items)
        
        random.seed(args.seed)
        sampled_data = random.sample(data, sample_count)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(sampled_data, f, ensure_ascii=False, indent=2)
            
        print(f"Successfully sampled {sample_count} items from {total_items} total items.")
        print(f"Saved to {output_path}")
        
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {input_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
