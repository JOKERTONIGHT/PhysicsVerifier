import json
import os
from collections import Counter
from pathlib import Path

def analyze_file(file_path):
    print(f"Analyzing {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"  File not found: {file_path}")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"  Error: Could not decode JSON file {file_path}")
        return
    except Exception as e:
        print(f"  Error reading file: {e}")
        return

    if not isinstance(data, list):
        print("  Error: JSON content is not a list")
        return

    json_parse_errors = 0
    error_type_counts = Counter()
    total_samples = len(data)
    samples_with_identified_errors = 0

    for item in data:
        # Check for top-level error or analysis-level error indicating failure to parse
        analysis = item.get('analysis')
        
        # Case 1: No analysis field or explicit error in analysis
        if analysis is None:
            # Check if there is a top level error
            if item.get('error'):
                json_parse_errors += 1
            else:
                # If neither analysis nor error, it's a malformed result
                json_parse_errors += 1
            continue
            
        # Case 2: Analysis exists but contains an error message (e.g. "Failed to parse JSON")
        if isinstance(analysis, dict) and analysis.get('error'):
            json_parse_errors += 1
            continue
            
        # Case 3: Valid analysis
        if isinstance(analysis, dict):
            errors = analysis.get('errors', [])
            if errors:
                samples_with_identified_errors += 1
                for err in errors:
                    # Normalize error type to lowercase
                    etype = err.get('type', 'unknown').lower()
                    error_type_counts[etype] += 1

    print(f"  Total Samples: {total_samples}")
    print(f"  JSON/System Errors: {json_parse_errors}")
    print(f"  Samples with Physics Errors Found: {samples_with_identified_errors}")
    print("  Error Type Breakdown:")
    if error_type_counts:
        for etype, count in error_type_counts.most_common():
            print(f"    - {etype}: {count}")
    else:
        print("    (No physics errors found)")
    print("-" * 40)

def main():
    base_dir = Path("PhysicsVerifier/results")
    files_to_analyze = [
        "pure_llm_eval_results_300.json"
    ]

    for filename in files_to_analyze:
        file_path = base_dir / filename
        analyze_file(file_path)

if __name__ == "__main__":
    main()
