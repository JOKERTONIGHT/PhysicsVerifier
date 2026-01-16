import argparse
import json
from pathlib import Path

def convert_combined_to_eval_format(input_file: Path, output_file: Path):
    """
    Converts data from combined_language_only.json format to a format
    similar to ipho25-0_eval.json.
    """
    try:
        with input_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_file}'")
        return

    if not isinstance(data, list):
        print("Error: Input data is not a list of records.")
        return

    converted_data = []
    for record in data:
        if not isinstance(record.get("samples"), list):
            continue

        for i, sample in enumerate(record["samples"]):
            try:
                metadata = sample.get("metadata", {})
                reward = sample.get("reward", {})
                original_id = metadata.get("id", "unknown")
                
                # Create a new unique ID by combining original ID and sample index
                new_id = f"{original_id}_{i}"

                new_record = {
                    "id": new_id,
                    "question": metadata.get("question"),
                    "prediction": sample.get("response"),
                    "answer": reward.get("extracted_gt"),
                }
                converted_data.append(new_record)
            except (AttributeError, KeyError, IndexError) as e:
                print(f"Skipping a sample due to missing data or unexpected structure: {e}")
                continue
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully converted {len(converted_data)} records.")
    print(f"Output saved to '{output_file}'")


def main():
    parser = argparse.ArgumentParser(
        description="Convert combined_language_only.json to an eval-like format."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input JSON file (e.g., combined_language_only.json).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="evaluation_input.json",
        help="Path to the output converted JSON file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output)

    convert_combined_to_eval_format(input_path, output_path)

if __name__ == "__main__":
    main()
