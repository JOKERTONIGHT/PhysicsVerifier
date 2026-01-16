import json
import argparse

def get_structure(data):
    """Recursively builds a structure of keys from the data."""
    if isinstance(data, dict):
        return {key: get_structure(value) for key, value in data.items()}
    elif isinstance(data, list):
        # If the list is not empty, show the structure of the first element.
        # This assumes all elements in the list have a similar structure.
        if data:
            return [get_structure(data[0])]
        else:
            return []
    else:
        # For primitive types, just show their type.
        return f"<{type(data).__name__}>"

def preview_json_file(file_path, count, keys_only):
    """Prints records from a JSON file, with an option to show only keys."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            display_count = min(count, len(data))
            print(f"Displaying the first {display_count} of {len(data)} records from '{file_path}':")

            if keys_only:
                # Show structure of the first record only, as they are likely uniform.
                if data:
                    print("\n--- Structure of the first record (keys only) ---")
                    structure = get_structure(data[0])
                    print(json.dumps(structure, indent=2, ensure_ascii=False))
                else:
                    print("File contains an empty list.")
            else:
                for i, record in enumerate(data[:display_count]):
                    print(f"\n--- Record {i+1} ---")
                    print(json.dumps(record, indent=2, ensure_ascii=False))
        else:
            print("The JSON file does not contain a list of records. Displaying the whole file:")
            if keys_only:
                structure = get_structure(data)
                print(json.dumps(structure, indent=2, ensure_ascii=False))
            else:
                print(json.dumps(data, indent=2, ensure_ascii=False))
            
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Preview the first N records of a JSON file.")
    parser.add_argument(
        "file_path",
        type=str,
        help="The path to the JSON file to preview."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="The number of records to display (ignored with --keys-only)."
    )
    parser.add_argument(
        "--keys-only",
        action="store_true",
        help="If set, displays only the key structure of the first record."
    )
    args = parser.parse_args()
    
    preview_json_file(args.file_path, args.count, args.keys_only)

if __name__ == "__main__":
    main()
