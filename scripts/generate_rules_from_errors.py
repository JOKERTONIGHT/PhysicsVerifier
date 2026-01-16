import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path to import pure_llm_verifier if needed, 
# but we will use openai directly for flexibility
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import openai
except ImportError:
    print("OpenAI package not found. Please run 'pip install openai'")
    sys.exit(1)

def get_llm_client():
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("Warning: No API key found.")
        return None
    
    return openai.OpenAI(base_url=base_url, api_key=api_key)

def generate_rule_for_error(client, question: str, prediction: str, error: Dict[str, Any], model: str = "gpt-5") -> Dict[str, Any]:
    
    error_type = error.get("type", "unknown")
    error_desc = error.get("description", "")
    
    system_prompt = (
        "You are an expert system designer for a physics verification engine. "
        "Your task is to generate specific verification rules that can detect errors in physics solutions. "
        "The rules should be defined in a structured format compatible with a rule-based verifier."
    )
    
    user_prompt = f"""
A student made the following error in a physics problem:

Problem:
{question[:1000]}... (truncated)

Student Solution Snippet:
{prediction[:1000]}... (truncated)

Identified Error:
Type: {error_type}
Description: {error_desc}

Please generate a specific verification rule that would catch this type of error or similar errors in the future.
The rule should be in JSON format with the following fields:
- `id`: A unique identifier string (e.g., "rule_momentum_01").
- `title`: A short title for the rule.
- `description`: A detailed description of what the rule checks.

Output ONLY the JSON object.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        if not content:
            print(f"Warning: LLM returned empty content for error type: {error_type}")
            return None
            
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print(f"Raw content: {content}")
            return None
        
    except Exception as e:
        print(f"Error generating rule: {e}")
        return None

def main():
    # 使用基于脚本位置的绝对路径，确保在任何目录下运行都能找到文件
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    input_path = project_root / "results" / "pure_llm_eval_results_300.json"
    output_path = project_root / "results" / "rules_300.json"
    
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print(f"Please ensure you have run the evaluation script first.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
        
    client = get_llm_client()
    if not client:
        return

    # 将原始数据加载移到循环外，提高效率
    # 优先尝试加载 300 样本文件，如果不存在则回退到 100
    data_path = project_root / "data" / "evaluation_sample_300.json"
    if not data_path.exists():
        data_path = project_root / "data" / "evaluation_sample_100.json"
    
    print(f"Loading original data from: {data_path}")
    try:
        with open(data_path, 'r', encoding='utf-8') as df:
            data_samples = json.load(df)
            data_map = {s["id"]: s for s in data_samples}
    except FileNotFoundError:
        print(f"Error: Original data file not found at {data_path}")
        return

    generated_rules = []
    
    # Filter for items with errors
    items_with_errors = [item for item in results if item.get("analysis") and item["analysis"].get("errors")]
    
    print(f"Found {len(items_with_errors)} samples with errors. Generating rules...")
    
    for i, item in enumerate(items_with_errors):
        sample_id = item.get("id")
        original_sample = data_map.get(sample_id)
        
        if not original_sample:
            print(f"Warning: Original sample {sample_id} not found in {data_path.name}")
            continue
            
        question = original_sample.get("question", "")
        prediction = original_sample.get("prediction", "")
        
        analysis = item.get("analysis", {})
        errors = analysis.get("errors", [])
        
        print(f"Processing Sample {sample_id} ({i+1}/{len(items_with_errors)})...")
        
        for error in errors:
            rule = generate_rule_for_error(client, question, prediction, error)
            if rule:
                # Add metadata
                rule["source_error_id"] = sample_id
                rule["source_error_type"] = error.get("type")
                generated_rules.append(rule)
                print(f"  Generated rule: {rule.get('title')}")
            
            # Rate limit protection
            time.sleep(1)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(generated_rules, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully generated {len(generated_rules)} rules. Saved to {output_path}")

if __name__ == "__main__":
    main()
