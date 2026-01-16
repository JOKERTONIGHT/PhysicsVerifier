import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path
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

def generate_rule_from_experience(client, experience: Dict[str, Any], model: str = "qwen3-30b-a3b") -> Optional[Dict[str, Any]]:
    question = experience.get("question_snippet", "")
    errors = experience.get("errors_found", [])
    
    if not errors:
        return None
        
    # Combine errors into a summary
    error_summary = "\n".join([f"- Type: {e.get('type')}, Desc: {e.get('description')}" for e in errors])
    
    system_prompt = (
        "You are an expert physics curriculum designer. "
        "Your task is to create a new verification rule based on observed student errors. "
        "The rule will be added to a rule-based verification system."
    )
    
    user_prompt = f"""
We have observed the following errors in a student's solution to a physics problem:

Problem Snippet:
{question}

Errors Observed:
{error_summary}

Please generate a new verification rule that would catch these types of errors.
The rule must be a JSON object with the following fields:
- `id`: A unique identifier string (e.g., "rule_topic_01").
- `title`: A short, descriptive title.
- `description`: A clear description of the rule.
- `check_logic`: Instructions for an AI on how to check this rule.
- `common_errors`: A list of strings describing common violations of this rule.
- `suggested_topic`: The name of the physics topic this rule belongs to (e.g., "Kinematics", "Thermodynamics", etc.).

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
        return json.loads(content)
        
    except Exception as e:
        print(f"Error generating rule: {e}")
        return None

def update_catalog(catalog: Dict, new_rules: List[Dict]):
    # Helper to find topic
    def find_topic(name):
        for d in catalog.get("domains", []):
            for t in d.get("topics", []):
                if t.get("name") == name:
                    return t
        return None

    for rule in new_rules:
        topic_name = rule.get("suggested_topic")
        # Remove suggested_topic from rule object before adding
        rule_to_add = {k: v for k, v in rule.items() if k != "suggested_topic"}
        
        # Ensure ID is unique
        # (Simple check, in production we'd be more robust)
        import uuid
        if not rule_to_add.get("id"):
            rule_to_add["id"] = f"gen_{uuid.uuid4().hex[:8]}"
            
        topic = find_topic(topic_name)
        if topic:
            if "rules" not in topic:
                topic["rules"] = []
            # Check if rule with same ID exists
            existing_ids = {r.get("id") for r in topic["rules"]}
            if rule_to_add["id"] not in existing_ids:
                topic["rules"].append(rule_to_add)
                print(f"Added rule '{rule_to_add['title']}' to topic '{topic_name}'")
            else:
                print(f"Rule ID {rule_to_add['id']} already exists in {topic_name}")
        else:
            print(f"Topic '{topic_name}' not found for rule '{rule_to_add['title']}'. Skipping or adding to General.")
            # Optionally add to a "General" topic in "General Physics" domain
            # For now, just log.

def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    exp_path = project_root / "results" / "error_experiences.json"
    catalog_path = project_root / "rules_catalog_top_down.json"
    
    if not exp_path.exists():
        print("No error experiences found.")
        return

    with open(exp_path, 'r') as f:
        experiences = json.load(f)
        
    if not catalog_path.exists():
        print("Catalog not found.")
        return
        
    with open(catalog_path, 'r') as f:
        catalog = json.load(f)
        
    client = get_llm_client()
    if not client:
        return

    new_rules = []
    print(f"Processing {len(experiences)} experiences...")
    for exp in experiences:
        # Only process if we haven't processed this sample before? 
        # For now, process all. In real system, we'd track status.
        rule = generate_rule_from_experience(client, exp)
        if rule:
            new_rules.append(rule)
            
    update_catalog(catalog, new_rules)
    
    # Save updated catalog
    with open(catalog_path, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
        
    print(f"Updated catalog saved to {catalog_path}")

if __name__ == "__main__":
    main()
