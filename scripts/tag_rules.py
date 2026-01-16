#!/usr/bin/env python3
"""
Script to tag physics rules using an LLM.
Tags include: Domain, Type, and Topic.
"""

import os
import json
import argparse
import time
from typing import List, Dict, Any
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
# Default model
DEFAULT_MODEL = "gpt-oss-120b"

SYSTEM_PROMPT = """You are an expert physics educator and curriculum designer. 
Your task is to categorize physics rules into structured tags.
For each rule provided, analyze its title and description and assign the following tags:

1. **Domain**: The broad field of physics. Choose ONE from:
   - Mechanics
   - Electromagnetism
   - Thermodynamics
   - Optics
   - Modern Physics
   - Relativity
   - Quantum Mechanics
   - Mathematical Physics
   - Other

2. **Type**: The nature of the rule. Choose ONE from:
   - Principle/Law (Fundamental physical laws, e.g., Conservation of Energy)
   - Formula (Specific equations or mathematical relationships)
   - Problem Solving Strategy (Heuristics, checks, or methods)
   - Approximation (Rules about when and how to approximate)
   - Definition (Clarification of terms or sign conventions)
   - Check (Sanity checks, dimensional analysis, limiting cases)

3. **Topic**: A specific sub-topic (1-3 words). Examples: "Orbital Mechanics", "Circuit Analysis", "Diffraction", "Kinematics".

Output your response as a JSON object mapping rule IDs to their tags.
Format:
{
    "rule_id_1": {
        "domain": "Mechanics",
        "type": "Formula",
        "topic": "Orbital Mechanics"
    },
    ...
}
"""

def load_rules(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_rules(rules: List[Dict[str, Any]], filepath: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)

def tag_batch(client: OpenAI, model: str, rules_batch: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """
    Tags a batch of rules using the LLM.
    """
    # Prepare the user message
    rules_text = ""
    for rule in rules_batch:
        rules_text += f"ID: {rule['id']}\nTitle: {rule['title']}\nDescription: {rule['description']}\n\n"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Please tag the following rules:\n\n{rules_text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Error tagging batch: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Tag physics rules using LLM.")
    parser.add_argument("--input", default="PhysicsVerifier/results/rules_300.json", help="Input JSON file")
    parser.add_argument("--output", default="PhysicsVerifier/results/rules_300_tagged.json", help="Output JSON file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model to use")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of rules per LLM call")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rules to process (for testing)")
    
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Initialize OpenAI client
    # Assumes OPENAI_API_KEY is set in environment
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment variables.")
    
    client = OpenAI()

    print(f"Loading rules from {args.input}...")
    rules = load_rules(args.input)
    
    if args.limit:
        rules = rules[:args.limit]
        print(f"Limiting to first {args.limit} rules.")

    print(f"Processing {len(rules)} rules...")
    
    tagged_rules = []
    
    # Process in batches
    for i in tqdm(range(0, len(rules), args.batch_size)):
        batch = rules[i : i + args.batch_size]
        tags_map = tag_batch(client, args.model, batch)
        
        for rule in batch:
            rule_id = rule['id']
            if rule_id in tags_map:
                rule['tags'] = tags_map[rule_id]
            else:
                print(f"Warning: No tags returned for rule {rule_id}")
                rule['tags'] = {"domain": "Unknown", "type": "Unknown", "topic": "Unknown"}
            tagged_rules.append(rule)
            
        # Sleep briefly to avoid rate limits if necessary
        # time.sleep(0.5)

    print(f"Saving tagged rules to {args.output}...")
    save_rules(tagged_rules, args.output)

    # Calculate statistics
    stats = {
        "domain": {},
        "type": {}
    }

    for rule in tagged_rules:
        tags = rule.get('tags', {})
        for key in stats:
            val = tags.get(key, "Unknown")
            stats[key][val] = stats[key].get(val, 0) + 1

    print("\n=== Tag Statistics ===")
    for category, counts in stats.items():
        print(f"\n{category.capitalize()}:")
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for val, count in sorted_counts:
            print(f"  {val}: {count}")

if __name__ == "__main__":
    main()
