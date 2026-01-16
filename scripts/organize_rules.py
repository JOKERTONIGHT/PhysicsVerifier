import json
import os
import argparse
from typing import List, Dict, Any, Union
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class RuleOrganizer:
    def __init__(self, model="gemini-3-flash-preview"):
        self.client = OpenAI()
        self.model = model

    def organize(self, rules: List[Dict[str, Any]], threshold=20) -> Union[Dict, List]:
        """
        Recursively organizes rules into a hierarchy.
        If number of rules > threshold, asks LLM to categorize them.
        """
        if len(rules) <= threshold:
            return rules

        # If list is large, process in batches to avoid context/output limits
        BATCH_SIZE = 50
        if len(rules) > BATCH_SIZE:
            print(f"List too large ({len(rules)}), processing in batches of {BATCH_SIZE}...")
            return self._organize_large_list(rules, threshold, BATCH_SIZE)

        print(f"Organizing batch of {len(rules)} rules...")
        
        # Prepare prompt
        # We use index to reference rules to save tokens and output complexity
        rules_list_str = ""
        for idx, rule in enumerate(rules):
            # Include ID and Title for context
            rules_list_str += f"[{idx}] {rule.get('title', 'No Title')}\n"

        prompt = f"""
You are an expert physics curriculum designer.
Your task is to organize the following list of physics verification rules into a hierarchical knowledge base.

The current list contains {len(rules)} rules.
Please categorize them into distinct, semantically meaningful sub-topics.
The sub-topics should be specific enough to be useful (e.g., "Orbital Mechanics", "Thermodynamics - First Law", "Kinematics").
Aim for 3 to 8 categories.

Output a JSON object where keys are the sub-topic names and values are lists of the indices (integers) of the rules that belong to that sub-topic.
Every index from 0 to {len(rules)-1} must appear exactly once.

Rules:
{rules_list_str}

JSON Output Format:
{{
  "Sub-topic Name 1": [0, 1, 5, ...],
  "Sub-topic Name 2": [2, 3, 4, ...]
}}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            content = response.choices[0].message.content
            mapping = json.loads(content)
            
            # Reconstruct hierarchy
            hierarchy = {}
            
            # Track assigned indices to ensure coverage
            assigned_indices = set()
            
            for topic, indices in mapping.items():
                # Filter valid indices
                valid_indices = [i for i in indices if isinstance(i, int) and 0 <= i < len(rules)]
                
                sub_rules = []
                for i in valid_indices:
                    if i not in assigned_indices:
                        sub_rules.append(rules[i])
                        assigned_indices.add(i)
                
                if sub_rules:
                    # Recurse
                    print(f"  -> Created category '{topic}' with {len(sub_rules)} rules.")
                    hierarchy[topic] = self.organize(sub_rules, threshold)
            
            # Handle unassigned rules
            unassigned_indices = set(range(len(rules))) - assigned_indices
            if unassigned_indices:
                print(f"  Warning: {len(unassigned_indices)} rules were not assigned to any category. Adding to 'Miscellaneous'.")
                misc_rules = [rules[i] for i in sorted(list(unassigned_indices))]
                hierarchy["Miscellaneous"] = self.organize(misc_rules, threshold)
            
            return hierarchy

        except Exception as e:
            print(f"Error during organization: {e}")
            # Fallback: return flat list
            return rules

    def _organize_large_list(self, rules, threshold, batch_size):
        """
        Handles large lists by batching the categorization step.
        """
        chunks = [rules[i:i + batch_size] for i in range(0, len(rules), batch_size)]
        
        rule_to_category = {}
        known_categories = [] # List of category names to encourage reuse
        
        print(f"Processing {len(chunks)} batches...")
        
        for i, chunk in enumerate(chunks):
            print(f"  Batch {i+1}/{len(chunks)}...")
            
            # Prepare prompt for batch
            rules_list_str = ""
            for idx, rule in enumerate(chunk):
                rules_list_str += f"[{idx}] {rule.get('title', 'No Title')}\n"
            
            known_cats_str = ", ".join(known_categories) if known_categories else "None yet"
            
            prompt = f"""
You are organizing a large set of physics rules.
Categorize the following {len(chunk)} rules into semantic topics.
Existing categories you should try to reuse if applicable: {known_cats_str}.
You can create new categories if needed.

Rules:
{rules_list_str}

Output JSON: {{ "Category Name": [indices] }}
"""
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                mapping = json.loads(response.choices[0].message.content)
                
                for cat, indices in mapping.items():
                    if cat not in known_categories:
                        known_categories.append(cat)
                    
                    for local_idx in indices:
                        if isinstance(local_idx, int) and 0 <= local_idx < len(chunk):
                            rule_id = chunk[local_idx]['id']
                            rule_to_category[rule_id] = cat
                            
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                # Assign to Miscellaneous
                for rule in chunk:
                    rule_to_category[rule['id']] = "Miscellaneous"

        # Group by category
        hierarchy = {}
        # Map rule_id back to rule object
        id_to_rule = {r['id']: r for r in rules}
        
        grouped_rules = {}
        for r_id, cat in rule_to_category.items():
            if cat not in grouped_rules:
                grouped_rules[cat] = []
            if r_id in id_to_rule:
                grouped_rules[cat].append(id_to_rule[r_id])
        
        # Handle any rules that might have been missed (shouldn't happen with logic above but good to be safe)
        for rule in rules:
            if rule['id'] not in rule_to_category:
                if "Miscellaneous" not in grouped_rules:
                    grouped_rules["Miscellaneous"] = []
                grouped_rules["Miscellaneous"].append(rule)

        # Recurse on each group
        for cat, sub_rules in grouped_rules.items():
            print(f"  -> Grouped category '{cat}' has {len(sub_rules)} rules. Recursing...")
            hierarchy[cat] = self.organize(sub_rules, threshold)
            
        return hierarchy

def main():
    parser = argparse.ArgumentParser(description="Organize physics rules into a hierarchy using LLM.")
    parser.add_argument("--input", default="PhysicsVerifier/results/rules_300.json", help="Input JSON file")
    parser.add_argument("--output", default="PhysicsVerifier/results/rules_hierarchy.json", help="Output JSON file")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="LLM model to use")
    parser.add_argument("--threshold", type=int, default=15, help="Max rules per leaf node")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file {args.input} not found.")
        return

    print(f"Loading rules from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        rules = json.load(f)

    # Optional: Limit for testing
    # rules = rules[:50]

    organizer = RuleOrganizer(model=args.model)
    hierarchy = organizer.organize(rules, threshold=args.threshold)

    print(f"Saving hierarchy to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(hierarchy, f, indent=2, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    main()
