import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import datetime

from rule_based_verifier import RuleBasedVerifier
from pure_llm_verifier import PureLLMVerifier

class TopDownVerifier:
    def __init__(
        self,
        rules_catalog_path: str = "rules_catalog_top_down.json",
        llm_model: str = "qwen3-30b-a3b",
        fallback_model: str = "qwen3-30b-a3b",
        log_dir: str = "logs",
        results_dir: str = "results"
    ):
        self.rules_catalog_path = rules_catalog_path
        self.llm_model = llm_model
        self.fallback_model = fallback_model
        self.log_dir = Path(log_dir)
        self.results_dir = Path(results_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.catalog = self._load_catalog()
        self.topics = self._flatten_topics()
        
        # Initialize the base verifiers
        # We will dynamically update rules for the rule-based verifier
        self.rule_verifier = RuleBasedVerifier(
            llm_model=self.llm_model,
            rule_mode='srd', # We will inject SRDs dynamically
            rule_translations_path="rule_translations.json" # Dummy path, we'll overwrite
        )
        # Clear initial translations as we will set them per request
        self.rule_verifier.rule_translations = {} 
        
        self.pure_verifier = PureLLMVerifier(
            model_name=self.fallback_model
        )
        
        self.error_experiences = []

    def _load_catalog(self) -> Dict:
        with open(self.rules_catalog_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _flatten_topics(self) -> List[Dict]:
        topics = []
        for domain in self.catalog.get("domains", []):
            domain_name = domain.get("name")
            for topic in domain.get("topics", []):
                t = topic.copy()
                t["domain"] = domain_name
                topics.append(t)
        return topics

    def classify_topic(self, question: str) -> Optional[Dict]:
        # Use LLM to classify the question into one of the topics
        topic_list_str = "\n".join([f"- {t['domain']}: {t['name']}" for t in self.topics])
        
        prompt = f"""
You are a physics expert. Classify the following physics problem into one of the provided domains and topics.
Return ONLY the JSON object with "domain" and "topic" keys.

Problem:
{question[:1000]}...

Available Topics:
{topic_list_str}

JSON Output:
"""
        # We can use the rule_verifier's LLM method for this
        response = self.rule_verifier._llm_json(
            system_prompt="You are a classifier.",
            user_prompt=prompt
        )
        
        if isinstance(response, dict):
            domain = response.get("domain")
            topic_name = response.get("topic")
            
            # Find the matching topic object
            for t in self.topics:
                if t["name"] == topic_name and (not domain or t["domain"] == domain):
                    return t
            
            # Fallback: try fuzzy match or just return None
            # For now, strict match on topic name
            for t in self.topics:
                if t["name"] == topic_name:
                    return t
                    
        return None

    def verify(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        question = sample.get("question", "")
        
        # 1. Classify
        topic = self.classify_topic(question)
        
        diagnostics = []
        used_rules = []
        verifier_used = "pure_llm"
        
        if topic:
            print(f"Classified into: {topic['domain']} - {topic['name']}")
            rules = topic.get("rules", [])
            
            if rules:
                # 2. Prepare Rule Verifier
                # Convert catalog rules to the format expected by RuleBasedVerifier
                # We use 'check_logic' as the SRD/description for the LLM
                current_translations = {}
                rule_ids = []
                for r in rules:
                    rid = r.get("id")
                    if not rid: continue
                    rule_ids.append(rid)
                    current_translations[rid] = {
                        "srd": f"Title: {r.get('title')}\nDescription: {r.get('description')}\nCheck Logic: {r.get('check_logic')}"
                    }
                
                self.rule_verifier.rules_to_check = rule_ids
                self.rule_verifier.rule_translations = current_translations
                
                # 3. Run Rule Check
                print(f"Running rule check with {len(rule_ids)} rules...")
                result = self.rule_verifier.analyze(sample)
                diagnostics = result.get("diagnostics", [])
                used_rules = rule_ids
                verifier_used = "rule_based"
        else:
            print("Could not classify topic or no topic found.")

        # 4. Fallback if no diagnostics found (or no rules applied)
        if not diagnostics:
            print("No rule violations found (or no rules). Falling back to Pure LLM...")
            verifier_used = "pure_llm_fallback"
            pure_result = self.pure_verifier.analyze(sample)
            analysis = pure_result.get("analysis", {})
            
            # Convert pure LLM errors to diagnostics format
            if analysis and isinstance(analysis, dict):
                errors = analysis.get("errors", [])
                for err in errors:
                    diagnostics.append({
                        "rule": "pure_llm_fallback",
                        "severity": err.get("severity", "major"),
                        "message": err.get("description", ""),
                        "location": "unknown"
                    })
                
                # 5. Generate Error Experience
                if errors:
                    self._record_error_experience(sample, topic, errors)
            
        return {
            "id": sample.get("id"),
            "topic": topic.get("name") if topic else None,
            "verifier": verifier_used,
            "diagnostics": diagnostics,
            "score": -1.0 * len(diagnostics) # Simple scoring
        }

    def _record_error_experience(self, sample: Dict, topic: Optional[Dict], errors: List[Dict]):
        experience = {
            "timestamp": datetime.datetime.now().isoformat(),
            "sample_id": sample.get("id"),
            "topic": topic.get("name") if topic else "Unknown",
            "question_snippet": sample.get("question", "")[:200],
            "errors_found": errors
        }
        self.error_experiences.append(experience)
        
        # Append to file immediately
        exp_file = self.results_dir / "error_experiences.json"
        
        all_exps = []
        if exp_file.exists():
            try:
                with open(exp_file, 'r') as f:
                    all_exps = json.load(f)
            except:
                pass
        
        all_exps.append(experience)
        
        with open(exp_file, 'w') as f:
            json.dump(all_exps, f, indent=2, ensure_ascii=False)

    def run_batch(self, samples: List[Dict]):
        results = []
        for s in samples:
            print(f"Verifying sample {s.get('id')}...")
            res = self.verify(s)
            results.append(res)
        return results

if __name__ == "__main__":
    # Test run
    import sys
    
    # Load sample data
    data_path = "data/evaluation_sample_30.json"
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        
    try:
        with open(data_path, 'r') as f:
            samples = json.load(f)
    except FileNotFoundError:
        print(f"File {data_path} not found.")
        sys.exit(1)

    # Limit to first few for testing if needed, or run all
    # samples = samples[:1] 

    verifier = TopDownVerifier()
    results = verifier.run_batch(samples)
    
    # Save results
    with open("results/top_down_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Done. Results saved to results/top_down_results.json")
