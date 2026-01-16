import os
import json
import time
import argparse
from typing import Dict, List, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MODEL_NAME = os.getenv("LLM_MODEL", "gpt-4o") 

class PhysicsFrameworkBuilder:
    def __init__(self, output_file: str):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL") # Support custom endpoints
        )
        self.output_file = output_file
        self.framework = {}

    def _call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        """Helper to call the LLM with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=temperature
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response content")
                
                # Clean markdown code blocks if present
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                elif content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                return content
            except Exception as e:
                print(f"Error calling LLM (Attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(2)
        return "{}"

    def step_1_generate_outline(self) -> Dict[str, Any]:
        """Generates the high-level domain and sub-topic structure."""
        print(">>> Step 1: Generating High-Level Outline...")
        
        system_prompt = """You are the Chief Architect of a Physics Verification Engine for IPhO-level competitions.
        Your goal is to design a comprehensive, granular hierarchical ontology of physics knowledge.
        This ontology will structure a rulebase for automatically verifying physics solutions.
        
        Output a JSON object containing a list of 'domains'. 
        Each 'domain' MUST have a 'name' and a non-empty list of 'sub_topics'.
        
        The 'sub_topics' must be specific and granular enough to host precise verification rules.
        Avoid broad categories.
        
        Required Domains:
        1. Mechanics (Kinematics, Dynamics, Statics, Rotational, Celestial, Fluid, etc.)
        2. Electromagnetism (Electrostatics, Circuits, Magnetism, Induction, EM Waves, etc.)
        3. Thermodynamics & Statistical Physics (Laws, Processes, Kinetic Theory, Phase Transitions, etc.)
        4. Optics (Geometric, Wave, Interference, Diffraction, etc.)
        5. Modern Physics (Relativity, Quantum, Nuclear, Particle, etc.)
        6. Experimental Physics (Error Analysis, Data Processing, etc.)
        """
        
        user_prompt = """Generate the detailed hierarchical outline.
        Ensure every domain has at least 5-10 specific sub-topics.
        
        Example Sub-topics for Mechanics:
        - "Kinematics in 1D/2D/3D"
        - "Newton's Laws and Free Body Diagrams"
        - "Work-Energy Theorem and Conservation"
        - "Linear Momentum and Collisions"
        - "Rotational Dynamics and Torque"
        - "Angular Momentum Conservation"
        - "Gravitation and Kepler's Laws"
        - "Non-inertial Reference Frames"
        - "Fluid Statics and Dynamics"
        - "Oscillations and Normal Modes"
        
        JSON Format:
        {
            "domains": [
                {
                    "name": "Mechanics",
                    "sub_topics": ["Kinematics", "Newton's Laws", ...]
                },
                ...
            ]
        }
        """
        
        content = self._call_llm(system_prompt, user_prompt)
        try:
            outline = json.loads(content)
            # Validate structure
            if "domains" not in outline or not outline["domains"]:
                print("Warning: 'domains' key missing or empty in Step 1 output.")
            return outline
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON from Step 1. Content preview: {content[:200]}... Error: {e}")
            return {"domains": []}

    def step_2_critique_and_refine_outline(self, outline: Dict[str, Any]) -> Dict[str, Any]:
        """Critiques the outline and refines it if necessary."""
        print(">>> Step 2: Critiquing and Refining Outline...")
        
        system_prompt = "You are a strict Physics Curriculum Auditor. Your job is to ensure the curriculum is exhaustive and structured for automated verification."
        user_prompt = f"""Review the following outline for completeness regarding the International Physics Olympiad (IPhO) syllabus.
        
        Current Outline:
        {json.dumps(outline, indent=2)}
        
        Tasks:
        1. Ensure all major IPhO topics are covered.
        2. Ensure 'sub_topics' lists are populated and granular (not just "Thermodynamics", but "First Law", "Entropy", "Heat Engines").
        3. If any domain has empty sub_topics, fill them in.
        
        Output the FULLY CORRECTED and POPULATED JSON.
        """
        
        content = self._call_llm(system_prompt, user_prompt, temperature=0.2)
        try:
            refined_outline = json.loads(content)
            return refined_outline
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON from Step 2. Content preview: {content[:200]}... Error: {e}")
            return outline

    def step_3_generate_rules_for_topic(self, domain: str, topic: str) -> List[Dict[str, Any]]:
        """Generates specific rules for a given sub-topic."""
        print(f"  > Generating rules for {domain} -> {topic}...")
        
        system_prompt = """You are an expert Physics Rule Architect.
        Create a comprehensive set of verification rules for the given sub-topic.
        These rules will be used to algorithmically check student solutions for errors.
        
        A 'Rule' consists of:
        - id: unique string (e.g., mech_energy_cons_check_01)
        - title: concise, action-oriented title
        - description: detailed explanation of the physical principle and when it applies.
        - check_logic: The specific condition to verify (e.g., "Check if non-conservative forces do work. If yes, E_final != E_initial").
        - common_errors: list of specific mistakes to flag (e.g., "Forgetting the 1/2 factor in kinetic energy", "Wrong sign for work done by friction").
        
        Design Principles:
        1. **Specificity**: Rules must be actionable. Avoid "Understand Newton's Law". Use "Sum of forces must equal mass times acceleration".
        2. **Completeness**: Cover definitions, sign conventions, conservation laws, constraints, and limiting cases.
        3. **Robustness**: Include checks for validity conditions (e.g., "Bernoulli's equation requires inviscid, incompressible flow").
        """
        
        user_prompt = f"""Generate at least 6-8 high-quality verification rules for:
        Domain: {domain}
        Sub-topic: {topic}
        
        Include rules for:
        - Fundamental Equations & Laws
        - Sign Conventions & Coordinate Systems
        - Constraints (Geometric, Kinematic)
        - Validity Conditions (When can this law be applied?)
        - Dimensional/Unit Consistency Checks specific to this topic
        
        JSON Format:
        {{
            "rules": [
                {{ "id": "...", "title": "...", "description": "...", "check_logic": "...", "common_errors": [...] }},
                ...
            ]
        }}
        """
        
        content = self._call_llm(system_prompt, user_prompt)
        try:
            data = json.loads(content)
            return data.get("rules", [])
        except:
            return []

    def step_4_critique_rules(self, domain: str, topic: str, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Critiques the generated rules for specificity and correctness."""
        # print(f"  > Critiquing rules for {topic}...")
        
        system_prompt = "You are a Quality Assurance Specialist for Physics Verification Rules."
        user_prompt = f"""Review these rules for {domain} - {topic}.
        
        Rules:
        {json.dumps(rules, indent=2)}
        
        Quality Check:
        1. Are the physics principles 100% correct?
        2. Are the 'check_logic' instructions clear enough for a verifier agent to follow?
        3. Are there any duplicates?
        4. Are important edge cases missing?
        
        Return a JSON object with a 'rules' list. 
        Remove incorrect rules. Improve vague rules. Add missing critical rules if necessary.
        Ensure the final list is high-quality and ready for production use.
        """
        
        content = self._call_llm(system_prompt, user_prompt, temperature=0.1)
        try:
            data = json.loads(content)
            return data.get("rules", [])
        except:
            return rules

    def run(self):
        # 1. Outline
        outline = self.step_1_generate_outline()
        outline = self.step_2_critique_and_refine_outline(outline)
        
        # 2. Expand
        full_framework = {"domains": []}
        
        for domain_obj in outline.get("domains", []):
            domain_name = domain_obj["name"]
            sub_topics = domain_obj.get("sub_topics", [])
            
            print(f"\nProcessing Domain: {domain_name}")
            
            domain_result = {
                "name": domain_name,
                "topics": []
            }
            
            for topic in sub_topics:
                # Generate
                rules = self.step_3_generate_rules_for_topic(domain_name, topic)
                # Critique
                rules = self.step_4_critique_rules(domain_name, topic, rules)
                
                domain_result["topics"].append({
                    "name": topic,
                    "rules": rules
                })
                
                # Sleep to avoid rate limits if necessary
                # time.sleep(1)
            
            full_framework["domains"].append(domain_result)
            
            # Intermediate save
            self.save_framework(full_framework)

        print("\n>>> Framework Construction Complete.")

    def save_framework(self, data):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved progress to {self.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="PhysicsVerifier/rules_catalog_top_down.json")
    args = parser.parse_args()
    
    builder = PhysicsFrameworkBuilder(args.output)
    builder.run()
