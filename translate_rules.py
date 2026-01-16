from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any
import hashlib
import tempfile

# Ensure the project root is in the Python path
try:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
except IndexError:
    # Fallback for running from a different structure
    sys.path.insert(0, str(Path.cwd()))

from rule_based_verifier import _BUILTIN_RULES_MAP, _load_rule_class

# Use OpenAI API directly
try:
    import openai
    from dotenv import load_dotenv
except ImportError:
    print("Required packages not found. Please run 'pip install openai python-dotenv'")
    sys.exit(1)

# --- Caching mechanism for LLM calls ---
_CACHE = {}
_CACHE_PATH = REPO_ROOT / ".cache" / "rule_translation_cache.json"

def _load_cache():
    global _CACHE
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if _CACHE_PATH.exists():
            _CACHE = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        _CACHE = {}

def _save_cache():
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, dir=_CACHE_PATH.parent, encoding="utf-8") as f:
            json.dump(_CACHE, f, ensure_ascii=False, indent=None)
        os.replace(f.name, str(_CACHE_PATH))
    except Exception:
        pass

def _cache_key(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

def _llm_call_with_cache(client, model, system_prompt, user_prompt, no_cache=False):
    if not openai or not client:
        raise SystemExit("OpenAI client is not available.")

    payload = {"system": system_prompt, "user": user_prompt, "model": model}
    if not no_cache:
        key = f"llm_translate:{_cache_key(payload)}"
        cached = _CACHE.get(key)
        if cached is not None:
            return cached

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # Debug: Print where we are connecting to (masking key)
        # print(f"    [DEBUG] Connecting to: {client.base_url}")
        
        resp_content = None
        
        # Attempt 1: Try with JSON mode
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            if response.choices and response.choices[0].message.content:
                resp_content = response.choices[0].message.content
        except Exception as e:
            print(f"    [DEBUG] Call with response_format failed ({e}).")

        # Attempt 2: Fallback to standard mode if Attempt 1 failed or returned empty
        if not resp_content:
            print(f"    [DEBUG] JSON mode failed or returned empty. Retrying with standard text mode...")
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=1024,
                )
                resp_content = response.choices[0].message.content
            except Exception as e:
                print(f"    [DEBUG] Standard mode call failed: {e}")
                return None
        
        if not resp_content:
            print("    [DEBUG] Received empty response content from LLM (both modes).")
            return None

        try:
            data = json.loads(resp_content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown or text
            import re
            match = re.search(r"\{[\s\S]*\}", resp_content)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    print(f"    [DEBUG] Failed to parse extracted JSON. Raw: {resp_content[:200]}...")
                    return None
            else:
                print(f"    [DEBUG] Response is not valid JSON. Raw: {resp_content[:200]}...")
                return None

        if not no_cache:
            _CACHE[key] = data
            _save_cache()
        return data
    except Exception as e:
        import traceback
        print(f"LLM call failed: {e}")
        print(f"    [DEBUG] Client Base URL: {client.base_url}")
        traceback.print_exc()
        return None


def get_srd_translation_prompt() -> tuple[str, str]:
    """
    Returns the system and user prompts for translating rules into Symbolic Rule Definitions (SRD).
    """
    system_prompt = (
        "You are a physics and logic expert. Your task is to translate a natural language rule "
        "into a clear, structured, and formal Symbolic Rule Definition (SRD) text. "
        "This SRD will be used by another AI to check a student's work. "
        "The SRD must be unambiguous and use a specific grammar. "
        "You must output a JSON object with a single key 'srd' containing the translation."
    )

    user_prompt_template = """
Translate the following rule into our Symbolic Rule Definition (SRD) format.

**SRD Grammar & Concepts:**
- **Keywords**: `FOR EACH`, `WHERE`, `IF`, `THEN`, `AND`, `OR`, `NOT`, `CHECK`, `REPORT`.
- **Entities**:
  - `symbol`: A physical quantity or variable.
  - `formula`: Any mathematical expression found in the text.
  - `equation`: A formula where the relation is '='.
  - `graph`: The dependency graph of symbols.
  - `node`: A node in the graph (representing a symbol).
  - `edge`: A dependency between symbols.
- **Attributes**:
  - `symbol.name`, `symbol.is_defined`, `symbol.usage_count`, `symbol.definition_count`.
  - `symbol.unit`, `symbol.dimension`, `symbol.is_constant`, `symbol.is_variable`.
  - `equation.lhs_symbols`, `equation.rhs_symbols`, `equation.is_dimensionally_consistent`.
  - `graph.has_cycles`, `graph.undefined_nodes`.
- **Operators**: `exists_in`, `count_is`, `is`, `is_not`, `contains`, `>`, `<`, `==`.
- **Structure**: You can write multiple logical checks separated by newlines or semicolons. If the rule is complex, break it down into steps.

**Rule to Translate:**
- **ID**: {rule_id}
- **Title**: {rule_title}
- **Description**: {rule_description}

**Examples:**

*Example 1: Self-Reference Check*
- **Rule**: "A symbol should not be used on both sides of an equation it defines."
- **SRD**: "FOR EACH equation: FOR EACH symbol IN equation.lhs_symbols: IF symbol exists_in equation.rhs_symbols THEN REPORT violation 'self_reference'."

*Example 2: Undefined Symbol Check*
- **Rule**: "Symbols used multiple times must be defined."
- **SRD**: "FOR EACH symbol: IF symbol.usage_count > 1 AND NOT symbol.is_defined THEN REPORT violation 'undefined_symbol'."

*Example 3: Dimensional Consistency*
- **Rule**: "All equations must be dimensionally consistent."
- **SRD**: "FOR EACH equation: IF NOT equation.is_dimensionally_consistent THEN REPORT violation 'dimensional_error'."

**Your Task:**
Provide a JSON object with the key "srd" containing the SRD string for the rule provided above. Ensure the SRD captures all aspects of the rule description.
"""
    return system_prompt, user_prompt_template


def translate_all_rules(model: str, max_llm_calls: int, no_cache: bool) -> dict:
    """
    Translates all built-in rules into Symbolic Rule Definition (SRD) strings.
    """
    # Load environment variables from .env file.
    # This will not override existing environment variables.
    load_dotenv()
    
    try:
        # The client automatically reads OPENAI_API_KEY and OPENAI_BASE_URL/OPENAI_API_BASE
        # from environment variables.
        # Explicitly pass base_url if OPENAI_API_BASE is set, as openai-python v1+ prefers OPENAI_BASE_URL
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        client = openai.OpenAI(base_url=base_url)
        
        # The OpenAI client lazy-loads the API key, so we must check it explicitly.
        if not client.api_key:
            raise ValueError(
                "OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable, "
                "or pass it to the client. Note that the variable name is case-sensitive."
            )
    except Exception as e:
        raise SystemExit(f"Failed to initialize OpenAI client: {e}")

    if not no_cache:
        _load_cache()

    system_prompt, user_prompt_template = get_srd_translation_prompt()
    translations = {}
    llm_calls = 0

    print("Translating built-in rules into Symbolic Rule Definitions (SRD)...")

    for rule_id, rule_spec in _BUILTIN_RULES_MAP.items():
        if llm_calls >= max_llm_calls:
            print("Reached maximum LLM calls. Stopping translation.")
            break
        try:
            rule_class = _load_rule_class(rule_spec)
            rule_instance = rule_class()
            
            print(f"  - Translating rule: '{rule_id}'...")
            
            user_prompt = user_prompt_template.format(
                rule_id=rule_instance.id,
                rule_title=rule_instance.title,
                rule_description=rule_instance.description,
            )
            
            response_data = _llm_call_with_cache(client, model, system_prompt, user_prompt, no_cache)
            llm_calls += 1
            
            srd_text = ""
            if response_data:
                if isinstance(response_data.get("srd"), str):
                    srd_text = response_data["srd"]
                else:
                    print(f"    [DEBUG] Unexpected response format for '{rule_id}': {json.dumps(response_data, ensure_ascii=False)}")
            else:
                print(f"    [DEBUG] No response data for '{rule_id}'")
            
            translations[rule_id] = {
                "title": rule_instance.title,
                "description": rule_instance.description,
                "srd": srd_text.strip() if srd_text else "Translation failed or returned empty.",
            }
        except Exception as e:
            print(f"    - Failed to load or translate rule '{rule_id}': {e}")
            translations[rule_id] = {"error": str(e)}
            
    return translations


def main():
    parser = argparse.ArgumentParser(
        description="Translate all built-in physics rules into symbolic operation plans using an LLM."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("PHYSICS_RULE_MODEL", "gpt-5"),
        help="LLM model for translation.",
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=10,
        help="Maximum number of LLM calls allowed for the entire process.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching for LLM calls.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="rule_translations.json",
        help="Output file to save the JSON results.",
    )
    args = parser.parse_args()

    all_translations = translate_all_rules(args.model, args.max_llm_calls, args.no_cache)

    # Save to file
    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_translations, f, ensure_ascii=False, indent=2)

    print(f"\n✅ All rules translated. Results saved to '{output_path}'.")
    
    # Also print to console for immediate feedback
    print("\n--- Translation Results ---")
    print(json.dumps(all_translations, ensure_ascii=False, indent=2))
    print("---------------------------\n")


if __name__ == "__main__":
    main()
