import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict
import openai
from tqdm import tqdm
from dotenv import load_dotenv

def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}\nRaw text: {text}")
        return {"diagnostics": []}

def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input dataset JSON path")
    parser.add_argument("--model", type=str, default="qwen3-30b-a3b-instruct-2507", help="Model name")
    parser.add_argument("--out_json", type=str, required=True, help="Output JSON path")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY", "dummy_key")
    
    if not base_url:
        print("WARNING: OPENAI_BASE_URL not set. Using default openai endpoint.")
        client = openai.OpenAI(api_key=api_key)
    else:
        print(f"Using base_url: {base_url}")
        client = openai.OpenAI(base_url=base_url, api_key=api_key)

    results = []
    
    system_prompt = """You are an expert physics evaluator.
Your task is to review a student's step-by-step solution to a physics problem and identify ANY physics errors, logic errors, or formula misapplications.
Do NOT flag minor rounding differences as errors. Only flag genuine physical or mathematical mistakes.
If you find errors, you MUST extract the exact quote from the student's solution where the error occurs.

Output your findings STRICTLY as a JSON object with the following schema:
{
  "diagnostics": [
    {
      "rule": "Short identifying name of the physical principle violated (e.g., 'Newton_Cooling_Misapplication')",
      "message": "Detailed explanation of exactly what is wrong and why it violates physics principles.",
      "evidence": {
        "quote": "The EXACT matching substring from the student's solution that contains the error. Do not truncate or modify it."
      }
    }
  ]
}
If the solution is completely correct and has no errors, output {"diagnostics": []}. 
Return ONLY the JSON object without any markdown formatting or extra text."""

    for item in tqdm(data, desc="Evaluating with LLM Baseline"):
        item_id = item.get("id")
        question = item.get("question", "")
        prediction = item.get("prediction", "")
        
        user_msg = f"--- Problem ---\n{question}\n\n--- Student Solution ---\n{prediction}\n\nReview the solution and output the JSON diagnostics."
        
        parsed = {"diagnostics": []}
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg}
                    ],
                    temperature=0.1,
                    max_tokens=2048,
                    timeout=30,
                )
                content = resp.choices[0].message.content
                parsed = extract_json(content)
                break
            except Exception as e:
                print(f"Error evaluating sample {item_id} (Attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(5)
            
        res_item = {
            "id": item_id,
            "topic": item.get("topic", "Unknown"),
            "verifier": f"baseline_llm_{args.model}",
            "diagnostics": parsed.get("diagnostics", [])
        }
        results.append(res_item)
        
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"Saved {len(results)} results to {out_path}")

if __name__ == "__main__":
    main()
