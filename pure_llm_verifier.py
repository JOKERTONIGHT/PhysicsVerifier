"""
Pure LLM-based Physics Verifier using Gemini 3 Pro.

This verifier relies entirely on the reasoning capabilities of a strong LLM (Gemini 3 Pro)
to identify errors in physics solutions, with a focus on modeling errors and logical flaws
that are difficult to catch with rigid rule-based systems.
"""

from __future__ import annotations
import os
import json
import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Try importing OpenAI client (compatible with many providers) or Google Generative AI
# For this implementation, we'll assume an OpenAI-compatible interface or similar
# is available, but since the user specifically asked for Gemini 3 Pro, we should
# ensure the model name is passed correctly.
try:
    import openai
except ImportError:
    print("OpenAI package not found. Please run 'pip install openai'")
    openai = None


class PureLLMVerifier:
    def __init__(
        self,
        model_name: str = "gemini-3-pro-preview", 
        temperature: float = 0.2,
        max_tokens: int = 8192,
        system_prompt: Optional[str] = None
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        
        if load_dotenv:
            load_dotenv()
            
        # Initialize client - assuming OpenAI-compatible endpoint for Gemini or similar
        # Adjust base_url if using a specific gateway for Gemini
        try:
            base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
            
            if not api_key:
                print("Warning: No API key found (OPENAI_API_KEY or GEMINI_API_KEY).")
            
            self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
            print(f"Initialized PureLLMVerifier with model: {self.model_name}")
        except Exception as e:
            print(f"Failed to initialize LLM client: {e}")

        self.default_system_prompt = system_prompt or (
            "You are an expert physics professor and grader of the highest caliber. "
            "Your task is to critically evaluate a student's solution to a physics problem. "
            "Focus specifically on identifying:\n"
            "1. Modeling errors (e.g., incorrect physical assumptions, invalid approximations, ignoring relevant forces).\n"
            "2. Logical flaws in the derivation.\n"
            "3. Conceptual misunderstandings.\n"
            "4. Calculation errors that lead to incorrect conclusions.\n\n"
            "Do NOT be pedantic about minor formatting or notation unless it causes ambiguity. "
            "If the solution is fundamentally correct, acknowledge it. "
            "If there are errors, explain them clearly and provide the correct reasoning.\n\n"
            "Output your analysis in the following JSON format:\n"
            "{\n"
            "  \"is_correct\": boolean,\n"
            "  \"errors\": [\n"
            "    {\n"
            "      \"type\": \"modeling|logic|calculation|concept\",\n"
            "      \"description\": \"Detailed explanation of the error\",\n"
            "      \"severity\": \"critical|major|minor\"\n"
            "    }\n"
            "  ]\n"
            "}"
        )

    def analyze(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if not self.client:
            return {"error": "LLM client not initialized"}

        question = sample.get("question", "")
        prediction = sample.get("prediction", "")
        answer = sample.get("answer", "") # Ground truth if available

        user_content = f"""
Problem Statement:
{question}

Student's Solution:
{prediction}

Reference Answer (for your verification only, do not reveal to student if not necessary):
{answer}

Please evaluate the student's solution.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.default_system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if not content:
                result = {
                    "response": content,
                    "finish_reason": response.choices[0].finish_reason
                }
            else:
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    # Fallback if model returns markdown code block
                    import re
                    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
                    if match:
                        result = json.loads(match.group(1))
                    else:
                        # Return raw response for refusals or non-JSON output
                        result = {"response": content}
            
            return {
                "id": sample.get("id"),
                "analysis": result
            }

        except Exception as e:
            print(f"Error analyzing sample {sample.get('id')}: {e}")
            return {
                "id": sample.get("id"),
                "error": str(e)
            }

    def analyze_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for i, sample in enumerate(samples):
            print(f"Analyzing sample {i+1}/{len(samples)} (ID: {sample.get('id')})...")
            results.append(self.analyze(sample))
        return results

if __name__ == "__main__":
    # Simple self-test
    verifier = PureLLMVerifier(model_name="gemini-3-pro-preview")
    print("Verifier initialized.")
