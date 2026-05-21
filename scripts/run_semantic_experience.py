from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    import openai
except ImportError as exc:  # pragma: no cover
    raise SystemExit("OpenAI package not found. Please run 'pip install openai'.") from exc


@dataclass
class TopicItem:
    domain: str
    topic: str


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _safe_id(value: Any) -> str:
    return str(value) if value is not None else "unknown"


def _empty_auxiliary() -> Dict[str, Any]:
    return {
        "node_summary": "",
        "scene_cues": [],
        "boundary_cues": [],
        "explore_cues": [],
        "evidence_sample_ids": [],
    }


def _clean_text_list(value: Any, limit: int) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        text = _normalize_text(item)
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _normalize_auxiliary(rule: Dict[str, Any]) -> Dict[str, Any]:
    raw = rule.get("auxiliary") if isinstance(rule.get("auxiliary"), dict) else {}
    return {
        "node_summary": _normalize_text(raw.get("node_summary", ""))[:160],
        "scene_cues": _clean_text_list(raw.get("scene_cues"), 4),
        "boundary_cues": _clean_text_list(raw.get("boundary_cues"), 4),
        "explore_cues": _clean_text_list(raw.get("explore_cues"), 3),
        "evidence_sample_ids": [],
    }


def _has_auxiliary_content(auxiliary: Dict[str, Any]) -> bool:
    return bool(
        _normalize_text(auxiliary.get("node_summary", ""))
        or auxiliary.get("scene_cues")
        or auxiliary.get("boundary_cues")
        or auxiliary.get("explore_cues")
    )


def _extend_unique(target: List[str], values: List[str], limit: int) -> None:
    for value in values:
        text = _normalize_text(value)
        if text and text not in target:
            target.append(text)
        if len(target) >= limit:
            break


def _merge_auxiliary(entry_aux: Dict[str, Any], rule_aux: Dict[str, Any], sample_id: str) -> None:
    summary = _normalize_text(rule_aux.get("node_summary", ""))
    current_summary = _normalize_text(entry_aux.get("node_summary", ""))
    if summary and (not current_summary or len(summary) < len(current_summary)):
        entry_aux["node_summary"] = summary

    _extend_unique(entry_aux["scene_cues"], rule_aux.get("scene_cues") or [], 8)
    _extend_unique(entry_aux["boundary_cues"], rule_aux.get("boundary_cues") or [], 8)
    _extend_unique(entry_aux["explore_cues"], rule_aux.get("explore_cues") or [], 8)

    if _has_auxiliary_content(rule_aux) and sample_id not in entry_aux["evidence_sample_ids"]:
        entry_aux["evidence_sample_ids"].append(sample_id)


def _load_topics(rules_catalog_path: Path) -> List[TopicItem]:
    data = json.loads(rules_catalog_path.read_text(encoding="utf-8"))
    out: List[TopicItem] = []
    for domain in data.get("domains", []) or []:
        domain_name = str(domain.get("name") or "Unknown")
        for topic in domain.get("topics", []) or []:
            topic_name = str(topic.get("name") or "Unknown")
            out.append(TopicItem(domain=domain_name, topic=topic_name))
    return out


def _topics_prompt(topics: List[TopicItem]) -> str:
    lines = []
    for item in topics:
        lines.append(f"- {item.domain} / {item.topic}")
    return "\n".join(lines)


def _load_existing(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"samples": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("samples", [])
            return data
    except Exception:
        pass
    return {"samples": []}


def _build_client() -> Any:
    if load_dotenv:
        load_dotenv()
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")
    return openai.OpenAI(base_url=base_url, api_key=api_key)


def _semantic_prompt(sample: Dict[str, Any], topics_block: str, max_rules_per_sample: int) -> Tuple[str, str]:
    question = _normalize_text(sample.get("question", ""))
    prediction = _normalize_text(sample.get("prediction", ""))
    answer = _normalize_text(sample.get("answer", ""))

    system_prompt = (
        "你是物理题语义审计与经验规则提炼器。"
        "目标：高精度、低冗余、可复用。"
        "必须仅输出 JSON 对象，不要任何解释文本。"
    )

    user_prompt = f"""
任务：对单个样本做纯语义审计，并提炼最多 {max_rules_per_sample} 条高价值经验规则。
请严格按 JSON schema 输出，不要新增字段。

可选知识主题（只能从这里选）：
{topics_block}

输入样本：
- id: {_safe_id(sample.get('id'))}
- question: {question[:3500]}
- prediction: {prediction[:3500]}
- reference_answer: {answer[:1800]}

输出 JSON schema：
{{
  "sample_id": "string",
  "topic_guess": {{"domain": "string", "topic": "string"}},
  "semantic_audit": {{
    "is_correct": true/false,
    "error_types": ["concept"|"logic"|"calculation"|"modeling"|"units"],
    "summary": "不超过80字",
    "key_errors": [
      {{"type": "concept|logic|calculation|modeling|units", "message": "不超过80字", "evidence": "不超过120字"}}
    ]
  }},
  "experience_rules": [
    {{
      "title": "不超过18字",
      "trigger": "出现何种表述/公式时触发，不超过60字",
      "check_logic": "可执行检查逻辑，不超过120字",
      "error_type": "concept|logic|calculation|modeling|units",
      "symbolic_hint": {{
        "primitive": "equation_equivalence|inequality_consistency|formula_pattern|power_law|none",
        "canonical": "若可符号化填写公式/关系，否则空字符串",
        "required_symbols": ["符号1", "符号2"]
      }},
      "auxiliary": {{
        "node_summary": "真实题目场景下的节点摘要，不超过60字",
        "scene_cues": ["来自题干或解答的真实场景触发线索，最多4条"],
        "boundary_cues": ["具体误匹配边界线索，最多4条"],
        "explore_cues": ["当前规则相关但需要继续下钻区分的线索，最多3条"]
      }}
    }}
  ]
}}

约束：
1) experience_rules 必须精简，优先可泛化规则，不要复述题目背景。
2) 如果样本基本正确，可返回空数组。
3) symbolic_hint.primitive 仅在确有可执行关系时设置为非 none。
4) topic_guess 必须来自候选列表的 domain/topic。
5) auxiliary 必须贴近当前真实题目，不要写泛泛学科边界；无法从样本支持时填空字符串或空数组。
"""
    return system_prompt, user_prompt


def _call_json(client: Any, model: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


def _fingerprint_rule(rule: Dict[str, Any], domain: str, topic: str) -> str:
    title = _normalize_text(rule.get("title", ""))
    trigger = _normalize_text(rule.get("trigger", ""))
    logic = _normalize_text(rule.get("check_logic", ""))
    base = f"{domain}|{topic}|{title}|{trigger}|{logic}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


def _build_distilled_library(samples_payload: List[Dict[str, Any]], min_count: int) -> Dict[str, Any]:
    bucket: Dict[str, Dict[str, Any]] = {}

    for item in samples_payload:
        topic_guess = item.get("topic_guess") if isinstance(item.get("topic_guess"), dict) else {}
        domain = str(topic_guess.get("domain") or "Unknown")
        topic = str(topic_guess.get("topic") or "Unknown")
        sample_id = _safe_id(item.get("sample_id"))

        for rule in item.get("experience_rules", []) or []:
            if not isinstance(rule, dict):
                continue
            fp = _fingerprint_rule(rule, domain, topic)
            entry = bucket.setdefault(
                fp,
                {
                    "rule_id": f"exp_{fp}",
                    "domain": domain,
                    "topic": topic,
                    "title": str(rule.get("title") or ""),
                    "trigger": str(rule.get("trigger") or ""),
                    "check_logic": str(rule.get("check_logic") or ""),
                    "error_type": str(rule.get("error_type") or "logic"),
                    "symbolic_hint": dict(rule.get("symbolic_hint") or {}),
                    "auxiliary": _empty_auxiliary(),
                    "count": 0,
                    "sample_ids": [],
                },
            )
            entry["count"] += 1
            if sample_id not in entry["sample_ids"]:
                entry["sample_ids"].append(sample_id)
            _merge_auxiliary(entry["auxiliary"], _normalize_auxiliary(rule), sample_id)

    rules = [v for v in bucket.values() if int(v.get("count", 0)) >= min_count]
    rules.sort(key=lambda x: (-int(x.get("count", 0)), x.get("domain", ""), x.get("topic", ""), x.get("rule_id", "")))

    by_topic: Dict[str, List[str]] = {}
    for rule in rules:
        key = f"{rule['domain']}::{rule['topic']}"
        by_topic.setdefault(key, []).append(rule["rule_id"])

    return {
        "summary": {
            "total_distilled_rules": len(rules),
            "min_count": min_count,
            "topic_buckets": len(by_topic),
        },
        "rules": rules,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pure semantic audit and distilled experience generation.")
    parser.add_argument("--input", type=str, default="data/evaluation_sample_300.json")
    parser.add_argument("--rules-catalog", type=str, default="catalogs/rules_catalog_top_down.json")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview-thinking")
    parser.add_argument("--output", type=str, default="results/semantic_experience_300.json")
    parser.add_argument("--distilled-output", type=str, default="results/semantic_experience_distilled_300.json")
    parser.add_argument("--max-rules-per-sample", type=int, default=2)
    parser.add_argument("--min-rule-count", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    input_path = Path(args.input)
    rules_catalog_path = Path(args.rules_catalog)
    output_path = Path(args.output)
    distilled_path = Path(args.distilled_output)

    samples = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise SystemExit("Input JSON must be a list of samples.")

    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    topics = _load_topics(rules_catalog_path)
    topics_block = _topics_prompt(topics)

    client = _build_client()

    existing_payload = _load_existing(output_path) if args.resume else {"samples": []}
    done_map = {str(item.get("sample_id")): item for item in existing_payload.get("samples", []) if isinstance(item, dict)}

    all_outputs: List[Dict[str, Any]] = []
    processed = 0

    for sample in samples:
        sid = _safe_id(sample.get("id"))
        if sid in done_map:
            all_outputs.append(done_map[sid])
            continue

        system_prompt, user_prompt = _semantic_prompt(sample, topics_block, args.max_rules_per_sample)

        ok = False
        last_error = None
        for _ in range(3):
            try:
                parsed = _call_json(
                    client=client,
                    model=args.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                parsed["sample_id"] = sid
                all_outputs.append(parsed)
                ok = True
                break
            except Exception as exc:
                last_error = exc
                time.sleep(1.0)

        if not ok:
            all_outputs.append(
                {
                    "sample_id": sid,
                    "topic_guess": {"domain": "Unknown", "topic": "Unknown"},
                    "semantic_audit": {
                        "is_correct": False,
                        "error_types": ["logic"],
                        "summary": "LLM调用失败，已记录重试占位。",
                        "key_errors": [{"type": "logic", "message": "LLM调用失败", "evidence": str(last_error)[:120]}],
                    },
                    "experience_rules": [],
                }
            )

        processed += 1
        if processed % 10 == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps({"samples": all_outputs}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Processed {processed}/{len(samples)}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"samples": all_outputs}, ensure_ascii=False, indent=2), encoding="utf-8")

    distilled = _build_distilled_library(all_outputs, min_count=max(1, int(args.min_rule_count)))
    distilled_path.parent.mkdir(parents=True, exist_ok=True)
    distilled_path.write_text(json.dumps(distilled, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done. Semantic sample output: {output_path}")
    print(f"Done. Distilled rule library: {distilled_path}")


if __name__ == "__main__":
    main()
