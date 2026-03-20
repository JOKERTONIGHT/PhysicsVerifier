from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    import openai
except ImportError as exc:  # pragma: no cover
    raise SystemExit("OpenAI package not found. Please run 'pip install openai'.") from exc


def _build_client() -> Any:
    if load_dotenv:
        load_dotenv()
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")
    return openai.OpenAI(base_url=base_url, api_key=api_key)


def _slug_hash(parts: List[str]) -> str:
    blob = "|".join(parts)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def _safe_func_name(raw: str) -> str:
    raw = re.sub(r"[^a-zA-Z0-9_]+", "_", raw).strip("_").lower()
    raw = re.sub(r"_+", "_", raw)
    if not raw:
        raw = "experience_rule"
    if raw[0].isdigit():
        raw = f"r_{raw}"
    return f"check_{raw}"[:72]


def _build_prompt(rule: Dict[str, Any], function_name: str) -> tuple[str, str]:
    system_prompt = (
        "你是物理规则代码翻译器。"
        "任务是把经验规则翻译为可执行的 Python 检查函数。"
        "必须只输出 JSON 对象，不要输出 markdown 或解释。"
    )

    user_prompt = f"""
请将下面经验规则翻译为 Python 函数，不依赖预置 primitive。

输入规则：
- domain: {rule.get('domain')}
- topic: {rule.get('topic')}
- rule_id: {rule.get('rule_id')}
- title: {rule.get('title')}
- trigger: {rule.get('trigger')}
- check_logic: {rule.get('check_logic')}
- error_type: {rule.get('error_type')}
- symbolic_hint: {json.dumps(rule.get('symbolic_hint') or {}, ensure_ascii=False)}

函数目标：
- 函数名固定为: {function_name}
- 签名固定为: def {function_name}(sample: dict) -> dict:
- 输入 sample 至少可能包含: question, prediction, answer, context
- 返回 dict 格式固定：
  {{
    "result": "pass|fail|inconclusive",
    "message": "简短说明",
    "evidence": "命中的文本片段或空串"
  }}

代码约束（必须遵守）：
1) 只用 Python 标准库。
2) 函数体必须纯文本匹配/正则/简单数值解析，不要网络、文件 IO、subprocess、eval/exec。
3) 代码要保守，拿不准返回 inconclusive。
4) 不要生成类，不要生成多个函数，不要导入第三方包。
5) 必须保证语法正确。
6) 不要包含 import 语句；正则可直接使用 __import__ 之外的内置字符串方法优先实现。
7) 不要使用多行三引号字符串，避免转义错误。

输出 JSON schema：
{{
  "should_translate": true/false,
  "reason": "不超过80字",
  "function_name": "{function_name}",
  "python_function": "完整函数源码字符串"
}}
"""
    return system_prompt, user_prompt


def _build_repair_prompt(function_name: str, bad_code: str, validate_reason: str) -> tuple[str, str]:
    system_prompt = (
        "你是 Python 代码修复器。"
        "必须只输出 JSON 对象。"
    )
    user_prompt = f"""
请修复下面函数代码，使其通过校验。

目标签名：def {function_name}(sample: dict) -> dict
返回结构：{{"result":"pass|fail|inconclusive","message":"...","evidence":"..."}}

当前校验失败原因：{validate_reason}

待修复代码：
{bad_code}

约束：
1) 不能有 import。
2) 不能用 eval/exec/open/compile/__import__。
3) 只能保留一个函数定义。
4) 语法必须正确。

输出 JSON schema：
{{
  "python_function": "完整函数源码字符串"
}}
"""
    return system_prompt, user_prompt


def _call_json(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    request_timeout: int,
) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        timeout=request_timeout,
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


def _validate_function(function_name: str, code: str) -> tuple[bool, str]:
    try:
        mod = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax_error: {exc}"

    funcs = [n for n in mod.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) != 1:
        return False, "must_contain_exactly_one_function"

    fn = funcs[0]
    if fn.name != function_name:
        return False, f"function_name_mismatch: {fn.name}"

    arg_names = [a.arg for a in fn.args.args]
    if arg_names != ["sample"]:
        return False, f"invalid_signature_args: {arg_names}"

    banned_nodes = (
        ast.Import,
        ast.ImportFrom,
        ast.With,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Global,
        ast.Nonlocal,
    )
    banned_calls = {"eval", "exec", "open", "compile", "__import__"}

    for node in ast.walk(mod):
        if isinstance(node, banned_nodes):
            return False, f"banned_node: {type(node).__name__}"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in banned_calls:
                return False, f"banned_call: {node.func.id}"

    # Basic runtime smoke check
    namespace: Dict[str, Any] = {"re": re}
    try:
        exec(compile(mod, filename="<generated_rule>", mode="exec"), {"re": re}, namespace)
        fn_obj = namespace.get(function_name)
        if not callable(fn_obj):
            return False, "not_callable"
        probe = fn_obj({"question": "", "prediction": "", "answer": "", "context": ""})
        if not isinstance(probe, dict):
            return False, "probe_not_dict"
        if str(probe.get("result")) not in {"pass", "fail", "inconclusive"}:
            return False, "probe_bad_result"
    except Exception as exc:
        return False, f"runtime_error: {exc}"

    return True, "ok"


def _sanitize_generated_code(function_name: str, code: str) -> str:
    fixed = str(code or "").strip()
    if not fixed:
        return fixed

    # Remove markdown code fences if present.
    fixed = re.sub(r"^```[a-zA-Z]*\s*", "", fixed)
    fixed = re.sub(r"\s*```$", "", fixed)

    # Drop import lines aggressively; imports are forbidden in generated function body.
    fixed = re.sub(r"(?m)^\s*(from\s+\S+\s+import\s+\S+|import\s+\S+)\s*$", "", fixed)

    # Force function name rewrite if needed.
    fixed = re.sub(r"def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", f"def {function_name}(", fixed, count=1)

    # Remove leading/trailing blank lines produced by cleanup.
    fixed = "\n".join([ln.rstrip() for ln in fixed.splitlines()]).strip()
    return fixed


def _try_repair_code(
    *,
    client: Any,
    model: str,
    function_name: str,
    bad_code: str,
    validate_reason: str,
    temperature: float,
    max_tokens: int,
    request_timeout: int,
) -> str:
    system_prompt, user_prompt = _build_repair_prompt(function_name, bad_code, validate_reason)
    parsed = _call_json(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
    )
    return _sanitize_generated_code(function_name, str(parsed.get("python_function") or ""))


def _fallback_function(function_name: str, rule: Dict[str, Any]) -> str:
    title = str(rule.get("title") or rule.get("rule_id") or "rule")
    trigger = str(rule.get("trigger") or "")
    logic = str(rule.get("check_logic") or "")
    token_blob = f"{title} {trigger} {logic}".strip().lower()
    return (
        f"def {function_name}(sample: dict) -> dict:\n"
        "    text = ' '.join([\n"
        "        str(sample.get('question', '')),\n"
        "        str(sample.get('prediction', '')),\n"
        "        str(sample.get('context', '')),\n"
        "        str(sample.get('answer', '')),\n"
        "    ]).lower()\n"
        f"    keys = {[k for k in token_blob.split() if len(k) >= 3][:6]}\n"
        "    for key in keys:\n"
        "        if key and key in text:\n"
        "            return {'result': 'fail', 'message': '触发经验规则(回退)', 'evidence': key}\n"
        "    return {'result': 'inconclusive', 'message': '未命中稳定模式', 'evidence': ''}\n"
    )


def _render_module(entries: List[Dict[str, Any]]) -> str:
    lines = [
        '"""Auto-generated experience symbolic checks.',
        "Generated by scripts/translate_experience_to_symbolic.py",
        '"""',
        "",
        "from __future__ import annotations",
        "import re",
        "",
    ]

    for item in entries:
        lines.append(f"# rule_id={item['rule_id']} | domain={item['domain']} | topic={item['topic']}")
        lines.append(item["python_function"].rstrip())
        lines.append("")

    lines.append("EXPERIENCE_CHECK_REGISTRY = [")
    for item in entries:
        lines.append(
            "    {"
            f"'rule_id': {item['rule_id']!r}, "
            f"'domain': {item['domain']!r}, "
            f"'topic': {item['topic']!r}, "
            f"'function': {item['function_name']}"
            "},"
        )
    lines.append("]")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate distilled experience rules into direct Python symbolic checks.")
    parser.add_argument("--input", type=str, default="results/semantic_experience_distilled_300.json")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview-thinking")
    parser.add_argument("--output-module", type=str, default="symbolic/generated_experience_checks.py")
    parser.add_argument("--output-manifest", type=str, default="results/experience_symbolic_program_manifest_300.json")
    parser.add_argument("--report", type=str, default="results/experience_symbolic_translation_report_300.json")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1300)
    parser.add_argument("--max-rules", type=int, default=0)
    parser.add_argument("--repair", action="store_true", help="Try one LLM repair pass before fallback.")
    parser.add_argument("--request-timeout", type=int, default=90)

    args = parser.parse_args()

    in_path = Path(args.input)
    module_path = Path(args.output_module)
    manifest_path = Path(args.output_manifest)
    report_path = Path(args.report)

    data = json.loads(in_path.read_text(encoding="utf-8"))
    rules = data.get("rules", []) if isinstance(data, dict) else []
    if not isinstance(rules, list):
        raise SystemExit("Input distilled JSON must contain a list field 'rules'.")

    if args.max_rules and args.max_rules > 0:
        rules = rules[: args.max_rules]

    client = _build_client()

    successful_entries: List[Dict[str, Any]] = []
    report_items: List[Dict[str, Any]] = []

    for rule in rules:
        if not isinstance(rule, dict):
            continue

        rule_id = str(rule.get("rule_id") or "unknown_rule")
        title = str(rule.get("title") or rule_id)
        domain = str(rule.get("domain") or "Unknown")
        topic = str(rule.get("topic") or "Unknown")

        function_name = _safe_func_name(f"{rule_id}_{_slug_hash([title, domain, topic])}")

        try:
            system_prompt, user_prompt = _build_prompt(rule, function_name)
            parsed = _call_json(
                client=client,
                model=args.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                request_timeout=args.request_timeout,
            )

            should_translate = bool(parsed.get("should_translate"))
            reason = str(parsed.get("reason") or "")
            if not should_translate:
                report_items.append({"rule_id": rule_id, "status": "skipped", "reason": reason or "model_skip"})
                continue

            fn_name_out = str(parsed.get("function_name") or function_name).strip()
            fn_code = _sanitize_generated_code(function_name, str(parsed.get("python_function") or ""))
            if fn_name_out != function_name:
                fn_code = re.sub(r"def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", f"def {function_name}(", fn_code, count=1)

            ok, validate_reason = _validate_function(function_name, fn_code)
            if not ok:
                repaired = ""
                repaired_reason = ""
                if args.repair:
                    try:
                        repaired = _try_repair_code(
                            client=client,
                            model=args.model,
                            function_name=function_name,
                            bad_code=fn_code,
                            validate_reason=validate_reason,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            request_timeout=args.request_timeout,
                        )
                        if repaired:
                            ok_repaired, repaired_reason = _validate_function(function_name, repaired)
                            if ok_repaired:
                                fn_code = repaired
                                ok = True
                                report_items.append(
                                    {
                                        "rule_id": rule_id,
                                        "status": "repaired",
                                        "reason": f"initial={validate_reason}",
                                        "function_name": function_name,
                                    }
                                )
                    except Exception as repair_exc:
                        repaired_reason = f"repair_error: {repair_exc}"

                if ok:
                    successful_entries.append(
                        {
                            "rule_id": rule_id,
                            "domain": domain,
                            "topic": topic,
                            "title": title,
                            "function_name": function_name,
                            "python_function": fn_code,
                        }
                    )
                    continue

                fn_code = _fallback_function(function_name, rule)
                ok2, validate_reason2 = _validate_function(function_name, fn_code)
                if not ok2:
                    report_items.append(
                        {
                            "rule_id": rule_id,
                            "status": "failed",
                            "reason": f"validation_failed: {validate_reason}; fallback_failed: {validate_reason2}",
                        }
                    )
                    continue
                report_items.append(
                    {
                        "rule_id": rule_id,
                        "status": "fallback",
                        "reason": f"model_code_invalid: {validate_reason}; repair={repaired_reason or 'not_used'}",
                        "function_name": function_name,
                    }
                )
            else:
                report_items.append(
                    {
                        "rule_id": rule_id,
                        "status": "translated",
                        "reason": reason,
                        "function_name": function_name,
                    }
                )

            successful_entries.append(
                {
                    "rule_id": rule_id,
                    "domain": domain,
                    "topic": topic,
                    "title": title,
                    "function_name": function_name,
                    "python_function": fn_code,
                }
            )

        except Exception as exc:
            report_items.append({"rule_id": rule_id, "status": "failed", "reason": str(exc)})

        if len(report_items) % 20 == 0:
            print(f"Processed {len(report_items)}/{len(rules)}")

    module_code = _render_module(successful_entries)
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(module_code, encoding="utf-8")

    manifest = {
        "summary": {
            "total_rules_input": len(rules),
            "total_functions_generated": len(successful_entries),
        },
        "checks": [
            {
                "rule_id": e["rule_id"],
                "domain": e["domain"],
                "topic": e["topic"],
                "title": e["title"],
                "function_name": e["function_name"],
            }
            for e in successful_entries
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "summary": {
            "total_rules_input": len(rules),
            "total_success": len(successful_entries),
            "total_events": len(report_items),
        },
        "report": report_items,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done. Generated module: {module_path}")
    print(f"Done. Manifest: {manifest_path}")
    print(f"Done. Report: {report_path}")


if __name__ == "__main__":
    main()
