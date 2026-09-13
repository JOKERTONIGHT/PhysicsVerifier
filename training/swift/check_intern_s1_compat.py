#!/usr/bin/env python3
"""Check whether Intern-S1-mini can be loaded for text-only vLLM / transformers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPORT_DEFAULT = Path("/home/jinjianhan/PhysicsVerifier/logs/intern_s1_compat.json")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/slow_share/jinjianhan/models/Intern-S1-mini")
    p.add_argument("--output", type=Path, default=REPORT_DEFAULT)
    args = p.parse_args()
    report: dict = {"model": args.model, "path_exists": Path(args.model).is_dir()}
    config_path = Path(args.model) / "config.json"
    report["config_exists"] = config_path.is_file()
    if config_path.is_file():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            report["model_type"] = cfg.get("model_type")
            report["architectures"] = cfg.get("architectures")
        except Exception as exc:
            report["config_error"] = str(exc)
    try:
        import vllm

        report["vllm_version"] = getattr(vllm, "__version__", "unknown")
    except Exception as exc:
        report["vllm_import"] = str(exc)
    names: set[str] = set()
    try:
        from vllm.model_executor.models.registry import ModelRegistry  # type: ignore

        registry = getattr(ModelRegistry, "_models", None) or getattr(ModelRegistry, "models", None)
        if isinstance(registry, dict):
            names.update(str(k) for k in registry)
        for attr in ("_ModelRegistry__models",):
            blob = getattr(ModelRegistry, attr, None)
            if isinstance(blob, dict):
                names.update(str(k) for k in blob)
        report["vllm_registry_size"] = len(names)
    except Exception as exc:
        report["vllm_registry_error"] = str(exc)
    if not names:
        try:
            from vllm.model_executor.models import _MODELS  # type: ignore

            names.update(str(k) for k in getattr(_MODELS, "keys", lambda: [])())
        except Exception as exc:
            report["vllm_models_fallback_error"] = str(exc)
    report["vllm_has_interns1"] = any("intern" in n.lower() and "s1" in n.lower() for n in names)
    report["vllm_intern_like"] = sorted(n for n in names if "intern" in n.lower())[:20]
    report["text_only_vllm_supported"] = bool(report["vllm_has_interns1"])
    if not report["text_only_vllm_supported"]:
        report["recommendation"] = (
            "vLLM 0.8.5 has no InternS1ForConditionalGeneration. "
            "Text-only path needs transformers generate or a newer vLLM. "
            "Do not migrate GRPO until a 5pp easy-tier lift is measured."
        )
    try:
        import transformers

        report["transformers_version"] = transformers.__version__
        from transformers import AutoConfig

        if config_path.is_file():
            AutoConfig.from_pretrained(args.model, trust_remote_code=True)
            report["transformers_config_ok"] = True
    except Exception as exc:
        report["transformers_error"] = str(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output.resolve() != Path("-").resolve():
        print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
