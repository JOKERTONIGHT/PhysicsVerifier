#!/usr/bin/env python3
"""Generate markdown report from scale-curve error-level experiment outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize error-level scale curve experiment.")
    parser.add_argument("--result-root", type=str, default="results/scale_curve_error")
    parser.add_argument("--split-manifest", type=str, default="data/derived/expansion3000_scale_seed20260508/split_manifest.json")
    parser.add_argument("--stamp-file", type=str, default="results/_scale_error_curve_local_stamp.txt")
    parser.add_argument("--llm-backend", type=str, default="local", choices=("local", "api"))
    parser.add_argument("--output", type=str, default="docs/规则库规模曲线实验报告_v3.md")
    args = parser.parse_args()

    backend_label = (
        "远程 OpenAI 兼容 API（Qwen3-30B-A3B-Instruct-2507）"
        if args.llm_backend == "api"
        else "本地 vLLM（Qwen3-30B-A3B-Instruct-2507 AWQ 4-bit）"
    )
    report_title = (
        "# 规则库规模曲线实验报告（远程 API 30B，错误级测评）"
        if args.llm_backend == "api"
        else "# 规则库规模曲线实验报告（本地 30B，错误级测评）"
    )

    root = Path(args.result_root)
    curve_json = root / "curve_metrics.json"
    curve_csv = root / "curve_metrics.csv"
    curve_png = root / "error_scale_curve.png"

    points: List[Dict[str, Any]] = []
    if curve_json.exists():
        payload = _load(curve_json)
        points = payload.get("points") if isinstance(payload.get("points"), list) else []

    manifest: Dict[str, Any] = {}
    manifest_path = Path(args.split_manifest)
    if manifest_path.exists():
        manifest = _load(manifest_path)

    stamp = ""
    stamp_path = Path(args.stamp_file)
    if stamp_path.exists():
        stamp = stamp_path.read_text(encoding="utf-8").strip()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: List[str] = [
        report_title,
        "",
        f"> 生成时间：{now}  ",
        f"> 实验批次 STAMP：`{stamp or 'unknown'}`",
        "",
        "## 实验设置",
        "",
        f"- **推理后端**：{backend_label}",
        "- **测评集**：`error_eval_dataset_100.json`（100 样本；与规则库按 question 去重后零重叠）",
        "- **备用 holdout**：`eval_holdout_200.json`（200 样本，不参与规则挖掘）",
        "- **规则挖掘数据**：`evaluation_sample_3000_expansion.json` 的非 blocked 子集（约 2691 样本；复用 `catalogs/semantic_experience.json`）",
        "- **规则库构建**：每 scale 点完整 unified pipeline（prepare-cluster，无 baseline seed → embedding 聚类 → LLM cluster 标注 → blueprint 生成 → rebuild）",
        "- **语义抽取**：跳过（使用已有 `catalogs/semantic_experience.json`）",
        "- **测评集**：dual-chain 原版 `error_eval_dataset_100`（prediction 与 GT span 对齐）",
        "- **评测模式**：错误级 location match，`--no-symbolic-check`",
        "- **规模梯度**：300 → 2700（步长 300；若连续 2 档 F1 增益 < 0.5% 则提前停止）",
        "",
    ]

    audit = manifest.get("overlap_audit") if isinstance(manifest.get("overlap_audit"), dict) else {}
    if audit:
        lines.extend(
            [
                "### 数据泄露防护",
                "",
                f"- 规则池 vs blocked/holdout 重叠：{len(audit.get('pool_vs_blocked_ids') or audit.get('expansion_vs_holdout') or [])} 条",
                f"- error_eval vs 规则池 question 重叠：{audit.get('pool_vs_error_eval_questions', 'n/a')} 条",
                f"- error_eval expansion IDs ⊆ holdout_200：{audit.get('error_eval_100_subset_of_holdout_200')}",
                f"- 审计通过：`{audit.get('passes')}`",
                "",
            ]
        )

    if not points:
        lines.extend(
            [
                "## 结果",
                "",
                "_实验尚未产出 metrics，或 `curve_metrics.json` 不存在。_",
                "",
            ]
        )
    else:
        lines.extend(["## 指标汇总", "", "| 扩充样本数 | 规则数 | Recall | Precision | F1 | 触发率 | 定位命中率 |", "|---:|---:|---:|---:|---:|---:|---:|"])
        for p in points:
            lines.append(
                f"| {int(p.get('expansion_size') or 0)} "
                f"| {int(p.get('catalog_rules') or 0)} "
                f"| {_fmt_pct(p.get('recall') or 0)} "
                f"| {_fmt_pct(p.get('precision') or 0)} "
                f"| {_fmt_pct(p.get('f1') or 0)} "
                f"| {_fmt_pct(p.get('sample_trigger_rate') or 0)} "
                f"| {_fmt_pct(p.get('location_hit_ratio') or 0)} |"
            )
        lines.extend(["", "## 曲线图", ""])
        if curve_png.exists():
            rel_png = curve_png.as_posix()
            lines.append(f"![错误级指标随规则库规模变化]({rel_png})")
        else:
            lines.append("_曲线图尚未生成。_")
        lines.extend(["", "## 原始产物", "", f"- CSV：`{curve_csv.as_posix()}`", f"- JSON：`{curve_json.as_posix()}`", f"- 各档结果目录：`{root.as_posix()}/scale_*`", ""])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "points": len(points)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
