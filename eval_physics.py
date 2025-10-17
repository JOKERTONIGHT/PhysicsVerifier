#!/usr/bin/env python3
"""
通用物理竞赛评测脚本 - 支持多种模型和数据格式

使用方法:
    # 评测所有可用数据集
    python eval_physics.py
    
    # 评测特定数据集
    python eval_physics.py --dataset panpho_2024
    
    # 评测JSON格式数据集
    python eval_physics.py --dataset apho_2025_json
    
    # 使用Judge模型进行marking评测（细粒度）
    python eval_physics.py --judge-model gpt-4o --api-key YOUR_API_KEY
    
    # 指定并行进程数
    python eval_physics.py --nproc 8
    
    # 禁用Judge模型（细粒度得分为0，仅使用答案匹配）
    python eval_physics.py --no-judge
    
    # 评测多次运行结果并计算统计信息
    python eval_physics.py --multi-runs --dataset apho_2025
    
    # 评测所有数据集的多次运行结果
    python eval_physics.py --multi-runs
    
    # 指定日志保存目录
    python eval_physics.py --log-dir my_logs

作者: AI Assistant
时间: 2025年
"""

import argparse
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# # 添加项目路径到sys.path
# sys.path.append(str(Path(__file__).parent))

# 加载.env文件中的API配置
from dotenv import load_dotenv
load_dotenv('.env')

from universal_physics_evaluator import UniversalPhysicsEvaluator
from variable_constant_verifier import VariableConstantVerifier
import json
from datetime import datetime as _dt

# 全局日志记录器
logger = None

def setup_logging(log_dir="logs", dataset_name=None, multi_runs=False):
    """设置日志记录系统"""
    global logger
    
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if dataset_name:
        dataset, model = dataset_name.split('/')
        if multi_runs:
            log_filename = f"eval_physics_multi_runs_{dataset}_{model}_{timestamp}.log"
        else:
            log_filename = f"eval_physics_{dataset_name}_{timestamp}.log"
    else:
        if multi_runs:
            log_filename = f"eval_physics_multi_runs_all_{timestamp}.log"
        else:
            log_filename = f"eval_physics_all_{timestamp}.log"
    
    log_file = log_path / log_filename
    
    # 配置日志记录器
    logger = logging.getLogger('eval_physics')
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    # 控制台不需要时间戳格式器，保持原有输出格式
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return log_file

def log_print(*args, **kwargs):
    """替代safe_print的日志打印函数"""
    global logger
    message = ' '.join(str(arg) for arg in args)
    if logger:
        logger.info(message)
    else:
        print(message, **kwargs)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="通用物理竞赛评测脚本 - 支持多种模型和数据格式")
    
    # 基本参数
    parser.add_argument(
        "--results-dir", 
        type=str, 
        default="results_reasoning",
        help="推理结果目录 (默认: results)"
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        help="指定要评测的数据集 (如果不指定，则评测所有可用数据集)"
    )
    
    parser.add_argument(
        "--nproc", 
        type=int, 
        default=8,
        help="并行进程数 (默认: 8)"
    )
    
    # Judge模型参数
    parser.add_argument(
        "--judge-model", 
        type=str, 
        default="gemini-2.5-flash",
        help="Judge模型名称 (例如: gpt-4o, gpt-4-turbo, claude-3-sonnet-20240229)"
    )
    
    parser.add_argument(
        "--api-key", 
        type=str, 
        help="API密钥 (可选，也可以通过环境变量设置)"
    )
    
    parser.add_argument(
        "--no-judge", 
        action="store_true",
        help="禁用Judge模型（细粒度得分为0，仍进行粗粒度评测）"
    )
    
    parser.add_argument(
        "--multi-runs", 
        action="store_true",
        help="评测多次运行结果并计算统计信息"
    )
    
    # 其他参数
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="evaluation_results",
        help="评测结果输出目录 (默认: evaluation_results)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="详细输出模式"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="试运行模式，仅检查数据集而不进行实际评测"
    )
    
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default="logs",
        help="日志文件保存目录 (默认: logs)"
    )

    # 变量/常量混淆检查器参数
    parser.add_argument(
        "--varconst-check",
        action="store_true",
        help="启用变量/常量混淆检查器，对模型作答进行静态+可选LLM辅助的符号一致性检查"
    )
    parser.add_argument(
        "--varconst-llm-model",
        type=str,
        default=None,
        help="用于辅助等价性/适用性判断的LLM模型标识（可选）。留空则仅使用规则检查"
    )
    parser.add_argument(
        "--varconst-max-llm-calls",
        type=int,
        default=0,
        help="可用于检查器的LLM调用上限（默认0=不调用）"
    )
    
    return parser.parse_args()

def setup_environment(args):
    """设置运行环境"""
    # 设置API密钥
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
        log_print(f"✅ 已设置API密钥")
    
    # 检查结果目录
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        log_print(f"❌ 结果目录不存在: {results_dir}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_print(f"📁 输出目录: {output_dir}")

def build_judge_kwargs(args):
    """构建Judge模型参数"""
    judge_kwargs = {}
    
    if args.no_judge:
        log_print("⚠️  禁用Judge模型，细粒度得分将为0，仍进行粗粒度评测")
        return judge_kwargs
    
    if args.judge_model:
        judge_kwargs['model'] = args.judge_model
        log_print(f"🤖 使用Judge模型: {args.judge_model}")
        
        # 检查API密钥
        if not os.getenv("OPENAI_API_KEY") and not args.api_key:
            log_print("⚠️  警告: 未设置API密钥，Judge模型可能无法工作")
            log_print("   请通过 --api-key 参数或 OPENAI_API_KEY 环境变量设置")
    else:
        log_print("ℹ️  未指定Judge模型，细粒度得分将为0，仅使用粗粒度评测")
    
    if args.nproc:
        judge_kwargs['nproc'] = args.nproc
    
    return judge_kwargs

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志系统
    log_file = setup_logging(
        log_dir=args.log_dir, 
        dataset_name=args.dataset, 
        multi_runs=args.multi_runs
    )
    
    # 打印启动信息
    log_print("🚀 通用物理竞赛评测系统启动")
    log_print("=" * 60)
    log_print(f"📂 推理结果目录: {args.results_dir}")
    log_print(f"📊 并行进程数: {args.nproc}")
    log_print(f"💾 输出目录: {args.output_dir}")
    log_print(f"📝 日志文件: {log_file}")
    if args.dataset:
        log_print(f"🎯 指定数据集: {args.dataset}")
    else:
        log_print(f"🎯 评测模式: 所有可用数据集")
    log_print("=" * 60)
    
    # 设置环境
    setup_environment(args)
    
    # 构建Judge参数
    judge_kwargs = build_judge_kwargs(args)
    
    # 初始化评测器
    evaluator = UniversalPhysicsEvaluator(
        args.results_dir,
        args.output_dir,
        nproc=args.nproc
    )
    log_print("✅ 评测器初始化成功")

    def _run_varconst_for_dataset(ds_key: str, run_payload: dict = None):
        """对指定数据集（可选指定单次运行数据）执行变量/常量检查并保存报告。

        run_payload: 可选，形如 {"run_name": str, "samples": List[Dict]}，若提供则直接使用样本列表；
                     否则通过 evaluator 加载该数据集的推理结果。
        """
        if not args.varconst_check:
            return
        try:
            verifier = VariableConstantVerifier(
                llm_model=args.varconst_llm_model,
                max_llm_calls=args.varconst_max_llm_calls,
                logger=logger,
            )

            # 加载样本
            samples = None
            run_name = None
            if run_payload and isinstance(run_payload, dict):
                samples = run_payload.get("samples")
                run_name = run_payload.get("run_name")
            if samples is None:
                # 尝试通过 evaluator 加载
                load_fn = getattr(evaluator, "load_inference_results", None)
                if callable(load_fn):
                    samples = load_fn(ds_key)
                else:
                    log_print(f"⚠️ 无法加载推理结果以运行变量/常量检查（缺少 load_inference_results 方法）: {ds_key}")
                    return

            # 规范化样本字段，挑选 question/context/prediction/id
            norm_samples = []
            for r in samples or []:
                norm_samples.append({
                    "id": (r.get("id") or r.get("problem_id") or r.get("sample_id") or "").strip(),
                    "question": r.get("question") or r.get("prompt") or r.get("instruction"),
                    "context": r.get("context") or r.get("passage") or r.get("materials"),
                    "prediction": r.get("prediction") or r.get("output") or r.get("final_answer") or r.get("answer_text") or "",
                    "meta": {k: v for k, v in r.items() if k not in {"id","problem_id","sample_id","question","prompt","instruction","context","passage","materials","prediction","output","final_answer","answer_text"}}
                })

            report = verifier.analyze_batch(norm_samples, dataset_key=ds_key)

            # 保存报告
            out_dir = Path(args.output_dir) / "varconst_reports"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            base = ds_key.replace("/", "_")
            if run_name:
                out_path = out_dir / f"{base}__{run_name}__varconst_{ts}.json"
            else:
                out_path = out_dir / f"{base}__varconst_{ts}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            log_print(f"🧪 变量/常量检查完成，报告已保存: {out_path}")
        except Exception as e:
            log_print(f"❌ 变量/常量检查失败（{ds_key}）: {e}")
    
    # 试运行模式：仅检查数据集
    if args.dry_run:
        log_print("\n🔍 试运行模式：检查可用数据集...")
        available_datasets = evaluator.detect_available_datasets()
        log_print(f"\n📊 发现 {len(available_datasets)} 个可用数据集:")
        for dataset_key in available_datasets:
            config = evaluator.DATASET_CONFIGS[dataset_key]
            log_print(f"   ✓ {config['display_name']} ({dataset_key})")
        log_print("\n✅ 试运行完成")
        return
    
    # 开始评测
    if args.multi_runs:
        # 多次运行评测模式
        if args.dataset:
            # 评测单个数据集的多次运行
            log_print(f"\n🔄 开始多次运行评测: {args.dataset}")
            
            # 检查是否有多次运行
            if not evaluator.has_multiple_runs(args.dataset):
                log_print(f"⚠️  数据集 {args.dataset} 没有多次运行结果")
                return
            
            multi_run_results = evaluator.evaluate_multiple_runs(args.dataset, judge_kwargs)
            
            if multi_run_results:
                overall = multi_run_results['overall_statistics']
                log_print(f"\n🎉 多次运行评测完成！")
                log_print(f"🔄 运行次数: {overall['num_runs']}")
                log_print(f"📈 平均得分率: {overall['mean_score_rate']:.2f}% ± {overall['std_score_rate']:.2f}%")
                # 同步运行变量/常量检查：对每个 run 单独生成报告
                if args.varconst_check:
                    load_multi = getattr(evaluator, "load_multiple_runs_results", None)
                    if callable(load_multi):
                        runs_map = load_multi(args.dataset)
                        for run_name, samples in (runs_map or {}).items():
                            _run_varconst_for_dataset(args.dataset, {"run_name": run_name, "samples": samples})
                    else:
                        log_print("⚠️ 未提供 load_multiple_runs_results，变量/常量检查仅对单次加载支持")
            else:
                log_print(f"❌ 数据集 {args.dataset} 多次运行评测失败")
        else:
            # 评测所有数据集的多次运行
            log_print(f"\n🌟 开始评测所有数据集的多次运行...")
            available_datasets = evaluator.detect_available_datasets()
            
            multi_run_datasets = []
            for dataset_key in available_datasets:
                if evaluator.has_multiple_runs(dataset_key):
                    multi_run_datasets.append(dataset_key)
            
            if not multi_run_datasets:
                log_print(f"❌ 未发现任何具有多次运行结果的数据集")
                return
            
            log_print(f"📊 发现 {len(multi_run_datasets)} 个具有多次运行的数据集")
            
            all_multi_run_results = {}
            for dataset_key in multi_run_datasets:
                log_print(f"\n{'='*60}")
                log_print(f"🔄 评测多次运行: {dataset_key}")
                log_print(f"{'='*60}")
                
                
                multi_run_results = evaluator.evaluate_multiple_runs(dataset_key, judge_kwargs)
                all_multi_run_results[dataset_key] = multi_run_results
                
                if multi_run_results:
                    overall = multi_run_results['overall_statistics']
                    log_print(f"✅ 完成: 平均得分率 {overall['mean_score_rate']:.2f}% ± {overall['std_score_rate']:.2f}%")
                    # 变量/常量检查：逐 run 报告
                    if args.varconst_check:
                        load_multi = getattr(evaluator, "load_multiple_runs_results", None)
                        if callable(load_multi):
                            runs_map = load_multi(dataset_key)
                            for run_name, samples in (runs_map or {}).items():
                                _run_varconst_for_dataset(dataset_key, {"run_name": run_name, "samples": samples})
            
            # 保存所有多次运行结果的汇总
            if all_multi_run_results:
                evaluator._save_all_multi_run_summary(all_multi_run_results)
                
    else:
        # 常规评测模式
        if args.dataset:
            # 评测单个数据集
            log_print(f"\n🎯 开始评测单个数据集: {args.dataset}")
            results = evaluator.evaluate_dataset(args.dataset, judge_kwargs)
            
            if results:
                config = evaluator.DATASET_CONFIGS[args.dataset]
                log_print(f"\n✅ {config['display_name']} 评测完成！")
                log_print(f"🏆 总体得分: {results['total_score']:.2f} / {results['max_possible_score']:.2f} ({results['score_rate']:.2f}%)")
                _run_varconst_for_dataset(args.dataset)
            else:
                log_print(f"❌ 数据集 {args.dataset} 评测失败")
        else:
            # 评测所有数据集
            log_print(f"\n🌟 开始评测所有可用数据集...")
            all_results = evaluator.evaluate_all_datasets(judge_kwargs)
            
            if all_results:
                log_print(f"\n🎉 所有数据集评测完成！")
                successful_count = sum(1 for r in all_results.values() if r is not None)
                log_print(f"📊 成功评测 {successful_count}/{len(all_results)} 个数据集")
                # 对成功的集合运行变量/常量检查
                if args.varconst_check:
                    for ds_key, r in (all_results or {}).items():
                        if r is not None:
                            _run_varconst_for_dataset(ds_key)
            else:
                log_print(f"❌ 未能成功评测任何数据集")
    
    log_print(f"\n🎯 评测完成！结果已保存到: {args.output_dir}")
    log_print(f"📝 完整日志已保存到: {log_file}")

if __name__ == "__main__":
    main()