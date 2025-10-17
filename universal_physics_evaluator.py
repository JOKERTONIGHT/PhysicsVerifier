import os
import re
import pandas as pd
import json
import glob
import warnings
import time
import threading
import datetime
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# 导入VLMEvalKit的相关工具
# import sys
# sys.path.append('VLMEvalKit')

# 加载.env文件中的API配置
from dotenv import load_dotenv
load_dotenv('.env')

from vlmeval.smp import load, dump, gpt_key_set
from vlmeval.dataset.physics_r1 import grade, extract_boxed_answer, get_answer_str, answer_tag_reward_fn_for_r1
from vlmeval.dataset.utils import build_judge
from vlmeval.utils import track_progress_rich

# 线程锁用于同步输出
output_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """线程安全的打印函数"""
    with output_lock:
        print(*args, **kwargs)

class LogBuffer:
    """日志缓存类，用于收集单个任务的所有日志"""
    def __init__(self, task_id):
        self.task_id = task_id
        self.logs = []
        self.start_time = datetime.datetime.now()
    
    def log(self, message):
        """添加日志消息"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.logs.append(f"[{timestamp}] [{self.task_id}] {message}")
    
    def flush(self):
        """一次性输出所有缓存的日志"""
        with output_lock:
            for log in self.logs:
                print(log)
            print()  # 添加空行分隔不同任务的输出


class UniversalPhysicsEvaluator:
    """
    通用物理竞赛评测系统
    
    支持多种物理竞赛数据集的自动评测：
    - PanPhO2024/2025: 泛亚物理奥林匹克
    - IPhO2024/2025: 国际物理奥林匹克
    - EuPhO2024/2025: 欧洲物理奥林匹克
    - APhO2025: 亚洲物理奥林匹克
    - FMA2024/2025: 法国数学竞赛
    - PanPhO_Mechanics2024/2025: 力学专项
    
    支持细粒度和粗粒度评测
    """
    
    # 数据集配置映射
    CONFIGS = {
        'datasets': ['IPhO_2025', 'IPhO_2024', 'APhO_2025', 'EuPhO_2025', 'EuPhO_2024', 'FMA_2024', 'FMA_2025', 'NBPhO_2024', 'NBPhO_2025', 'PanPhO_2024', 'PanPhO_2025', 'PanMechanics_2024', 'PanMechanics_2025'],
        'models': ['qwen3-vl-235b-a22b-thinking', 'Physics-235B-1001', 'Physics-235B-0929', 'Physics-30B-0929', 'Gemini-2.5-Flash-Thinking', 'Gemini-2.5-Pro-Thinking', 'gpt-5-2025-08-07', 'deepseek-r1', 'deepseek-v3', 'qwen3-8b', 'qwen3-30b', 'qwen3-32b', 'qwen3-235b', 'Physics-p2', 'Physics-0915', 'Qwen3-30B', 'Qwen3-235B', 'Qwen-235B-2507', 'Physics-30B-0915', 'Physics-0921', 'Physics-0923', 'Physics-119'] 
    }

    def __init__(self, infer_dir="results_reasoning_no_thinking", eval_dir="evaluation_results_no_thinking", nproc: int = 4):
        """
        初始化评测器
        
        Args:
            infer_dir: 推理结果目录
            eval_dir: 评测结果目录
            nproc: 并行进程数
        """
        self.infer_dir = Path(infer_dir)
        self.eval_dir = Path(eval_dir)  
        self.nproc = nproc
        
        if not self.infer_dir.exists():
            raise FileNotFoundError(f"Inference directory not found: {self.infer_dir}")
        if not self.eval_dir.exists():
            raise FileNotFoundError(f"Evaluation directory not found: {self.eval_dir}")
    
    def detect_available_datasets(self) -> List[str]:
        """检测可用的数据集"""
        available_datasets = []

        for dataset in self.CONFIGS['datasets']:
            for model in self.CONFIGS['models']:
                infer_dir = self.infer_dir / dataset / model
                eval_dir = self.eval_dir / dataset / model
                if infer_dir.exists() and not eval_dir.exists():
                    available_datasets.append(f"{dataset}/{model}")
        return available_datasets

    def detect_multiple_runs(self, dataset_key: str) -> List[str]:
        """检测数据集的多次运行结果""" 
        dataset, model = dataset_key.split('/')
        result_dir = self.infer_dir / dataset / model
        run_files = list(result_dir.glob("*.json")) # run_files type: list[PosixPath]
        run_files = [str(run_file) for run_file in run_files]
        return run_files

    def has_multiple_runs(self, dataset_key: str) -> bool:
        """检查数据集是否有多次运行结果"""
        return len(self.detect_multiple_runs(dataset_key)) > 1

    def load_inference_results(self, dataset_key: str, infer_file: Optional[str] = None) -> List[Dict]:
        """加载推理结果"""
        dataset, model = dataset_key.split('/')
        result_dir = self.infer_dir / dataset / model
        safe_print(f"📁 加载推理结果: {result_dir}")
        
        if infer_file:
            # 加载指定的运行结果
            inference_file = infer_file
            if not os.path.exists(inference_file):
                raise FileNotFoundError(f"No inference_results.json found in {result_dir}/{infer_file}.json")
            safe_print(f"   使用指定文件: {inference_file}")
        else:
            # 查找run_*目录中的inference_results.json文件
            inference_files = list(result_dir.glob("*.json"))
            if not inference_files:
                raise FileNotFoundError(f"No *.json files found in {result_dir}")
            
            # 加载最新的推理结果
            inference_file = max(inference_files, key=lambda x: x.parent.name)
            safe_print(f"   使用最新文件: {inference_file}")
        
        with open(inference_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        safe_print(f"   推理结果数量: {len(results)}")
        return results

    def load_multiple_runs_results(self, dataset_key: str) -> Dict[str, List[Dict]]:
        """加载多次运行的所有推理结果"""
        run_files = self.detect_multiple_runs(dataset_key)
        if not run_files:
            raise ValueError(f"No multiple runs found for dataset: {dataset_key}")
        
        all_runs_results = {}
        for run_file in run_files:
            try:
                results = self.load_inference_results(dataset_key, run_file)
                all_runs_results[str(run_file)] = results
                safe_print(f"✅ 加载 {run_file}: {len(results)} 条结果")
            except Exception as e:
                safe_print(f"⚠️  跳过 {run_file}: {e}")
        
        return all_runs_results

    def prepare_evaluation_data(self, dataset_key: str) -> pd.DataFrame:
        """准备评测数据，直接从JSON文件加载"""
        # 直接加载推理结果JSON，里面已经包含了所有需要的字段
        inference_results = self.load_inference_results(dataset_key)
        
        # 转换为DataFrame
        eval_data = pd.DataFrame(inference_results)
        
        safe_print(f"✅ 直接加载评测数据，共 {len(eval_data)} 条记录")
        safe_print(f"📊 数据列名: {list(eval_data.columns)}")
        
        # 检查必要字段
        required_fields = ['prediction', 'answer']
        for field in required_fields:
            if field not in eval_data.columns:
                raise ValueError(f"Missing required field: {field}")
        
        return eval_data

    def _safe_parse_json_field(self, field_value):
        """安全解析JSON字段"""
        # 如果已经是列表，直接返回
        if isinstance(field_value, list):
            return field_value
        
        # 检查None和NaN
        if field_value is None:
            return []
        
        try:
            if pd.isna(field_value):
                return []
        except (TypeError, ValueError):
            # 处理无法用pd.isna检查的情况
            pass
        
        if field_value == '':
            return []
        
        field_str = str(field_value).strip()
        if field_str.startswith('[') and field_str.endswith(']'):
            try:
                return json.loads(field_str)
            except json.JSONDecodeError:
                return [field_str]
        else:
            return [field_str] if field_str != 'nan' else []
    
    def _safe_parse_points_field(self, points_value):
        """安全解析points字段"""
        # 如果已经是列表，直接转换
        if isinstance(points_value, list):
            return [float(p) for p in points_value if p is not None]
        
        # 检查None
        if points_value is None:
            return [0.0]
            
        # 检查NaN
        try:
            if pd.isna(points_value):
                return [0.0]
        except (TypeError, ValueError):
            # 处理无法用pd.isna检查的情况
            pass
        
        if isinstance(points_value, (int, float)):
            return [float(points_value)]
        
        points_str = str(points_value).strip()
        if points_str.startswith('[') and points_str.endswith(']'):
            try:
                parsed = json.loads(points_str)
                return [float(p) for p in parsed if p is not None]
            except (json.JSONDecodeError, ValueError):
                pass
        
        try:
            return [float(points_str)]
        except ValueError:
            return [0.0]

    def _has_valid_marking(self, marking):
        """检查marking是否包含有效的评分标准"""
        if not marking:
            return False
        
        if not isinstance(marking, list):
            return False
        
        if len(marking) == 0:
            return False
        
        for item in marking:
            if item is None:
                continue
            
            if isinstance(item, list):
                if len(item) > 0:
                    return True
            elif isinstance(item, str):
                stripped = item.strip()
                if stripped and stripped.lower() not in ['', 'nan', 'none', 'null']:
                    return True
            else:
                return True
        
        return False

    def evaluate_dataset(self, dataset_key: str, judge_kwargs: Optional[Dict] = None) -> Dict:
        """评测单个数据集"""
        if judge_kwargs is None:
            judge_kwargs = {}
        
        safe_print(f"🚀 开始评测数据集: {dataset_key}")
        
        # 准备评测数据
        eval_data = self.prepare_evaluation_data(dataset_key)
        
        # 初始化judge模型
        judge_model = self._init_judge_model(judge_kwargs)
        
        safe_print(f"📊 开始并行评测，共{len(eval_data)}题...")
        
        # 构建任务列表
        tasks = []
        indices = []
        for i in range(len(eval_data)):
            row = eval_data.iloc[i]
            task_kwargs = judge_kwargs.copy()
            task = (judge_model, row, i, task_kwargs)
            tasks.append(task)
            indices.append(i)
        
        safe_print(f"🔄 启动并行评测，任务数: {len(tasks)}")
        
        # 设置中间结果保存文件
        dataset, model = dataset_key.split('/')
        output_dir = self.eval_dir / dataset / model
        output_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = output_dir / "parallel_tmp.pkl"
        
        # 并行评测所有题目
        parallel_results = track_progress_rich(
            self._evaluate_single_problem,
            tasks,
            nproc=self.nproc,
            chunksize=max(1, self.nproc//2),
            keys=indices,
            save=str(tmp_file)
        )
        
        safe_print(f"✅ 并行评测完成，开始汇总结果...")
        
        # 汇总结果
        results = self._aggregate_results(parallel_results, eval_data, dataset_key)
        
        # 保存结果
        self._save_evaluation_results(results, dataset_key, eval_data, parallel_results)
        
        # 清理临时文件
        try:
            if tmp_file.exists():
                tmp_file.unlink()
                safe_print(f"🗑️  清理临时文件: {tmp_file}")
        except Exception as e:
            safe_print(f"⚠️  清理临时文件失败: {e}")
        
        return results

    def _init_judge_model(self, judge_kwargs):
        """初始化judge模型"""
        safe_print(f"🔧 开始初始化Judge模型")
        
        judge_model_name = judge_kwargs.get('model', None)
        safe_print(f"   🤖 指定的模型名称: {judge_model_name}")
        
        if judge_model_name and judge_model_name != 'exact_matching':
            safe_print(f"   🔑 检查API密钥...")
            if gpt_key_set():
                safe_print(f"   ✅ API密钥已设置")
                try:
                    model_kwargs = {
                        'model': judge_model_name,
                        'timeout': 300,
                        'retry': 2,
                        'temperature': 0.6,
                        'max_output_tokens': 16384,
                        'verbose': False,
                        **{k: v for k, v in judge_kwargs.items() if k not in ['model', 'nproc']}
                    }
                    test_model = build_judge(**model_kwargs)
                    if test_model.working():
                        safe_print(f"🤖 使用Judge模型: {judge_model_name}")
                        return test_model
                    else:
                        safe_print(f"   ❌ Judge API不工作")
                        warnings.warn('Judge API不工作，跳过过程评测')
                except Exception as e:
                    safe_print(f"   ❌ 模型初始化失败: {e}")
                    warnings.warn(f'模型初始化失败: {e}，跳过过程评测')
            else:
                safe_print(f"   ❌ API_KEY未设置或无效")
                warnings.warn('API_KEY无效，跳过过程评测')
        else:
            safe_print("⚠️  未指定Judge模型，仅进行最终答案评测")
        return None

    def _evaluate_single_problem(self, judge_model, row, index, judge_kwargs):
        """评测单个题目的函数（用于并行调用）"""
        task_id = f"题目{index + 1}"
        log_buffer = LogBuffer(task_id)
        
        try:
            log_buffer.log(f"📖 开始评测 - ID: {row.get('id', 'N/A')}")
            
            # 提取字段
            prediction = str(row['prediction']).strip()
            ground_truth = self._safe_parse_json_field(row.get('answer', ''))
            answer_type = self._safe_parse_json_field(row.get('answer_type', 'Open-End'))
            unit = self._safe_parse_json_field(row.get('unit', ''))
            # 尝试读取points字段，如果不存在则尝试point字段（向后兼容）
            points_value = row.get('points', row.get('point', 0))
            points = self._safe_parse_points_field(points_value)
            # 记录使用的字段名（用于调试）
            points_field_used = 'points' if 'points' in row else ('point' if 'point' in row else 'default')
            marking = self._safe_parse_json_field(row.get('marking', ''))
            
            log_buffer.log(f"📝 题目信息:")
            log_buffer.log(f"   - 预测答案长度: {len(prediction)} 字符")
            log_buffer.log(f"   - 标准答案: {ground_truth}")
            log_buffer.log(f"   - 分值: {points} (字段: {points_field_used})")
            log_buffer.log(f"   - marking标准数量: {len(marking) if marking else 0}")
            
            item_total_points = sum(points) if points else 0.0
            log_buffer.log(f"   - 本题总分: {item_total_points}")
            
            # 总是进行细粒度和粗粒度评测（完全对齐EuPhO2024逻辑）
            # 细粒度评测
            log_buffer.log(f"🔍 开始细粒度评测...")
            fine_grained_score, marking_detailed_scores = self._evaluate_fine_grained_with_buffer(
                prediction, marking, points, judge_model, row.get('question', ''), log_buffer
            )
            log_buffer.log(f"✅ 细粒度得分: {fine_grained_score}")
            
            # 粗粒度评测（传入fine_grained_score，完全对齐EuPhO2024逻辑）
            log_buffer.log(f"🎯 开始粗粒度评测...")
            coarse_grained_score, extracted_pred = self._evaluate_coarse_grained_with_buffer(
                prediction, ground_truth, answer_type, unit, points, 
                fine_grained_score, row.get('question', ''), log_buffer
            )
            log_buffer.log(f"✅ 粗粒度得分: {coarse_grained_score}")
            log_buffer.log(f"📤 提取的预测答案: {extracted_pred}")
            
            # 计算最终得分（取两者最大值）
            final_score = max(fine_grained_score, coarse_grained_score)
            log_buffer.log(f"📊 最终得分: {final_score} (细粒度: {fine_grained_score}, 粗粒度: {coarse_grained_score})")
            
            result = {
                'index': index,
                'fine_grained_score': fine_grained_score,
                'coarse_grained_score': coarse_grained_score,
                'extracted_pred': extracted_pred,
                'marking_detailed_scores': marking_detailed_scores,
                'item_total_points': item_total_points,
                'ground_truth': ground_truth,
                'answer_type': answer_type,
                'unit': unit,
                'points': points,
                'marking': marking,
                'prediction': prediction,
                'earned_points': final_score  # 添加earned_points字段，等于最大得分
            }
            
            log_buffer.log(f"✅ 评测完成，最终得分: {final_score}")
            log_buffer.flush()
            return result
            
        except Exception as e:
            log_buffer.log(f"❌ 评测失败: {e}")
            import traceback
            log_buffer.log(f"📄 错误详情: {traceback.format_exc()}")
            log_buffer.flush()
            return None

    def _evaluate_fine_grained_with_buffer(self, prediction, marking, points, judge_model, question, log_buffer):
        """细粒度评测 - 带重测机制（带日志缓存版本）"""
        log_buffer.log(f"   🔍 细粒度评测开始")
        log_buffer.log(f"      - marking数量: {len(marking) if marking else 0}")
        log_buffer.log(f"      - judge_model: {'有' if judge_model else '无'}")
        
        if not marking or not judge_model:
            log_buffer.log(f"   ⚠️  跳过细粒度评测：{'无marking标准' if not marking else '无judge模型'}")
            return 0.0, []
        
        # 检查是否有多套marking标准（对齐EuPhO2024逻辑）
        has_multiple_marking_sets = self._has_multiple_marking_sets(marking)
        if has_multiple_marking_sets:
            log_buffer.log(f"   📋 检测到多套marking标准，共 {len(marking)} 套")
            return self._evaluate_multiple_marking_sets_with_buffer(prediction, marking, points, judge_model, question, log_buffer)
        else:
            log_buffer.log(f"   📋 单套marking标准")
            return self._evaluate_single_marking_set_with_buffer(prediction, marking, points, judge_model, question, log_buffer)
    
    def _has_multiple_marking_sets(self, marking):
        """检查是否有多套marking标准"""
        if not marking or len(marking) == 0:
            return False
        
        # 如果第一个元素是列表，则认为有多套标准
        return isinstance(marking[0], list)
    
    def _evaluate_multiple_marking_sets_with_buffer(self, prediction, marking_sets, points, judge_model, question, log_buffer):
        """评测多套marking标准，取最高分"""
        best_score = 0.0
        best_detailed_scores = []
        all_marking_results = []
        
        max_possible_score = sum(points) if points else 0.0
        
        for set_idx, marking_set in enumerate(marking_sets):
            log_buffer.log(f"   📊 评测第 {set_idx + 1} 套marking标准")
            
            score, detailed_scores = self._evaluate_single_marking_set_with_buffer(
                prediction, marking_set, points, judge_model, question, log_buffer
            )
            
            # 记录每套标准的结果
            marking_result = {
                'marking_set_index': set_idx + 1,
                'score': score,
                'detailed_scores': detailed_scores,
                'max_possible_score': max_possible_score
            }
            all_marking_results.append(marking_result)
            
            log_buffer.log(f"      ✅ 第 {set_idx + 1} 套标准得分: {score:.2f}")
            
            # 更新最佳分数
            if score > best_score:
                best_score = score
                best_detailed_scores = detailed_scores
                # 在最佳详细得分中添加标记
                for detailed_score in best_detailed_scores:
                    detailed_score['best_marking_set'] = set_idx + 1
        
        log_buffer.log(f"   🏆 多套标准最终得分: {best_score:.2f} (来自第 {[r['marking_set_index'] for r in all_marking_results if r['score'] == best_score][0]} 套标准)")
        
        return round(best_score, 2), best_detailed_scores
    
    def _evaluate_single_marking_set_with_buffer(self, prediction, marking, points, judge_model, question, log_buffer):
        """评测单套marking标准 - 带重测机制（带日志缓存版本）"""        
        scoring_criteria = self._parse_marking_criteria(marking)
        max_possible_score = sum(points) if points else 0.0
        max_retries = 3
        
        log_buffer.log(f"      📊 评测配置:")
        log_buffer.log(f"         - 评分标准数量: {len(scoring_criteria)}")
        log_buffer.log(f"         - 最大总分: {max_possible_score}")
        log_buffer.log(f"         - 最大重测次数: {max_retries}")
        
        for attempt in range(max_retries + 1):
            log_buffer.log(f"      🔄 开始第 {attempt + 1} 次评测")
            scores = []
            detailed_scores = []
            
            # 对每个marking标准进行评分
            for i, criterion in enumerate(scoring_criteria):
                log_buffer.log(f"         📏 评测标准 {i+1}/{len(scoring_criteria)}: {criterion['description'][:50]}{'...' if len(criterion['description']) > 50 else ''}")
                score, response = self._evaluate_single_criterion_with_buffer(
                    prediction, criterion, judge_model, question, 
                    max_total_score=max_possible_score, 
                    current_attempt=attempt,
                    log_buffer=log_buffer
                )
                scores.append(score)
                log_buffer.log(f"            ➡️ 得分: {score}")
                
                # 记录详细得分
                detailed_scores.append({
                    'marking_criterion': criterion['description'],
                    'score': round(score, 2),
                    'index': criterion['index'],
                    'attempt': attempt + 1,
                    'judge_response': response
                })
            
            total_score = sum(scores)
            log_buffer.log(f"      📊 第 {attempt + 1} 次评测总分: {total_score} (各项得分: {scores})")
            
            # 检查是否超过最大分数
            if total_score <= max_possible_score or max_possible_score == 0:
                # 分数合理，添加成功标记
                for detailed_score in detailed_scores:
                    detailed_score['retry_info'] = f"第{attempt + 1}次评测成功" if attempt > 0 else "首次评测成功"
                    detailed_score['total_attempts'] = attempt + 1
                    detailed_score['final_success'] = True
                
                if attempt > 0:
                    log_buffer.log(f"      ✅ 第{attempt + 1}次评测成功，总分 {total_score:.2f} <= {max_possible_score:.2f}")
                else:
                    log_buffer.log(f"      ✅ 首次评测成功，总分 {total_score:.2f} <= {max_possible_score:.2f}")
                
                return round(total_score, 2), detailed_scores
            else:
                # 分数超限，准备重测
                if attempt < max_retries:
                    log_buffer.log(f"      ⚠️  第{attempt + 1}次评测超分: {total_score:.2f} > {max_possible_score:.2f}，进行第{attempt + 2}次重测...")
                else:
                    # 达到最大重测次数，强制调整
                    log_buffer.log(f"      ❌ 已达最大重测次数({max_retries + 1})，总分仍超限: {total_score:.2f} > {max_possible_score:.2f}")
                    log_buffer.log(f"      📊 强制按比例调整分数...")
                    
                    scale_factor = max_possible_score / total_score
                    adjusted_scores = []
                    
                    log_buffer.log(f"         📐 调整系数: {scale_factor:.3f}")
                    for i, score in enumerate(scores):
                        adjusted_score = score * scale_factor
                        adjusted_scores.append(adjusted_score)
                        log_buffer.log(f"            标准{i+1}: {score:.2f} -> {adjusted_score:.2f}")
                        detailed_scores[i]['original_score'] = detailed_scores[i]['score']
                        detailed_scores[i]['score'] = round(adjusted_score, 2)
                        detailed_scores[i]['retry_info'] = f"重测{max_retries + 1}次后强制调整"
                        detailed_scores[i]['total_attempts'] = max_retries + 1
                        detailed_scores[i]['forced_adjustment'] = True
                        detailed_scores[i]['scale_factor'] = round(scale_factor, 3)
                        detailed_scores[i]['final_success'] = False
                    
                    return round(sum(adjusted_scores), 2), detailed_scores
        
        return 0.0, []

    def _evaluate_coarse_grained_with_buffer(self, prediction, ground_truth, answer_type, unit, points, fine_grained_score, question, log_buffer):
        """粗粒度评测（完全对齐EuPhO2024逻辑）"""
        log_buffer.log(f"   🎯 粗粒度评测开始")
        log_buffer.log(f"      - 标准答案: {ground_truth}")
        log_buffer.log(f"      - 细粒度得分: {fine_grained_score}")
        
        extracted_pred = ""
        
        if ground_truth:
            log_buffer.log(f"      ✅ 有标准答案，开始答案匹配评测")
            try:
                # 提取预测答案用于显示
                num_expected_answers = len(ground_truth)
                log_buffer.log(f"      📤 提取预测答案（期望{num_expected_answers}个答案）")
                extracted_pred = self._extract_prediction_for_display(prediction, num_expected_answers)
                log_buffer.log(f"      📝 提取结果: {extracted_pred}")
                
                # 多答案评测
                log_buffer.log(f"      🔍 开始多答案匹配评测")
                answer_score = self._evaluate_multiple_answers_with_buffer(prediction, ground_truth, points, question, log_buffer)
                log_buffer.log(f"      📊 答案匹配得分: {answer_score}")
                
                if answer_score > 0:
                    # 答案正确，使用答案得分
                    log_buffer.log(f"      ✅ 答案正确，使用答案得分: {answer_score}")
                    return round(answer_score, 2), extracted_pred
                else:
                    # 答案错误，直接使用已计算的细粒度得分（避免重复marking评分）
                    log_buffer.log(f"      ❌ 答案错误，使用细粒度得分: {fine_grained_score}")
                    return round(fine_grained_score, 2), extracted_pred
            except Exception as e:
                # 评测失败，使用已计算的细粒度得分
                log_buffer.log(f"      ⚠️  答案评测失败: {e}，使用细粒度得分: {fine_grained_score}")
                return round(fine_grained_score, 2), extracted_pred
        
        # 如果没有标准答案，尝试提取预测答案用于显示
        log_buffer.log(f"      ⚠️  无标准答案，尝试提取预测答案用于显示")
        if not extracted_pred:
            try:
                extracted_pred = self._extract_prediction_for_display(prediction, 10)
                log_buffer.log(f"      📝 提取的预测答案: {extracted_pred}")
            except Exception as e:
                log_buffer.log(f"      ❌ 提取预测答案失败: {e}")
                extracted_pred = ""
        
        # 没有标准答案时，使用细粒度得分
        log_buffer.log(f"      📊 最终使用细粒度得分: {fine_grained_score}")
        return round(fine_grained_score, 2), extracted_pred

    def _evaluate_coarse_grained_simple_with_buffer(self, prediction, ground_truth, answer_type, unit, points, question, log_buffer):
        """简化的粗粒度评测 - 仅进行答案匹配评测（带日志缓存版本）"""
        log_buffer.log(f"   🎯 粗粒度评测开始")
        log_buffer.log(f"      - 标准答案: {ground_truth}")
        
        extracted_pred = ""
        
        if ground_truth:
            log_buffer.log(f"      ✅ 有标准答案，开始答案匹配评测")
            try:
                # 提取预测答案用于显示
                num_expected_answers = len(ground_truth)
                log_buffer.log(f"      📤 提取预测答案（期望{num_expected_answers}个答案）")
                extracted_pred = self._extract_prediction_for_display(prediction, num_expected_answers)
                log_buffer.log(f"      📝 提取结果: {extracted_pred}")
                
                # 多答案评测
                log_buffer.log(f"      🔍 开始多答案匹配评测")
                answer_score = self._evaluate_multiple_answers_with_buffer(prediction, ground_truth, points, question, log_buffer)
                log_buffer.log(f"      📊 答案匹配得分: {answer_score}")
                
                return round(answer_score, 2), extracted_pred
                
            except Exception as e:
                log_buffer.log(f"      ⚠️  答案评测失败: {e}，返回0分")
                return 0.0, extracted_pred
        
        # 如果没有标准答案，尝试提取预测答案用于显示
        log_buffer.log(f"      ⚠️  无标准答案，尝试提取预测答案用于显示")
        if not extracted_pred:
            try:
                extracted_pred = self._extract_prediction_for_display(prediction, 10)
                log_buffer.log(f"      📝 提取的预测答案: {extracted_pred}")
            except Exception as e:
                log_buffer.log(f"      ❌ 提取预测答案失败: {e}")
                extracted_pred = ""
        
        log_buffer.log(f"      📊 无标准答案，返回0分")
        return 0.0, extracted_pred

    def _evaluate_multiple_answers_with_buffer(self, prediction, ground_truth_list, points_list, question="", log_buffer=None):
        """多答案评测（带日志缓存版本）"""
        if not ground_truth_list:
            return 0.0
            
        # 确保数据长度一致
        actual_length = min(len(ground_truth_list), len(points_list))
        ground_truth_list = ground_truth_list[:actual_length]
        points_list = points_list[:actual_length]
        
        try:
            # 使用physics_r1的多答案评测函数
            total_score, total_point, extracted_preds, extracted_gts, scored_by_list = answer_tag_reward_fn_for_r1(
                prediction, ground_truth_list, problem=question, points=points_list, 
                use_xverify=True, debug=True, log_callback=log_buffer.log if log_buffer else None
            )
            if log_buffer:
                log_buffer.log(f"         📊 评分详情: scored_by={scored_by_list}, total_point={total_point}")
            return total_point
        except Exception as e:
            if log_buffer:
                log_buffer.log(f"[DEBUG] Exception in answer evaluation: {str(e)}")
            # 回退到逐个评测
            return self._fallback_individual_evaluation_with_buffer(prediction, ground_truth_list, points_list, question, log_buffer)

    def _fallback_individual_evaluation_with_buffer(self, prediction, ground_truth_list, points_list, question="", log_buffer=None):
        """回退的逐个评测方法（带日志缓存版本）"""
        try:
            num_answers = len(ground_truth_list)
            extracted_answers = get_answer_str(prediction, return_origin=False, num_answers=num_answers)
            
            total_earned_score = 0.0
            for extracted_ans, gt_answer, points in zip(extracted_answers, ground_truth_list, points_list):
                if extracted_ans and extracted_ans.strip():
                    try:
                        is_correct, _, _, _ = grade(extracted_ans, gt_answer, False, problem=question, 
                                                 use_xverify=True, debug=True, 
                                                 log_callback=log_buffer.log if log_buffer else None)
                        if is_correct:
                            total_earned_score += points
                    except Exception as e:
                        if log_buffer:
                            log_buffer.log(f"[DEBUG] Exception in grade call: {str(e)}")
                        pass
            
            return total_earned_score
        except Exception as e:
            if log_buffer:
                log_buffer.log(f"[DEBUG] Exception in fallback evaluation: {str(e)}")
            return 0.0

    def _parse_marking_criteria(self, marking_list):
        """解析marking评分标准"""
        criteria = []
        if not marking_list:
            return criteria
        
        for i, marking_criterion in enumerate(marking_list):
            if marking_criterion and str(marking_criterion).strip():
                criteria.append({
                    'description': str(marking_criterion).strip(),
                    'index': i
                })
        
        return criteria

    def _evaluate_single_criterion_with_buffer(self, prediction, criterion, judge_model, question, max_total_score=None, current_attempt=0, log_buffer=None):
        """使用judge模型评测单个标准 - 带重测机制（带日志缓存版本）"""
        log_buffer.log(f"         🤖 调用Judge模型评测标准")
        
        # 构建总分限制提示
        total_score_warning = ""
        if max_total_score is not None and max_total_score > 0:
            total_score_warning = f"""
⚠️  IMPORTANT TOTAL SCORE CONSTRAINT:
- This question has a maximum total score of {max_total_score} points
- ALL marking criteria scores combined MUST NOT exceed {max_total_score} points
- You are evaluating ONE criterion among multiple criteria for this question
- Be conservative in your scoring to ensure the total doesn't exceed the limit
- This is attempt #{current_attempt + 1} of evaluation"""

        prompt = f"""You are an expert physics competition grader. Evaluate the student's solution against the specific grading criterion.

PHYSICS PROBLEM:
{question}

STUDENT'S SOLUTION:
{prediction}

GRADING CRITERION:
{criterion['description']}{total_score_warning}

INSTRUCTIONS:
1. Carefully analyze the student's solution for physics concepts, mathematical derivations, and calculations.
2. Compare the solution against the specific grading criterion provided.
3. Award points strictly according to the criterion, including partial credit when specified.
4. BE CONSERVATIVE - remember this is one of multiple criteria being evaluated simultaneously.

SCORING FORMAT:
- Read the grading criterion carefully to understand the maximum points and conditions for partial credit
- Evaluate whether the student's solution meets the full criteria, partial criteria, or no criteria
- Output your score using the exact format: \\boxed{{score}}
- The score should be a number (e.g., 0.4, 0.2, 0.1, 0.0)

CRITICAL REQUIREMENTS:
- You MUST output your final score in the format: \\boxed{{score}}
- The score must be a single number only (no text inside the boxed)
- BE CONSERVATIVE to avoid exceeding the total score limit

⚠️ CRITICAL INSTRUCTION: 
- Output ONLY: \\boxed{{score}}
- NO explanations, NO analysis, NO reasoning
- Just the number in the exact format \\boxed{{score}}

RESPOND WITH ONLY THE BOXED SCORE:"""
        
        try:
            log_buffer.log(f"         ⏳ 调用Judge模型...")
            start_time = time.time()
            
            response = judge_model.generate(prompt).strip()
            
            elapsed_time = time.time() - start_time
            log_buffer.log(f"         ⏱️  响应耗时: {elapsed_time:.2f}秒")
            
            # 提取分数
            score = self._extract_score_from_response(response)
            log_buffer.log(f"         🔍 提取的分数: {score}")
            
            return score, response
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            log_buffer.log(f"         ❌ Judge模型调用失败 (耗时 {elapsed_time:.2f}秒): {e}")
            return 0.0, f"Judge模型调用失败: {str(e)}"

    def _extract_score_from_response(self, response):
        """从模型响应中提取分数的辅助函数"""
        if not response:
            return 0.0
            
        response = response.strip()
        
        # 优先使用boxed格式提取分数
        boxed_patterns = [
            r'\\boxed\{([^}]+)\}',
            r'boxed\{([^}]+)\}',
        ]
        
        for pattern in boxed_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in reversed(matches):
                match = match.strip()
                if match:
                    try:
                        score = float(match)
                        return round(score, 2)
                    except ValueError:
                        nums = re.findall(r'\d+\.?\d*', match)
                        if nums:
                            try:
                                score = float(nums[-1])
                                return round(score, 2)
                            except ValueError:
                                continue
        
        # 查找特定格式的分数
        score_patterns = [
            r'(?:Score|Final Score|Total|Points?):\s*([0-9]*\.?[0-9]+)',
            r'([0-9]*\.?[0-9]+)\s*(?:points?|pts?)',
        ]
        
        for pattern in score_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                try:
                    score = float(matches[-1])
                    return round(score, 2)
                except ValueError:
                    continue
        
        # 提取所有数字，取最后一个
        all_numbers = re.findall(r'[0-9]*\.?[0-9]+', response)
        if all_numbers:
            try:
                score = float(all_numbers[-1])
                return round(score, 2)
            except ValueError:
                pass
        
        return 0.0
    
    def _extract_prediction_for_display(self, prediction, num_answers=10):
        """提取预测答案用于显示"""
        try:
            extracted_answers = get_answer_str(prediction, return_origin=False, num_answers=num_answers)
            valid_answers = []
            
            for ans in extracted_answers:
                if ans and ans.strip():
                    cleaned_ans = ' '.join(ans.strip().replace('\n', ' ').replace('\r', ' ').split())
                    if cleaned_ans:
                        valid_answers.append(cleaned_ans)
            
            return ", ".join(valid_answers) if valid_answers else ""
        except Exception:
            # 回退到extract_boxed_answer
            try:
                extracted = extract_boxed_answer(prediction)
                if extracted and extracted.strip():
                    cleaned = ' '.join(extracted.strip().replace('\n', ' ').replace('\r', ' ').split())
                    return cleaned if cleaned else ""
            except Exception:
                pass
            return ""

    def _aggregate_results(self, parallel_results, eval_data, dataset_key):
        """汇总并行评测结果（对齐EuPhO2024逻辑）"""
        fine_grained_total_score = 0.0
        coarse_grained_total_score = 0.0
        total_score = 0.0  # 使用earned_points（最大值）作为总分
        max_possible_score = 0.0
        
        for i, result in enumerate(parallel_results):
            if result is None:
                safe_print(f"⚠️  题目 {i+1} 评测失败，跳过")
                continue
                
            fine_score = result['fine_grained_score']
            coarse_score = result['coarse_grained_score']
            earned_points = result.get('earned_points', max(fine_score, coarse_score))
            item_points = result['item_total_points']
            
            # 累加各种得分（对齐EuPhO2024逻辑）
            fine_grained_total_score = round(fine_grained_total_score + fine_score, 2)
            coarse_grained_total_score = round(coarse_grained_total_score + coarse_score, 2)
            total_score = round(total_score + earned_points, 2)  # 总分使用earned_points
            
            max_possible_score += item_points
        
        # 计算最终结果
        max_possible_score = round(max_possible_score, 2)
        fine_rate = round((fine_grained_total_score / max_possible_score * 100), 2) if max_possible_score > 0 else 0.0
        coarse_rate = round((coarse_grained_total_score / max_possible_score * 100), 2) if max_possible_score > 0 else 0.0
        total_rate = round((total_score / max_possible_score * 100), 2) if max_possible_score > 0 else 0.0
        
        return {
            'dataset_key': dataset_key,
            'fine_grained_total_score': fine_grained_total_score,
            'fine_grained_score_rate': fine_rate,
            'coarse_grained_total_score': coarse_grained_total_score,
            'coarse_grained_score_rate': coarse_rate,
            'total_score': total_score,  # 这是earned_points的总和
            'score_rate': total_rate,
            'max_possible_score': max_possible_score,
            'total_count': len(parallel_results),
        }

    def _save_evaluation_results(self, results, dataset_key, eval_data, parallel_results, file_name=None):
        """保存评测结果"""
        dataset, model = dataset_key.split('/')
        output_dir = self.eval_dir / dataset / model
        if file_name:
            output_dir = output_dir / file_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存汇总结果
        score_file = output_dir / f"{file_name}_score.json"
        dump(results, str(score_file))
        
        # 构建详细结果
        detailed_results = []
        for i, result in enumerate(parallel_results):
            if result is None:
                continue
            
            row = eval_data.iloc[i]
            # 使用earned_points字段，如果没有则计算最大值（对齐EuPhO2024逻辑）
            earned_points = result.get('earned_points', max(result['fine_grained_score'], result['coarse_grained_score']))
            
            detailed_item = {
                "id": str(row.get('id', f"{dataset_key}_{i+1}")),
                "context": str(row.get('context', '')).strip(),
                "question": str(row.get('question', '')).strip(),
                "solution": str(row.get('solution', '')).strip(),
                "marking": result['marking'] if result['marking'] else [],
                "marking_detailed_scores": result['marking_detailed_scores'] if result['marking_detailed_scores'] else [],
                "answer": [f"\\boxed{{{ans}}}" for ans in result['ground_truth']] if result['ground_truth'] else [''],
                "answer_type": result['answer_type'] if result['answer_type'] else ['Open-End'],
                "unit": result['unit'] if result['unit'] else [''],
                "points": result['points'] if result['points'] else [0.0],
                "modality": str(row.get('modality', 'text')).strip(),
                "field": str(row.get('field', '')).strip(),
                "subfield": str(row.get('subfield', '')).strip(),
                "source": dataset_key,
                "test_result": str(result['prediction']),
                "test_answer": [f"\\boxed{{{ans.strip()}}}" for ans in result['extracted_pred'].split(", ") if ans.strip()] if result['extracted_pred'] else [''],
                "fine_grained_score": result['fine_grained_score'],
                "coarse_grained_score": result['coarse_grained_score'],
                "earned_points": earned_points
            }
            detailed_results.append(detailed_item)
        
        # 保存详细结果
        detailed_file = output_dir / f"{file_name}_detailed_results.json"
        dump(detailed_results, str(detailed_file))
        
        # 保存Excel格式（带评测结果）
        try:
            eval_data_with_results = eval_data.copy()
            eval_data_with_results['fine_grained_score'] = [r['fine_grained_score'] for r in detailed_results]
            eval_data_with_results['coarse_grained_score'] = [r['coarse_grained_score'] for r in detailed_results]
            eval_data_with_results['earned_points'] = [r['earned_points'] for r in detailed_results]
            eval_data_with_results['extracted_prediction'] = [", ".join(r['test_answer']).replace("\\boxed{", "").replace("}", "") for r in detailed_results]
            # 将marking详细得分转换为可读字符串格式保存到Excel
            eval_data_with_results['marking_detailed_scores'] = [
                json.dumps(r['marking_detailed_scores'], ensure_ascii=False) if r['marking_detailed_scores'] else '[]' 
                for r in detailed_results
            ]
            
            detailed_xlsx_file = output_dir / f"{file_name}_detailed.xlsx"
            dump(eval_data_with_results, str(detailed_xlsx_file))
        except Exception as e:
            safe_print(f"⚠️  保存详细Excel文件失败: {e}")
        
        safe_print(f"💾 评测结果已保存到: {output_dir}")

    def evaluate_all_datasets(self, judge_kwargs: Optional[Dict] = None) -> Dict[str, Dict]:
        """评测所有可用的数据集"""
        available_datasets = self.detect_available_datasets()
        
        if not available_datasets:
            safe_print("❌ 未发现任何可用的数据集")
            return {}
        
        safe_print(f"🎯 开始评测 {len(available_datasets)} 个数据集...")
        
        all_results = {}
        for dataset_key in available_datasets:
            safe_print(f"\n{'='*60}")
            safe_print(f"📊 正在评测: {dataset_key}")
            safe_print(f"{'='*60}")
            
            try:
                results = self.evaluate_dataset(dataset_key, judge_kwargs)
                all_results[dataset_key] = results
                
                # 打印单个数据集的总结
                safe_print(f"\n✅ {dataset_key} 评测完成！")
                safe_print(f"🏆 总体得分: {results['total_score']:.2f} / {results['max_possible_score']:.2f} ({results['score_rate']:.2f}%)")
                safe_print(f"📊 细粒度总分: {results['fine_grained_total_score']:.2f} ({results['fine_grained_score_rate']:.2f}%)")
                safe_print(f"🎯 粗粒度总分: {results['coarse_grained_total_score']:.2f} ({results['coarse_grained_score_rate']:.2f}%)")
                safe_print(f"📈 评测题目数: {results['total_count']}")
                
            except Exception as e:
                safe_print(f"❌ 评测 {dataset_key} 失败: {e}")
                import traceback
                safe_print(f"错误详情: {traceback.format_exc()}")
                all_results[dataset_key] = None
        
        # 保存汇总结果
        self._save_summary_results(all_results)
        
        return all_results

    def evaluate_multiple_runs(self, dataset_key: str, judge_kwargs: Optional[Dict] = None) -> Dict:
        """评测多次运行结果并计算统计信息"""
        if judge_kwargs is None:
            judge_kwargs = {}
        
        safe_print(f"🚀 开始评测多次运行结果: {dataset_key}")
        
        # 加载所有运行的推理结果
        all_runs_results = self.load_multiple_runs_results(dataset_key)
        
        # 对每次运行分别进行评测
        run_evaluation_results = {}
        for run_file, inference_results in all_runs_results.items():
            safe_print(f"\n📈 评测 {run_file}...")
            
            # 转换为DataFrame进行评测
            eval_data = pd.DataFrame(inference_results)
            
            # 初始化judge模型
            judge_model = self._init_judge_model(judge_kwargs)
            
            # 构建任务列表
            tasks = []
            indices = []
            for i in range(len(eval_data)):
                row = eval_data.iloc[i]
                task_kwargs = judge_kwargs.copy()
                task = (judge_model, row, i, task_kwargs)
                tasks.append(task)
                indices.append(i)
            
            # 设置中间结果保存文件
            output_dir = Path(f"{self.eval_dir}/{dataset_key}")
            output_dir.mkdir(parents=True, exist_ok=True)
            tmp_file = output_dir / f"parallel_tmp.pkl"
            
            # 并行评测所有题目
            parallel_results = track_progress_rich(
                self._evaluate_single_problem,
                tasks,
                nproc=self.nproc,
                chunksize=max(1, self.nproc//2),
                keys=indices,
                save=str(tmp_file)
            )
            
            # 汇总单次运行结果
            run_results = self._aggregate_results(parallel_results, eval_data, run_file.split('/')[-1].replace('.json', ''))
            self._save_evaluation_results(run_results, dataset_key, eval_data, parallel_results, run_file.split('/')[-1].replace('.json', ''))
            run_evaluation_results[run_file] = {
                'results': run_results,
                'detailed_results': parallel_results,
                'eval_data': eval_data
            }
            
            # 清理临时文件
            try:
                if tmp_file.exists():
                    tmp_file.unlink()
            except Exception:
                pass
            
            safe_print(f"   ✅ {run_file} 评测完成: {run_results['total_score']:.2f}/{run_results['max_possible_score']:.2f} ({run_results['score_rate']:.2f}%)")
        
        # 计算多次运行的统计信息
        multi_run_stats = self._calculate_multi_run_statistics(run_evaluation_results, dataset_key)
        
        # 保存多次运行评测结果
        self._save_multi_run_results(multi_run_stats, dataset_key, run_evaluation_results)
        
        return multi_run_stats

    def _calculate_multi_run_statistics(self, run_evaluation_results: Dict, dataset_key: str) -> Dict:
        """计算多次运行的统计信息"""
        safe_print(f"\n📊 计算多次运行统计信息...")
        
        run_files = list(run_evaluation_results.keys())
        num_runs = len(run_files)
        
        # 获取第一次运行的题目信息作为基准
        first_run = list(run_evaluation_results.values())[0]
        first_eval_data = first_run['eval_data']
        num_questions = len(first_eval_data)
        
        # 初始化统计数据结构
        question_stats = {}
        
        # 为每道题目初始化统计信息
        for i in range(num_questions):
            row = first_eval_data.iloc[i]
            question_id = str(row.get('id', f"{dataset_key}_{i+1}"))
            question_stats[question_id] = {
                'question_id': question_id,
                'question': str(row.get('question', '')).strip(),
                'context': str(row.get('context', '')).strip(),
                'answer': row.get('answer', []),
                'points': row.get('points', row.get('point', [0.0])),
                'max_points': sum(self._safe_parse_points_field(row.get('points', row.get('point', [0.0])))),
                'runs': {},
                'statistics': {}
            }
        
        # 收集每次运行的结果
        for run_file, run_data in run_evaluation_results.items():
            detailed_results = run_data['detailed_results']
            eval_data = run_data['eval_data']
            
            for i, result in enumerate(detailed_results):
                if result is None:
                    continue
                
                row = eval_data.iloc[i]
                question_id = str(row.get('id', f"{dataset_key}_{i+1}"))
                
                if question_id in question_stats:
                    # 确定使用的分数（细粒度优先）
                    has_marking = result['marking'] and len(result['marking']) > 0 and self._has_valid_marking(result['marking'])
                    earned_score = max(result['fine_grained_score'], result['coarse_grained_score']) if has_marking else result['coarse_grained_score']
                    
                    question_stats[question_id]['runs'][run_file] = {
                        'fine_grained_score': result['fine_grained_score'],
                        'coarse_grained_score': result['coarse_grained_score'],
                        'earned_score': earned_score,
                        'prediction': result['prediction'],
                        'extracted_prediction': result['extracted_pred'],
                        'evaluation_method': 'fine_grained' if has_marking else 'coarse_grained'
                    }
        
        # 计算每道题目的统计信息
        for question_id, question_data in question_stats.items():
            runs_data = question_data['runs']
            if not runs_data:
                continue
            
            scores = [run_data['earned_score'] for run_data in runs_data.values()]
            fine_scores = [run_data['fine_grained_score'] for run_data in runs_data.values()]
            coarse_scores = [run_data['coarse_grained_score'] for run_data in runs_data.values()]
            
            question_data['statistics'] = {
                'num_runs': len(scores),
                'mean_score': round(np.mean(scores), 2),
                'std_score': round(np.std(scores), 2),
                'min_score': round(np.min(scores), 2),
                'max_score': round(np.max(scores), 2),
                'mean_fine_score': round(np.mean(fine_scores), 2),
                'std_fine_score': round(np.std(fine_scores), 2),
                'min_fine_score': round(np.min(fine_scores), 2),
                'max_fine_score': round(np.max(fine_scores), 2),
                'mean_coarse_score': round(np.mean(coarse_scores), 2),
                'std_coarse_score': round(np.std(coarse_scores), 2),
                'min_coarse_score': round(np.min(coarse_scores), 2),
                'max_coarse_score': round(np.max(coarse_scores), 2),
                'score_rate': round(np.mean(scores) / question_data['max_points'] * 100, 2) if question_data['max_points'] > 0 else 0.0
            }
        
        # 计算整体统计信息
        overall_stats = self._calculate_overall_multi_run_stats(run_evaluation_results, question_stats)
        
        return {
            'dataset_key': dataset_key,
            'num_runs': num_runs,
            'run_files': run_files,
            'question_statistics': question_stats,
            'overall_statistics': overall_stats,
            'run_results': {run_file: data['results'] for run_file, data in run_evaluation_results.items()}
        }

    def _calculate_overall_multi_run_stats(self, run_evaluation_results: Dict, question_stats: Dict) -> Dict:
        """计算整体多次运行统计信息"""
        run_files = list(run_evaluation_results.keys())
        num_runs = len(run_files)
        
        # 收集每次运行的总体得分
        run_total_scores = []
        run_max_scores = []
        run_score_rates = []
        
        for run_file, run_data in run_evaluation_results.items():
            results = run_data['results']
            run_total_scores.append(results['total_score'])
            run_max_scores.append(results['max_possible_score'])
            run_score_rates.append(results['score_rate'])
        
        # 计算每道题目的平均得分
        question_mean_scores = []
        question_max_points = []
        
        for question_data in question_stats.values():
            if question_data['runs']:
                question_mean_scores.append(question_data['statistics']['mean_score'])
                question_max_points.append(question_data['max_points'])
        
        total_mean_score = sum(question_mean_scores)
        total_max_score = sum(question_max_points)
        
        return {
            'num_runs': num_runs,
            'num_questions': len(question_stats),
            'mean_total_score': round(np.mean(run_total_scores), 2),
            'std_total_score': round(np.std(run_total_scores), 2),
            'min_total_score': round(np.min(run_total_scores), 2),
            'max_total_score': round(np.max(run_total_scores), 2),
            'mean_score_rate': round(np.mean(run_score_rates), 2),
            'std_score_rate': round(np.std(run_score_rates), 2),
            'question_based_mean_score': round(total_mean_score, 2),
            'question_based_max_score': round(total_max_score, 2),
            'question_based_score_rate': round(total_mean_score / total_max_score * 100, 2) if total_max_score > 0 else 0.0
        }

    def _save_summary_results(self, all_results):
        """保存所有数据集的汇总结果"""
        # 保存完整结果
        summary_file = self.eval_dir / "all_datasets_summary.json"
        dump(all_results, str(summary_file))
        
        # 创建简化的汇总表
        summary_table = []
        for dataset_key, results in all_results.items():
            if results is None:
                continue
                
            summary_table.append({
                'dataset_key': dataset_key,
                'total_questions': results['total_count'],
                'total_score': results['total_score'],
                'max_possible_score': results['max_possible_score'],
                'score_rate': results['score_rate'],
                'total_count': results['total_count'],
                'fine_grained_score': results['fine_grained_total_score'],
                'fine_grained_rate': results['fine_grained_score_rate'],
                'coarse_grained_score': results['coarse_grained_total_score'],
                'coarse_grained_rate': results['coarse_grained_score_rate']
            })
        
        # 保存汇总表
        summary_table_file = self.eval_dir / "summary_table.json"
        dump(summary_table, str(summary_table_file))
        
        # 打印最终汇总
        safe_print(f"\n{'='*80}")
        safe_print(f"🏆 所有数据集评测完成！汇总结果:")
        safe_print(f"{'='*80}")
        
        total_score_all = sum(r['total_score'] for r in all_results.values() if r)
        total_max_all = sum(r['max_possible_score'] for r in all_results.values() if r)
        overall_rate = round((total_score_all / total_max_all * 100), 2) if total_max_all > 0 else 0.0
        
        safe_print(f"📊 整体表现: {total_score_all:.2f} / {total_max_all:.2f} ({overall_rate:.2f}%)")
        safe_print(f"📁 详细结果已保存到: {self.eval_data}/")
        safe_print(f"💾 汇总文件: {summary_file}")
        safe_print(f"📋 汇总表格: {summary_table_file}")
        
        for item in summary_table:
            safe_print(f"   {item['dataset_key']}: {item['score_rate']:.2f}% ({item['total_score']:.1f}/{item['max_possible_score']:.1f})")

    def _save_multi_run_results(self, multi_run_stats: Dict, dataset_key: str, run_evaluation_results: Dict):
        """保存多次运行评测结果"""
        dataset, model = dataset_key.split('/')
        output_dir = self.eval_dir / dataset / model
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存完整的多次运行统计结果
        stats_file = output_dir / f"multi_run_statistics.json"
        dump(multi_run_stats, str(stats_file))
        
        # 保存每道题目的详细统计（Excel格式）
        question_stats_list = []
        fine_grained_question_stats_list = []
        coarse_grained_question_stats_list = []
        for question_id, question_data in multi_run_stats['question_statistics'].items():
            stats = question_data['statistics']
            if not stats:
                continue
            
            # 收集每次运行的得分
            run_scores = {}
            fine_grained_run_scores = {}
            coarse_grained_run_scores = {}
            for run_file in multi_run_stats['run_files']:
                if run_file in question_data['runs']:
                    run_scores[f"{run_file}_score"] = question_data['runs'][run_file]['earned_score']
                    fine_grained_run_scores[f"{run_file}_score"] = question_data['runs'][run_file]['fine_grained_score']
                    coarse_grained_run_scores[f"{run_file}_score"] = question_data['runs'][run_file]['coarse_grained_score']
                else:
                    run_scores[f"{run_file}_score"] = 0.0
                    fine_grained_run_scores[f"{run_file}_score"] = 0.0
                    coarse_grained_run_scores[f"{run_file}_score"] = 0.0
            
            question_stats_list.append({
                'question_id': question_id,
                'question': question_data['question'][:100] + '...' if len(question_data['question']) > 100 else question_data['question'],
                'max_points': question_data['max_points'],
                'mean_score': stats['mean_score'],
                'std_score': stats['std_score'],
                'min_score': stats['min_score'],
                'max_score': stats['max_score'],
                'score_rate': stats['score_rate'],
                'num_runs': stats['num_runs'],
                **run_scores
            })
            fine_grained_question_stats_list.append({
                'question_id': question_id,
                'question': question_data['question'][:100] + '...' if len(question_data['question']) > 100 else question_data['question'],
                'max_points': question_data['max_points'],
                'mean_score': stats['mean_fine_score'],
                'std_score': stats['std_fine_score'],
                'min_score': stats['min_fine_score'],
                'max_score': stats['max_fine_score'],
                'score_rate': stats['score_rate'],
                'num_runs': stats['num_runs'],
                **fine_grained_run_scores
            })
            coarse_grained_question_stats_list.append({
                'question_id': question_id,
                'question': question_data['question'][:100] + '...' if len(question_data['question']) > 100 else question_data['question'],
                'max_points': question_data['max_points'],
                'mean_score': stats['mean_coarse_score'],
                'std_score': stats['std_coarse_score'],
                'min_score': stats['min_coarse_score'],
                'max_score': stats['max_coarse_score'],
                'score_rate': stats['score_rate'],
                'num_runs': stats['num_runs'],
                **coarse_grained_run_scores
            })
        
        # 保存题目统计Excel
        if question_stats_list:
            question_stats_df = pd.DataFrame(question_stats_list)
            question_stats_file = output_dir / f"question_statistics.xlsx"
            question_stats_df.to_excel(question_stats_file, index=False)
            safe_print(f"📊 题目统计已保存: {question_stats_file}")

        if fine_grained_question_stats_list:
            fine_grained_question_stats_df = pd.DataFrame(fine_grained_question_stats_list)
            fine_grained_question_stats_file = output_dir / f"fine_grained_question_statistics.xlsx"
            fine_grained_question_stats_df.to_excel(fine_grained_question_stats_file, index=False)
            safe_print(f"📊 细粒度题目统计已保存: {fine_grained_question_stats_file}")
            
        if coarse_grained_question_stats_list:
            coarse_grained_question_stats_df = pd.DataFrame(coarse_grained_question_stats_list)
            coarse_grained_question_stats_file = output_dir / f"coarse_grained_question_statistics.xlsx"
            coarse_grained_question_stats_df.to_excel(coarse_grained_question_stats_file, index=False)
            safe_print(f"📊 粗粒度题目统计已保存: {coarse_grained_question_stats_file}")
        
        # 保存运行汇总统计
        run_summary_list = []
        for run_file in multi_run_stats['run_files']:
            if run_file in run_evaluation_results:
                results = run_evaluation_results[run_file]['results']
                run_summary_list.append({
                    'run_id': run_file.split('/')[-1].replace('.json', ''),
                    'total_score': results['total_score'],
                    'max_possible_score': results['max_possible_score'],
                    'score_rate': results['score_rate'],
                    'total_count': results['total_count'],
                    'fine_grained_score': results['fine_grained_total_score'],
                    'fine_grained_rate': results['fine_grained_score_rate'],
                    'coarse_grained_score': results['coarse_grained_total_score'],
                    'coarse_grained_rate': results['coarse_grained_score_rate']
                })
        
        if run_summary_list:
            run_summary_df = pd.DataFrame(run_summary_list)
            run_summary_file = output_dir / f"run_summary.xlsx"
            run_summary_df.to_excel(run_summary_file, index=False)
            safe_print(f"📈 运行汇总已保存: {run_summary_file}")
        
        # 打印多次运行汇总信息
        overall = multi_run_stats['overall_statistics']
        safe_print(f"\n🏆 多次运行评测完成！")
        safe_print(f"📊 数据集: {multi_run_stats['dataset_key']}")
        safe_print(f"🔄 运行次数: {overall['num_runs']}")
        safe_print(f"📝 题目数量: {overall['num_questions']}")
        safe_print(f"📈 平均总分: {overall['mean_total_score']:.2f} ± {overall['std_total_score']:.2f}")
        safe_print(f"🎯 平均得分率: {overall['mean_score_rate']:.2f}% ± {overall['std_score_rate']:.2f}%")
        safe_print(f"📋 基于题目平均: {overall['question_based_mean_score']:.2f}/{overall['question_based_max_score']:.2f} ({overall['question_based_score_rate']:.2f}%)")
        safe_print(f"💾 详细结果已保存到: {output_dir}")
        
        return multi_run_stats

    def _save_all_multi_run_summary(self, all_multi_run_results: Dict):
        """保存所有数据集多次运行结果的汇总"""
        # 保存完整结果
        summary_file = self.eval_dir / "all_datasets_multi_run_summary.json"
        dump(all_multi_run_results, str(summary_file))
        
        # 创建汇总表格
        summary_table = []
        for dataset_key, multi_run_results in all_multi_run_results.items():
            if multi_run_results is None:
                continue
            
            overall = multi_run_results['overall_statistics']
            
            summary_table.append({
                'dataset_key': dataset_key,
                'num_runs': overall['num_runs'],
                'num_questions': overall['num_questions'],
                'mean_total_score': overall['mean_total_score'],
                'std_total_score': overall['std_total_score'],
                'mean_score_rate': overall['mean_score_rate'],
                'std_score_rate': overall['std_score_rate'],
                'question_based_mean_score': overall['question_based_mean_score'],
                'question_based_max_score': overall['question_based_max_score'],
                'question_based_score_rate': overall['question_based_score_rate']
            })
        
        # 保存汇总表格
        if summary_table:
            summary_table_df = pd.DataFrame(summary_table)
            summary_table_file = self.eval_dir / "multi_run_summary_table.xlsx"
            summary_table_df.to_excel(summary_table_file, index=False)
            safe_print(f"📋 多次运行汇总表已保存: {summary_table_file}")
        
        # 打印最终汇总
        safe_print(f"\n{'='*80}")
        safe_print(f"🏆 所有数据集多次运行评测完成！汇总结果:")
        safe_print(f"{'='*80}")
        
        for item in summary_table:
            safe_print(f"   {item['dataset_key']}: {item['mean_score_rate']:.2f}% ± {item['std_score_rate']:.2f}% ({item['num_runs']} runs)")
        
        safe_print(f"📁 详细结果已保存到: {self.eval_dir}/")
        safe_print(f"💾 汇总文件: {summary_file}")
        safe_print(f"📋 汇总表格: {summary_table_file}")

    def fix_existing_statistics(self, base_results_dir=None):
        """修复已有的多次运行统计结果中的earned_score错误"""
        if base_results_dir is None:
            base_results_dir = "evaluation_results"
        
        base_path = Path(base_results_dir)
        fixed_count = 0
        
        # 查找所有多次运行统计文件
        stat_files = list(base_path.glob("**/*multi_run_statistics.json"))
        
        safe_print("Found {} statistics files to fix".format(len(stat_files)))
        
        for stat_file in stat_files:
            # try:
            safe_print("Fixing file: {}".format(stat_file))
            
            # 读取原始统计数据
            with open(stat_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            
            modified = False
            
            # 修复每道题的每次运行的earned_score
            if 'question_statistics' in stats:
                for question_id, question_data in stats['question_statistics'].items():
                    if 'runs' in question_data:
                        for run_id, run_data in question_data['runs'].items():
                            # 重新计算earned_score为最大值
                            fine_score = run_data.get('fine_grained_score', 0)
                            coarse_score = run_data.get('coarse_grained_score', 0)
                            correct_earned_score = max(fine_score, coarse_score)
                            
                            if run_data.get('earned_score') != correct_earned_score:
                                safe_print("    {} {}: {} -> {}".format(question_id, run_id, run_data.get('earned_score'), correct_earned_score))
                                run_data['earned_score'] = correct_earned_score
                                modified = True
            
            # 总是重新计算汇总统计（确保统计数据与earned_score一致）
            if 'question_statistics' in stats:
                stats_updated = False
                for question_id, question_data in stats['question_statistics'].items():
                    if 'runs' in question_data:
                        runs_data = question_data['runs']
                        if runs_data:
                            # 重新计算统计指标
                            scores = [run_data['earned_score'] for run_data in runs_data.values()]
                            fine_scores = [run_data['fine_grained_score'] for run_data in runs_data.values()]
                            coarse_scores = [run_data['coarse_grained_score'] for run_data in runs_data.values()]
                            
                            new_mean = round(np.mean(scores), 4)
                            new_std = round(np.std(scores), 4)
                            new_min = round(min(scores), 4)
                            new_max = round(max(scores), 4)
                            new_median = round(np.median(scores), 4)
                            new_mean_fine = round(np.mean(fine_scores), 4)
                            new_mean_coarse = round(np.mean(coarse_scores), 4)
                            
                            # 计算score_rate
                            max_points = question_data.get('max_points', 1)
                            new_score_rate = round((new_mean / max_points * 100) if max_points > 0 else 0, 2)
                            
                            # 更新statistics子字段
                            if 'statistics' not in question_data:
                                question_data['statistics'] = {}
                            
                            question_data['statistics']['mean_score'] = new_mean
                            question_data['statistics']['std_score'] = new_std
                            question_data['statistics']['min_score'] = new_min
                            question_data['statistics']['max_score'] = new_max
                            question_data['statistics']['median_score'] = new_median
                            question_data['statistics']['score_rate'] = new_score_rate
                            question_data['statistics']['mean_fine_score'] = new_mean_fine
                            question_data['statistics']['std_fine_score'] = new_std_fine
                            question_data['statistics']['min_fine_score'] = new_min_fine
                            question_data['statistics']['max_fine_score'] = new_max_fine
                            question_data['statistics']['median_fine_score'] = new_median_fine
                            question_data['statistics']['mean_coarse_score'] = new_mean_coarse
                            question_data['statistics']['std_coarse_score'] = new_std_coarse
                            question_data['statistics']['min_coarse_score'] = new_min_coarse
                            question_data['statistics']['max_coarse_score'] = new_max_coarse
                            question_data['statistics']['median_coarse_score'] = new_median_coarse
                            question_data['statistics']['num_runs'] = len(scores)
                            
                            # 也更新顶级字段（向后兼容）
                            question_data['mean_score'] = new_mean
                            question_data['std_score'] = new_std
                            question_data['min_score'] = new_min
                            question_data['max_score'] = new_max
                            question_data['median_score'] = new_median
                            
                            stats_updated = True
                
                # 如果统计数据被更新了，标记为需要保存
                if stats_updated and not modified:
                    modified = True
                    safe_print("    Statistics recalculated")
            
            # 重新计算整体统计
            if 'question_statistics' in stats and modified:
                all_scores = []
                for question_data in stats['question_statistics'].values():
                    if 'runs' in question_data:
                        for run_data in question_data['runs'].values():
                            all_scores.append(run_data['earned_score'])
                
                if all_scores:
                    stats['overall_statistics'] = {
                        'total_runs': len(all_scores),
                        'mean_score': round(np.mean(all_scores), 4),
                        'std_score': round(np.std(all_scores), 4),
                        'min_score': round(min(all_scores), 4),
                        'max_score': round(max(all_scores), 4),
                        'median_score': round(np.median(all_scores), 4)
                    }
            
            # 保存修复后的文件和重新生成Excel
            if modified:
                # 备份原文件
                backup_file = stat_file.with_suffix('.json.backup')
                import shutil
                shutil.copy2(stat_file, backup_file)
                
                # 保存修复后的文件
                with open(stat_file, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, ensure_ascii=False, indent=4)
                
                safe_print("    Fixed, backup saved as: {}".format(backup_file.name))
                fixed_count += 1
            else:
                safe_print("    No fix needed")
            
            # 总是重新生成Excel文件（确保Excel与JSON一致）
            # try:
            self._regenerate_excel_files(stat_file.parent, stats)
            safe_print("    Excel files regenerated")
            # except Exception as e:
            #     safe_print("    Excel regeneration failed: {}".format(e))
                    
            # except Exception as e:
            #     safe_print("    Fix failed: {}".format(e))
        
        # 重新生成总汇总表格
        if fixed_count > 0:
            self._regenerate_summary_table(base_path)
        
        safe_print("Fix completed! Fixed {} statistics files".format(fixed_count))
        return fixed_count

    def _regenerate_excel_files(self, output_dir, multi_run_stats):
        """重新生成Excel统计文件"""
        # try:
        import shutil
        # 提取文件前缀（从数据集名称获取，不依赖目录名）
        file_prefix = multi_run_stats.get('dataset_key', 'dataset')
        print(f"{file_prefix=}")
        if not file_prefix or file_prefix == 'dataset':
            # 如果没有dataset_key，从目录名推断数据集名称
            dir_name = output_dir.name
            if "_multiple_runs" in dir_name:
                # 从目录名中提取数据集名称（去掉模型前缀）
                temp_name = dir_name.replace("_multiple_runs", "")
                # 查找已知的数据集名称
                known_datasets = ['apho_2025', 'apho_2024', 'ipho_2025', 'ipho_2024', 
                                'eupho_2025', 'eupho_2024', 'nbpho_2025', 'nbpho_2024',
                                'panpho_2025', 'panpho_2024', 'panpho_mechanics_2025', 'panpho_mechanics_2024',
                                'fma_2025', 'fma_2024']
                for dataset in known_datasets:
                    if dataset in temp_name:
                        file_prefix = dataset
                        break
                else:
                    file_prefix = temp_name
            else:
                file_prefix = dir_name
        
        # 重新生成题目统计Excel
        question_stats_list = []
        fine_grained_question_stats_list = []
        coarse_grained_question_stats_list = []
        if 'question_statistics' in multi_run_stats:
            for question_id, question_data in multi_run_stats['question_statistics'].items():
                # 收集每次运行的得分（使用修复后的earned_score）
                run_scores = {}
                fine_grained_run_scores = {}
                coarse_grained_run_scores = {}
                for run_dir in multi_run_stats.get('run_files', []):
                    if run_dir in question_data.get('runs', {}):
                        run_scores["{}_score".format(run_dir)] = question_data['runs'][run_dir]['earned_score']
                        fine_grained_run_scores["{}_score".format(run_dir)] = question_data['runs'][run_dir]['fine_grained_score']
                        coarse_grained_run_scores["{}_score".format(run_dir)] = question_data['runs'][run_dir]['coarse_grained_score']
                    else:
                        run_scores["{}_score".format(run_dir)] = 0.0
                        fine_grained_run_scores["{}_score".format(run_dir)] = 0.0
                        coarse_grained_run_scores["{}_score".format(run_dir)] = 0.0
                
                question_text = question_data.get('question', '')
                if len(question_text) > 100:
                    question_text = question_text[:100] + '...'
                
                # 从statistics子字段获取统计信息，如果不存在则从question_data直接获取
                stats = question_data.get('statistics', question_data)
                
                question_stats_list.append({
                    'question_id': question_id,
                    'question': question_text,
                    'max_points': question_data.get('max_points', 0),
                    'mean_score': stats.get('mean_score', 0),
                    'std_score': stats.get('std_score', 0),
                    'min_score': stats.get('min_score', 0),
                    'max_score': stats.get('max_score', 0),
                    'score_rate': stats.get('score_rate', 0),
                    'num_runs': stats.get('num_runs', 0),
                    **run_scores
                })
                fine_grained_question_stats_list.append({
                    'question_id': question_id,
                    'question': question_text,
                    'max_points': question_data.get('max_points', 0),
                    'mean_score': stats.get('mean_fine_score', 0),
                    'std_score': stats.get('std_fine_score', 0),
                    'min_score': stats.get('min_fine_score', 0),
                    'max_score': stats.get('max_fine_score', 0),
                    'score_rate': stats.get('score_rate', 0),
                    'num_runs': stats.get('num_runs', 0),
                    **fine_grained_run_scores
                })
                coarse_grained_question_stats_list.append({
                    'question_id': question_id,
                    'question': question_text,
                    'max_points': question_data.get('max_points', 0),
                    'mean_score': stats.get('mean_coarse_score', 0),
                    'std_score': stats.get('std_coarse_score', 0),
                    'min_score': stats.get('min_coarse_score', 0),
                    'max_score': stats.get('max_coarse_score', 0),
                    'score_rate': stats.get('score_rate', 0),
                    'num_runs': stats.get('num_runs', 0),
                    **coarse_grained_run_scores
                })
        
        if question_stats_list:
            question_stats_df = pd.DataFrame(question_stats_list)
            question_stats_file = output_dir / "question_statistics.xlsx"
            print(f"{question_stats_file=}")
            # 备份原Excel文件
            if question_stats_file.exists():
                backup_excel = question_stats_file.with_suffix('.xlsx.backup')
                shutil.copy2(question_stats_file, backup_excel)
            
            question_stats_df.to_excel(question_stats_file, index=False)
            safe_print("    Regenerated question statistics Excel: {}".format(question_stats_file.name))
        
        if fine_grained_question_stats_list:
            fine_grained_question_stats_df = pd.DataFrame(fine_grained_question_stats_list)
            fine_grained_question_stats_file = output_dir / "fine_grained_question_statistics.xlsx"
            fine_grained_question_stats_df.to_excel(fine_grained_question_stats_file, index=False)
            safe_print("    Regenerated fine_grained question statistics Excel: {}".format(fine_grained_question_stats_file.name))
        
        if coarse_grained_question_stats_list:
            coarse_grained_question_stats_df = pd.DataFrame(coarse_grained_question_stats_list)
            coarse_grained_question_stats_file = output_dir / "coarse_grained_question_statistics.xlsx"
            coarse_grained_question_stats_df.to_excel(coarse_grained_question_stats_file, index=False)
            safe_print("    Regenerated coarse_grained question statistics Excel: {}".format(coarse_grained_question_stats_file.name))
        
        # 重新生成运行汇总Excel
        run_summary_list = []
        if 'run_files' in multi_run_stats and 'question_statistics' in multi_run_stats:
            for run_dir in multi_run_stats['run_files']:
                total_score = 0
                max_possible = 0
                for question_data in multi_run_stats['question_statistics'].values():
                    if run_dir in question_data.get('runs', {}):
                        total_score += question_data['runs'][run_dir]['earned_score']
                    max_possible += question_data.get('max_points', 0)
                
                score_rate = (total_score / max_possible * 100) if max_possible > 0 else 0
                
                run_summary_list.append({
                    'run_dir': run_dir,
                    'total_score': round(total_score, 4),
                    'max_possible_score': round(max_possible, 4),
                    'score_rate': round(score_rate, 2)
                })
        
        if run_summary_list:
            run_summary_df = pd.DataFrame(run_summary_list)
            run_summary_file = output_dir / "run_summary.xlsx"
            # 备份原Excel文件
            if run_summary_file.exists():
                backup_excel = run_summary_file.with_suffix('.xlsx.backup')
                shutil.copy2(run_summary_file, backup_excel)
            
            run_summary_df.to_excel(run_summary_file, index=False)
            safe_print("    Regenerated run summary Excel: {}".format(run_summary_file.name))
                
        # except Exception as e:
        #     safe_print("    Excel file regeneration failed: {}".format(e))

    def _regenerate_summary_table(self, base_path):
        """重新生成总汇总表格"""
        # try:
        import shutil
        safe_print("Regenerating summary table...")
        
        # 收集所有修复后的统计数据
        summary_table = []
        stat_files = list(base_path.glob("**/*multi_run_statistics.json"))
        
        for stat_file in stat_files:
            # try:
            with open(stat_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            
            # 计算总体统计（基于修复后的earned_score）
            all_scores = []
            total_max_points = 0
            
            if 'question_statistics' in stats:
                for question_data in stats['question_statistics'].values():
                    total_max_points += question_data.get('max_points', 0)
                    if 'runs' in question_data:
                        for run_data in question_data['runs'].values():
                            all_scores.append(run_data['earned_score'])
            
            if all_scores:
                total_score = sum(all_scores)
                mean_score = np.mean(all_scores)
                std_score = np.std(all_scores)
                num_runs = len(stats.get('run_files', []))
                print(f"{total_score=}, {total_max_points=}, {num_runs=}")
                score_rate = (total_score / (total_max_points * num_runs) * 100) if total_max_points > 0 else 0
                
                summary_table.append({
                    'dataset': stats.get('dataset_name', stats.get('dataset_key', 'Unknown')),
                    'num_runs': num_runs,
                    'total_score': round(total_score, 2),
                    'max_possible_score': round(total_max_points * num_runs, 2),
                    'mean_score': round(mean_score, 4),
                    'std_score': round(std_score, 4),
                    'mean_score_rate': round(score_rate, 2),
                    'std_score_rate': round(std_score / total_max_points * 100 if total_max_points > 0 else 0, 2)
                })
                    
            # except Exception as e:
            #     safe_print("    Error processing {}: {}".format(stat_file, e))
        
        # 保存总汇总表格
        if summary_table:
            summary_table_df = pd.DataFrame(summary_table)
            summary_table_file = base_path / "multi_run_summary_table.xlsx"
            
            # 备份原文件
            if summary_table_file.exists():
                backup_file = summary_table_file.with_suffix('.xlsx.backup')
                shutil.copy2(summary_table_file, backup_file)
            
            summary_table_df.to_excel(summary_table_file, index=False)
            safe_print("    Summary table updated: {}".format(summary_table_file))
        
        # except Exception as e:
        #     safe_print("    Summary table generation failed: {}".format(e))