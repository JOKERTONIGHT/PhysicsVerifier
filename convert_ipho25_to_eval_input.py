import json
import re
from pathlib import Path

def extract_boxed_answers(text):
    """
    从 LaTeX 文本中提取所有 \boxed{...} 内容，返回字符串列表。
    """
    return re.findall(r"\\boxed\{([^}]*)\}", text)

# 输入输出路径
in_path = Path("ipho25-0.json")
out_path = Path("ipho25-0_eval.json")

with in_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

results = []
for i, item in enumerate(data):
    # 题目 id（可用序号或 hash）
    qid = f"IPhO25-Q{i+1}"
    # 题干
    question = item.get("input", "")
    # 作答全文
    prediction = item.get("output", "")
    # 答案（从 output 中提取所有 boxed）
    answers = extract_boxed_answers(prediction)
    # 分值（优先 point 字段，否则默认 1.0）
    points = item.get("point", 1.0)
    if isinstance(points, (int, float)):
        points = [float(points)] * max(1, len(answers))
    elif isinstance(points, list):
        points = [float(p) for p in points]
    else:
        points = [1.0] * max(1, len(answers))
    # 标准答案（如有 extracted_gt 字段则用，否则留空）
    gt = item.get("extracted_gt", None)
    if gt:
        try:
            gt_list = json.loads(gt) if isinstance(gt, str) else gt
        except Exception:
            gt_list = [str(gt)]
    else:
        gt_list = []
    # 构造评测输入
    results.append({
        "id": qid,
        "question": question,
        "prediction": prediction,
        "answer": answers if answers else gt_list,
        "points": points,
    })

with out_path.open("w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"已生成评测输入: {out_path}，共 {len(results)} 条样本")
