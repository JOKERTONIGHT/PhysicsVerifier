"""规则插件基础接口与运行时封装。

核心目标：
1. 统一输入：符号节点网络 (SymbolGraph)、原始样本文本、预提取上下文。
2. 统一输出：diagnostics 列表，每项包含 {severity, rule, symbol, message, evidence}。
3. 统一运行时：提供 LLM JSON 调用、缓存、日志等能力给规则使用。

扩展方式：
实现一个继承 RulePlugin 的类，并注册到 RuleRegistry 或通过动态导入路径引用。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol


@dataclass
class RuleContext:
    sample_id: Optional[str]
    dataset_key: Optional[str]
    text_all: str
    lines: List[str]
    symbols: List[str]
    formulas_raw: List[str]
    graph: Any  # 使用 Any 以避免循环导入（实际为 SymbolGraph）
    snippets: Dict[str, List[str]]
    sym_stats: Dict[str, Any]
    precondition_cues: List[str]


@dataclass
class RuleRuntime:
    llm_model: Optional[str]
    max_llm_calls: int
    llm_calls_used: int = 0
    llm_temperature: float = 0.2
    llm_max_output_tokens: int = 4096
    logger: Any = None
    cache_get: Any = None
    cache_set: Any = None
    llm_json: Any = None

    def log(self, *args):
        if self.logger is not None:
            try:
                self.logger.info(" ".join(map(str, args)))
            except Exception:
                pass

    def llm_available(self) -> bool:
        return callable(self.llm_json) and self.llm_calls_used < self.max_llm_calls


class RulePlugin(Protocol):
    id: str
    title: str
    description: str

    def run(self, ctx: RuleContext, rt: RuleRuntime) -> List[Dict[str, Any]]:
        """执行规则并返回规范化 diagnostics 列表。
        若无问题，返回空列表。"""
        ...


class RuleRegistry:
    """简单注册表，支持按 id 查找或动态添加。"""
    _rules: Dict[str, RulePlugin] = {}

    @classmethod
    def register(cls, plugin: RulePlugin) -> None:
        cls._rules[plugin.id] = plugin

    @classmethod
    def get(cls, rule_id: str) -> Optional[RulePlugin]:
        return cls._rules.get(rule_id)

    @classmethod
    def list_ids(cls) -> List[str]:
        return sorted(cls._rules.keys())
