"""
PhysicsVerifier.rules

可插拔规则插件系统：
- base: 定义 RulePlugin 接口、RuleContext 数据结构、RuleRuntime 运行时工具。
- llm_rules: 以 LLM 为主的规则实现示例（变量/常量一致性、公式正确性、前提一致性）。
- graph_consistency: 纯本地静态图一致性检查，示例如何实现非 LLM 规则。

使用方式：
- 在 RuleBasedVerifier 中通过规则规范字符串加载（内置别名或模块路径）。
"""

from .base import RulePlugin, RuleContext, RuleRuntime  # noqa: F401
