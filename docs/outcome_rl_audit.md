# Outcome-only GRPO 审计（P0）

生成时间：`2026-09-13T13:59:38.633820+00:00`

## 本机

- `qwen3-8b-outcome-only-rl` 存在：`False`
- `qwen3-8b-hybrid-outcome-rl` 内容：`['launch.lock', 'runs', 'plots', 'hybrid_pipeline.log']`
- waiter 日志存在：`True`

waiter 尾部：

```
[grpo-wait] preflight ok mixed=0.785 trunc=0.054 boxed=0.943
[grpo-wait] prompts=/home/jinjianhan/PhysicsVerifier/data/rl/swift_prompts_hybrid_band.jsonl actor=/slow_share/jinjianhan/models/Qwen3-8B reward=outcome_only
[grpo-wait] waiting for 4 idle GPUs
```

## 缺陷对照（共享代码，与机器无关）

| 缺陷 | 本机证据 | 他机预判 |
|---|---|---|
| D1 奖励非同构 | `_check_answer` 二值 any-of | 同一份代码则同样命中 |
| D2 步数不足 | onset 100 步 kl≤0.0011 | 若 MAX_STEPS≤100 则同样 |
| D3 长度预算 | onset clipped_ratio≈0.48 @1536 | full 模式原 3072，评测 8192 |
| D4 测评分辨率 | 65 题 SE=0.054 | 若用 heldout_eval_trusted.jsonl 则同样 |

If the other-machine run used launch_hybrid_grpo_4gpu.sh MODE=full before this overhaul, it inherited max_completion_length=3072, MAX_STEPS=30, binary any-of grading, and the 65-item heldout.

## 搜索命中

- `/slow_share/jinjianhan/ckpt/qwen3-8b-deepseek-v4-flash-grpo-onset-smoke/swift_launch_report.json`
  `{"phase": "launched", "reward_mode": "llm_step_score", "max_steps": 2, "max_completion_length": 1536, "num_generations": 6, "overlong_filter": null}`
- `/slow_share/jinjianhan/ckpt/qwen3-8b-deepseek-v4-flash-grpo-onset/swift_launch_report.json`
  `{"phase": "launched", "reward_mode": "llm_step_score", "max_steps": 100, "max_completion_length": 1536, "num_generations": 6, "overlong_filter": null}`
- `/slow_share/jinjianhan/ckpt/qwen3-8b-physics-swift/swift_launch_report.json`
  `{"phase": "launched", "reward_mode": null, "max_steps": 0, "max_completion_length": null, "num_generations": 6, "overlong_filter": null}`
- `/slow_share/jinjianhan/ckpt/qwen3-8b-hybrid-outcome-pilot10/swift_launch_report.json`
  `{"phase": "launched", "reward_mode": "hybrid_llm_outcome", "max_steps": 10, "max_completion_length": 2048, "num_generations": 6, "overlong_filter": "false"}`
- `/slow_share/jinjianhan/ckpt/qwen3-8b-hybrid-outcome-pilot10/physics_reward_metrics.jsonl`
- `/slow_share/jinjianhan/ckpt/qwen3-8b-hybrid-outcome-pilot10/v0-20260831-062701/logging.jsonl`
  `{"n_reward_rows": 1, "reward_first": 0.06558334, "reward_last": 0.06558334, "kl_max": 0.0, "clipped_ratio_mean": 0.20833333, "last_keys": ["clip_ratio/high_max", "clip_ratio/high_mean", "clip_ratio/low_mean", "clip_ratio/low_min", "clip_ratio/region_mean", "completions/clipped_ratio", "completions/max_length", "completions/mean_length", "completions/min_length", "elapsed_time", "epoch", "frac_reward_zero_std", "global_step/max_steps", "grad_norm", "kl", "learning_rate", "loss", "memory(GiB)", "remaining_time", "reward"], "epoch_last": 0.00173611, "max_completion_hint": 2048.0}`

