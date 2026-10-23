# 训练环境与资产准备

## 环境

核心安装使用 `pip install -e .`，包含 PyTorch、NumPy 和 OpenCV。世界模型视觉分支可安装 `pip install -e '.[visual]'`。完整模型训练建议 Linux、NVIDIA CUDA 与匹配的 PyTorch。

第三方依赖独立放置：

- `Dreamerv4-MC/`：与所用权重兼容的推理源码；按该项目安装 diffusers、FlashAttention 等依赖。
- `third_party/Video-Pre-Training/`：VPT 官方实现及其要求的依赖。
- Minecraft 评估环境：按 VPT/MineRL 的版本组合准备，配置 Java 8 和 Xvfb。

配置 `JAVA_HOME` 指向 Java 8；`DREAMER_EVAL_PYTHON` 可选择评估环境解释器，默认当前 Python。第三方仓库、模型、数据的授权信息见 THIRD_PARTY.md。

## 资产约定

源码保留原实验的相对路径，首次运行需要生成本机清单：

| 资产 | 位置 / 入口 |
|---|---|
| VPT 网络配置 | `checkpoints/foundation-model-3x.model` |
| BC 策略起点 | `checkpoints/bc-early-game-3x.weights` |
| 世界模型权重与哈希 | `outputs/acceptance_20260913/assets.json` |
| 动作规格 | `outputs/acceptance_20260913/action_probe.json` |
| 来源/任务记录 | `outputs/p1c_dreamer4_table4_audit_v1/training_records.jsonl` |
| VPT 特征缓存 | 默认 `outputs/v7_vpt3x_feat_20260926/`，或环境变量 `DREAMER_FEATURE_CACHE` |
| 想象起点 | 特征缓存目录下 `roots/*.npz` |

数据准备入口为 `sources`、`prepare_task_bc`、`vpt_feature_cache`；资产与动作验收入口为 `q0_assets`、`p0_actions`、`verify_world_adapter`。用 `python -m dreamer_agent.<模块> --help` 查看各入口参数。来源审核、动作对齐及奖励校准沿用研究计划的要求。

特征缓存包含 `selection.jsonl`、`files/<stem>.h.npy`、`files/<stem>.npz`。读写格式由 `vpt_feature_cache.py` 和 `v7_data.py` 定义。下载权重后重新登记本机路径与 SHA-256。

## 训练命令

以下命令在资产、数据与校准准备完成后执行。每次使用新的输出目录。

```bash
python -m dreamer_agent.train_vpt_heads --output outputs/public_heads
python -m dreamer_agent.train_vpt_skill_bc --output outputs/public_skill_bc

torchrun --standalone --nproc_per_node=4 -m dreamer_agent.train_vpt_imagination   --heads outputs/public_heads/heads.pt   --rule outputs/public_heads/rule.json   --skill-policy outputs/public_skill_bc/skill_policy.pt   --output outputs/public_imagination
```

想象 RL 的资源配置由实际吞吐、显存和起点数确定。世界模型微调的独立命令见 `python -m dreamer_agent.wm_finetune --help` 和 `python -m dreamer_agent.wm_visual_trial --help`。

完整环境依赖按模型版本记录锁定；CPU 测试只安装核心包即可执行。使用已审核来源的 checkpoint 和 pickle 模型配置。
