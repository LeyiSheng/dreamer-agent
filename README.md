# Dreamer Agent

基于预训练世界模型的 Minecraft 离线智能体研究：VPT 技能条件行为克隆、校准奖励、想象强化学习，以及世界模型 LoRA / 视觉微调。

## 作者

**Leyi Sheng · Jiaheng Wei · Xiaobo Xia\***



本仓库的自有源码位于 `dreamer_agent/`。本次公开准备版本从实验主工作目录提取，源文件清单及哈希见 [SOURCE_SNAPSHOT](docs/SOURCE_SNAPSHOT.json)。

## 训练逻辑导航

| 路径 | 自有实现 |
|---|---|
| 数据与任务采样 | 会话分组、20%→全量子集、相关/通用片段混合、历史和 MTP 边界 |
| 奖励链路 | 事件与库存信念头、完成确认、阈值校准、去重账本 |
| 技能 BC | 冻结 VPT 特征、技能 FiLM、动作头、通用行为 KL 保持 |
| 想象 RL | 动作时序适配、技能起点、PMPO 风格更新、KL + BC 约束、two-hot value |
| 世界模型微调 | 后四层 LoRA、flow/shortcut 目标、可微当前帧解码、视觉对照 |

具体文件、损失和入口见 **[TRAINING_LOGIC](docs/TRAINING_LOGIC.md)**；环境与资产准备见 **[TRAINING_SETUP](docs/TRAINING_SETUP.md)**。

## 安装与测试

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
python -m dreamer_agent.audit --root .
```

GitHub Actions 配置保存在 [CI 模板](docs/ci.example.yml)。具有 workflow 权限后，将其放入 `.github/workflows/ci.yml`，同步更新发布清单及忽略规则，即可启用自动测试。

核心测试使用 CPU。分布式回归使用本机多进程、共享内存和 Gloo。训练使用 Linux/CUDA；世界模型、VPT 和 Minecraft 环境分别按上游要求安装。

## 目录

- `dreamer_agent/`：自有算法、训练、评估和兼容适配。
- `dreamer_agent/paper_agent/`：内部 agent-token 路线的独立研究模块。
- `tests/`：核心语义、数据、奖励、时序及分布式回归。
- `docs/`：训练导航、环境说明、来源记录及历史研究计划。
- `scripts/release.py`：发布清单检查、源码导出和 SHA-256 清单。

V6 与 V5 保留研究基础规格；V7 文档记录后续实验路线。本轮仅整理发布与路径可移植性，算法调整和计划变更另外记录。数据、权重、第三方工作副本及运行输出通过本地资产目录管理。

## 发布与结果

```bash
python scripts/release.py check
python scripts/release.py export --output dist/dreamer-agent-source.tar.gz
```

软件验证和研究结果分别记录。实验发布字段见 [REPRODUCIBILITY](docs/REPRODUCIBILITY.md)。训练文件中的历史输出名便于追溯原实验；在新环境中按准备指南生成对应资产清单与特征缓存。

## 许可证与研究参考

自有代码采用 [MIT](LICENSE)，第三方材料见 [THIRD_PARTY](docs/THIRD_PARTY.md)。贡献方式见 [CONTRIBUTING](CONTRIBUTING.md)。

- [Dreamer 4](https://arxiv.org/abs/2509.24527)
- [Dreamerv4-MC](https://github.com/IamCreateAI/Dreamerv4-MC)
- [OpenAI VPT](https://github.com/openai/Video-Pre-Training)
- [RLVR-World](https://github.com/thuml/RLVR-World)

## 引用本项目

如果本项目的代码或训练流程对你的研究有帮助，请使用以下软件引用格式：

> Wei, J., & Xia, X. (2026). *Dreamer Agent* [Computer software]. https://github.com/LeyiSheng/dreamer-agent

BibTeX：

```bibtex
@misc{wei2026dreameragent,
  author       = {},
  title        = {{Dreamer Agent}},
  year         = {2026},
  howpublished = {Computer software},
  url          = {https://github.com/LeyiSheng/dreamer-agent}
}
```

复现实验时，请同时记录所使用的 Git commit 或发布版本。涉及上游方法与模型时，请一并引用上方列出的对应研究。
