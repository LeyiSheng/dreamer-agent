# 自有训练逻辑与源码地图

本页按同步到本地的实际源码整理。算法来源分别引用 Dreamer、VPT 等工作；项目特色主要体现为数据契约、训练组合、校准链路与模型适配。

## 1. 数据与任务采样

- `sources.py`、`alignment_diagnostic.py`、`video_audit.py`：来源登记、动作时序与视频检查。
- `event_coverage.py`、`task_mix_data.py`、`task_mix_windows.py`：任务事件、分层子集选择、通用覆盖、连续窗口与 MTP 标签边界。
- `train_task_mix_v2.py`：任务加权与跨卡有效样本质量归一化，含空 rank 的梯度一致性测试。
- `vpt_feature_cache.py`、`v7_data.py`：冻结 VPT 特征与离线事件缓存；原始会话哈希划分 fit/select/confirm；从事件前窗口产生技能标签。

这些模块承接先小子集对照、再扩大语料的实验流程。真实评估数据与训练来源隔离。

## 2. 奖励与信念

- `vpt_heads.py`：`SignalHeads`、`SkillPolicy`、`TwoHotValue`。
- `train_vpt_heads.py`：集成事件/信念头训练，独立会话阈值搜索和规则校准。
- `reward_rule.py`：检测、确认、阈值及事件匹配。
- `v7_agent.py`：推断账本、技能选择与执行状态。

奖励从感知信号和确认规则产生；真实库存用于离线标签与评估评分。部署策略使用像素及其历史表征。

## 3. 技能行为克隆

入口 `train_vpt_skill_bc.py`：冻结 VPT 骨干，训练复制的动作头与零初始化技能 FiLM。`SkillSampler` 混合技能相关与通用帧；动作监督使用 VPT joint log-prob，通用帧加入相对原策略的 KL。`offline_rl.py` 提供可选离线 AWR 权重对照。

## 4. 想象强化学习

入口 `train_vpt_imagination.py`，由 `torchrun` 启动：

1. 加载经过登记的世界模型、VPT、奖励头和规则。
2. 从离线数据的技能起点预填充历史。
3. `world_v7.py` 的 `AlignedWorldAdapter` 管理动作进入/离开帧的关系；`vpt_rl.py` 转换动作协议。
4. 世界模型生成画面，经 `vpt_features.py` 统一预处理送入 VPT。
5. 计算逐步奖励、lambda-return 与优势。
6. 优化策略项 + KL 先验项 + 离线 BC 项，以及 two-hot 价值损失；多卡同步梯度。

代码中的策略项按正负优势分别归一，并设 deadzone；周期性 prior rollout 用于对照记录。恢复逻辑、采样器、校准文件与 checkpoint 一起纳入实验协议。

## 5. 世界模型微调分支

- `wm_finetune.py`：冻结基础矩阵，给后部空间注意力、FFN 和输出层加入低秩更新；显式区分旧目标和 flow/shortcut 目标版本。
- `wm_visual.py`：当前帧可微解码、L1/LPIPS、最后去噪步骤反传；历史缓存及时间分支的梯度范围在模块注释中声明。
- `wm_visual_trial.py`：原模型、shortcut、shortcut+视觉三臂配对实验。
- `verify_wm_visual.py`：前向等价、梯度和显存预检。

最近小试验采用 rank 16、alpha 32、最后四层、每臂 16 更新。其原始工程记录保存在工作目录，公开结果需配套完整协议和授权后的产物。

## 6. 其他研究路径与测试

`paper_agent/` 提供 agent-token 路线的模型与目标函数模块；部署集成和真实训练验收按该分支协议推进。

`train_bc*`、`train_reward*`、`train_imagination.py`、`vpt_direct_rl.py`、`vpt_budget_rl.py` 保留早期实验实现和相应回归，阅读主线时优先上述 VPT 技能路径。

`eval_long.py`、`complete_vpt_evaluation.py` 等提供环境评估与续跑管理。测试中的合成输入用于核对梯度、边界、缓存和统计语义。
