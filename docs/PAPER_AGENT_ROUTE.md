# 论文机制复现路线 R（2026-09-21）

状态：R0可运行的合成机制原型；不是论文数值复现，不是已接入公开预训练权重的Agent。主目录dreamer_agent/paper_agent/，独立于HistoryPolicy与VPT探索分支。用户授权尝试论文机制作为新路线，因此本路线包含必要的监督Agent适配阶段；此前VPT直接RL“不做监督训练”的限制仍适用于那个独立试验，不将两者混用。

## 来源与逐项对应

主要规格：https://arxiv.org/html/2509.24527v1#S3.SS3（Agent适配与RL），第3.4节（时空注意力与训练长度），第4.1节（Minecraft任务/动作/混合目标）。twohot核对作者Dreamer3：https://github.com/danijar/dreamerv3/blob/main/embodied/jax/outs.py ，以及configs.yaml的symexp_twohot配置。未复制第三方训练实现。

| 机制 | 本次实现 | 限制/未完成 |
|---|---|---|
| task embedding→Agent Token | 每帧一个Agent Token，20任务 | 模态数量/维度为原型配置 |
| 单向注意力 | Agent读当前及历史世界/Agent；世界不得读Agent | dense参考实现；未复现论文轴向时空层/GQA/RMSNorm/RoPE |
| 防标签泄漏 | 动作输入由标签右移；episode起点显式unknown；未来全屏蔽 | 这是原型的时间定义，真实WM动作条件偏移仍需认证 |
| 世界输出任务不变 | 在固定权重、输入动作、噪声时检查任务干预差为0 | 联合优化共享参数仍会改变动力学；注意力隔离不保证训练不遗忘 |
| Policy MTP | 默认23个二类分布+121类camera，8目标 | 二类softmax与Bernoulli分布等价，但不是单logit参数化；真实动作到WM codec尚未接通 |
| Reward/Value symexp twohot | 原始标量在symexp间隔bin内线性分配概率；对称求和解码 | 255 bins/±20为明确工程配置，未认定为Dreamer4精确超参；value只建结构，未实训RL |
| 联合监督目标 | relevant片段BC，uniform片段基本flow x预测，已知奖励MTP；未知奖励mask | smoke数据50/50按序列；正式数据采样、权重、shortcut bootstrap尚未实现 |
| MTP边界 | task切换/episode/无效帧截断；切走再切回也不跨越 | 8目标采用n=0..7；论文称L=8而公式0..L存在索引歧义，保留待核对 |
| RL冻结方式 | freeze_for_rl仅开放policy/value，测试其余参数不变 | prior、分布价值lambda-return和Agent Token rollout collector尚需组装，不调用旧VPT奖励 |

## 阶段与验收

R0（本次）：可测试注意力、Agent Token、动作/奖励MTP、twohot以及合成联合训练。原型禁止自动加载真实checkpoint；synthetic_only.pt不得进入正式训练或评估。

R1：公开模型接入。重新实现兼容的训练前向和单向空间/时间注意力，严格映射预训练参数；保留原始模型和权重。原实现SpatialAttention是帧内全连接，TemporalAttention只让图像Token读时间cache，且每帧273Token硬编码，不能直接拼一个Agent Token。需独立处理Agent位置编码和cache、任务干预不变、未来泄漏、reset/branch重放以及原模型输出等价；真实权重通过这些测试后，才声称完成结构接入。不用strict=False掩盖缺失骨干参数。

R2：离线Agent适配。复用审核6xx—10xx来源及原split，重建20任务、23按键/121相机与WM动作映射；保持固定时间轴。优先小规模过拟合/独立验证，之后才20%→全量。动作、奖励与视频联合损失；奖励标签不能由task名字自动生成，原木“看到/获得”困难负例必须存在。固定骨干只训新增模块可作为低成本对照，但不称论文完整联合微调。现有GRU的128窗口不直接决定此路线；192上下文和64/256动力学训练长度分别登记，依据硬件测试决定首次可承受配置。

R3：校准与真实短任务BC基线。检查奖励精度、召回、会话隔离及高分误报，评估动作与物品持久性。分布式twohot不是可靠奖励证书。

R4：组装Agent Token原生想象RL。固定监督末轮prior，冻结Transformer/reward，训练policy/value；接PMPO、分布价值lambda-return、明确截断、root采样、完整续训及同种子真实对照。短horizon16先验收，再32/64。固定任务或手写顺序允许，P3自主调度和P4搜索仍由真实增益解锁。

## 运行与证据

`<LOCAL_ASSET_PATH> -m dreamer_agent.paper_agent.smoke --output <新目录>`

本次输出outputs/paper_agent_mechanisms_20260921/。固定合成batch、CPU40次优化，仅验证目标有梯度及任务不污染世界读出；loss下降不代表泛化、真实奖励能力或钻石成功。tests.log为工程回归。既有训练进程及已有资产未被替换。
