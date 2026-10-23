# P2 想象 RL 代码准备（2026-09-17）

本次交付为可测试的算法核心、冻结模型桥接接口和带证据检查的运行入口。尚未提供已验收的真实资产 factory，不能称真实模型端到端 RL 已跑通；不启动训练，不改变正在执行的 P1 任务。

- imagination.py：固定 BC history 与 prior；从 BC 初始化可训练 task embedding/MTP 读出；当前动作采样、独立价值网络、PMPO、TD/lambda-return 与离线 MTP BC 混合更新。每次只接受当前策略生成的一条新轨迹，不重复利用旧 rollout。
- imagination_backend.py：FrozenWorldAdapter 桥接；生成像素重新编码到 P1 特征；BC 每32帧重置但保留上一动作输入；奖励 GRU 通过完整转移前缀重算保持记忆。BC 分块边界不确认奖励。root 必须从可追溯 BC 块边界开始，奖励完成语义从 root 最后真实观测开始。
- train_imagination.py：证据核验、显式 module:function factory、逐更新日志、原子检查点。保存 optimizer/RNG 供审计，但尚未实现自动续训；不要以 latest.pt 存在宣称可以无损恢复。

默认配置 configs/p2_imagination_v1.json：horizon16，gamma0.997，lambda0.95，PMPO alpha0.5，KL0.3，BC混合权重1，lr3e-4，连续确认2步。新增候选参数尚未调优。32/64 horizon 必须分别获得证据批准。单固定任务、每root最多一次完成奖励；技能完成不当环境终止。uncertainty_cut 丢弃当前转移并在有效前缀末端使用[0,1]自举；该范围来自本配置至多一次单位奖励，不用于其它重复奖励任务。

真实运行需要显式提供：审核过的离线训练 roots 与 BC batches；原始像素转换和与 P1 完全一致的独立帧 encode_pixels；动作到 WM 观测的时序约定；不读取真实库存的状态一致性函数；冻结的 BC/Tokenizer/Dynamics/奖励权重。风险截断可选，未接入时不声称已估计不确定性。当前后端不从完成头推断死亡；模型终止检测器尚未建立，因此真实桥接默认只按 rollout 时限截断。

证据 JSON schema_version=1, stage=P2, approved_horizons=[16]；bindings 包含 bc/reward/tokenizer/dynamics/action_spec/feature_spec/plan/integration_code 的 path 与 sha256。四份 evidence：world_model_semantics、reward_calibration、real_bc_baseline、integration_contract，同样各含 path/sha256。每份审核报告需 kind、status=accepted、reviewer、synthetic=false 及完全一致的 artifact_hashes。相对路径以证据清单所在目录为基准。审核结论必须来源于可追溯真实记录，布尔状态本身不是实验认证。

仅检查命令：
```
python -m dreamer_agent.train_imagination --config configs/p2_imagination_v1.json --evidence <真实证据清单> --check
```
通过后才可使用 --factory <已审核模块:函数> --output <新目录> --updates 100。没有默认factory、旧权重或跳过门槛的选项。factory负责按证据绑定严格加载资产和核验数据；本次没有伪造可通过的证据清单。

当前限制：未完成真实生成画面→独立帧编码→历史→奖励的集成数值验收；缺真实 BC 基线与合格奖励证据时仍不能启动 P2。CPU合成测试仅证明软件语义，不代表真实任务成功或世界模型语义可靠。

## 2026-09-17 审计后接口变更

以上初版描述由本节覆盖：ValueHead新增confirmation输入；完成目标轨迹结束且目标剩余自举为0、terminated仍保留真实环境语义。OfflineBatch替代tuple并验证训练来源及manifest binding；OfflineRoot新增provenance，必须为有源哈希/tensor绑定的单帧task-origin。完整配置/timebase/seeds/tasks/thresholds须进入证据清单，bindings新增training_data与roots，报告提供raw_records、奖励独立确认统计。run也执行检查，factory必须绑定实际artifact_hashes。

新增vpt_rl.py是显式VPT变体，冻结net、训练pi_head与新value；targets=1，保留原生buttons/camera。配置p2_vpt_experiment_v1.json和readiness.json记录小实验预算与未通过项。没有可直接启动的已验收factory；必须先准备真实root/BC manifest、动作语义和新起点奖励确认。107项软件测试及真实VPT权重读出梯度预检不替代这些实验。
