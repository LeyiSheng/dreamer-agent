# V6/V4.1综合优化实施说明

## 取舍

目标是冻结世界模型下的可靠技能学习与低成本执行。V4.1的技能抽象、鲁棒选择、搜索蒸馏作为方向保留；V6的历史状态、持续时间折扣、独立校准、风险截断和分阶段对照作为实施规则。不存在由计划文字直接证明的“最优架构”。本轮选择优先消除已确认的接口与重复计算问题；搜索放在低层能力和模型语义有证据后实施。

|V4.1思想|采用方式|不采用的假设|
|---|---|---|
|技能级规划|固定序列collector、持续时间宏观回报、episode级去重账本|技能名字本身代表已会执行；宏观决策压缩等于WM推演提速|
|技能可行性约束|soft penalty + 显式高置信排除 + 全屏蔽回退|用不可靠库存推断无条件hard mask|
|防止想象失真|保留无效转移丢弃/有界bootstrap，接入显式outcome信号|与起始画面的视觉差即为预测误差|
|价值保守性|保持后续独立风险正则消融|使用无约束的线性V差并称CQL|
|搜索教师与蒸馏|维持P4及等算力验收|现在接入MCTS就能弥补奖励不可靠|

## 实际改动

1. learning.bc_loss新增normalization，默认valid_mean；anchor_sum对所有有效目标NLL求和、按至少一个有效标签的锚点数归一化。先计算计数比例再缩放，避免中间低精度大数溢出。RLConfig保存同名bc_normalization，正式证据完整配置比较会包含该值。
2. HistoryPolicy.encode_step及TemporalRewardCompletionHeads.forward_step提供冻结eval专用的增量推理，不改参数结构或监督forward。StreamingReadout分别保存policy/reward state；不同实例不共享可变状态。
3. HistoryRewardContract按名称映射策略/奖励任务，绑定context、权重、特征、动作及重置协议。verify_training_metadata要求原训练产物记录一致；缺少元数据的旧奖励不能通过补维度/改名字直接认证。
4. FrozenFeatureBackend新增contract路径；不传contract继续原32帧实现。新路径对root.runtime_contract_sha256、特征指纹、已加载模块和训练语义进行核验，允许已绑定context128。当前仍只接受单帧task-origin root。
5. outcome(prev_history,next_history,action,task)可返回terminated/timeout/uncertainty；校验类型/范围并在终止后使backend失效。回调不是已训练的终止或风险检测器；没有回调时仍沿用历史默认，必须在真实集成中声明缺失。
6. begin_skill保留WM上下文，在当前帧按task-origin契约重置低层及奖励历史；decision_history提供交接后的同一因果历史定义。
7. collect_fixed_skills连接真实backend接口、动作采样和SkillExecutor，受总模型调用/每技能步数双预算限制。只接受offline train root和可映射的显式技能/阈值；确认完成后才推进，失败、截断或超时不伪装为完成。真实库存不作为输入。
8. MacroTransition记录实际gamma^duration、折扣里程碑奖励、起止历史及achieved_before/after。高层完成后仍可bootstrap；低层单目标完成后价值0的旧语义不改变。风险边界必须提供适合全局回报的上下界。
9. feasible_logits是独立高层接口准备：不确定技能软惩罚，高置信不可行技能屏蔽，全屏蔽时明确开放观察/重选fallback并返回标记。当前没有学习式高层策略/信念模型调用它。

## 接入约定

- 新正式factory在加载并冻结BC/reward后读取HistoryRewardContract，向FrozenFeatureBackend传入contract、policy_metadata、reward_metadata，以及已审核consistency/outcome。
- contract的policy_state_sha256/reward_state_sha256通过module_sha256计算，指纹含完整state_dict名称、dtype、shape、数值及buffer。它只证明内容绑定，不证明训练质量。
- policy_metadata至少含tasks和context。reward_metadata必须来自奖励训练记录，含policy_state_sha256、tasks、context、feature_contract_sha256、action_spec_sha256、history_protocol、reward_protocol；禁止为旧权重虚填这些记录以通过核验。
- 原root内容审核保持，并额外携带runtime_contract_sha256。policy任务索引按名称映射到reward索引，不要求二者恰好相同；general_behavior不得进入reward。
- collect_fixed_skills的thresholds仍需真实校准；本轮没有提供“默认成功阈值”或可绕过正式训练门槛的启动器。合成测试使用0阈值只用于确定性验证控制流，不保存为训练配置。
- 新配置字段/代码hash属于新的运行协议，旧严格resume出现版本不一致应拒绝，而不是删除检查；旧权重和输出保留。未修改VPT评估器、vpt_rl适配文件或已完成RL权重。

## 验收与局限

本轮测试覆盖：两种BC归一化手算与无效mask梯度、权重/任务/元数据/root不兼容拒绝、32/128历史跨界输出、独立实例状态、线性递归工作量、目标切换保留WM、跨技能奖励去重、持续时间折扣、完成与timeout bootstrap、terminal取消、自疑截断丢弃及前缀约束、全部技能屏蔽回退和硬预算。

CPU计数与计时只衡量policy/reward读出；不包含Tokenizer、Dynamics或Minecraft，因此不推断整体GPU节省比例。真实权重+合成特征的等价性诊断只能说明数值兼容，不能说明真实像素校准。

尚缺：新BC配套奖励训练/校准及真实语义、长真实上下文root、正式runner完整恢复、高层自主策略/价值训练、可行性学习、搜索与蒸馏。本次没有启动监督训练、想象RL或另一次Minecraft评估。原有真实评估是否完成按其独立输出判断，不由本轮单测替代。
