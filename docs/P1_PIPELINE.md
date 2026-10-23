# P1 监督训练流水线

远端主目录：<SSH_HOST>:<PROJECT_ROOT>。所有命令在该目录运行。

## 当前阶段补充（2026-09-14）

第一阶段基础BC不需要技能或奖励标签；使用prepare_bc生成policy_mode=unconditioned配置。run_p1默认--stage bc，不再强制训练奖励；下文任务条件完整流水线需显式--stage all。详见P1A_ALIGNMENT_REVIEW_20260914.md及PLAN_V6第14节。下文reviewed/task/event要求只适用于任务条件阶段，不是所有BC的门槛。

## 已实现的步骤

1. prepare_annotations：从筛查注册表选取不同原始会话，生成未审核标注模板，不触碰test集合。
2. export_vpt：校验审核声明、来源哈希、会话切分和片段不重叠；完整解码视频，按官方同序号协议配对。保留空动作与20Hz时间轴，T+1情况下最后一帧不进入有标签BC样本。
3. train_bc：用冻结视觉特征、过去动作和技能标签训练时序策略，保存固定版本BC checkpoint。
4. train_reward：严格载入对应BC并冻结；训练独立事件/完成logit头。未知标签mask、任务切换、clip/context末尾无后继的转移不参与监督。
5. 奖励校准：校准会话ID排序后，偶数位置用于阈值选择，奇数位置用于固定阈值确认。训练前规则固定，确认数据不参与选阈值；test未用于阈值拟合。报告每技能、每头、每会话指标、Brier、精确率区间及高分尾部误报。帧相关性仍限制IID区间解释，报告不会自动授权RL。

## 标注协议

annotations.json包含schema_version=1、origin=offline、reviewed、reviewer、pairing_protocol=vpt_same_index_v1、tasks技能名列表、clips记录。

每条clip包含episode_id、source_episode_id、split、origin、video_path/actions_path/labels_path及对应SHA256，以及零基[start,stop)范围。路径相对annotations文件解析。同一原始会话不能跨split，原始文件同内容不能跨split，同视频clip不能重叠。

labels文件是长度stop-start的JSON数组。每项严格为：
```json
{"task": 0, "event": null, "completion": null}
```

task是tasks列表索引，必须有审核依据。event与completion为0/1/null；null表示未知，不当负例。第t项event表示frame[t]到frame[t+1]的技能事件，completion表示该转移后的技能完成；二者不代表环境终止。事件证据来自已审核离线记录，禁止把真实评估轨迹回流。给定原始记录的库存或统计可作为离线监督依据，但不进入策略输入。

模板默认reviewed=false且task=null，不能直接训练。补齐真实标注后需更新labels_sha256并记录reviewer。不得为跑通而伪造审核、默认技能、或把所有未知事件标成0。训练要求每个声明任务的两个头都具有正负有效转移；若数据不足，应依据证据缩小任务集。

## 冻结视觉提取

本版采用独立单帧encoder调用，输出全部token展平为8192维；历史由BC的GRU承担。原始RGB 640×360映射到[-1,1]后上下各补12行数值0，形成384×640，不叠加外部GUI光标。该约定写入特征指纹，属于本项目P1初始方案，不等同VPT的128×128预处理。

独立单帧路径避免跨episode缓存污染和未来信息。复用相同提取器是部署要求。它的真实单帧数值接口已验证，不证明GUI技能可学性；若改变时间上下文、空间读出或光标处理，须重新导出并重训，不能复用旧特征指纹。

奖励历史与BC采用相同context切段和零隐状态；不跨段拼接后继。不通过技能ID推断事件，不输入库存/坐标。原始源审核报告与特征、标签分别保存。

## 运行

先生成待审核模板：
```sh
python3 -m dreamer_agent.prepare_annotations \
  --registry outputs/p0_sources_20260913/registry.json \
  --output outputs/NEW_ANNOTATIONS --per-split 2 --frames 64
```

审核真实标注后，一次运行全部监督步骤：
```sh
python3 -m dreamer_agent.run_p1 \
  --annotations outputs/REVIEWED_ANNOTATIONS/annotations.json \
  --output outputs/NEW_P1_RUN \
  --device cuda --stage all --context 32 --bc-epochs 1 --reward-epochs 1
```

GPU由CUDA_VISIBLE_DEVICES指定；实际Tokenizer后端需要可用CUDA环境。上述epoch数量是默认执行参数，不是已验证收敛预算。可分别调用export_vpt、train_bc、train_reward，参数见各自--help。每个运行使用新输出目录，失败报告保留，不覆盖旧结果。

主要产物：
- features/manifest.json：特征、动作、任务、帧号及来源。
- features/source_*.json：已审核配对协议记录；physical alignment_verified仍false，避免与协议审核混用。
- features/reward_labels.json：独立监督侧文件及哈希。
- bc/bc.pt：BC参数与manifest/config/code指纹。
- reward/reward.pt：奖励/完成参数与固定BC指纹。
- reward/calibration_scores.json、calibration.json：选择和独立会话确认结果。
- pipeline_report.json：步骤状态。completed仅表示监督流水线执行完成，p1_passed仍false。

## 本轮验证与未完成事项

60项CPU软件测试通过，包含合成端到端导出→BC→奖励训练→校准、源污染拒绝、未知标签、冻结梯度和独立确认失败用例。真实冻结Tokenizer单帧8192维输出有限且重复一致。均不是Minecraft成功率。

本轮真实数据模板见outputs/p1_pipeline_20260914/annotation_template/，未审核、不用于训练。目前缺少经审核的技能/事件标签，因此未启动正式BC/奖励训练。P0语义测试、预注册真实BC环境评估和P2想象RL不由此流水线替代。
