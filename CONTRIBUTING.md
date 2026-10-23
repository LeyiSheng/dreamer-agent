# 贡献指南

使用独立虚拟环境并按根 README 安装。算法主线位于 `dreamer_agent/`，测试位于 `tests/`。

提交前运行：

```bash
python -m unittest discover -s tests -v
python scripts/release.py check
```

修改说明包含问题、行为变化和验证方式。数据、权重、运行日志保存在本地资产目录；公开配置使用可配置路径。算法主线调整同步更新研究计划，实验记录注明训练和评估的数据划分。

新增公开文件时更新 `release_manifest.json`。根 `.gitignore` 采用选择性收录，原有工作文件仍可在本地保留。

提交自有代码时采用项目 MIT 许可证；引入第三方材料时注明来源、固定版本与许可证，并保留其版权声明。
