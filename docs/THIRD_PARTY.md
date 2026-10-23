# 第三方材料

根 MIT 许可证适用于本项目自有代码。源码发布清单收录 `dreamer_agent/` 自有训练、适配与测试；第三方工作副本、模型权重及数据由使用者按来源条款获取。

| 来源 | 用途 | 发布处理 |
|---|---|---|
| [Dreamerv4-MC](https://github.com/IamCreateAI/Dreamerv4-MC) | 世界模型兼容性研究 | 本地副本的 pyproject 引用 `LICENSE.txt`，当前副本缺少该文件；再分发前核实授权 |
| [Dreamerv4-MC 权重](https://huggingface.co/IamCreateAI/Dreamerv4-MC) | 模型资产 | 单独核对模型卡、revision 与权重使用条款 |
| [OpenAI VPT](https://github.com/openai/Video-Pre-Training) | 动作协议、预训练策略与数据来源 | 分别核对代码、模型和数据条款 |
| [RLVR-World](https://github.com/thuml/RLVR-World) | 研究参考 | 引用项目与论文，实际引入材料时登记版本和许可 |
| PyTorch | 核心运行依赖 | 通过包管理器安装并遵循上游许可 |

本地 `dreamer4/`、`dreamer4-lucidrains/` 及 `abc/` 中存在独立实现副本。它们保留在工作目录，源码导出清单单独选择本项目文件。若以后引入其中的代码，随具体文件保留上游版权和许可证。

Minecraft、游戏资产及相关商标的权利归其权利人。数据下载、使用与再分发分别检查适用条款。
