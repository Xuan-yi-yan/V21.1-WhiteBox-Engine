# V21.1 白盒中文认知引擎

**49天独立开发 | 12模块 | ~9.9M参数 | RTX 5070 12GB | 52K训练对**

白盒模块化中文纠错引擎。从CharEmbed到P6解码, 11个可解释模块全链路。P3属性栈(128D×126槽)、P7交叉注意力(RMSNorm Q/K)、ABC三阶推理、384D拼接Gate、ABC'内容加压。

---

## GitHub 技术贡献

| # | 仓库 | Issue | 诊断 |
|---|------|-------|------|
| 1 | Qwen/qwen-code (25K⭐) | [#5083](https://github.com/QwenLM/qwen-code/issues/5083) | CoT语义压缩替代物理截断 |
| 2 | DeepSeek-V3 (104K⭐) | [#1368](https://github.com/deepseek-ai/DeepSeek-V3/issues/1368) | 安全引力替代权重博弈 |
| 3 | DeepSeek-V3 (104K⭐) | [#1420](https://github.com/deepseek-ai/DeepSeek-V3/issues/1420) | 语义标注替代KV裸游 |
| 4 | DeepSeek-V3 (104K⭐) | [#1125](https://github.com/deepseek-ai/DeepSeek-V3/issues/1125) | 注意力锁定,非目标偏移 |
| 5 | DeepSeek-V3 (104K⭐) | [#1460](https://github.com/deepseek-ai/DeepSeek-V3/issues/1460) | CE loss倒逼质量 + 引力子锚点 |

**外部引用**: AmoebaFPS独立研究实验室将蒸馏诊断和CE loss门控采纳入临床报告附录, 引力子锚点被选定为Report III原型方向并被形式化为数学框架。

---

## 目录

| 目录 | 内容 |
|------|------|
| [docs/](./docs/) | 架构全景、技术能力、开发记录、FLOPS分析、GitHub对话记录 |
| [architecture/](./architecture/) | 黑盒V2多元组、双核辩论Agent |
| [agent/](./agent/) | 法律咨询Agent教学示例 |
| [training/](./training/) | V21.1训练脚本、P7交叉注意力 |
| [eval/](./eval/) | 评测脚本 |
| [resume/](./resume/) | 简历 |

---

## 架构链路

```
A句输入 → CharEmbed → P3属性栈(128D×126槽) → P7交叉注意力(RMSNorm Q/K)
→ P3-L属性联动 → ABC'(内容加压) → ABC(控制) → 384D Gate → P6解码(128头)
→ B句输出
```

## 训练状态

- 52K训练对, batch=14, FP32
- 从E56 checkpoint续训 (CE=5.99)
- V21.1新增: ABC'内容加压(86K) + abc_to_sent加法注入(33K)

---

*Built with PyTorch, 49 days of solo development, June 2026.*
