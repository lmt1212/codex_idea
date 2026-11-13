# AutoDL部署与真实实验记录

> 以下保留第一轮试跑记录。其中“缺少第二个模型”、事实数 0、55 项测试为旧阶段状态。之后 Qwen3-4B 已完整加载，产生 11 条候选事实，上一轮测试增至 59 项；真实多跳 QA 自动通过仍为 0。当前统一状态见 [项目进展记录](项目进展记录.md)。

## 已部署环境

- 项目目录：`/root/autodl-tmp/codex_idea`。
- Conda Python：`/root/miniconda3/envs/sft311/bin/python`，Python 3.11.18。
- GPU：NVIDIA GeForce RTX 3090，24576 MiB。
- 实际检查：PyTorch 2.15.0 / CUDA 13.2，CUDA可用；Transformers 5.22.2、PEFT 0.21.2。
- 已补充 PyMuPDF、RapidOCR、ONNX Runtime、openpyxl；未重装现有CUDA或PyTorch。
- 用户自行完成的模型下载：`/root/autodl-tmp/Qwen2.5-3B-Instruct`，两片 safetensors 已完整，可以加载。
- 使用新配置的SSH密钥登录；密钥与本机.env未同步到项目。

## 已执行的真实数据检查

经用户明确授权，已上传book中的6本PDF。首次解析只覆盖各书前8个物理页，共48页小样本，不代表全文验收。

1. `real_01_ingest`：6份文档、1377个Chunk、408个待审Chunk、0个解析错误。原文页码和边界框保留。旧批次报告的limitation字段写有“no OCR”，属于静态说明错误；该批次实际使用了RapidOCR日志与OCR产物，代码现已修正报告字段。
2. `real_02_tokenized`：1401个Chunk，正文按BGE tokenizer限制为480 token，保持原文偏移。
3. `real_03_vectors_retry`：1397条真实BGE中文向量，4个超预算原子表格跳过并保留原因。第一次向量运行因原子表格超预算中止；重试复用981条已完成缓存。
4. `real_05_gpu_smoke`：Qwen2.5-3B合成输入LoRA单步测试通过。loss约3.1003，梯度范数约3.2394，1843200个可训练参数，峰值6423664128字节。该结果不表示真实SFT收益。
5. `real_09_single`：3次真实正文调用，2条候选QA，0个格式错误，待人工审核。
6. `real_10_adjacent`：3次调用均返回空题，候选数0，保留原始响应，未伪造对照产出。
7. `real_12_extracted`：3次本地事实抽取均返回空数组，事实数0。格式已符合协议，但模型未提供可用事实，因此不能声称完整路径挖掘已经在真实书籍上成功。

以上原始产物保存在服务器output目录，已下载的报告与基线数据保存在本机 `output/autodl_real/`。本轮未向外部API发送正文，符合用户只使用本地模型的要求。

## 人工标注与后续阻塞

- `output/autodl_real/real_13_qa_annotation`：2条真实QA盲评任务，未标注。
- `output/autodl_real/real_14_source_annotation`：30条原文审核任务，未标注。
- `output/autodl_real/real_15_quality_pending`：状态为 `awaiting_human_annotation`，没有自动填充通过率。

当前缺少：独立人工质量标签、互补证据对的正负样本、校准集、来源隔离的训练/验证/测试数据，以及第二个独立本地验证模型。完整多跳数据验收、校准后的质量指标和正式四组SFT结果尚不能生成。现有代码会拒绝不满足这些条件的训练或评测请求。

## 代码检查

本机与服务器最终均通过55项测试，包含核心流程、解析、向量缓存、隐含候选、标注、校准、SFT损失掩码、基线连续性和仅本地模型策略检查。服务器最终模型预检ready=true，缺失文件和.incomplete文件均为空，torch/transformers/peft均可用。

最终服务器已同步本地真实BGE向量至 `output/real_bge_vectors`，相应token切分文档位于 `output/real_bge_tokenized`。项目根目录 `project_policy.json` 禁止外部正文API调用。
