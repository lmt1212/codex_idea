# EPC-QG：证据优先的多跳问答构建

开发入口：[项目进展记录](项目进展记录.md)（当前状态、问题与变更日志）｜[后续开发计划](后续开发计划.md)（执行步骤与验收条件）｜[开发约定](AGENTS.md)。每次修改代码必须同步更新进展记录。历史报告中的旧统计以进展记录注明的最新批次为准。

当前服务器：[服务器连接说明](服务器连接说明.md)，SSH 端口 `35276`，本机可用 `ssh codex-idea-autodl`。项目已同步 GitHub，后续修复与真实试跑状态见进展记录。

最新开发批次：本地 schema 解码、Transformers 5 兼容层、冻结请求双模型对照和化学式后缀检查已接入；83 项测试通过。完整必要性曾产生 1 条模型自动候选，但已被新规则复核发现公式遗漏并拦截，仍无正式验收样本。完整状态以项目进展记录为准。

本地推理需安装 `requirements-training.txt` 中的 `lm-format-enforcer==0.11.3`；`generate` 默认 `--json-mode schema`。保留 `--json-mode unconstrained` 用于诊断。冻结请求对照示例：`python scripts/replay_verifier.py output/dev_context_qa_04 --models /root/autodl-tmp/Qwen2.5-3B-Instruct /root/autodl-tmp/Qwen3-4B --output output/new_replay`。该工具只比较请求输出，不创建人工标签或正式通过结果。

依据 [多跳问答自动构建方案](多跳问答自动构建方案.md)，实现保留原文出处的文档解析、真实向量生成、证据路径搜索、问答生成与验证、人工盲评、评分校准和 SFT 实验接口。

最新修复：完整`generate`命令已接入双本地模型，三步抽取在12个真实块上获得11条可追溯候选事实。真实多跳QA尚未获得验证通过的产出；用户暂不人工标注，因此正式校准和四组SFT尚未执行。详见[本地生成修复记录](本地生成修复记录.md)。下表保留此前第一轮试跑结果。

当前本地和 AutoDL 版本使用同一代码。用户要求：**只使用本机及 AutoDL 的本地模型，不向外部 API 发送正文**。代码保留可选 API 适配器，但实际本轮实验未使用外部模型 API。

## 当前真实验证结果

| 项目 | 实际结果 |
| --- | --- |
| 服务器 | RTX 3090 24GB，Conda `sft311` |
| 文档试跑 | 6 本 PDF，各前 8 页；1,377 个 Chunk，408 个待原文审核，解析错误 0 |
| Token 切分 | 1,401 个 Chunk，目标 480 token，保留原文偏移 |
| BGE 中文向量 | 1,397 条；4 个超预算原子表格记录为跳过，无截断 |
| Qwen2.5-3B 单块基线 | 正文小样本 3 次调用，2 条候选 QA，尚未人工审核 |
| 相邻块基线 | 3 次调用，模型全部返回无题；产出 0，未人为填充 |
| 本地事实抽取 | 简化协议后 3 次调用均返回空事实；产出 0，未视为成功知识抽取 |
| GPU LoRA 检查 | 合成输入单步反向与优化通过；峰值显存约 6GB；不是正式 SFT 结果 |
| 人工标注 | 已生成 2 条 QA 盲评任务、30 条原文审核任务，全部待标注 |

这些结果来自 `output/autodl_real/`，不代表全文准确率或真实多跳通过率。实际质量报告保持 `awaiting_human_annotation`，没有伪造人工标签、阈值或四组 SFT 改进结论。

## 快速检查

```powershell
python -m unittest discover -s tests -v
python -m codes.multihop_qa --help
python -m codes.multihop_qa demo --output tmp/new_demo
```

基础测试仅依赖标准库，PDF/XLSX/OCR 与真实模型运行需要对应依赖。`demo` 明确使用合成测试响应，不能正式导出。所有新输出目录必须为空，避免覆盖已有数据。

## 真实解析与向量

以下命令在服务器项目目录执行，Python 使用 `sft311` 环境。

```bash
python -m codes.multihop_qa ingest book --pdf-backend layout --pages 1:8 --ocr auto --provisional-native --output output/run01
python -m codes.multihop_qa rechunk output/run01 --tokenizer /path/to/BGE --max-tokens 480 --output output/run02
python -m codes.multihop_qa embed output/run02 --backend local --model /path/to/BGE --device cuda --cache output/bge_cache.jsonl --output output/run03
```

`--pages` 指物理页码，范围包括首尾。布局解析保留页码、边界框，抽取原生表格，扫描页使用 RapidOCR；OCR、表格及公式风险仍需审核。`--provisional-native` 只允许干净原生文本进入试验，不代表有人审过。跨页表格自动合并仍未实现。

支持 TXT、Markdown、HTML、DOCX、PDF、XLSX。`--pdf-backend text` 保留简单原生文本后端。Token 切分不静默截断；原子表格超预算会保留待审并跳过向量化。向量缓存绑定模型、池化方式和原文指纹，重跑可复用已完成记录。

## 单块／相邻块与其他基线

```bash
python -m codes.multihop_qa baseline output/run02 --method single --backend local --model /root/autodl-tmp/Qwen2.5-3B-Instruct --live --limit 10 --output output/single
python -m codes.multihop_qa baseline output/run02 --method adjacent --backend local --model /root/autodl-tmp/Qwen2.5-3B-Instruct --live --limit 10 --output output/adjacent
```

不加 `--live` 仅准备证据选择计划。支持 single、adjacent、random、entity、dense、graph；dense 需 `--vectors`，entity/graph 需事实或路径产物。相邻基线不会跨章节，也不会越过待审块拼接不相邻证据。基线输出 `baseline_unverified`，不能冒充高质量多跳 QA。

## 事实、隐含关联与路径

```bash
python -m codes.multihop_qa extract output/run02 --backend local --model /root/autodl-tmp/Qwen2.5-3B-Instruct --live --limit 10 --output output/facts
python -m codes.multihop_qa mine output/facts --embeddings output/run03/vectors.jsonl --implicit --beam-width 16 --path-limit 20 --output output/paths
```

抽取保存原文连续引用、角色、条件、时间、否定和不确定性，必须匹配原文。只有空事实时不会凭空创建路径。手工审核过的事实缓存可用 `extract --facts-cache`，但不能把程序或模型标注冒充人工结果。

`--implicit` 打开有界的跨实体互补角色召回；有互补角色或语义联系的候选不再因为实体不重合被一律丢弃。它们始终是候选，进入问答生成前还需独立语义核验。支持固定 2～4 步结构、Beam Search 与有界 BFS，按连贯性、角色覆盖、来源多样性、互补角色及冗余评分。

完整 QA 流程默认要求生成与验证模型独立，当前服务器已有Qwen2.5-3B和Qwen3-4B。调用示例：

```bash
python -m codes.multihop_qa generate output/paths --live --generator-backend local --generator-model /root/autodl-tmp/Qwen3-4B --verifier-backend local --verifier-model /root/autodl-tmp/Qwen2.5-3B-Instruct --limit 2 --output output/local_qa
```

`--review-mode same-model-diagnostic`仅用于明确标记的诊断，不会产生独立验证通过的数据。API路线继续受项目本地模型策略限制。

## 人工标注和真实质量评测

创建一个 JSON 配置，例如 `runs.json`：

```json
[
  {"folder":"output/single","method":"single","split":"val"},
  {"folder":"output/adjacent","method":"adjacent","split":"val"}
]
```

```bash
python -m codes.multihop_qa annotate runs.json --kind qa --output output/annotation
python -m codes.multihop_qa quality output/annotation --output output/quality
```

标注者查看 `tasks.jsonl` 的问题、答案和原文出处，在 `responses.jsonl` 填写真实 reviewer、decision、各质量布尔标签与实际 hop_count。`private_mapping.jsonl` 供负责人汇总，盲评时不展示方法名称。没有真实标签时只报告待标注；不计算虚假的通过率。

`--kind pair` 可准备互补证据对标注任务。QA 配置可加 `include_rejected:true` 纳入有答案的拒绝样本，避免只评已通过样本的选择偏差。质量报告导出 QA 审核、证据对训练标签和可用的必要性校准样本。

## 评分训练与阈值校准

```bash
python -m codes.multihop_qa train-scorer output/quality/pair_labels.jsonl --output output/scorer
python -m codes.multihop_qa calibrate output/quality/calibration_samples.jsonl --min-precision 0.9 --output output/calibration
```

训练互补评分要求真实正负标签、训练／验证来源隔离。必要性阈值只在有正负样本的人工验证集上选择；测试集不参与选阈值，同源跨集会被拒绝。完成校准后 `generate --calibration output/calibration/calibration.json` 可使用校准结果。标注不足时命令明确报错，不输出假模型或假阈值。

## SFT 实验

```bash
python -m codes.multihop_qa sft-check /root/autodl-tmp/Qwen2.5-3B-Instruct --output output/model_check
python scripts/gpu_lora_smoke.py --model /root/autodl-tmp/Qwen2.5-3B-Instruct --output output/new_gpu_smoke
```

`gpu_lora_smoke.py` 只验证合成输入的模型加载、答案掩码、LoRA 梯度和单步优化，不构成正式训练或质量实验。

正式数据必须先审核：完整方法用 `export`，对照基线用 `release-baseline --reviews ...`，均保留来源与消息指纹。`release-baseline` 允许真实单跳对照，但不能据此声称是多跳数据。

```bash
python -m codes.multihop_qa sft-train output/audited_dataset --model /root/autodl-tmp/Qwen2.5-3B-Instruct --steps 100 --max-tokens 2048 --output output/adapter
python -m codes.multihop_qa benchmark independent_test.jsonl --model /root/autodl-tmp/Qwen2.5-3B-Instruct --adapter output/adapter/adapter --training-release output/audited_dataset --output output/test_results
```

训练只在答案 token 上计算损失，超长样本明确拒绝，保留步数、学习率、实际答案 token 数和显存峰值。评测需要人类确认的参考答案，并检查测试来源不与训练／验证重叠；EM 和字符 F1 只表示表面答案匹配，不能替代推理与幻觉人工评测。

`sft-plan arms.json --model ... --output ...` 要求 single、direct_multihop、graph、full 四组真实审核数据，生成等量训练子集和一致训练预算命令。当前没有这些已审核数据，正式四组实验尚未执行。

## 环境与边界

`requirements-server.txt` 提供 PDF/OCR/XLSX 依赖；`requirements-training.txt` 提供本地向量与训练依赖，AutoDL 现有 CUDA PyTorch 应保留。实际服务器环境与结果见 [AutoDL部署与实验记录](AutoDL部署与实验记录.md)。

尚待真实人工工作：原文质量审核、互补正负样本、必要性阈值校准、独立评测集及正式四组 SFT。尚待增强的工程能力包括复杂跨页表格合并、自动共指/近重复归组、神经互补模型、大规模增量索引和完整断点恢复。历史文档中旧入口与旧统计不适用于本版本。
