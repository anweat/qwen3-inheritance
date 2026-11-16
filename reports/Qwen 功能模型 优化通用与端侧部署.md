# 让 Qwen 功能模型共享内核，而不是权重

现有的 Qwen3-ASR、Qwen3-TTS 与 Qwen3 LLM 只能在**架构、算子和运行时**层面共享，**不能在权重层面共享**。本地 E0 实测显示，ASR thinker 相对 Qwen3-1.7B-Base 的相对漂移约 0.30（余弦 0.96），TTS talker 约 1.0（余弦 0.25，范数减半）。差量都不是低秩的，连 embedding 表也漂了 0.30，所以“一个量化主干加若干小差量包”的方案对这些已发布模型不成立。三个功能的体积是三个量化模型之和，不是“接近一个模型”。这不是坏消息：同形状、同量化格式的三个模型可以共用一套 kernel、一套 CUDA graph 结构和一个分页 KV 池。在 6 GB 的 RTX 4050 上，最划算的做法是让三个 4-bit 模型**常驻显存**，在单进程里用优先级流交错执行，而不是追求快速换入换出〔估算：合计约 3–3.6 GB 权重，加外围组件后可能在 5.5 GB 以内〕。社区对 Qwen3 解码器的优化（GGUF/MLX/MNN 的量化格式、FlashAttention、KV 量化、CUDA graph）会自动惠及 ASR 和 TTS 的主干。真正的瓶颈在外围组件：AuT 音频编码器、TTS 的 code_predictor、codec 解码器（SnakeBeta、因果转置卷积）、ViT 加 DeepStack、M-RoPE 变体，以及 Qwen3.5 的 Gated DeltaNet，这些都要逐个运行时单独适配。端侧方面，Qwen3 稠密模型（0.6B–4B）和 Qwen3-VL-4B 已经能在骁龙 NPU 上跑出可用速度，Qwen3-ASR 在 Apple 芯片上远快于实时。两块短板是 **Qwen3-TTS**（M5 Pro 上 RTF 仍有 0.76–0.95）和 **Qwen3.5**（厂商 NPU 工具链还不支持 GDN）。

> **更正（2026-11-05，依据 `experiments/e1_block_atlas/`）。** 本报告把所有模型都和 Base 比较，因此有两处说法过宽。第一，“一个基座加小差量”只在跨功能时不成立；同一分支内部差量很小，例如 VL 的 4 个变体相互距离 0.05–0.10、TTS-1.7B 的 3 个变体 0.08–0.16、0.6B 的 Instruct/Guard-Stream/Reranker 0.01–0.04（实测范围为 0.6B 的 3 层、1.7B 的第 14 层）。第二，“TTS 从 base 初始化”只适用于 1.7B；TTS-0.6B 与 Base 余弦为 0.000，看起来是独立训练的。第三，“ASR 差量几乎省不下空间”也被实测否定：ASR-0.6B 在 Q4 Base 上用 2-bit 残差，质量基本与独立 Q4 相同，详见 `Qwen3 结构与功能溯源实验.md`。

**读法说明。** 带链接的句子是有来源的事实。〔实测〕指用户本地 E0 的测量。〔估算〕是由公开参数推出的数字。〔推断〕是没有直接来源的工程判断。〔未核实〕是研究笔记中凭先验知识写下、本轮没有重新查证的内容。

## E0 实测：差量既不小也不低秩，共享只能停在内核层

E0 用 HTTP range 读取 safetensors，逐层比较了四个模型 28 层的 7 个投影矩阵：Qwen3-1.7B-Base、Qwen3-1.7B instruct、Qwen3-ASR-1.7B 的 thinker、Qwen3-TTS-12Hz-1.7B-Base 的 talker。漂移按每个矩阵前 64 行计算，SVD 用的是完整 q_proj。结果分成三个梯度〔实测〕。**instruct 的相对漂移约 0.10**，余弦约 0.99，范数比 0.99。**ASR 约 0.30**，余弦均值 0.96，最低 0.91，范数比 0.85。ASR 的 embedding 漂移同样是 0.30，lm_head 与自己的 embedding 绑定。**TTS 约 1.0**，余弦均值只有 0.25，最低 0.05，范数比 0.50，layernorm 相关系数降到 0.90，embedding 漂移 1.11。TTS 每层与 base 余弦最高的仍是同一层，说明它确实从 base 初始化而来，只是被大规模重写了。漂移从浅层到深层缓慢上升，没有出现集中在少数几层、其余可以共享的情况。

“不是低秩”的证据也很明确〔实测〕。q_proj 差量要达到 90% 能量，需要 2048 维中的 **700–875 个奇异值**，前 64 个奇异值只占 22–31% 的能量。连漂移最小的 instruct 也是如此，所以 LoRA 式的低秩近似对哪一个都不成立。BitDelta 式代理测试用“符号 × 每矩阵平均幅值”，没有做缩放蒸馏，属于偏悲观的上界。在这个测试里，三者的差量重建误差都约 0.61–0.63，但折算到权重层面相差很大：instruct 0.059，ASR 0.21，TTS 1.28。BitDelta 论文只在指令或对话微调这类小漂移上验证了 1-bit 差量 ([BitDelta, arXiv 2402.10193](https://arxiv.org/abs/2402.10193v3))，E0 的 instruct 结果与之一致。ASR 和 TTS 已在这个适用范围之外。ASR 的大漂移有明确来由：Qwen3-ASR 技术报告说它的基础是 Qwen3-Omni 的音频理解能力，而不是原始的 Qwen3-1.7B ([Qwen3-ASR 技术报告](https://arxiv.org/html/2601.21337v2))。

这个结果对用户方案文档中的每条策略都有直接影响。

**共享量化主干加压缩差量**，对 TTS 不成立，对 ASR 收益很小。一个稠密差量在 b bit 下的体积和完整矩阵在 b bit 下完全相同，因为形状一样。所以差量方案只有在差量能压到比模型本身更低的比特、或者能稀疏化时才省空间〔推断〕。漂移为 ε 时，差量的重建误差在权重层面会被缩小到 ε 倍。ASR 的 ε≈0.3，意味着差量量化误差可以放宽约 3 倍，2–3 bit 的差量才可能与 Q4 完整模型的误差相当〔估算，需 P2-1 实验验证〕。TTS 的 ε≈1，差量与模型本身一样大，存差量没有意义。DeltaZip 报告全量微调差量可压约 10 倍、服务吞吐提升 2–12 倍 ([DeltaZip, arXiv 2312.05215](https://arxiv.org/abs/2312.05215v3))，但它依赖可压缩的差量和已知的基座，这个前提在 TTS 上不存在。

**LoRA 功能包**，对现有模型不适用。没有现成的低秩近似可提取，只能在未来重新训练时把功能约束为冻结主干上的 LoRA。

**共享 embedding**，不可行。ASR 的 embedding 漂了 0.30，TTS 漂了 1.11。研究笔记曾估算共享 embedding 可省约 0.6 GB（BF16），这个收益不存在。

**多 LoRA 批处理**（Punica SGMV 报告 12 倍吞吐）同样不适用 ([Punica](https://github.com/punica-ai/punica))。这类技术的前提都是“共享基座每步只读一次”。

**仍然成立的部分**都与权重值无关〔推断〕：相同的 GEMM 形状、同一套量化 kernel、同一种 CUDA graph 拓扑、同一种 KV 块形状，以及同一套导出、编译和调优流水线。方案文档中“基座版本锁定”“差量相对量化基座重新校准”等风险，对现有模型自然消失了。代价是，体积目标从“约 1 个主干加 3 个小包”改为“约 3 个量化模型加外围组件”。

## 6 GB 显卡上，常驻 4-bit 三模型比任何切换技巧都划算

### 加载：先让切换不发生

研究笔记的估算如下〔估算；RTX 4050 Laptop 的 PCIe 4.0 ×8 链路和 192 GB/s 显存带宽都是未核实的规格，需实测〕。Qwen3-1.7B 在 BF16 下约 3.4 GB，4-bit 下约 1.0–1.2 GB。三个 BF16 模型约 10 GB，放不进 6 GB 显存；三个 4-bit 模型约 3–3.6 GB，放得下，还能留出 KV 池和外围组件的空间。社区 GGUF 给了实际体积参照：ggml-org 的 Qwen3-ASR-1.7B Q8_0 为 2.17 GB ([ggml-org/Qwen3-ASR-1.7B-GGUF](https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF))。qwentts.cpp 的 TTS talker 在 Q4_K_M 下 1.2 GB，codec tokenizer 在 Q4_K_M 下 255 MB，因为 codebook 和投影保留 F32 ([Serveurperso/Qwen3-TTS-GGUF](https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF))。

如果显存不够、必须换入，权重放在**锁页主机内存**时，4-bit 模型的拷贝约 0.08–0.1 s，BF16 约 0.25–0.3 s。从冷页缓存的 NVMe 读取则要 0.5–1.2 s〔估算〕。可以借鉴的成熟做法有三类。vLLM sleep mode 的一级休眠把权重卸到 CPU 内存，小模型唤醒约 0.1 s（A4000），比冷启动快 18–20 倍以上 ([vLLM sleep mode 博客](https://vllm.ai/blog/2025-10-26-sleep-mode))。ServerlessLLM 用免解析的加载友好格式和多级加载器，README 称比 safetensors 快 5–10 倍 ([ServerlessLLM, OSDI'24](https://usenix.org/conference/osdi24/presentation/fu))。逐层流水加载让第 i+1 层的拷贝与第 i 层的计算重叠，但对 1.7B 短提示的 prefill，计算只有几十毫秒，能藏住的时间有限〔推断〕。还有一个环境限制：vLLM 只支持 Linux，而这台笔记本没有 WSL 发行版，所以 sleep mode 只能在 AutoDL 上验证，只有思路能移植〔未核实〕。BlitzScale、WarmServe 等依赖多 GPU 网络的方案与笔记本无关 ([BlitzScale](https://arxiv.org/html/2412.17246v2))。

### 切换：同架构让 CUDA graph 可以复用

同架构的真正红利在切换路径上。`cudaGraphExecKernelNodeSetParams` 可以“以极小代价”改写已实例化 graph 中 kernel 节点的参数（包括指针），不需要重新实例化。前提是 kernel 函数和拓扑不变 ([NVIDIA CUDA Graphs 博客](https://developer.nvidia.com/blog/?p=52463); [API 限制说明](https://docs.rs/rcudnn-sys/latest/rcudnn_sys/fn.cudaGraphExecKernelNodeSetParams.html))。同形状、同量化格式的三个模型正好满足这个前提。由此有三种模式〔推断〕：

- **A：每个模型、每个批大小档位各捕获一张 graph。** 最简单，对 1.7B 模型额外开销可以忽略。这是首选。
- **B：只捕获一张 graph，把下一个模型的权重拷进同一组缓冲区。** 适合只能常驻一个模型的情况。
- **C：三套权重都常驻，通过 SetParams 重定向约 200 个 GEMM 节点的指针。** 比 A 只省 graph 实例化的时间和内存，收益很小。

三种模式都要求三个模型使用**同一种量化格式和分组大小**。哪怕只把 TTS 换成 Q8，它也要单独一套 graph。

### 并发：单进程、优先级流、按 token 交错

MIG 只有数据中心 GPU 有，MPS 面向 Linux，Jetson 也不支持 MPS ([NVIDIA 论坛](https://forums.developer.nvidia.com/t/using-green-context-in-cuda-on-jetson-devices-with-ampere-architecture/330055))。Windows WDDM 下的部分 CUDA 行为也与 Linux 不同 ([cuda-python #206](https://github.com/NVIDIA/cuda-python/issues/206))。所以在这台笔记本上，并发只能靠**单进程内的多条 CUDA 流加 CUDA graph**。多进程靠驱动时间片轮转，切换粒度粗，不可取〔推断〕。

推荐的调度方式如下〔推断〕：

1. 三个模型常驻，共用一个分页 KV 池。
2. TTS 的分块生成和 ASR 的部分结果放在高优先级流（`cudaStreamCreateWithPriority`），LLM decode 填补空隙。
3. LLM 输出到子句边界（约 5–15 个 token）时就开始 TTS。

共享 KV 池在 E0 之后依然成立，因为它只要求块形状相同，不要求权重相似。Qwen3-1.7B 形状的 KV 约为每 token 112–114 KB（BF16），1 GB 池约可放 9k token〔估算〕。数据中心的 Prism 和 SeaLLM 正是靠跨模型共享显存池取得收益 ([Prism, arXiv 2505.04021](https://arxiv.org/abs/2505.04021v1))。

把三个模型叠成“三个专家”做 grouped GEMM，在 decode 阶段收益很小。batch 1–4 的 decode 受带宽限制，三套权重的字节数一个都省不掉，能省的只是 kernel 启动开销，而 CUDA graph 已经把启动开销消掉了。研究笔记预计收益不到 10–20%，并且没有找到任何“多个完整模型合批”的公开基准〔估算〕。

带宽决定了速度上限〔估算〕：BF16 下单模型 decode 上限约 55 tok/s，4-bit 下约 170–190 tok/s。流水线中三个模型各走一步，就要读三份权重。端到端参照只有数据中心的数据：流式级联 P50 首音约 947 ms ([arXiv 2603.05413](https://arxiv.org/html/2603.05413v1))，NVIDIA 用三张 H100 做到 0.79 s ([NVIDIA Nemotron FAQ](https://perspectives.nvidia.com/nemotron-speech/task/faq/which-speech-models-run-voice-assistants-500-milliseconds/))。研究笔记没有找到 ≤8 GB 单卡的公开端到端基准。对 4050，笔记估算结束说话后 300–600 ms 出首音，这必须实测。

### 值得增强的 GPU 算子

按收益排序，第一位是 **W4 仅权重的 GEMV/GEMM**，这是带宽受限的 4050 上最大的杠杆。可选 GGUF k-quant（Windows 原生）、Marlin、AWQ 或 GPTQ。Ada 架构（sm_89）支持原生 FP8 W8A8；NVFP4 需要 Blackwell ([vLLM FP8 文档](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/))。

第二位是 **TTS code_predictor 的逐帧小循环**。它在每个 80 ms 帧内要跑 15 次 5 层小 transformer，是受 kernel 启动开销限制的延迟瓶颈。Qwen3-Omni 报告 MTP 每 token 14 ms，codec 每个 code 只要 3 ms ([Qwen3-Omni 技术报告](https://arxiv.org/html/2509.17765))。这一步需要 CUDA graph、persistent kernel 或融合的小模型执行器。

第三位是 **codec 解码器**。官方流式实现 `chunked_decode(chunk_size=300, left_context_size=25)` 每次都重算 25 帧左上下文 ([官方 tokenizer_12hz 代码](https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py))。按 4 帧一包计算，冗余约 7 倍〔估算〕。改成每层卷积保存 (k−1)·dilation 的状态环形缓冲，加上窗口 72 的滑窗 KV，就能消除这部分冗余。SnakeBeta 无法融进厂商卷积的 epilogue，llama.cpp 为此专门加了融合 snake 算子 ([PR #26254](https://github.com/ggml-org/llama.cpp/pull/26254))。

第四位是编码器用的**非因果 varlen/分块窗口注意力**（AuT 的窗口为 1–8 s），以及 RMSNorm、QK-norm 与 RoPE 的融合。

如果要上 Qwen3.5，还需要 **GDN 的 chunked prefill 加 fused recurrent decode**。fla 已有 Triton 和 FlashQLA 后端 ([fla-org](https://github.com/fla-org/flash-linear-attention))。

## 主干优化自动迁移，七类外围组件要逐个适配

社区对 Qwen3 解码器的优化会自动迁移到 ASR 和 TTS 的主干，迁移的是代码和格式，不是权重产物。llama.cpp 把 ASR 的 GGUF 当作 `qwen3vl` 架构加载，从而直接获得 M-RoPE 和 Qwen3 解码块 ([ggml-org/Qwen3-ASR-1.7B-GGUF](https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF))。TTS talker 在 libllama 里作为 Qwen3 文本模型运行，codec embedding 并入文本 embedding 表，再接一个 `codec_head` ([PR #26254](https://github.com/ggml-org/llama.cpp/pull/26254))。于是 K-quant、IQ、Q8_0、FlashAttention、KV 量化以及 CUDA、Vulkan、Metal 后端都可以直接用。MLX 按 Linear 层做量化，MNN 的 llmexport 量化流水线（HQQ、AWQ、GPTQ、QAT）也一样适用 ([MNN releases](https://github.com/alibaba/MNN/releases))。vLLM 的 `Qwen3ASRForConditionalGeneration` 原生实现了 SupportsLoRA 和 SupportsMRoPE ([vLLM qwen3_asr API](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen3_asr/))。

**权重层面的产物不能迁移**〔推断〕，包括量化后的 checkpoint、imatrix 和校准数据、LoRA、Eagle3 草稿头。E0 表明权重不同，所以每个功能模型都要重新校准。研究笔记粗估，共享算子集约占各模型 FLOPs 和参数的 85–90%〔估算〕。

量化敏感度在不同功能之间差别很大，不能拿 LLM 的结论直接套用。M5 Pro 上的社区测试 ([soniqo 基准](https://soniqo.audio/benchmarks)) 显示：

| 模型 | 量化 | 测得结果 |
|---|---|---|
| Qwen3-TTS-1.7B | 4-bit / 8-bit | round-trip WER 3.47% / 3.66%，几乎不掉 |
| Qwen3-TTS-0.6B | 8-bit / 4-bit | round-trip WER 9.74% / 15.58% |
| Qwen3-ASR（FLEURS） | 4-bit / 8-bit | 韩语 WER 19.95 / 6.89，低资源语言被 4-bit 严重拖累 |
| Qwen3-ASR-0.6B | Core ML 转换 | 重建前 WER 一度达到 24.88% |

Core ML 那一行说明转换链路本身就很脆弱。另外，audio.cpp 报告 Qwen3-TTS 用 q8_0 后峰值显存降约 25%，但速度只快 3.8% ([audio.cpp](https://github.com/JanWerder/audio.cpp))。这说明 TTS 的耗时不在主干带宽上，而在自回归小循环和 codec 上〔推断〕。

需要单独适配的组件可以归为七类，下表按“迁移难度”列出。

| 组件 | 用于 | 为什么不能自动迁移 | 现状（有来源） |
|---|---|---|---|
| AuT 音频编码器（1.7B 配约 300M、hidden 1024，Conv2D 8 倍下采样，1–8 s 动态窗口） | ASR、Omni | 双向注意力加窗口 mask，Conv2D 前端，Fbank 预处理通常在 CPU | 各运行时各写一套编码器图，常保留 Q8 或 BF16 ([Qwen3-ASR 报告](https://arxiv.org/html/2601.21337)) |
| code_predictor（5 层，15 个 lm_head） | TTS、Omni talker | 每帧 15 步串行，受启动开销限制 | llama.cpp 手写独立图，刻意不走 MTP 基础设施 ([PR #26254](https://github.com/ggml-org/llama.cpp/pull/26254))；MNN 3.6.2 为其加了 KV cache ([MNN](https://github.com/alibaba/MNN/releases)) |
| codec 解码器（SnakeBeta、因果 ConvTranspose1d、膨胀卷积、ConvNeXt、RVQ） | TTS、Omni Code2Wav | 周期激活对 INT8 不友好；24 kHz 低通道卷积受带宽限制；流式需要卷积状态 | ggml 新增融合 snake 算子；多个栈仍把它交给 ONNX Runtime 或 Python ([CSGLite](https://github.com/OpenCSGs/csglite)) |
| ECAPA-TDNN 说话人编码器 | TTS | CNN 结构，每段参考音频只跑一次 | 作为 mtmd 音频编码器处理 |
| ViT（SigLIP2）加 DeepStack | VL、Omni | 2D-RoPE 加插值绝对位置、2×2 merger，前 3 层 LLM 要注入视觉残差 | llama.cpp PR #16780 修好了 DeepStack ([PR #16780](https://github.com/ggml-org/llama.cpp/pull/16780))；Qwen3.5 去掉了 DeepStack ([HF config](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json)) |
| M-RoPE 变体（三段式、交错式、partial rotary 0.25） | VL、ASR、Omni、3.5 | 每个后端都要单独实现一种模式；写错不会崩溃，只会悄悄掉精度 | 已实现于 ggml、MLX、FlashInfer |
| Gated DeltaNet（3 层 GDN 配 1 层门控注意力，FP32 状态） | Qwen3.5 | 新的循环状态算子；chunked prefill 需要三角求解 | llama.cpp 有 `GATED_DELTA_NET` 算子 ([PR #19504](https://github.com/ggml-org/llama.cpp/pull/19504))；MNN 有 LinearAttention 算子 |

此外，**流式编排**本身也是一类缺口。只有 vLLM-Omni 的 async chunk（Omni 的 TTFP 从 2790 ms 降到 655 ms）([vLLM-Omni async_chunk](https://docs.vllm.ai/projects/vllm-omni/en/stable/design/feature/async_chunk/))、MNN 和 HaujetZhao 的方案实现了流式。主线 llama.cpp 的 TTS 只能输出 WAV，不能流式，而且只支持 1.7B-Base，还有无法在 `codec_eos` 处停止的 bug ([PR #26254](https://github.com/ggml-org/llama.cpp/pull/26254))。

由此可以判断，“一个基底覆盖所有功能”在运行时层面应定义为**一份契约**，而不是一个共享实例〔推断〕。这份契约包括：统一的 Qwen3 解码器算子集；四种缓存类型（KV、滑窗 KV、卷积状态、SSM 状态）；形状为 [3,T] 的位置张量；编码器的三种 mask 模式；残差注入钩子；多头、多步的 decode 循环。卷积状态缓存同时服务于 codec 流式和 GDN 的短卷积，值得作为一等抽象。在现有引擎中，llama.cpp（桌面和 GPU）与 MNN（移动端）是唯二用一套原生 C++ 栈覆盖 LLM、VL、ASR 和 TTS 的。目前还没有任何运行时让多个功能共享同一个已加载的解码器，原因正是权重不同。

## 端侧：Qwen3 稠密模型能上 NPU，Qwen3-TTS 与 Qwen3.5 是两块短板

**手机 NPU。** 只有高通发布了系统化的厂商数据。在 8 Elite Gen 5 上，w4a16 GENIE、4K 上下文的成绩是：

| 模型 | decode | TTFT |
|---|---|---|
| Qwen3-0.6B | 132.8 tok/s | — |
| Qwen3-1.7B | **58.3 tok/s** | 0.029–0.93 s |
| Qwen3-4B | 28.6 tok/s | — |
| Qwen3-VL-4B（GENIEX_QAIRT，512 上下文） | 21.6 tok/s | — |

来源：[qualcomm/Qwen3-1.7B](https://huggingface.co/qualcomm/Qwen3-1.7B)、[qualcomm/Qwen3-VL-4B-Instruct](https://huggingface.co/qualcomm/Qwen3-VL-4B-Instruct)。

独立论文的结论更保守。在同代 SoC 上，Qwen2.5-1.5B 用 GENIE 的 NPU decode 只有 23 tok/s，比 llama.cpp 的 CPU decode（55.7 tok/s）还慢，NPU decode 的能耗常常高 2–4 倍。NPU 真正的优势在 prefill：超过 1400 tok/s ([arXiv 2607.05475](https://arxiv.org/html/2607.05475v1))。厂商数据和学术数据相差两倍以上，规划时应按保守值算。

AI Hub 上没有 Qwen3-ASR 或 Qwen3-TTS。Qwen3.5-2B 的页面状态是“不支持” ([AI Hub qwen3_5_2b](https://aihub.qualcomm.com/models/qwen3_5_2b))。社区的 Hexagon DeltaNet kernel 在强制 NPU 时 decode 只有 2.0 tok/s，同一模型在 CPU 上是 64.4 tok/s ([ara142/llama-cpp-hexagon-npu](https://github.com/ara142/llama-cpp-hexagon-npu))。所以在 NPU 上，结论是用 Qwen3、不用 Qwen3.5。Qwen3.5 在 CPU 和 GPU 上（llama.cpp、MLX、MNN）没有问题。

**语音组件。** Qwen3-ASR 在 M5 Pro 上用 MLX 时，RTF 为 0.012–0.033，内存 1–2 GB ([soniqo](https://soniqo.audio/benchmarks))。sherpa-onnx 官方提供 0.6B int8 版本，1.7B 版本由社区导出 ([sherpa-onnx #3535](https://github.com/k2-fsa/sherpa-onnx/issues/3535))。Qwen3-TTS 是最弱的一环：在 M5 Pro 上，MLX 的 RTF 为 0.76–0.85，MNN CPU 为 0.75–0.95 ([qwen3-tts-mnn](https://pypi.org/project/qwen3-tts-mnn/))。MLX 测试中约 92% 的时间花在自回归生成上，codec 解码占 244–457 ms ([soniqo](https://soniqo.audio/benchmarks))。手机 CPU 一般比 M5 Pro 慢 2–4 倍，所以纯 CPU 跑 Qwen3-TTS 大概率达不到实时〔估算〕。研究笔记没有找到任何手机上的 Qwen3-TTS 或 Qwen3-ASR 的 RTF 数据。Qwen3-Omni 是 30B-A3B，Q4 下约 15–17 GB〔估算〕，不适合手机。

**嵌入式与笔记本。** Jetson Orin Nano 8GB 用 llama.cpp CUDA 跑 Qwen3-1.7B Q4_K_M，decode 39.4 tok/s，prefill 1408 tok/s ([Turing Pi](https://turingpi.com/jetson-orin-nano-8gb-llm-benchmark/))。NVIDIA 的 TensorRT Edge-LLM 列出了 Qwen3-Omni，并附有 TTS 示例 ([TensorRT Edge-LLM](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/supported-models.html))。RK3588 用 RKLLM w8a8 跑 Qwen3-VL-2B 为 11.5 tok/s、3.1 GB，但没有 Qwen3-1.7B 的实测数据 ([Qengineering](https://github.com/Qengineering/Qwen3-VL-2B-NPU))。离 4050 最近的参照是 HaujetZhao 的方案：talker 用 GGUF，codec 用 ONNX Runtime，在 RTX 5050 上 RTF 0.35，首音 ≤300 ms ([HaujetZhao/Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF))。

**内存预算**〔估算〕：全 1.7B 栈常驻约 4.9–5.7 GB，超过 8 GB iPhone 的前台上限（约 4.8–5.2 GB，来自未经验证的博客 ([Lapis](https://lapis.lvrpiz.com/en/blog/local-llm-iphone-16-pro))）。均衡栈（ASR-0.6B、LLM-1.7B、TTS-0.6B 或 Kokoro）约 2.5–3.8 GB。由此给出的推荐组合如下〔推断，每行都需要在目标设备上实测〕。

| 平台 | LLM | ASR | TTS | 运行时 | 可行性判断 |
|---|---|---|---|---|---|
| RTX 4050 6 GB / Windows | Qwen3-1.7B Q4_K_M | Qwen3-ASR-1.7B（解码器 Q4/Q5，编码器 Q8） | Qwen3-TTS-12Hz-1.7B talker Q4/Q8，codec F16 | llama.cpp CUDA；流式 TTS 先用 HaujetZhao 或 qwentts.cpp | 可行，三者常驻，显存预计 ≤5.5 GB |
| Mac（Apple 芯片） | Qwen3-1.7B/4B 或 Qwen3.5（mlx-lm） | ASR-1.7B 5-bit（mlx-audio） | TTS-1.7B 4-bit（mlx-audio `--stream`） | MLX | 可行；TTS 余量小，低端芯片可能跟不上实时 |
| 骁龙 8 Elite 手机 | Qwen3-1.7B w4a16（GenieX，HTP） | ASR-0.6B int8（sherpa-onnx，CPU） | Kokoro 或 Supertonic；Qwen3-TTS-0.6B 经 MNN 属高风险 | GenieX + sherpa-onnx/MNN | LLM 和 ASR 可行，Qwen3-TTS 不建议 |
| 8 GB iPhone | Qwen3-1.7B 4-bit（MLX） | ASR-0.6B 4-bit | Kokoro | MLX + Core ML | 均衡栈可行，需申请内存 entitlement |
| Jetson Orin Nano 8GB | Qwen3-1.7B Q4 | ASR-0.6B/1.7B | TTS-1.7B | llama.cpp CUDA 或 TensorRT Edge-LLM | 可行，8 GB 统一内存偏紧 |
| RK3588 | Qwen3-1.7B 或 VL-2B（RKLLM w8a8） | ASR-0.6B（sherpa-onnx） | Kokoro | RKLLM + sherpa-onnx | LLM 和 ASR 可行，Qwen3-TTS 不建议 |

## 结论

E0 改变了问题的重心。原方案把压缩当作主杠杆，指望靠差量和 LoRA 把三个功能压成一个；实测之后，主杠杆变成了**统一运行时契约加外围组件工程**。三个功能可以共用一套 kernel、量化格式、graph 拓扑和 KV 池，但必须各带一份权重。对现有模型而言，收益来自“一次优化，三处生效”，而不是“一次存储，三处使用”。而且 TTS 的瓶颈根本不在主干带宽上：q8_0 只带来 3.8% 的提速，92% 的时间耗在自回归循环上。所以最值得投入的 GPU 工作是 code_predictor 的图化，以及 codec 的有状态流式化，而不是更激进的主干量化。

如果仍然想要权重共享，它必须被**设计进训练**，而不能从现有 checkpoint 中提取。做法是在一个冻结、已量化的 Qwen3 基座上，用 LoRA 或受约束的差量去训练新的功能模型。这正是方案文档中原本作为后备的“可嫁接预训练”思路，现在应升为主路径。它的可行性要用一个小规模的 ASR LoRA 试训来回答，而不是靠更多的权重分析。至于主干选哪一代，结论有分叉：端侧 NPU 选 Qwen3；CPU、GPU 和 Apple 芯片可以考虑 Qwen3.5，因为它的 GDN 状态是常量大小，适合同时挂多个流式任务。但 Qwen3.5 目前没有任何 ASR 或 TTS 变体，暂时只能承担 LLM 这一个角色。

## 优先实验清单

标注说明：【4050】可在 Windows 笔记本上跑（llama.cpp CUDA 或原生 PyTorch，不依赖 vLLM）；【Mac】可在本机跑；【AutoDL】需要租用 Linux GPU（vLLM、vLLM-Omni、训练）。动用 GPU 前先查看 `nvidia-smi`，长任务用 `Start-Process` 后台运行并轮询日志。

**P0-1 三模型常驻容量与单模型基线【4050】。** 目标是确认“全部常驻”能否成立。设置：在 llama.cpp CUDA 上加载 Qwen3-1.7B Q4_K_M；把 ggml-org 的 Qwen3-ASR-1.7B GGUF 重新量化为 Q4_K_M（mmproj 保持 Q8）；加载 Qwen3-TTS-12Hz-1.7B-Base GGUF 的 Q4_K_M。先逐个测，再三者同时驻留测。指标：峰值显存、各自的 decode tok/s、ASR 的 RTF、TTS 的 RTF。判据：三者总显存 ≤5.5 GB，且单独运行时 TTS 的 RTF ≤0.5，就采纳常驻方案；超出则把 ASR 换成 0.6B，或转向锁页内存换入方案（见 P0-3）。

**P0-2 分功能量化敏感度【4050 或 Mac】。** 目标是为每个功能选出最低可用比特，而不是统一用 Q4。设置：ASR 用 AISHELL 或 FLEURS-zh 中的 200 条、LibriSpeech test-clean 中的 200 条，比较 BF16、Q8、Q5、Q4（编码器分别试 Q8 和 F16）。TTS 用 1.7B 和 0.6B 各合成 100 句，测 Q8 和 Q4。Mac 上可以直接用 mlx-audio 复现。指标：ASR 的 CER 和 WER；TTS 的 round-trip WER（用 ASR 转写）和说话人相似度（ECAPA 余弦）。判据：相对 BF16 劣化 ≤0.5 个百分点的最低比特即为选型。如果 TTS 需要 Q8 而另外两个能用 Q4，就接受 TTS 单独一套 graph。

**P0-3 切换、换入与并发微基准【4050】。** 目标是量化 CUDA graph 模式 A/B 的成本，以及单 GPU 并发时的互相干扰。设置：用 PyTorch 先确认 PCIe 链路宽度（`nvidia-smi -q`），测锁页和可分页内存的 H2D 带宽；用 3 个同形状的 Qwen3-1.7B 解码器（4-bit 或 BF16）各捕获一张 decode graph；比较“graph 间切换”“同缓冲区拷入新权重”“双流优先级交错（LLM decode 加 TTS talker 步）”三种情况。Windows 上 PyTorch CUDA graph 的可用性要先确认。指标：切换延迟，4-bit 和 BF16 的换入时间，并发时 LLM tok/s 的下降幅度和 TTS 每帧延迟。判据：4-bit 换入 <150 ms，则换入方案可作为显存不足时的后备；并发时 LLM 降速 <30%，且 TTS 每帧 <80 ms（12.5 Hz 的实时预算），就采纳单进程交错调度。

**P1-1 code_predictor 图化【4050】。** 目标是验证它是不是 TTS 的延迟瓶颈。设置：在官方 `qwen-tts` 中分别计时 talker 一步、code_predictor 的 15 步、codec 解码；然后把 code_predictor 用 `torch.cuda.graphs` 捕获，或改用 `torch.compile(mode="reduce-overhead")`。指标：每帧各段耗时（ms）。判据：eager 模式下 code_predictor 超过帧预算的 30%（>24 ms）时，图化就是必做项，并在基底运行时里抽象成“多步 decode 循环”。

**P1-2 codec 有状态流式解码【4050】。** 目标是消除 25 帧左上下文的重复计算。设置：实现每层卷积的状态环形缓冲，加上滑窗 72 的 KV，与官方 `chunked_decode` 对比，测 4 帧一包和 8 帧一包。指标：每包解码耗时，以及与离线整段解码输出之间的 SNR。判据：提速 ≥3 倍且 SNR ≥40 dB，就把卷积状态缓存定为基底的一等缓存类型；GDN 的短卷积可以复用同一抽象。

**P1-3 端到端语音首音延迟【4050】。** 目标是拿到单张 6 GB 卡的真实端到端数字。设置：VAD → ASR → LLM（到子句边界）→ TTS 流式，使用 P0-3 选定的调度方式，跑 50 轮对话。指标：从说话结束到首音的 P50 和 P95，以及有无音频欠载。判据：P50 <800 ms 即达到数据中心级流式级联的水平；超过 1.2 s 时，按每段耗时回到 P1-1 或 P1-2 优化。

**P2-1 ASR 差量可压性的真实验证【Mac 做权重统计，4050 跑 WER】。** 目标是确认 ASR 在 0.30 的漂移下，能否用低比特或稀疏差量换来体积收益。设置：对 ASR thinker 的全部矩阵计算 Δ = W_asr − W_base，分别做 2-bit 和 3-bit 分组量化，以及 50% 稀疏加 4-bit；用“Q4 基座 + 压缩差量”重建权重，再跑 P0-2 的 ASR 测试集。可选：用同样方法测 Qwen3-VL-2B/4B 的 LLM 部分（先核对形状），补齐 VL 的漂移数据。指标：WER，以及“差量字节数 ÷ Q4 完整模型字节数”。判据：差量体积 ≤50% 且 WER 不差于独立的 Q4 ASR，就保留 ASR 的差量路径；否则放弃差量方案，所有功能都独立量化。TTS 不做此实验，因为漂移约 1.0，已经足以判定。

**P2-2 Mac 全栈常驻【Mac】。** 目标是给出 Apple 端的可行性结论。设置：mlx-lm 跑 Qwen3-1.7B 4-bit，mlx-audio 跑 ASR-1.7B 5-bit 和 TTS-1.7B 4-bit，三者常驻同一进程，并开启 `--stream`。指标：总 RSS，并发时各组件的 RTF，首音延迟。判据：TTS 并发时 RTF <0.7 即可行；否则 Mac 端 TTS 改用 0.6B 8-bit 或其他轻量 TTS。

**P2-3 服务端多模型参照【AutoDL】。** 目标是给本地方案提供上限参照，并验证 sleep mode 和多 LoRA。设置：在单张 24 GB GPU 上，用 vLLM 跑 Qwen3-ASR（开启 SupportsLoRA）和 Qwen3-1.7B，用 vLLM-Omni 跑 Qwen3-TTS，开启 `--enable-sleep-mode` 测试互相切换。指标：唤醒时间，切换后首个 token 的耗时，流式首音时间。判据：如果唤醒 <0.3 s，就在本地 llama.cpp 进程中仿照实现“卸到锁页内存”的休眠接口。

**P3-1 LoRA 约束的功能模型试训【AutoDL】。** 目标是回答“如果未来想共享权重，代价多大”。设置：冻结 Q4 量化的 Qwen3-1.7B 作为基座（QLoRA），复用 Qwen3-ASR 的 AuT 编码器和投影器，只训练 r=64/128 的 LoRA，以及必要时的投影器；在中文和英文 ASR 数据的子集上训练。指标：与 Qwen3-ASR-1.7B 的相对 WER 差距、适配器大小。判据：相对差距 ≤10%，就把“冻结基座加 LoRA”定为今后新功能的默认训练范式，届时 Punica 式批处理和共享 embedding 才重新成立；超过 25%，则接受“共享内核、独立权重”为长期架构。

**P3-2 三模型 grouped GEMM【4050】。** 目标是验证研究笔记“收益 <20%”的估算，决定要不要投入。设置：把三套同形状权重叠成 [3, out, in]，用 Triton 或 CUTLASS 的 grouped GEMM，与三次独立的 graph decode 对比，batch 取 1、4、16。指标：每轮三个模型各走一步的总耗时。判据：提速 <15% 就放弃，只保留模式 A。

**P3-3 Qwen3.5 作为 LLM 角色【4050 和 Mac】。** 目标是评估 GDN 常量状态对多路流式并发的价值。设置：用 llama.cpp CUDA 和 MLX 分别跑 Qwen3.5-0.8B/2B 与 Qwen3-0.6B/1.7B，在 4K 和 16K 上下文下测 decode 速度和状态内存，并测 4 路并发。指标：tok/s、每路的状态和 KV 字节数、质量（与当前用例相关的任务集）。判据：Qwen3.5 在同等质量下内存更省、并发更稳，就在桌面和 Mac 端的 LLM 角色上切换到 Qwen3.5；NPU 端维持 Qwen3。
