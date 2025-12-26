---
layout: distill
title: "An Evaluation of DeepSpeed ZeRO, Compilation, and Offloading"
description: "This blog contains the final project of the course Large-Scale AI Engineering."

tags: DeepSpeed LSAIE Finetuning
giscus_comments: false
date: 2025-12-26
featured: false
mermaid:
  enabled: true
  zoomable: true
code_diff: true
map: true
chart:
  chartjs: true
  echarts: true
  vega_lite: true
tikzjax: true
typograms: true

authors:
  - name: Rahul Steiger
    affiliations:
      name: ETH Zurich
  - name: Bernhard Zuidema
    affiliations:
      name: ETH Zurich
  - name: Lampros Dionysiou
    affiliations:
      name: ETH Zurich

bibliography:

toc:
  - name: Abstract
  - name: Introduction
  - name: Experimental Setup
  - name: Memory Optimization via ZeRO Stages
    subsections:
      - name: Results
  - name: Compute Efficiency with Compilation
    subsections:
      - name: torch.compile
      - name: DeepCompile
      - name: Throughput Analysis
  - name: Extending Memory Limits
    subsections:
      - name: ZeroOffload
      - name: SuperOffload
      - name: Running SuperOffload
      - name: Gradient Clipping with SuperOffload
      - name: VRAM Savings
      - name: Throughput Trade-off
      - name: Convergence Issues with SuperOffload
      - name: DeepCompile + Offloading
      - name: Scalability Across GPU Counts
  - name: Conclusion

_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Abstract

Training Large Language Models (LLMs) on memory-constrained hardware requires careful orchestration of memory optimization, compilation, and offloading techniques. We present a systematic evaluation of DeepSpeed's optimization suite for training a 6.7B parameter model on four NVIDIA GH200 GPUs. Our analysis yields three key findings. First, ZeRO Stage 3 reduces peak memory usage by 10% compared to Stage 1, enabling training of models that cause out-of-memory errors under standard data parallelism. Second, DeepCompile delivers substantial throughput gains—12–13% for ZeRO Stages 1–2 and 61% for Stage 3—by optimizing communication scheduling alongside computation. Third, while CPU offloading extends trainable model size, we uncover critical stability issues: DeepCompile combined with offloading fails to converge, and SuperOffload exhibits multi-GPU training failures on our platform. Based on these findings, we recommend ZeRO Stage 3 with DeepCompile for throughput-critical workloads and ZeRO Stage 3 with ZeroOffload for memory-constrained scenarios requiring stable training. The code is available on [GitHub](https://github.com/ImaGoodFella/lsaie_project).

## Introduction

The rapid growth in Large Language Model (LLM) parameter counts has outpaced improvements in GPU memory capacity, making memory optimization a central challenge in modern deep learning. State-of-the-art models routinely exceed tens of billions of parameters, yet even high-end GPUs provide only 80 GB of VRAM—insufficient to hold the parameters, gradients, and optimizer states required for training. Standard Distributed Data Parallel (DDP) approaches exacerbate this problem by replicating the entire model state across all devices.

Several techniques address these constraints. DeepSpeed's Zero Redundancy Optimizer (ZeRO) partitions model state across workers, reducing per-GPU memory requirements. Compilation frameworks such as DeepCompile optimize computation graphs with awareness of communication patterns. For extreme memory constraints, CPU offloading techniques like ZeroOffload and SuperOffload leverage host RAM as extended memory.

While these techniques are well-documented individually, their interactions remain underexplored—particularly on tightly-coupled CPU-GPU architectures like the NVIDIA GH200. In this work, we systematically evaluate DeepSpeed's optimization suite on four GH200 GPUs, training a GPT-style model scaled up to 6.7B parameters. We find that DeepCompile provides up to 61% throughput improvement on ZeRO Stage 3, but combining it with CPU offloading causes convergence failures. We also identify multi-GPU training instabilities in SuperOffload and contribute a bug fix merged into DeepSpeed.

## Experimental Setup

### Hardware

All experiments were conducted on the ALPS supercomputer at CSCS, using up to four NVIDIA GH200 Superchip nodes. Each GH200 pairs a 72-core ARM-based Grace CPU with 128 GB of LPDDR5X memory and an H100 GPU with 96 GB HBM3, connected via a 900 GB/s NVLink-C2C interconnect.

### Model

We train a GPT-style decoder-only Transformer with 32 layers, 32 attention heads, 8 key-value heads (grouped-query attention), and SwiGLU activations with FFN multiplier 1.3. We use RMSNorm for layer normalization, rotary position embeddings (RoPE) with θ = 500,000, and a vocabulary size of 128k tokens. To evaluate memory scaling, we vary the hidden dimension $d_{\text{model}}$ across configurations:

| Hidden Dim ($d_{\text{model}}$) | Parameters |
| :-----------------------------: | :--------: |
|              2048               |    1.7B    |
|              3072               |    3.7B    |
|              4096               |    6.7B    |
|              5120               |   10.4B    |

_Table 1: Model configurations used in our experiments. Parameter counts exclude embedding layers._

### Training

We train for up to 1000 steps with a micro-batch size of 1 per GPU and a sequence length of 2048 tokens. We use AdamW with learning rate $5 \times 10^{-5}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, and 100 linear warmup steps. For throughput measurements, we report tokens per second averaged over steps ≥ 6 to exclude compilation and initialization overhead.

### Software

We use PyTorch 2.6 (NVIDIA container 25.01), DeepSpeed 0.18.x, and the Hugging Face Transformers tokenizer. All experiments use BF16 mixed precision.

## Memory Optimization via ZeRO Stages

Our initial objective was to scale the model hidden dimension from $d_{\text{model}} = 2048$ to $d_{\text{model}} = 4096$ while keeping the hardware configuration constant at four GPUs. Under the baseline configuration (Stage 0), increasing the hidden dimension to 4096 resulted in immediate CUDA out-of-memory (OOM) errors. This behavior is expected, as doubling the hidden dimension approximately quadruples the number of parameters and intermediate activations, increasing the model size from roughly 1.7B to 6.7B parameters.

To mitigate this, we evaluated DeepSpeed's Zero Redundancy Optimizer (ZeRO) stages:

- **Stage 1 (Optimizer State Partitioning):** Partitions the optimizer states across data parallel workers. This provides significant memory savings as optimizer states often consume the largest portion of VRAM (e.g., 12 bytes per parameter for Adam).

- **Stage 2 (Gradient Partitioning):** In addition to optimizer states, gradients are also partitioned. This further reduces the memory footprint, allowing for larger batch sizes or model dimensions.

- **Stage 3 (Parameter Partitioning):** Partitions the model parameters themselves. Each GPU stores only a fraction of the model, fetching necessary partitions just in time during the forward and backward passes.

### Results

We observed that while Stage 0 failed for the 4096-dimension model, enabling higher ZeRO stages allowed the model to fit comfortably in memory.

| Configuration      | Model Dim | Status  | Peak VRAM (GB) |
| :----------------- | :-------: | :-----: | :------------: |
| Stage 0 (Baseline) |   4096    | **OOM** |      N/A       |
| Stage 1            |   4096    | Success |      58.8      |
| Stage 2            |   4096    | Success |      58.6      |
| Stage 3            |   4096    | Success |      52.6      |

_Table 2: Comparison of training stability and memory usage for a 4096-dimension model across DeepSpeed ZeRO stages on 4 GPUs._

Notably, Stage 3 achieves approximately 10% lower peak memory than Stages 1 and 2. This difference arises from how each stage partitions model state. In Stages 1 and 2, each GPU retains a full copy of the model parameters throughout training—only optimizer states (Stage 1) or optimizer states and gradients (Stage 2) are partitioned across workers. In contrast, Stage 3 partitions the parameters themselves: each GPU stores only 1/N of the parameters persistently (where N is the number of GPUs) and reconstructs full parameter tensors on-demand via all-gather operations during forward and backward passes. The gathered parameters are discarded after use, freeing memory. The observed 6 GB reduction corresponds directly to eliminating this redundant parameter storage. However, this memory benefit comes at the cost of increased communication volume, a trade-off we address through compilation in the next section.

## Compute Efficiency with Compilation

Memory optimizations such as ZeRO partitioning and offloading often introduce communication overhead that can limit training throughput. To counteract this and improve hardware utilization, we evaluated DeepSpeed's compilation support through `torch.compile` and DeepCompile.

### torch.compile

PyTorch's built-in compiler converts models into optimized computation graphs, enabling operator fusion and memory reuse. However, `torch.compile` operates independently of distributed training logic—it optimizes individual kernels without visibility into the communication patterns introduced by ZeRO's parameter partitioning. As a result, the communication overhead that dominates ZeRO Stage 3 performance remains unaddressed.

### DeepCompile

DeepCompile is a compiler-driven framework that optimizes distributed training by transforming the computation graph with awareness of both memory usage and communication patterns. Unlike `torch.compile`, which focuses on single-GPU optimizations, DeepCompile specifically targets the fully sharded approach used by ZeRO-3 and FSDP.

DeepCompile applies a sequence of profiling-guided optimization passes:

- **Proactive Prefetching:** Initiates all-gather operations as early as possible based on dynamic memory availability, maximizing overlap between communication and computation.

- **Selective Unsharding:** Keeps frequently accessed parameters unsharded when memory permits, reducing redundant all-gather operations—particularly effective with gradient accumulation.

- **Adaptive Offloading:** Offloads only the minimum necessary optimizer states to CPU and schedules transfers to overlap with computation, reducing the performance penalty of offloading.

The authors report up to 1.28× throughput improvement over ZeRO-3 on Llama-3 70B, and up to 7× improvement when using offloading on memory-constrained configurations.

### Throughput Analysis

We measured training throughput with and without compilation across different ZeRO stages. Our results demonstrate that DeepCompile provides consistent speedups across all configurations. Specifically, DeepCompile improves throughput by approximately 12% for ZeRO Stage 1 and 13% for Stage 2 relative to the uncompiled baseline, increasing throughput from 19.7k to 22.1k tokens/sec and from 19.6k to 22.2k tokens/sec, respectively. The benefits are substantially more pronounced for ZeRO Stage 3, where DeepCompile achieves a ~61% speedup, boosting throughput from 14.6k to 23.5k tokens/sec.

{% include figure.liquid loading="eager" path="assets/img/lsaie/throughput_comparison_new.png" class="img-fluid rounded z-depth-1" zoomable=true %}
_Figure 1: Throughput comparison across DeepSpeed ZeRO stages using different compilation methods. Values are averaged over training steps ≥6, excluding the initial warm-up phase._

While compilation generally improves performance, the magnitude of improvement varies considerably by compiler. `torch.compile` yields only marginal gains, likely because it optimizes computational kernels without addressing the communication bottlenecks that dominate overall performance. In contrast, DeepCompile incorporates communication constraints into its optimization strategy, enabling it to substantially outperform the baseline across all configurations. We hypothesize that the disproportionate gains observed in Stage 3 stem from additional optimization passes that DeepCompile applies specifically to ZeRO-3 workloads.

We note that compilation introduces overhead during the first 5 training steps while the computation graph is traced and optimized. In our experiments, this warm-up phase added approximately 90 seconds total, but this cost amortizes quickly over longer training runs.

## Extending Memory Limits

Even with ZeRO Stage 3 partitioning optimizer states, gradients, and parameters across GPUs, available VRAM remains a hard constraint on maximum model size. This is particularly problematic when access to additional computational resources is limited.

### ZeroOffload

ZeroOffload addresses this by offloading optimizer states and computation to CPU memory, effectively treating system RAM as an extension of GPU memory. During training, gradients are transferred to the CPU where optimizer updates are computed, and the updated parameters are then moved back to the GPU. This approach significantly reduces VRAM usage at the cost of throughput, as data movement is bottlenecked by interconnect bandwidth (PCIe on traditional systems, though GH200's NVLink-C2C substantially reduces this penalty).

### SuperOffload

The emergence of tightly coupled heterogeneous GPU/CPU architectures—often referred to as _Superchips_—such as the NVIDIA GH200, GB200, and AMD MI300A, presents new optimization opportunities for large-scale machine learning. These systems feature high-bandwidth CPU-GPU interconnects (e.g., 900 GB/s for NVLink-C2C on GH200) that fundamentally change the performance tradeoffs of offloading strategies. Existing solutions like ZeroOffload were designed for slower interconnects (e.g., 64 GB/s for PCIe Gen4) and are therefore suboptimal on Superchips.

SuperOffload addresses this gap with techniques specifically designed for Superchip architectures. Built on DeepSpeed ZeRO Stage 3, it enables full-parameter fine-tuning of 20B-parameter models on a single GH200 and Llama-70B on four GH200s, achieving up to 4× higher throughput than ZeroOffload. SuperOffload introduces four key optimizations:

- **Speculation-then-Validation:** Speculatively executes optimizer updates on CPU in parallel with GPU backward computation, avoiding synchronization stalls from gradient norm clipping and NaN/Inf checks.

- **Heterogeneous Optimizer Computation:** Partitions optimizer work across GPU and CPU to reduce idle time and unnecessary data transfers.

- **Superchip-Aware Casting:** Performs precision casting on GPU and transfers in FP32, exploiting the high interconnect bandwidth rather than minimizing transfer size.

- **GraceAdam:** An Adam optimizer implementation optimized for the ARM-based Grace CPU, achieving 3× speedup over PyTorch CPU Adam.

### Running SuperOffload

During our experiments, we encountered a bug in the DeepSpeed library that prevented SuperOffload from functioning at all. We identified and fixed the issue, submitting a pull request ([PR#7715](https://github.com/deepspeedai/DeepSpeed/pull/7715)) that has since been merged into DeepSpeed.

### Gradient Clipping with SuperOffload

While SuperOffload provided significant performance improvements, these gains were only realized when gradient clipping was disabled. The Speculation-then-Validation mechanism executes optimizer updates speculatively during backward propagation, then validates whether gradient clipping or NaN/Inf corrections are needed. If corrections are required, the speculative updates must be rolled back.

When fine-tuning pre-trained models, rollbacks are rare after warmup—the SuperOffload authors demonstrate this with BLOOM-176B. However, training from scratch produces large, unstable gradient norms that trigger clipping at nearly every step, causing constant rollbacks that negate the benefits of speculation. While rollback frequency reportedly decreases after ~1000 steps, we could not verify this due to limited compute budget and instead disabled gradient clipping for subsequent experiments.

{% include figure.liquid loading="eager" path="assets/img/lsaie/grad_clip_comparison.png" class="img-fluid rounded z-depth-1" zoomable=true %}
_Figure 2: Training throughput for SuperOffload with and without gradient clipping enabled. Gradient clipping triggers frequent rollbacks in early training, negating speculative execution benefits._

### VRAM Savings

Enabling CPU offloading substantially reduced GPU memory usage compared to standard ZeRO Stage 3.

{% include figure.liquid loading="eager" path="assets/img/lsaie/stage3_offload_metrics_time.png" class="img-fluid rounded z-depth-1" zoomable=true %}
_Figure 3: Peak and average GPU memory usage for ZeRO Stage 3 with and without CPU offloading. Both offloading strategies substantially reduce VRAM requirements, with ZeroOffload achieving the lowest average memory footprint._

As shown in Figure 3, offloading dramatically reduces GPU memory usage. ZeroOffload reduces peak memory from 52.6 GB (Stage 3 baseline) to 24.2 GB—a 54% reduction—while SuperOffload achieves 30.5 GB peak memory (42% reduction). Average memory during training drops even more substantially: from 33.3 GB to 4.9 GB for ZeroOffload and 6.8 GB for SuperOffload. The large gap between peak and average memory with offloading reflects the model initialization phase, during which parameters are fully materialized on GPU before offloading takes effect.

### Throughput Trade-off

CPU offloading incurs a performance cost due to data transfers between CPU and GPU. The increased movement of data and the resulting synchronization overhead can lead to stalling, which significantly reduces throughput.

{% include figure.liquid loading="eager" path="assets/img/lsaie/average_throughput_steps_6_plus.png" class="img-fluid rounded z-depth-1" zoomable=true %}
_Figure 4: Average training throughput (tokens per second) measured after the warm-up phase (steps ≥ 6) for different DeepSpeed configurations._

Compared to Stage 3 training without offloading, SuperOffload reduces throughput by approximately 20%, while ZeroOffload results in a much larger degradation of about 45%, as shown in Figure 4. Despite this cost, offloading remains valuable when model size exceeds available VRAM—a 20–45% throughput reduction is preferable to being unable to train at all.

### Convergence Issues with SuperOffload

While SuperOffload improved throughput compared to standard ZeroOffload, we observed that training loss did not decrease in multi-GPU configurations. Investigation revealed that gradients were computed correctly (non-zero values), but model parameters were not being updated properly. Notably, the same configuration converged normally on a single GPU.

{% include figure.liquid loading="eager" path="assets/img/lsaie/loss_comparison.png" class="img-fluid rounded z-depth-1" zoomable=true %}
_Figure 5: Training loss for Multi-GPU ZeroOffload and SuperOffload across GPU configurations. SuperOffload configurations fail to decrease loss despite correct gradient computation._

We attempted to reproduce this issue using the official [DeepSpeedExamples](https://github.com/deepspeedai/DeepSpeedExamples), which exhibited identical convergence failures. This suggests the issue stems from incorrect gradient aggregation or a synchronization bug in multi-GPU settings, rather than our specific configuration. We verified that gradients were non-zero and correctly computed on each rank, but parameter updates did not propagate correctly after synchronization.

We were unable to determine whether this issue is specific to the ALPS system or affects other GH200-based clusters, as we did not have access to alternative Superchip hardware for comparison.

### DeepCompile + Offloading

We investigated whether DeepCompile could be combined with CPU offloading to achieve both memory savings and improved throughput. Running SuperOffload with DeepCompile yielded a significant throughput improvement over SuperOffload alone. However, given the convergence issues identified with SuperOffload in the previous section, we cannot determine whether this speedup results from legitimate optimization or from dependency violations causing incorrect execution.

{% include figure.liquid loading="eager" path="assets/img/lsaie/combined_metrics_labeled.png" class="img-fluid rounded z-depth-1" zoomable=true %}
_Figure 6: Performance comparison of DeepSpeed offloading strategies with and without DeepCompile. (a) Average training throughput (tokens/sec) for ZeroOffload and SuperOffload configurations. (b) Training loss convergence over time for the ZeroOffload strategy, comparing the Baseline (No Compile) against DeepCompile._

Further investigation revealed that ZeroOffload combined with DeepCompile also fails to converge, as shown in Figure 6. Since ZeroOffload without DeepCompile converges normally, this suggests the issue lies in the interaction between DeepCompile and offloading rather than offloading itself. This points to a broader incompatibility between DeepCompile's graph transformations and DeepSpeed's offloading mechanisms, at least on our system configuration.

We also observed that when using DeepSpeed's model initialization, enabling DeepCompile results in higher peak memory than expected. We hypothesize that DeepCompile's graph transformations inadvertently bypass DeepSpeed's memory-efficient initialization routines, though further investigation is needed to confirm this.

### Scalability Across GPU Counts

Beyond extending memory via partitioning and offloading, we also evaluate how these techniques scale with the number of available GPUs. Specifically, we ask: _given the same hardware generation, how does increasing the GPU count affect the maximum trainable model size?_

To answer this, we vary the number of GPUs from 1 to 4 and measure the largest model hidden dimension ($d_{\text{model}}$) that can be trained without encountering OOM errors under different DeepSpeed configurations.

| Configuration          | 1 GPU | 2 GPUs | 3 GPUs | 4 GPUs |
| :--------------------- | :---: | :----: | :----: | :----: |
| Stage 0 (Baseline)     | 2048  |  2048  |  2048  |  2048  |
| Stage 1                | 2048  |  3072  |  4096  |  4096  |
| Stage 2                | 2048  |  3072  |  4096  |  4096  |
| Stage 3                | 3072  |  3072  |  4096  |  4096  |
| Stage 3 + ZeroOffload  | 5120  |  5120  |  5120  |  5120  |
| Stage 3 + SuperOffload | 5120  |  5120  |  5120  |  5120  |

_Table 3: Maximum Trainable Model Hidden Dimension ($d_{\text{model}}$) by DeepSpeed Stage. Values represent the largest dimension that fit in memory without OOM._

The results highlight that ZeRO Stage 3 primarily improves _memory efficiency per GPU_, while CPU offloading determines the _absolute upper bound_ on model size. Notably, offloading enables identical model sizes across all GPU counts, indicating that host memory, rather than GPU count, becomes the dominant constraint in these configurations. With offloading enabled, maximum model size becomes independent of GPU count: host memory (512 GB per node) rather than aggregate VRAM determines capacity, and each GPU offloads to its own local CPU memory.

## Conclusion

In this study, we systematically evaluated DeepSpeed's optimization suite for training Large Language Models on memory-constrained hardware. Our results demonstrate that while modern distributed training techniques can effectively overcome memory and compute limitations, they introduce complex trade-offs between performance and training stability.

**Memory Scalability.** The transition from standard data parallelism to ZeRO Stage 3 was the single most critical factor for model scaling. We successfully increased the model hidden dimension from 2048 to 4096 and beyond—infeasible with standard data parallelism on our 4-GPU setup. This confirms that parameter partitioning is essential for LLM training on consumer and mid-range hardware.

**Compilation.** DeepCompile provided consistent throughput improvements across all ZeRO stages (12–13% for Stages 1–2, 61% for Stage 3) by optimizing communication scheduling and overlapping data transfers with computation. However, we discovered that combining DeepCompile with CPU offloading (both ZeroOffload and SuperOffload) results in convergence failures, indicating an incompatibility between graph-level optimizations and offloading mechanisms.

**CPU Offloading.** ZeroOffload successfully extended our effective memory capacity by utilizing system RAM, enabling training of larger models at the cost of reduced throughput. SuperOffload, designed for tightly coupled CPU-GPU architectures like the GH200, showed promising throughput improvements over ZeroOffload but exhibited convergence failures in multi-GPU configurations on ALPS. We were unable to determine whether this is a system-specific issue or a broader bug, as we lacked access to alternative GH200-based systems for comparison.

**Recommendation.** For memory-constrained environments requiring stable training, we recommend ZeRO Stage 3 with ZeroOffload. For configurations where memory is sufficient, ZeRO Stage 3 with DeepCompile offers the best throughput. However, users should avoid combining compilation with offloading until these compatibility issues are resolved.

**Future Work.** Further investigation is needed to identify the root cause of convergence failures when combining DeepCompile with offloading, and to determine whether SuperOffload's multi-GPU issues are specific to ALPS or affect other platforms. Additionally, validating these findings on alternative GH200 or GB200 systems would help isolate system-specific factors from broader software bugs.

---

## References

1. Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2020). ZeRO: Memory optimizations toward training trillion parameter models. _SC20: International Conference for High Performance Computing, Networking, Storage and Analysis_.

2. Tanaka, M., et al. (2025). DeepCompile: A compiler-driven approach to optimizing distributed deep learning. _arXiv preprint_.

3. Ren, J., et al. (2021). ZeRO-Offload: Democratizing billion-scale model training. _USENIX Annual Technical Conference_.

4. SuperOffload Authors (2026). SuperOffload: Efficient offloading for Superchip architectures. _Technical Report_.

5. Ansel, J., et al. (2024). PyTorch 2: Faster machine learning through dynamic Python bytecode transformation and graph compilation. _ASPLOS_.
