---
layout: distill
title: "Scaling LLM Training with DeepSpeed"
description: "Making the most of limited GPU memory with DeepSpeed."

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

## Introduction

How far can we scale training on four GH200 GPUs using DeepSpeed's memory partitioning, compilation, and CPU offloading features? We explored this on the ALPS supercomputer, measuring memory use and throughput while checking whether training loss decreased.

ZeRO made configurations fit that ran out of memory under our baseline, and DeepCompile increased measured throughput by up to 61% without offloading. CPU offloading reduced GPU memory use further, but several combinations showed flat training loss or parameter-update problems. Those runs illustrate why throughput alone is not enough to evaluate a training configuration.

This post describes the configurations we tested, their measured tradeoffs, and the unresolved failures we encountered. The findings apply to these experiments and software versions, rather than establishing general compatibility limits. The code is available on [GitHub](https://github.com/ImaGoodFella/lsaie_project).

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

<div class="caption">Table 1: Model configurations used in our experiments. Parameter counts exclude embedding layers.</div>

### Training

We train for up to 1000 steps with a micro-batch size of 1 per GPU and a sequence length of 2048 tokens. We use AdamW with learning rate $5 \times 10^{-5}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, and 100 linear warmup steps. For throughput measurements, we report tokens per second averaged over steps ≥ 6 to exclude compilation and initialization overhead.

### Software

We use PyTorch 2.6 (NVIDIA container 25.01), DeepSpeed 0.18.x, and the Hugging Face Transformers tokenizer. All experiments use BF16 mixed precision.

## Memory Optimization via ZeRO Stages

Our initial objective was to scale the model hidden dimension from $d_{\text{model}} = 2048$ to $d_{\text{model}} = 4096$ while keeping the hardware configuration constant at four GPUs. Under the baseline configuration (Stage 0), increasing the hidden dimension to 4096 resulted in immediate CUDA out-of-memory (OOM) errors. This behavior is expected, as doubling the hidden dimension approximately quadruples the number of parameters in the Transformer layers and also increases activation memory, increasing the model size from roughly 1.7B to 6.7B parameters.

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

<div class="caption">Table 2: Comparison of whether the model fits and peak memory usage for a 4096-dimension model across DeepSpeed ZeRO stages on 4 GPUs.</div>

Notably, Stage 3 achieves approximately 10% lower peak memory than Stages 1 and 2. This difference arises from how each stage partitions model state. In Stages 1 and 2, each GPU retains a full copy of the model parameters throughout training—only optimizer states (Stage 1) or optimizer states and gradients (Stage 2) are partitioned across workers. In contrast, Stage 3 partitions the parameters themselves: each GPU stores only 1/N of the parameters persistently (where N is the number of GPUs) and reconstructs full parameter tensors on-demand via all-gather operations during forward and backward passes. The gathered parameters are discarded after use, freeing memory. The lower peak memory is consistent with reduced parameter replication, although these measurements do not isolate its contribution from other buffers and temporary allocations. However, this memory benefit comes at the cost of increased communication volume, a trade-off we address through compilation in the next section.

## Compute Efficiency with Compilation

Memory optimizations such as ZeRO partitioning and offloading often introduce communication overhead that can limit training throughput. To counteract this and improve hardware utilization, we evaluated DeepSpeed's compilation support through `torch.compile` and DeepCompile.

<a id="torch-compile"></a>

### torch.compile

PyTorch's `torch.compile` captures computation graphs and applies compiler optimizations. In our measurements, enabling it alone produced smaller throughput gains than DeepCompile. These throughput measurements do not isolate which operations or communication patterns limited performance.

### DeepCompile

[DeepCompile](https://arxiv.org/abs/2504.09983) applies profiling-guided graph transformations to coordinate memory use and communication with computation. Its optimizations include proactive parameter prefetching, selective unsharding, and adaptive offloading. This provides a framework for exploring distributed training optimizations beyond the compiler configuration used in our `torch.compile` baseline.

### Throughput Analysis

We measured training throughput with and without compilation across different ZeRO stages. Our results demonstrate that DeepCompile provides speedups across the three tested ZeRO stages without offloading. Specifically, DeepCompile improves throughput by approximately 12% for ZeRO Stage 1 and 13% for Stage 2 relative to the uncompiled baseline, increasing throughput from 19.7k to 22.1k tokens/sec and from 19.6k to 22.2k tokens/sec, respectively. The benefits are substantially more pronounced for ZeRO Stage 3, where DeepCompile achieves a ~61% speedup, boosting throughput from 14.6k to 23.5k tokens/sec.

{% include figure.liquid loading="eager" path="assets/img/lsaie/throughput_comparison_new.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">Figure 1: Throughput comparison across DeepSpeed ZeRO stages using different compilation methods. Values are averaged over training steps ≥6, excluding the initial warm-up phase.</div>

DeepCompile's largest measured gain was at Stage 3. Better overlap between parameter communication and computation is a possible explanation, but confirming the cause would require profiling these runs. The results establish a throughput difference between the tested configurations, rather than identifying the contribution of each optimization.

We note that our compilation runs incurred overhead during the first 5 training steps while the computation graph is traced and optimized. In our experiments, this warm-up phase added approximately 90 seconds total, but this cost amortizes quickly over longer training runs.

## Extending Memory Limits

Even with ZeRO Stage 3 partitioning optimizer states, gradients, and parameters across GPUs, available VRAM remains a hard constraint on maximum model size. This is particularly problematic when access to additional computational resources is limited.

### ZeroOffload

ZeroOffload addresses this by offloading optimizer states and computation to CPU memory, effectively treating system RAM as an extension of GPU memory. During training, gradients are transferred to the CPU where optimizer updates are computed, and the updated parameters are then moved back to the GPU. This approach significantly reduces VRAM usage at the cost of throughput, as data movement is bottlenecked by interconnect bandwidth (PCIe on traditional systems, though GH200's NVLink-C2C substantially reduces this penalty).

### SuperOffload

[SuperOffload](https://arxiv.org/abs/2509.21271) targets tightly coupled CPU-GPU systems such as the GH200. It coordinates CPU, GPU, and interconnect use through adaptive weight offloading, bucket repartitioning, Superchip-aware casting, speculative execution, and an Adam optimizer optimized for Grace CPUs. The paper reports up to 2.5× throughput improvement over its offloading baselines; those results use different workloads and should be distinguished from our measurements below.

### Running SuperOffload

During setup, we encountered a crash in `SuperOffloadOptimizer_Stage3` caused by a missing `param_names` argument. We submitted a fix in [PR #7715](https://github.com/deepspeedai/DeepSpeed/pull/7715), which was merged into DeepSpeed. This setup fix is separate from the training-loss issues discussed below.

### Gradient Clipping with SuperOffload

In our runs, SuperOffload's measured throughput was higher with gradient clipping disabled. Its speculative execution mechanism overlaps optimizer work with backward computation, with corrections needed when the speculative updates cannot be used. Frequent clipping could therefore reduce the benefit of speculation, but the throughput comparison alone does not establish how often corrections occurred.

We disabled gradient clipping for subsequent SuperOffload experiments. This changes the training configuration and should be considered when comparing results. As discussed below, some SuperOffload runs also showed flat training loss, so higher throughput should not be interpreted as faster successful training.

{% include figure.liquid loading="eager" path="assets/img/lsaie/grad_clip_comparison.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">Figure 2: Training throughput for SuperOffload with and without gradient clipping enabled. Disabling clipping increased measured throughput in these runs; this comparison does not establish training correctness.</div>

### VRAM Savings

Enabling CPU offloading substantially reduced GPU memory usage compared to standard ZeRO Stage 3.

{% include figure.liquid loading="eager" path="assets/img/lsaie/stage3_offload_metrics_time.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">Figure 3: Peak and average GPU memory usage for ZeRO Stage 3 with and without CPU offloading. Both offloading strategies substantially reduce VRAM requirements, with ZeroOffload achieving the lowest average memory footprint.</div>

As shown in Figure 3, offloading dramatically reduces GPU memory usage. ZeroOffload reduces peak memory from 52.6 GB (Stage 3 baseline) to 24.2 GB—a 54% reduction—while SuperOffload achieves 30.5 GB peak memory (42% reduction). Average memory during training drops even more substantially: from 33.3 GB to 4.9 GB for ZeroOffload and 6.8 GB for SuperOffload. The gap between peak and average memory shows that transient allocations matter; attributing the peak specifically to initialization would require a closer review of the memory traces.

### Throughput Trade-off

CPU offloading incurs a performance cost due to data transfers between CPU and GPU. The increased movement of data and the resulting synchronization overhead can lead to stalling, which significantly reduces throughput.

{% include figure.liquid loading="eager" path="assets/img/lsaie/average_throughput_steps_6_plus.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">Figure 4: Average training throughput (tokens per second) measured after the warm-up phase (steps ≥ 6) for different DeepSpeed configurations.</div>

Compared to Stage 3 training without offloading, SuperOffload reduces throughput by approximately 20%, while ZeroOffload results in a much larger degradation of about 45%, as shown in Figure 4. These are execution-rate measurements. The SuperOffload training-loss issues described next prevent us from treating its throughput advantage as a validated improvement in training efficiency.

### Convergence Issues with SuperOffload

While SuperOffload improved throughput compared to standard ZeroOffload, we observed that training loss did not decrease in multi-GPU configurations. We observed non-zero gradients but problems with parameter updates. Non-zero gradients alone do not establish that the gradients were correct. In the corresponding single-GPU run, training loss decreased.

{% include figure.liquid loading="eager" path="assets/img/lsaie/loss_comparison.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">Figure 5: Training loss for Multi-GPU ZeroOffload and SuperOffload across GPU configurations. The tested multi-GPU SuperOffload runs showed flat loss despite non-zero gradients.</div>

We also observed similar training-loss behavior when using the official [DeepSpeedExamples](https://github.com/deepspeedai/DeepSpeedExamples). This gives another reproduction on our system, but does not identify the root cause. Gradient aggregation, synchronization, and optimizer updates are possible areas to investigate; the observations do not establish which component was responsible.

We were unable to determine whether this issue is specific to the ALPS system or affects other GH200-based clusters, as we did not have access to alternative Superchip hardware for comparison.

<a id="deepcompile-offloading"></a>

### DeepCompile + Offloading

We investigated whether DeepCompile could be combined with CPU offloading to achieve both memory savings and improved throughput. Running SuperOffload with DeepCompile yielded a significant throughput improvement over SuperOffload alone. However, given the convergence issues identified with SuperOffload in the previous section, we cannot determine whether this speedup results from legitimate optimization or from dependency violations causing incorrect execution.

{% include figure.liquid loading="eager" path="assets/img/lsaie/combined_metrics_labeled.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">Figure 6: Performance comparison of DeepSpeed offloading strategies with and without DeepCompile. (a) Average training throughput (tokens/sec) for ZeroOffload and SuperOffload configurations. (b) Training loss convergence over time for the ZeroOffload strategy, comparing the Baseline (No Compile) against DeepCompile.</div>

The ZeroOffload run with DeepCompile also showed flat training loss, as shown in Figure 6. Since ZeroOffload without DeepCompile showed decreasing loss, this suggests the issue lies in the interaction between DeepCompile and offloading rather than offloading itself. This comparison identifies a failing combination in our setup; it does not establish a general incompatibility or its root cause.

We also observed that when using DeepSpeed's model initialization, enabling DeepCompile results in higher peak memory than expected. The source of this additional memory use remains unresolved.

### Scalability Across GPU Counts

Beyond extending memory via partitioning and offloading, we also evaluate how these techniques scale with the number of available GPUs. Specifically, we ask: _given the same hardware generation, how does increasing the GPU count affect the largest tested model size that fits in memory?_

To answer this, we vary the number of GPUs from 1 to 4 and report the largest tested model hidden dimension ($d_{\text{model}}$) that can be trained without encountering OOM errors under different DeepSpeed configurations.

| Configuration          | 1 GPU | 2 GPUs | 3 GPUs | 4 GPUs |
| :--------------------- | :---: | :----: | :----: | :----: |
| Stage 0 (Baseline)     | 2048  |  2048  |  2048  |  2048  |
| Stage 1                | 2048  |  3072  |  4096  |  4096  |
| Stage 2                | 2048  |  3072  |  4096  |  4096  |
| Stage 3                | 3072  |  3072  |  4096  |  4096  |
| Stage 3 + ZeroOffload  | 5120  |  5120  |  5120  |  5120  |
| Stage 3 + SuperOffload | 5120  |  5120  |  5120  |  5120  |

<div class="caption">Table 3: Largest Tested Model Hidden Dimension That Fits ($d_{\text{model}}$) by DeepSpeed Stage. Values represent the largest tested dimension that fit without OOM, not a validated training outcome for every configuration.</div>

With offloading, a hidden dimension of 5120 fit at every tested GPU count. This shows that offloading expanded the range of configurations that fit in these experiments. It does not establish an absolute model-size limit, show that host memory was the bottleneck, or imply that capacity is generally independent of GPU count. Fitting in memory also does not resolve the training-loss issues described above.

## Conclusion

ZeRO Stage 1 was already sufficient to fit the 4096-dimension model on four GPUs; Stage 3 reduced peak memory further, from 58.8 GB to 52.6 GB. Without offloading, DeepCompile increased measured throughput across the three tested ZeRO stages, with the largest gain at Stage 3.

CPU offloading reduced GPU memory use and allowed larger tested configurations to fit. However, some SuperOffload and compilation-plus-offloading runs showed flat training loss or parameter-update problems. Their execution rates should therefore be reported separately from demonstrated training progress.

Among the configurations tested, uncompiled ZeRO Stage 3 with ZeroOffload reduced GPU memory use while showing decreasing loss. DeepCompile without offloading delivered the highest measured throughput in the compilation comparison. These observations provide starting points for further experiments, rather than universal recommendations about current DeepSpeed releases.

The next step is to reproduce the failing combinations with exact software revisions and investigate the parameter updates. Profiling would also help distinguish the causes of throughput and memory differences. Until then, the conclusions are limited to the behavior observed in these runs.

---

## References

1. Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2020). ZeRO: Memory optimizations toward training trillion parameter models. _SC20: International Conference for High Performance Computing, Networking, Storage and Analysis_.

2. Tanaka, M., Li, D., Chand, U., Zafar, A., Shen, H., & Ruwase, O. (2025). [DeepCompile: A Compiler-Driven Approach to Optimizing Distributed Deep Learning Training](https://arxiv.org/abs/2504.09983). _arXiv:2504.09983_.

3. Ren, J., et al. (2021). ZeRO-Offload: Democratizing billion-scale model training. _USENIX Annual Technical Conference_.

4. Lian, X., Tanaka, M., Ruwase, O., & Zhang, M. (2025). [SuperOffload: Unleashing the Power of Large-Scale LLM Training on Superchips](https://arxiv.org/abs/2509.21271). _arXiv:2509.21271_.

5. Ansel, J., et al. (2024). PyTorch 2: Faster machine learning through dynamic Python bytecode transformation and graph compilation. _ASPLOS_.
