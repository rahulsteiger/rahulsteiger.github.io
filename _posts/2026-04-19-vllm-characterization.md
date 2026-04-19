---
layout: distill
title: "Profiling vLLM on the GH200"
description: "What we learned benchmarking vLLM on the NVIDIA GH200.<br>Part 2 of 3 from my Master's thesis at ETH Zurich."

tags: llm-serving vllm gh200 profiling cuda-graphs nvlink
giscus_comments: false
date: 2026-04-19
featured: false

authors:
  - name: Rahul Steiger
    affiliations:
      name: ETH Zurich
  - name: Foteini Strati
    affiliations:
      name: ETH Zurich
  - name: Xiaozhe Yao
    affiliations:
      name: ETH Zurich
  - name: Prof. Ana Klimovic
    affiliations:
      name: ETH Zurich

bibliography: 2026-04-19-vllm-characterization.bib

toc:
  - name: Introduction
  - name: vLLM Architecture
    subsections:
      - name: Overview
      - name: Scheduler
      - name: KV Cache Manager
      - name: Model Executor
      - name: CPU Offloading
  - name: Experimental Setup
  - name: Memory System Performance
    subsections:
      - name: Copy Performance
      - name: Managed vs. Pinned Memory
  - name: Execution Optimizations
    subsections:
      - name: CUDA Graph Performance
      - name: Offloading Overhead
      - name: Prefix Caching Effectiveness
  - name: Multimodal Inference
  - name: Implications for System Design

---

## Introduction

In [Part 1](/notes/2026/llm-serving-background/), we covered the background on LLM serving, the NVIDIA GH200 architecture, and the existing offloading landscape. In this post, we present an empirical analysis of vLLM's performance on the GH200 Grace Hopper Superchip.

vLLM <d-cite key="kwon2023efficient"></d-cite> integrates the key optimizations described in Part 1 — continuous batching, chunked prefill, PagedAttention, and automatic prefix caching — into a unified, production-ready system. Its widespread adoption in both research and production environments, combined with its modular architecture and active development, makes it a representative target for studying LLM serving behavior on the GH200. Our analysis is based on vLLM v0.13.0, which uses the V1 engine architecture by default.

Our findings reveal several important implications: the GH200's NVLink-C2C interconnect provides substantial bandwidth but vLLM's current offloading mechanisms fail to exploit it, CUDA graphs are even more critical on the GH200 than on x86 systems, and a static offloading configuration is fundamentally limited. These observations motivate the design of our dynamic parameter offloading system, which we present in [Part 3](/notes/2026/llm-serving-dynamic-offloading/).

## vLLM Architecture

vLLM has evolved substantially since its initial release, with features such as chunked prefill, prefix caching, and CUDA graphs transitioning from experimental to enabled by default. This section provides an overview of the current vLLM V1 architecture.

### Overview

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/vllm-characterization/vllm_architecture.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="caption">Figure 1: vLLM V1 engine architecture. Figure from <d-cite key="gordic2025vllm"></d-cite>.</div>

The figure above illustrates the high-level architecture of vLLM's V1 engine. Arriving requests are tokenized by the processor and forwarded to the engine core client, which communicates with the engine core via inter-process communication. This separation allows CPU-intensive tasks such as tokenization and detokenization to run concurrently with GPU-bound model execution. The engine core runs on the CPU and serves as the central coordinator, containing the scheduler, model executor, and KV cache manager. After inference completes, generated tokens are forwarded to the output processor for detokenization and streaming back to the client.

The engine core is the central component of vLLM and has a modular structure containing the scheduler, model executor, and KV cache manager. It runs a busy loop that repeatedly performs an engine step, where each step consists of invoking the scheduler to select which requests to advance, followed by dispatching the scheduled batch to the model executor for a forward pass.

### Scheduler

vLLM handles two types of workloads: prefill and decode requests. The V1 scheduler can mix both types within a single batch but prioritizes decode requests, i.e., those already in the running queue. For each running request, the scheduler computes the number of new tokens to generate, allocates the required KV cache blocks, and updates the available token budget. If insufficient KV cache blocks are available, lower-priority requests are preempted and returned to the waiting queue, freeing their KV cache blocks for recomputation in a later step.

After processing decode requests, the scheduler handles prefill requests from the waiting queue. For each prefill request, the scheduler determines the required KV cache blocks. If prefix caching is enabled and requests share common prefixes, the corresponding blocks are shared rather than duplicated. Requests with sufficient KV cache capacity are promoted to the running queue and included in the batch dispatched to the model executor.

### KV Cache Manager

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/vllm-characterization/kv_cache_management.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="caption">Figure 2: KV Cache Management in vLLM. Figure from <d-cite key="gordic2025vllm"></d-cite>.</div>

The KV cache manager handles KV cache block allocation and is central to enabling both PagedAttention and prefix caching. Importantly, the manager only maintains indexing structures, whereas the physical GPU memory is allocated and managed by the model workers.

The core data structure is a free list of available block IDs. Each block typically stores 16 tokens of KV cache data. When the scheduler requires blocks for a request, they are popped from this free list and associated with the request. When a request completes, its blocks are returned to the pool.

To support prefix caching, the manager additionally maintains a hash table mapping token sequences to their corresponding KV blocks. When a new request arrives, its token sequence is split into block-sized chunks and hashed. The scheduler looks up these hashes to identify blocks that have already been computed, either by active requests or by recently completed ones. Matching blocks are shared rather than recomputed, avoiding redundant prefill computation for common prefixes.

To track sharing, each block maintains a reference counter that increments when assigned to a request and decrements upon completion. Blocks are only returned to the free list when their reference count reaches zero, and cached blocks remain valid until they must be reallocated to accommodate new requests.

### Model Executor

The model executor is responsible for running the forward pass of the model by coordinating workers, typically one per GPU. During initialization, each worker performs three key tasks: initializing its assigned device, loading the model weights, and allocating the KV cache. By default, vLLM allocates a separate KV cache tensor for each model layer, enabling compatibility with hybrid architectures that may have different KV cache requirements across layers (e.g., models combining attention and state-space layers).

### CPU Offloading

**KV Cache Offloading.** vLLM supports asynchronous offloading and loading of KV cache data to CPU memory via the offloading connector API <d-cite key="ozeri2026kv"></d-cite>. This capability serves two primary use cases: (1) caching KV values for prefix sharing, where loading from CPU is faster than recomputation, and (2) preserving KV data for preempted requests, avoiding costly recomputation when requests are later rescheduled. vLLM recently introduced a new contiguous KV cache memory layout that stores all layers' KV data in contiguous physical blocks, replacing the previous per-layer allocated layout. This change increased the effective block size from a few kilobytes to 0.5–2MB depending on the model and number of tokens in a block, improving direct memory access (DMA) transfer efficiency by an order of magnitude.

**Weight Offloading.** vLLM also supports offloading model parameters to CPU memory using unified virtual addressing (UVA), which allows the GPU to access CPU memory directly without explicit copies. While this approach enables serving models larger than GPU memory, it introduces a key limitation: data is fetched on-demand when accessed, preventing prefetching optimizations and resulting in higher latency for model weights not resident in GPU memory.

Neither KV cache offloading nor weight offloading is enabled by default in vLLM.

## Experimental Setup

All experiments were conducted on the Alps supercomputer at the Swiss National Supercomputing Centre (CSCS).

**Hardware.** Each node contains four GH200 Grace Hopper Superchips. Each superchip pairs a 72-core ARM Neoverse V2 CPU with an H100 GPU, connected by a 900 GB/s cache-coherent NVLink-C2C interconnect. Memory per superchip comprises 96GB of HBM3 (4 TB/s bandwidth) on the GPU and 128GB (500 GB/s) of LPDDR5X on the CPU. The four superchips are fully connected via NVLink for GPU-to-GPU communication, and the node operates as a single NUMA system with 288 CPU cores and 4 GPUs.

**Models and Workloads.** We evaluate three model configurations: Qwen3-32B <d-cite key="yang2025qwen3"></d-cite>, a dense text-only transformer; Llama 3.1-70B <d-cite key="grattafiori2024llama3"></d-cite> with FP8 quantization, which fits on a single GH200 due to its reduced parameter size; and Mixtral-8x22B <d-cite key="jiang2024mixtral"></d-cite>, a sparse mixture-of-experts model with 141B total parameters. Qwen3-32B and Llama 3.1-70B fit on a single GH200, while Mixtral-8x22B requires tensor parallelism across all four GPUs.

For text workloads, we use the ShareGPT dataset <d-cite key="chiang2023vicuna"></d-cite>, which provides a representative distribution of real conversational prompt and completion lengths. For our end-to-end benchmarks, we generate an open-loop workload by sending requests to the vLLM engine according to a Poisson arrival process.

**Metrics.** We report the following latency metrics for each request:

- **Time to First Token (TTFT):** Time from request arrival to generation of the first output token, reflecting responsiveness.
- **Time Between Tokens (TBT):** Average time between successive output tokens during decoding.
- **End-to-End Latency (E2E Latency):** Time from request arrival to completion.

## Memory System Performance

The defining feature of the GH200 is its high-bandwidth NVLink-C2C interconnect between CPU and GPU. To characterize real-world transfer performance, we conducted microbenchmarks measuring achievable bandwidth in both directions.

### Copy Performance

We implemented a PyTorch-based benchmark measuring transfer rates between CPU and GPU memory. Following best practices for maximizing DMA throughput, we used pinned (page-locked) CPU memory and asynchronous transfers via non-blocking copy operations. Bidirectional measurements used separate CUDA streams for each direction to enable concurrent transfers.

<div class="row mt-3 justify-content-center">
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/vllm-characterization/transfer_bandwidth.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="caption">Figure 3: GH200 transfer bandwidth as a function of transfer size.</div>

**Directional asymmetry.** As shown in the figure above, transfer performance exhibits significant asymmetry: CPU to GPU transfers achieve over 85% of peak theoretical bandwidth, whereas GPU to CPU transfers reach only approximately 35%. This disparity arises from the cache coherency protocol: transfers into CPU memory require additional coherency traffic to maintain consistency with the CPU cache hierarchy, overhead that is absent in the reverse direction. Similar behavior has been reported in prior work <d-cite key="fusco2024understanding"></d-cite>.

**Size-dependent efficiency.** Peak bandwidth is only attained for sufficiently large transfers: CPU to GPU requires tensors exceeding 128MB, while GPU to CPU saturates at 8MB. This difference reflects substantial fixed overhead that must be amortized, and suggests that small transfers will underutilize the interconnect.

**Bidirectional contention.** When transferring simultaneously in both directions, average per-direction bandwidth drops approximately 10% compared to the unidirectional GPU to CPU case, indicating contention for shared interconnect resources.

These findings suggest that applications should favor large, unidirectional transfers where possible to maximize effective bandwidth utilization. As we show in [Part 3](/notes/2026/llm-serving-dynamic-offloading/), our system is designed to exploit each of these characteristics: weight offloading leverages the higher CPU to GPU bandwidth, KV cache transfers are batched into large contiguous blocks to exceed saturation thresholds, and transfer scheduling avoids bidirectional contention wherever possible.

### Managed vs. Pinned Memory

Related work <d-cite key="fusco2024understanding"></d-cite> demonstrated that transfers from HBM to managed memory allocated on the CPU achieve nearly twice the bandwidth of transfers from HBM to pinned CPU memory. While we were able to reproduce this result, we also observed significant performance degradation when performing repeated consecutive transfers. We hypothesize that this degradation stems from page table thrashing, potentially caused by the managed memory runtime's page migration and coherency tracking mechanisms. This result is notable given that pinned memory is conventionally recommended for maximizing transfer bandwidth, suggesting that the GH200's coherency architecture may favor managed memory allocations in certain access patterns.

This page migration overhead has also been identified as a critical bottleneck for LLM inference on the GH200 <d-cite key="yu2026superinfer"></d-cite>.

## Execution Optimizations

vLLM provides several configurable optimizations that significantly affect serving performance. We evaluate CUDA graphs, prefix caching, and KV cache offloading to quantify their impact and identify limitations.

### CUDA Graph Performance

Kernel launch overhead is particularly significant on the GH200, where the ARM-based Grace CPU exhibits lower single-threaded performance compared to x86 CPUs in traditional loosely-coupled systems <d-cite key="vellaisamy2025characterizing"></d-cite>. This makes CUDA graph capture especially important for minimizing CPU-side bottlenecks during decode iterations.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/vllm-characterization/cuda_graphs_impact.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="caption">Figure 4: Impact of CUDA graphs on median and P99 TBT for Qwen3-32B on ShareGPT across request rates.</div>

Enabling CUDA graphs reduces median decode time per token by over 40% at low request rates. While the absolute gap persists at higher load, the relative improvement narrows as GPU compute becomes the dominant bottleneck. Notably, piecewise CUDA graphs alone account for the majority of the observed benefit, with full graph capture providing only a marginal additional improvement. This is practical, as piecewise graphs offer greater flexibility in handling varying batch sizes without requiring capture of every possible shape.

### Offloading Overhead

vLLM supports CPU offloading, enabling the use of CPU memory to store KV cache data and model parameters. Due to the overhead associated with transferring data between CPU and GPU, offloading is typically employed as a last resort for models or KV cache that exceed GPU memory capacity.

Notably, vLLM's offloading implementations differ significantly between KV cache and parameter offloading. For KV cache offloading, vLLM employs asynchronous transfers with optimized memory layouts — reorganizing KV data into large, contiguous blocks to maximize DMA throughput <d-cite key="ozeri2026kv"></d-cite>. This creates a secondary caching tier in CPU DRAM from which previously computed KV data can be loaded back to the GPU faster than recomputation. However, this primarily benefits prefix-heavy workloads where requests share common prefixes. Crucially, KV cache offloading does not free GPU memory for additional concurrent requests, as active requests still require their KV data to reside on the GPU during decoding.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/vllm-characterization/offloading_overhead.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="caption">Figure 5: Impact of parameter and KV cache offloading on median E2E Latency, TTFT, TBT for Qwen3-32B on ShareGPT at 18.0 target arrival rate.</div>

Parameter offloading relies on UVA, which allows the GPU to access CPU memory directly without explicit copies. While this simplifies implementation, data is fetched on-demand at the point of access rather than prefetched asynchronously. The freed GPU memory can then be used for KV cache, allowing for more requests to run concurrently. Each layer's parameters are loaded only when needed, stalling execution while the transfer completes. This synchronous access pattern prevents the data movement overlap that the GH200's NVLink-C2C interconnect could otherwise support.

The figure above confirms that neither strategy improves end-to-end latency under high load. KV offloading tracks the baseline closely, with only marginal wait time reductions from preemption recovery. UVA offloading frees enough GPU memory to eliminate queuing beyond approximately 23GB but the steadily increasing decode time per token from synchronous parameter fetching more than offsets this gain.

### Prefix Caching Effectiveness

Agentic workloads, system prompts, and document understanding tasks often share common prefixes across requests. Prefix caching exploits this redundancy in two ways: it avoids redundant prefill computation, and it reduces memory usage by sharing KV cache blocks between requests.

<div class="row mt-3 justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/vllm-characterization/prefix_caching_cdf.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="caption">Figure 6: Prefill time CDF with and without prefix caching.</div>

To evaluate this, we construct a synthetic document QA workload where many requests share long common prefixes (≈ 4k tokens). The figure above shows the cumulative distribution function (CDF) of prefill times: with caching enabled, over 50% of requests achieve near-zero prefill latency due to cache hits, while cache misses show similar latency to the uncached baseline.

For deployments where the prefix cache exceeds GPU memory capacity, KV cache offloading to CPU DRAM could extend this benefit by providing a larger secondary cache tier, though at the cost of higher reload latency compared to GPU-resident cache hits.

## Multimodal Inference

Multimodal serving introduces additional caching layers for image preprocessing and encoding, avoiding redundant computation when the same image appears across multiple requests. While benchmarking multimodal inference on the GH200, we identified a significant performance bottleneck: cache lookup required decoding the input image (e.g., JPEG to raw pixels) before computing its hash. For large 4K images, this decoding step alone can exceed 100ms, adding substantial latency even for cached images.

<div class="row mt-3 justify-content-center">
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/vllm-characterization/image_hashing.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="caption">Figure 7: Comparison of image hashing strategies across image resolutions.</div>

We addressed this by hashing the raw image bytes directly, bypassing the decoding step for cache hits. This reduces TTFT significantly for repeated images. Our fix was [merged into vLLM](https://github.com/vllm-project/vllm/pull/29621) <d-cite key="vllm_hash_fix"></d-cite>, which bypasses image decoding for cache lookups and additionally reduces hash computation time by operating on the smaller compressed byte representation rather than decoded pixel data.

## Implications for System Design

Our characterization reveals several key findings that motivate the design choices presented in [Part 3](/notes/2026/llm-serving-dynamic-offloading/).

The GH200's NVLink-C2C interconnect provides substantial CPU to GPU bandwidth but vLLM's current parameter offloading via UVA fails to exploit this. UVA fetches parameters synchronously at the point of access, preventing any overlap between data movement and computation. No prefetching is performed, and the interconnect's full bandwidth is only realized for transfers exceeding 128MB, far larger than typical per-layer parameter accesses.

CUDA graphs are critical for decode performance on the GH200, reducing per-token latency by over 40%. Any offloading mechanism must therefore be compatible with CUDA graph capture to preserve this benefit rather than forcing fallback to eager execution.

Finally, while parameter offloading frees GPU memory for additional KV cache capacity, our results show that the decode penalty grows steadily with the amount offloaded, dominating any throughput gains from larger batch sizes. A static offloading configuration is therefore fundamentally limited: at low load, unnecessary offloading degrades decode performance without providing useful KV cache capacity, while at high load, insufficient offloading leaves throughput on the table. The optimal operating point depends on workload intensity, which varies at runtime.

These observations motivate the following design requirements:

**Asymmetry-aware prefetching.** Store offloaded parameters in contiguous CPU tensors and prefetch them asynchronously to the GPU, exploiting the favorable CPU to GPU bandwidth and enabling overlap with computation.

**CUDA graph compatibility.** Integrate parameter offloading with CUDA graph capture to maintain low kernel launch overhead during decode.

**Dynamic adaptation.** Adjust the offloading configuration at runtime based on KV cache pressure and iteration time, offloading parameters only when additional KV cache capacity is needed and sufficient compute exists to hide the data movement latency.

In [Part 3](/notes/2026/llm-serving-dynamic-offloading/), we present a system that addresses all three requirements.