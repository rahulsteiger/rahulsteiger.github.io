---
layout: distill
title: "The GPU Memory Wall in LLM Serving"
description: "Why GPU memory is the bottleneck, and what the GH200 changes.<br>Part 1 of 3 from my Master's thesis at ETH Zurich."

tags: llm-serving vllm gh200 gpu-memory kv-cache offloading
giscus_comments: false
date: 2026-04-11

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

bibliography: 2026-04-11-llm-serving-background.bib

toc:
  - name: Introduction
  - name: LLM Architecture
    subsections:
      - name: Decoder-Only Transformers
      - name: Key-Value Caching
  - name: LLM Serving
    subsections:
      - name: Continuous Batching and Mixed Prefill
      - name: Chunked Prefill
      - name: CUDA Graphs
      - name: Paged Attention
      - name: Prefix Caching
      - name: Quantization
  - name: The NVIDIA GH200 Grace Hopper Superchip
    subsections:
      - name: Superchip Architecture
      - name: Memory Architecture
      - name: Memory Model
  - name: The Offloading Landscape
    subsections:
      - name: Offloading to accelerate LLM Serving
      - name: Elastic Memory Management
      - name: Optimizing LLM Serving on Superchips
  - name: "What's Next"
---

## Introduction

The widespread adoption of large language models has placed increasing demands on GPU memory, which must hold both model parameters and the key-value (KV) cache used during autoregressive generation. Because the KV cache grows linearly with sequence length and batch size, it competes directly with model parameters for limited high-bandwidth memory (HBM), making GPU memory the primary bottleneck for serving throughput <d-cite key="gholami2024ai"></d-cite>. Under high load, insufficient KV cache capacity forces serving systems to queue or preempt requests, degrading latency and reducing the number of requests that can be served concurrently.

A natural strategy is to extend available memory by offloading model parameters to CPU DRAM. Prior work <d-cite key="ren2021zero"></d-cite><d-cite key="sheng2023flexgen"></d-cite> has demonstrated this approach for serving models that exceed GPU capacity, but on conventional PCIe systems the limited interconnect bandwidth (approximately 128 GB/s for PCIe Gen5) makes such transfers a severe bottleneck, restricting offloading to a capacity fallback rather than a throughput optimization.

The NVIDIA GH200 Grace Hopper Superchip fundamentally changes this tradeoff. With up to 900 GB/s bidirectional CPU–GPU bandwidth over NVLink-C2C — roughly 7× that of PCIe Gen5 — the GH200 transforms offloading from a last resort into a viable strategy for actively improving serving throughput. Recent systems <d-cite key="li2025oneiros"></d-cite><d-cite key="xu2024pie"></d-cite><d-cite key="yu2026superinfer"></d-cite> have demonstrated significant throughput gains by exploiting this interconnect to offload KV cache or model parameters on the GH200. However, these approaches sacrifice compatibility with critical production optimizations: they disable CUDA graphs, which reduce decode latency by over 40% on the GH200's ARM-based CPU, and do not support prefix caching, which eliminates redundant computation for workloads with shared prefixes.

This is the first post in a three-part series based on my Master's thesis at ETH Zurich. In this post, I provide the necessary background on LLM serving and the GH200 architecture, and survey the existing offloading landscape. In Part 2, I will characterize vLLM's behavior on the GH200 and identify the design requirements for an effective offloading system. In Part 3, I will present our dynamic parameter offloading system that achieves 10–22% throughput improvements over default vLLM while maintaining full compatibility with CUDA graphs and prefix caching.

## LLM Architecture

Modern large language models are built on the transformer architecture, originally introduced for machine translation <d-cite key="vaswani2017attention"></d-cite>. The architecture's key innovation is the self-attention mechanism which enables each token to attend to all other tokens in a sequence, capturing long-range dependencies more effectively than recurrent approaches. Unlike RNNs and LSTMs, which process sequences step-by-step, transformers process all positions in parallel during training, enabling more efficient utilization of modern GPU hardware. This parallelism, combined with straightforward scaling across multiple devices, has enabled training models with hundreds of billions of parameters.

### Decoder-Only Transformers

Decoder-only transformers <d-cite key="liu2018generating"></d-cite> simplify the original encoder-decoder design by removing the encoder and cross-attention components, retaining only the masked self-attention decoder stack. This architecture was shown to generalize well across tasks when pretrained on large text corpora and fine-tuned for specific applications <d-cite key="radford2018improving"></d-cite>.

<div class="row mt-3 justify-content-center">
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/llm-serving-background/decoder_only_architecture.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="caption">Figure 1: The GPT architecture. Figure from <d-cite key="radford2018improving"></d-cite>.</div>

The core operation in each transformer block is masked multi-head self-attention, which computes pairwise relationships between all input token embeddings, producing an $n \times n$ attention matrix for a sequence of $n$ tokens. A causal mask restricts each token to attend only to itself and preceding tokens, preventing the model from accessing future positions during generation. This masking enables autoregressive generation, where each new token is predicted based solely on preceding tokens. From a computational perspective, the quadratic complexity of attention in sequence length significantly increases both memory consumption and compute cost, particularly for long contexts.

### Key-Value Caching

Naive autoregressive generation would recompute attention over all preceding tokens at each decode step, leading to substantial redundant computation. Key-value (KV) caching eliminates this redundancy by storing intermediate attention states for each token. During decoding, only the new token computes attention against these cached states, avoiding recomputation between previous tokens and reducing each decode step from $O(n^2)$ to $O(n)$ <d-cite key="kwon2023efficient"></d-cite>.

With KV caching, text generation divides into two phases with distinct computational characteristics. During the _prefill_ phase, the model processes the entire input prompt in a single forward pass, computing attention across all input tokens and populating the KV cache. This phase is compute-bound, with large matrix multiplications that exhibit high arithmetic intensity. The _decode_ phase then generates output tokens one at a time, with each step attending to the cached KV states. Despite reduced computation per token, decode is typically memory-bound: each step must load model weights and the entire KV cache from memory while performing relatively little computation.

<div class="row mt-3 justify-content-center">
    <div class="col-sm-11 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/llm-serving-background/prefill_decode.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="caption">Figure 2: Prefill and Decode Phase with KV Caching. Figure from <d-cite key="merkel2025prefill"></d-cite>.</div>

However, KV cache memory grows linearly with both sequence length and batch size, often becoming the primary memory bottleneck in LLM serving. At long context lengths or large batch sizes, KV cache memory can exceed the memory required for model parameters. This tension between cache capacity and throughput motivates much of the memory management work discussed later in this post.

## LLM Serving

As LLM inference is projected to consume an increasing fraction of global datacenter capacity, significant research effort has focused on improving serving efficiency.

### Continuous Batching and Mixed Prefill

Static batching requires all requests in a batch to complete before new requests can be scheduled, leading to GPU underutilization when request lengths vary. Orca <d-cite key="orca2022"></d-cite> addresses this by making scheduling decisions at each autoregressive generation step. This allows completed requests to exit early and new requests to continuously enter the batch, significantly improving throughput. This approach is known as iteration-level scheduling or continuous batching.

<div class="row mt-3 justify-content-center">
    <div class="col-sm-11 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/llm-serving-background/continuous_batching.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="caption">Figure 3: Baseline vs Continuous batching. Figure from <d-cite key="yun2024duplex"></d-cite>.</div>

Early continuous batching implementations processed either prefill or decode requests in each step, but not both simultaneously. Modern systems extend this by allowing prefill and decode requests to be combined within the same iteration.

### Chunked Prefill

Combining prefill and decode requests in the same batch introduces a scheduling challenge: prefill is compute-bound and processes many tokens in parallel, while decode is memory-bound and generates tokens sequentially. Without careful management, long prefills can stall ongoing decodes for several seconds, causing latency spikes. Sarathi <d-cite key="sarathi2023"></d-cite> addresses this with chunked prefills, which split prefill requests into smaller chunks that can be interleaved with decode iterations. Sarathi-Serve <d-cite key="sarathiserve2024"></d-cite> extends this approach with stall-free scheduling, enabling new requests to enter a batch without pausing ongoing decodes.

<div class="row mt-3 justify-content-center">
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/llm-serving-background/chunked_prefill.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="caption">Figure 4: Scheduling without vs with Chunked Prefill. Figure from <d-cite key="sarathiserve2024"></d-cite>.</div>

### CUDA Graphs

During the decode phase, each generation step processes only a single token per request. This results in very brief GPU computations. Consequently, the time taken by the CPU to launch these kernels often exceeds the actual execution time on the GPU. This phenomenon is known as kernel launch overhead. CUDA Graphs address this bottleneck by capturing a sequence of kernel operations into a directed acyclic graph. This allows the entire sequence to be launched with a single CPU operation, drastically reducing overhead and improving overall GPU utilization.

<div class="row mt-3 justify-content-center">
    <div class="col-sm-11 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/llm-serving-background/cuda_graphs.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="caption">Figure 5: Combining kernel launches with CUDA Graphs.</div>

However, CUDA Graphs impose strict execution constraints. They require static memory addresses and can only capture operations that run entirely on the GPU. If the forward pass contains unsupported operations, such as certain attention variants or CPU logic, modern serving engines employ piecewise CUDA Graphs. This technique splits the execution into capturable and uncapturable segments. This allows the system to benefit from reduced launch overhead where possible while maintaining correctness.

Support for CUDA graphs depends heavily on the underlying attention backend. Some backends support full graph capture for all computations, while others require piecewise graphs or eager execution for batches with varying sizes <d-cite key="vllm_cuda_graphs"></d-cite>. Most importantly for memory optimization, because graph capture fundamentally relies on static memory pointers, it often conflicts with dynamic memory allocation strategies. This tension is a central challenge in modern serving architecture.

### Paged Attention

Prior to dynamic memory management, LLM serving systems allocated contiguous memory for each request's KV cache based on the maximum supported sequence length. However, most requests generate far fewer tokens than this maximum, leading to significant GPU memory waste due to internal fragmentation — studies showed that effective memory utilization could be as low as 20% <d-cite key="kwon2023efficient"></d-cite>.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/llm-serving-background/paged_attention.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="caption">Figure 6: KV cache of two requests with paged attention. Figure from <d-cite key="kwon2023efficient"></d-cite>.</div>

PagedAttention addresses this by applying virtual memory concepts to KV cache management. Instead of allocating contiguous memory, it partitions the KV cache into fixed-size blocks and allocates them on demand. Analogous to how operating systems map virtual pages to physical frames, a block table maps logical blocks to physical blocks. This approach reduces memory waste to under 4% and enables flexible memory sharing across requests. For example, when multiple requests share a common prefix, their KV cache blocks can point to the same physical memory using reference counting and copy-on-write semantics. PagedAttention is the core innovation behind vLLM <d-cite key="kwon2023efficient"></d-cite>, which achieves 2–4× higher throughput compared to prior systems like Orca <d-cite key="orca2022"></d-cite>.

### Prefix Caching

Many real-world LLM workloads exhibit identical prefixes — multi-turn conversations reuse chat history, few-shot prompting repeats the same examples, and API calls often share system prompts. Without prefix caching, the KV cache for these shared prefixes must be recomputed for each request, wasting computation and increasing latency. SGLang <d-cite key="sglang"></d-cite> introduces RadixAttention, which stores KV cache entries in a radix tree data structure, enabling automatic and efficient prefix matching across requests. When a new request arrives, the system traverses the radix tree to find the longest matching prefix and reuses its cached KV tensors. An LRU eviction policy manages memory when the cache is full. vLLM implements a similar feature called Automatic Prefix Caching, which uses block-level hashing to identify shared prefixes. Both approaches significantly improve throughput and reduce time-to-first-token, particularly for workloads with high prefix overlap.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/llm-serving-background/prefix_caching.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="caption">Figure 7: Prefix Caching.</div>

### Quantization

LLM inference is typically performed in 16-bit floating point formats such as FP16 or BF16, where each parameter occupies 2 bytes. Quantization reduces parameter precision to fewer bits, decreasing both memory footprint and memory bandwidth requirements. For a model with $P$ parameters, moving from 16-bit to 8-bit representation halves the memory required from $2P$ to $P$ bytes.

Multiple quantization methods exist. Weight-only quantization methods <d-cite key="lin2023awq"></d-cite><d-cite key="frantar2023gptq"></d-cite> compress model parameters while keeping activations in higher precision. These methods reduce model size and memory bandwidth during decode but gain no additional computational speedup from low-precision tensor cores. Weight-and-activation quantization <d-cite key="xiao2023smoothquant"></d-cite> reduces both weights and activations to lower precision, enabling hardware-accelerated matrix multiplications on GPUs with dedicated low-precision compute units.

The H100 GPU features fourth-generation Tensor Cores that support FP8 arithmetic at twice the peak throughput of FP16 or BF16 <d-cite key="micikevicius2022fp8"></d-cite>. In practice, the end-to-end inference speedup from FP8 is less than 2x because only matrix multiplications execute on FP8 tensor cores, while operations such as softmax, layer normalization, and residual additions remain in higher precision. Nonetheless, FP8 quantization provides two complementary benefits for LLM serving: it reduces GPU memory consumption, allowing either larger batch sizes or models that would otherwise not fit in HBM, and it reduces the volume of data that must be transferred when model parameters are offloaded to CPU memory.

## The NVIDIA GH200 Grace Hopper Superchip

The growing demand for GPU memory by LLMs, combined with challenges in manufacturing large HBM capacities, has driven hardware vendors to explore tightly coupled heterogeneous architectures. NVIDIA's GH200 Grace Hopper Superchip integrates a 72-core Grace CPU with an H100 GPU via NVLink-C2C, a coherent interconnect delivering 900 GB/s bidirectional bandwidth — approximately 7× that of PCIe Gen5 <d-cite key="fusco2024understanding"></d-cite>.

### Superchip Architecture

The Grace CPU features 72 Arm Neoverse V2 cores with up to 480 GB of LPDDR5X memory providing approximately 500 GB/s bandwidth. The Hopper GPU provides 96 GB of HBM3 with over 4 TB/s bandwidth. NVLink-C2C enables cache-coherent access across both memory domains at 64-byte granularity, supporting direct loads, stores, and atomic operations between CPU and GPU memory without explicit copies. It provides 900 GB/s bidirectional bandwidth.

Multiple GH200 units can be connected via NVLink Switch, with up to 32 superchips forming a single cache-coherent system where all GPUs communicate at 900 GB/s bidirectional bandwidth.

<div class="row mt-3 justify-content-center">
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/llm-serving-background/superchip.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="caption">Figure 8: NVIDIA GH200 Grace Hopper architecture. Figure from <d-cite key="fusco2024understanding"></d-cite>.</div>

### Memory Architecture

The GH200 presents a hierarchical memory system with distinct bandwidth characteristics depending on the data path. The Grace CPU provides up to 480 GB of LPDDR5X memory with approximately 500 GB/s bandwidth, while the Hopper GPU provides 96 GB of HBM3 with over 4 TB/s bandwidth. This creates an asymmetric system: HBM offers roughly 8× higher bandwidth but significantly less capacity per dollar than CPU memory <d-cite key="fusco2024understanding"></d-cite>.

The theoretical bandwidth for data movement between memory domains depends on the transfer direction and initiating processor. GPU-initiated copies between CPU and GPU memory achieve up to 450 GB/s, bounded by the C2C interconnect.

### Memory Model

The GH200 exposes its memory as two NUMA nodes: LPDDR5X memory with affinity to Grace (NUMA node 0) and HBM3 memory with affinity to Hopper (NUMA node 1). Memory placement follows a first-touch policy for standard allocations, or can be explicitly controlled via `numactl` or `libnuma`.

CUDA provides several approaches for CPU-GPU data movement. Explicit copies via `cudaMemcpy` give the programmer full control over when data moves. Managed memory (`cudaMallocManaged`) provides a unified address space where the CUDA driver automatically migrates pages between CPU and GPU on demand — when a processor accesses a non-local page, a fault triggers migration. Pinned memory (`cudaMallocHost`) allows the GPU to directly access CPU memory without migration, though on PCIe systems this incurs high latency and limited bandwidth.

The GH200 changes the performance tradeoffs of these options. NVLink-C2C provides cache-coherent access with address translations handled by the Address Translation Service (ATS) at cache-line granularity (64 bytes). The table below summarizes the available memory types.

| Type    | API                 | Placement   | GPU Access |
| ------- | ------------------- | ----------- | ---------- |
| System  | `malloc`, `mmap`    | First touch | ATS        |
| Device  | `cudaMalloc`        | HBM         | Direct     |
| Managed | `cudaMallocManaged` | First touch | ATS/Direct |
| Pinned  | `cudaMallocHost`    | DDR         | DMA        |

<div class="caption">Table 1: Memory allocation types available on GH200 <d-cite key="fusco2024understanding"></d-cite>.</div>

## The Offloading Landscape

Historically, systems relying on standard PCIe buses treated offloading strictly as a capacity fallback mechanism to avoid out-of-memory errors. Due to severe bandwidth bottlenecks, developers avoided data transfers between the CPU and GPU whenever possible. The extreme bandwidth of tightly coupled Superchips fundamentally changes this paradigm. It transforms offloading from a mere fallback into an active strategy for maximizing throughput. Existing systems that leverage the NVLink-C2C connection often target the highly volatile KV cache, which creates significant bidirectional memory traffic. Additionally, systems that focus on weight offloading break compatibility with crucial serving optimizations like prefix caching and CUDA graphs.

The table below summarizes recent LLM memory optimization systems based on their offloading strategies, targeted assets, and the interconnect they were designed for and evaluated on.

| System                                                     | Strategy               | Asset                 | Interconnect   | Prefix Caching | CUDA Graphs |
| ---------------------------------------------------------- | ---------------------- | --------------------- | -------------- | -------------- | ----------- |
| ZeRO-Inference <d-cite key="deepspeed_inference"></d-cite> | Static Offloading      | Weights               | PCIe           | No             | No          |
| FlexGen <d-cite key="sheng2023flexgen"></d-cite>           | Static Offloading      | Weights, KV           | PCIe           | No             | No          |
| LMCache <d-cite key="cheng2025lmcache"></d-cite>           | Dynamic Offloading     | KV Cache              | PCIe / Network | Yes            | Yes         |
| kvcached <d-cite key="yu2025prism"></d-cite>               | Dynamic Allocation     | KV Cache              | None           | No             | Yes         |
| xLLM <d-cite key="liu2025xllm"></d-cite>                   | Dynamic Offloading     | KV Cache              | PCIe / SSD     | Yes            | Yes         |
| eLLM <d-cite key="xu2025ellm"></d-cite>                    | Dynamic Allocation     | KV Cache, Activations | None           | No             | Yes         |
| Pie <d-cite key="xu2024pie"></d-cite>                      | Dynamic Offloading     | KV Cache              | NVLink-C2C     | No             | No          |
| SuperInfer <d-cite key="yu2026superinfer"></d-cite>        | Dynamic Offloading     | KV Cache              | NVLink-C2C     | No             | Yes         |
| Oneiros <d-cite key="li2025oneiros"></d-cite>              | Dynamic Offloading     | Weights               | NVLink-C2C     | No             | No          |
| **Ours**                                                   | **Dynamic Offloading** | **Weights**           | **NVLink-C2C** | **Yes**        | **Yes**     |

<div class="caption">Table 2: Comparison of LLM serving memory optimization systems.</div>

### Offloading to accelerate LLM Serving

As model sizes grow and GPU HBM remains expensive and capacity-limited, offloading model weights and KV cache to CPU memory or storage has emerged as a practical approach for both running large models on resource-constrained systems and improving serving efficiency on standard hardware.

**ZeRO-Inference** <d-cite key="deepspeed_inference"></d-cite> adapts the ZeRO Stage 3 memory optimization techniques from training to inference workloads. It offloads model weights to CPU memory or NVMe storage, streaming them back to GPU on demand. A key design choice is full offloading rather than partial offloading: keeping the entire model off-GPU enables larger batch sizes, which improves throughput for latency-insensitive applications. ZeRO-Inference uses prefetching to overlap weight transfers with computation and can parallelize layer fetching across multiple PCIe interconnects when using multiple GPUs.

**FlexGen** <d-cite key="sheng2023flexgen"></d-cite> targets throughput-oriented inference scenarios where latency is less critical, such as batch processing and benchmarking. It aggregates memory from GPU, CPU, and disk, and uses a linear programming optimizer to search for efficient tensor placement and access patterns across this memory hierarchy. FlexGen also compresses weights and KV cache to 4 bits with negligible accuracy loss. When running OPT-175B on a single 16GB GPU, FlexGen achieves up to 100× higher throughput compared to DeepSpeed ZeRO-Inference.

**LMCache** <d-cite key="cheng2025lmcache"></d-cite> addresses the complementary challenge of KV cache management. It extracts KV caches from modern inference engines (vLLM and SGLang) and stores them in a hierarchy of storage devices including CPU memory, local disk, remote disk, and Redis. It supports both cache offloading for prefix reuse across queries and prefill-decode disaggregation for cross-engine KV cache transfer. Evaluation shows up to 15× throughput improvement for workloads with high prefix overlap.

### Elastic Memory Management

Modern LLM serving systems manage KV caches through page table-based virtualization (e.g., PagedAttention), but the total memory reserved for KV cache is still statically allocated. Recent work addresses this limitation by employing CUDA virtual memory APIs to enable truly dynamic allocation.

**kvcached** <d-cite key="yu2025prism"></d-cite> extends virtual memory techniques to multi-model serving scenarios. It provides an OS-style virtual memory abstraction that allows serving engines to reserve contiguous virtual space but back only active portions with physical GPU pages on demand. When requests complete or models go idle, pages are unmapped and returned to a shared pool for immediate reuse by other models.

**xLLM** <d-cite key="liu2025xllm"></d-cite> implements a global multi-level KV cache management across HBM, DRAM, and SSD. Beyond text-only models, xLLM extends this hierarchical caching to multimodal workloads. It reports 2.2× throughput improvement over vLLM in production deployments.

**eLLM** <d-cite key="xu2025ellm"></d-cite> introduces a unified elastic memory management framework inspired by OS memory ballooning. It virtualizes GPU tensors to enable dynamic memory inflation and deflation at runtime, allowing activation memory and KV cache to share a common pool rather than maintaining separate static reservations.

### Optimizing LLM Serving on Superchips

The emergence of tightly coupled heterogeneous GPU/CPU architectures — often referred to as _Superchips_ — such as the NVIDIA GH200, GB200, GB300, VR300, and AMD MI300A, presents new optimization opportunities for large-scale machine learning. These systems feature high-bandwidth CPU-GPU interconnects (e.g., 900 GB/s for NVLink-C2C on GH200) that fundamentally change the performance tradeoffs of offloading strategies. Existing solutions such as ZeRO-Inference were designed for slower interconnects (e.g., 64 GB/s for PCIe Gen4) and are therefore suboptimal on Superchips.

**Pie** <d-cite key="xu2024pie"></d-cite> introduces performance-transparent swapping to leverage CPU memory for KV cache expansion without impacting foreground computation. Unlike prior offloading systems that trigger data transfers on-demand and stall computation, Pie exploits the predictable layer-by-layer execution pattern of LLM inference to prefetch data before it is needed. On the GH200's high-bandwidth NVLink-C2C interconnect, this allows concurrent CPU-GPU data movement that is fully hidden behind GPU computation. Compared to vLLM and FlexGen, Pie achieves up to 1.9× higher throughput and 2× lower latency while expanding effective KV cache capacity by up to 4×. However, Pie offloads KV cache rather than model parameters, requiring bidirectional transfers every iteration, and does not support prefix caching. Its dynamic per-layer swap decisions are also incompatible with CUDA graphs.

**SuperInfer** <d-cite key="yu2026superinfer"></d-cite> employs two co-designed techniques for responsive, full-duplex offloading: RotaSched and DuplexKV. RotaSched is an OS-inspired rotary scheduler that actively rotates requests in and out of the running state by offloading their KV cache to DRAM based on their service level objective (SLO) progress. This SLO-aware serving system achieves up to 74.7% higher TTFT SLO attainment rate under high request rates while maintaining comparable throughput. Similar to Pie, SuperInfer does not support prefix caching and requires bidirectional transfers. Furthermore, the transfer budget must be manually configured, and its setting becomes critical under high load.

**Oneiros** <d-cite key="li2025oneiros"></d-cite> takes a different approach by observing that model parameters remain constant during inference while KV cache is dynamically updated. Rather than swapping KV cache between CPU and GPU, Oneiros remaps memory allocated to model parameters for KV cache use. When remapped parameters are needed for computation, they are fetched from CPU memory, with transfers overlapped with layer execution. Evaluated on GH200, Oneiros reduces tail time-between-tokens latency by 44.8–82.5% and increases throughput by 6.6–86.7% compared to vLLM. However, Oneiros lacks support for prefix caching, and inspection of its source code reveals that CUDA graphs are disabled by default to enable dynamic parameter remapping <d-cite key="oneiros_source"></d-cite>. During periods of low traffic, this design choice incurs significant performance overhead due to increased kernel launch latency.

## What's Next

This post covered the essential background for understanding LLM serving and the memory challenges that motivate offloading. In the [next post](/notes/2026/llm-serving-characterization/), I will present our characterization of vLLM on the GH200, including bandwidth microbenchmarks, the impact of CUDA graphs, and why existing offloading mechanisms fall short. In the [final post](/notes/2026/llm-serving-dynamic-offloading/), I will present our dynamic parameter offloading system and its evaluation.
