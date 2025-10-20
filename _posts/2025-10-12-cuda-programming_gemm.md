---
layout: distill
title: CUDA Programming (Optimizing GEMM)
description: This blog contains the notes of me (attempting) to create a fast GEMM CUDA implementation.

tags: CUDA GH200 C++
giscus_comments: false
date: 2025-10-12
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

bibliography:

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Setup
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Baseline implementation
  - name: Global Memory Coalescing
  - name: Shared Memory Cache-Blocking
  - name: 1D Blocktiling

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
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

These are my notes for Chapter 7 of the CUDA Programming Course – High-Performance Computing with GPUs on [YouTube](https://www.youtube.com/watch?v=86FAWCzIe_4&t). For all experiments, I will be working on the ALPS cluster, which features a GH200 System.

This part of the course covers building a fast CUDA SGEMM from scratch. The approach is based on this [blog](https://siboehm.com/articles/22/CUDA-MMM) post, and the associated code can be found on [GitHub](https://github.com/siboehm/SGEMM_CUDA?tab=readme-ov-file). The original blog is very well written and features excellent figures, so I recommend checking it out before/ while reading my notes. 

Since I am working with the GH200, I can do some Hopper specific optimizations that are not mentioned in the course or original blog post. But that is an ambitious goal. 

This is still a work in progress.

## Setup

I will use the testing infrastructure from the original blog post’s [GitHub](https://github.com/siboehm/SGEMM_CUDA?tab=readme-ov-file) repository. The only change I need to make is setting the `cmake` `CUDA_COMPUTE_CAPABILITY` parameter to 90.

## Baseline implementation

The baseline kernel that we will be improving upon looks as follows:

```c++
__global__ void sgemm_baseline(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // if statement is necessary to make things work under tile quantization
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}
```

The kernel is launched with:

```c++
// create as many blocks as necessary to map all of C
int BLOCK_SIZE = 32
dim3 gridDim(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE), 1);
// 32 * 32 = 1024 thread per block
dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
// launch the asynchronous execution of the kernel on the device
// The function call returns immediately on the host
sgemm_baseline<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```

For $m=n=k=4096$, this kernel achieves 502.5 GFLOPs. In comparison, cuBLAS achieves 50058.0 GFLOPs for the same matrix sizes, which is nearly 100x faster than the baseline implementation. 

## Global Memory Coalescing

Looking at how we are launching the kernel and assuming $m=n=k=4096$, the warps will be executed with the following threads: 

```bash
Warp 0: (y=0, x=0..31)
Warp 1: (y=0, x=32..63)
...
```

Consequently, each thread in a warp accesses a different row of A, the same column of B, and a different entry of C (along the column dimension). GPUs support 128B memory operations on contiguous data. Such load instructions can be issued not only on data used by a single thread, but on the entire data used during warp execution. Assuming each thread in a warp loads a different 32-bit float and that data is stored in contiguous memory, four of them can be coalesced into a single transaction.

Since the matrices are stored in row-major order, we can take advantage of 128B loading for matrix A. However, since we access non-contiguous memory in matrices B and C, we cannot exploit this optimization. The solution is to redefine `x` and `y` to ensure that warps access data that can be coalesced. 

We can achieve this as follows:
```
const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
```

This ensures that each thread within a warp accesses a consecutive set of 32 columns of B and the same consecutive entries of C. Consequently, we reduce the number of load transactions each warp needs to perform while keeping the total number of warps unchanged. 

For $m=n=k=4096$, this kernel achieves 6352.2 GFLOPs, a nearly 12x improvement, but still 8x slower than the cuBlas implementation.


## Shared Memory Cache-Blocking

Each block is executed on a Streaming Multiprocessor (SM). Each SM has some (very fast) shared memory that can be accessed by all threads within the same block. Every block computes a $32 \times 32$ block of C. For this, each block uses 32 consecutive rows of A and 32 consecutive columns of B. However, these rows and columns are currently reloaded by every single warp. 

We will move a $32 \times 32$ chunk of A and B into shared memory and let each warp compute its part before continuing to the next chunks. 

```c++
template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  // the output block that we want to compute in this threadblock
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // advance pointers to the starting positions
  A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCKSIZE;                        // row=0, col=cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}
```

For $m=n=k=4096$, this kernel achieves 9174.1 GFLOPs, a 1.5x improvement over the previous version, but still more than 5x slower than the cuBLAS implementation.

## 1D Blocktiling

Instead of computing a single entry per thread, the next optimization computes multiple entries of C per thread. This has the advantage that our arithmetic intensity ratio (arithmetic operations vs memory operations) is higher, since the result of a single load can be used to compute multiple entries. The main change occurs in the computation loop: 

```c++
// allocate thread-local cache for results in registerfile
float threadResults[TM] = {0.0};

// outer loop over block tiles
for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
  // populate the SMEM caches
  As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
  Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
  __syncthreads();

  // advance blocktile
  A += BK;
  B += BK * N;

  // calculate per-thread results
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // we make the dot product loop the outer loop, which facilitates
    // reuse of the Bs entry, which we can cache in a temporary variable
    float tmpB = Bs[dotIdx * BN + threadCol];
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
      threadResults[resIdx] +=
          As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
    }
  }
  __syncthreads();
}

// write out the results
for (uint resIdx = 0; resIdx < TM; ++resIdx) {
  C[(threadRow * TM + resIdx) * N + threadCol] =
      alpha * threadResults[resIdx] +
      beta * C[(threadRow * TM + resIdx) * N + threadCol];
}
```

For $m=n=k=4096$ and `TM`$=8$, this kernel achieves 17040.9 GFLOPs, a 1.85x improvement over the previous version, but still nearly 3x slower than the cuBLAS implementation.

TODO. 