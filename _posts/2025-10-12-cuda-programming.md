---
layout: distill
title: CUDA Programming - Fundamentals
description: "The fundamentals of GPU programming, from first principles."

tags: CUDA GH200 C++
giscus_comments: false
date: 2025-09-16
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
  - name: The CUDA basics
  - name: CUDA API

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

These are my notes for the course CUDA Programming Course – High-Performance Computing with GPUs on [YouTube](https://www.youtube.com/watch?v=86FAWCzIe_4&t) Chapters 1-6. The authors made the course code available on [GitHub](https://github.com/Infatoshi/cuda-course). I will be working on the ALPS cluster with a GH200 System.

## Setup

The first four chapters are mostly an overview of GPU Programming, the Deep Learning ecosystem and a recap of C++.

I am running my experiments in a standard container from the NVIDIA registry, specifically I am using the following one: nvcr.io/nvidia/cuda:12.9.0-devel-ubuntu24.04. This means that I will work with CUDA 12.9.0 for this course. As a first step, the course recommends taking a look around the NVIDIA CUDA samples on [GitHub](https://github.com/NVIDIA/cuda-samples). One thing to keep in mind is that you select the correct release, i.e. CUDA Samples v12.9.

Looking at the utilities, I can use the sample in `Samples/1_Utilities/deviceQuery` to print the properties of the system. The system has 4 GH200 GPUs, however to keep things short, here is the overview of a single GH200 GPU.

```bash
Device 0: "NVIDIA GH200 120GB"
  CUDA Driver Version / Runtime Version          12.9 / 12.9
  CUDA Capability Major/Minor version number:    9.0
  Total amount of global memory:                 96768 MBytes (101468602368 bytes)
  (132) Multiprocessors, (128) CUDA Cores/MP:    16896 CUDA Cores
  GPU Max Clock rate:                            1980 MHz (1.98 GHz)
  Memory Clock rate:                             2619 Mhz
  Memory Bus Width:                              6144-bit
  L2 Cache Size:                                 62914560 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        233472 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 3 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Enabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   9 / 1 / 0
```

The GH200 system has a higher compute capability compared to the one used in the Tutorial. This means that the system I am working on has some additional features that are not covered in the tutorial. How to properly use those features is for me to figure out and will be the topic of a future note.

Additionally, I created a private fork of the course's repo following the instructions [here](https://gist.github.com/0xjac/85097472043b697ab57ba1b1c7530274).

## The CUDA basics

The fifth chapter covers CUDA fundamentals.

### Thread Hierarchy

**Thread:**

- Smallest unit of execution in CUDA
- Executes kernel code independently
- Has local memory (registers) private to the thread

**Block:**

- Group of threads that can cooperate and synchronize
- Threads share data via shared memory
- Can be 1D, 2D, or 3D
- Enables efficient memory access through coordination of threads

**Grid:**

- Complete set of threads launched for a kernel invocation
- Grid dimensions determine the number of blocks created
- Each block represents a subset of the problem that can be solved independently
- Can be 1D, 2D, or 3D

**CUDA Variables:**

- `gridDim`: number of blocks in the grid
- `blockIdx`: index of the current block
- `blockDim`: number of threads per block
- `threadIdx`: index of the current thread within its block

There are a `dim3` (3D), `dim2` (2D), int (1D) types that you can pass to the kernel launch configuration.

**Warps:**

- Each block is divided into warps
- A warp executes 32 threads in parallel (SIMT)
- All threads in a warp execute the same instruction simultaneously

Note: CUDA Compute Capability 9.0 introduced the notion of Clusters. In short, Clusters are a group of thread blocks that can communicate with each other. Since this course only considers GPUs with lower Compute Capabilities, this will be the topic of another future note.

### Indexing

This is a cool example and analogy that explains quite clearly how CUDA threads are organized.

```cpp
__global__ void whoami(void) {
    int block_id =
        blockIdx.x +    // apartment number on this floor (points across)
        blockIdx.y * gridDim.x +    // floor number in this building (rows high)
        blockIdx.z * gridDim.x * gridDim.y;   // building number in this city (panes deep)

    int block_offset =
        block_id * // times our apartment number
        blockDim.x * blockDim.y * blockDim.z; // total threads per block (people per apartment)

    int thread_offset =
        threadIdx.x +
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y;

    int id = block_offset + thread_offset; // global person id in the entire apartment complex

    printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
        id,
        blockIdx.x, blockIdx.y, blockIdx.z, block_id,
        threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
}
```

### Memory Hierarchy

**Global Memory**:

- Main GPU memory, accessible by all threads.
- **Largest** but **slowest**.
- For data shared across threads (e.g., arrays).

**Shared Memory**:

- Fast, small memory shared within a block.
- For frequent data exchange between threads in a block (e.g., temp variables).

**Registers**:

- Fastest, private to each thread.
- For local variables (e.g., loop counters).
- Limited, use carefully.

**Constant Memory**:

- Read-only, cached, accessible by all threads.
- For constants/configuration that don't change during execution.

**Local Memory**:

- Used when registers overflow.
- **Slow** (stored in global memory).
- Avoid if possible (e.g., large arrays).

### cudaMalloc & cudaMemcpy & cudaMallocHost

A good convention for data variable names is to start them with `h_` if they are on the Host (CPU) and start them with `d_` if they are on the Device (GPU). Allocating memory on the host and device can be done as follows:

```cpp
int size = 2048;
float *d_x;
cudaMalloc(&d_x, size * sizeof(float));
```

To transfer data between host and device, use `cudaMemcpy`:

```cpp
int size = 2048;
float *d_x, *h_x;
cudaMalloc(&d_x, size * sizeof(float));
h_x = (float*)malloc(size * sizeof(float));

cudaMemcpy(d_x, h_x, size * sizeof(float), cudaMemcpyHostToDevice);   // Host → Device
cudaMemcpy(h_x, d_x, size * sizeof(float), cudaMemcpyDeviceToHost);   // Device → Host
cudaMemcpy(h_x, h_x, size * sizeof(float), cudaMemcpyHostToHost);     // Host → Host, equivalent to memcpy
cudaMemcpy(d_x, d_x, size * sizeof(float), cudaMemcpyDeviceToDevice); // Device → Device
```

Pinned (page-locked) memory is host memory that the OS cannot move, allowing the GPU to access it directly and transfer data faster. Allocate pinned memory with:

```cpp
float* h_data;
cudaMallocHost((void**)&h_data, size);
```

### Launching a kernel:

One can launch a kernel with

```cpp
<<<gridDim, blockDim, Ns, S>>>my_kernel(my_param)
```

- `gridDim` defines the size of the block grid
- `blockDim` defines the size of each block
- `Ns` specifies the number of bytes in shared memory that are dynamically allocated per block (usually omitted)
- `S` specifies the associated CUDA Stream (see definition later)

### Thread Synchronization

By default, the CPU and GPU execute independently. To ensure the CPU waits for all GPU work to finish (e.g., after launching a kernel), call `cudaDeviceSynchronize()`. This blocks the CPU until all preceding GPU tasks are complete.

In many cases, this is overkill since you might want to do some more fine-grained synchronization. This can be done as follows:

- Synchronize all threads in a block: `__syncthreads()`.
- Synchronize all threads in a warp: `__syncwarps()`.

### Vector addition

Standard C:

```cpp
void vector_add_cpu(float *a, float *b, float *c, int n) {
  for (int i = 0; i < n; i++)
    c[i] = a[i] + b[i];
}
```

CUDA kernel:

```cpp
__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = a[i] + b[i];
}
```

Assuming `d_a`, `d_b`, and `d_c` are device arrays of length `N`, launch the kernel:

```cpp
int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
```

This splits the work into blocks, each handling up to `BLOCK_SIZE` elements.

Note: Use the simplest indexing for best performance. For example, 3D indexing:

```cpp
__global__ void vector_add_gpu_3d(float *a, float *b, float *c, int nx, int ny, int nz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (i < nx && j < ny && k < nz) {
    int idx = i + j * nx + k * nx * ny;
    if (idx < nx * ny * nz)
      c[idx] = a[idx] + b[idx];
  }
}
```

3D indexing adds overhead and reduces performance. For a vector with 1,000,000 elements:

```bash
Speedup (CPU vs GPU 1D): 365.5x
Speedup (CPU vs GPU 3D): 308.6x
```

That's over 15% slower just from using 3D indexing!

### Nsight systems & compute

NVIDIA provides powerful tools for profiling and analyzing CUDA applications. Use [Nsight Systems](https://developer.nvidia.com/nsight-systems) to visualize application performance and [Nsight Compute](https://developer.nvidia.com/nsight-compute) for detailed kernel analysis. For command-line profiling, try [NVIDIA nsys](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-launch). These tools help you identify bottlenecks and optimize your GPU code efficiently.

### Atomic operations

Atomic operations in CUDA ensure that updates to shared memory are performed safely, preventing race conditions when multiple threads modify the same variable. While they can reduce parallel efficiency, they are essential for correctness in concurrent code.

CUDA provides built-in atomic functions like `atomicAdd`, `atomicSub`, `atomicMin`, and `atomicMax` for integers and floats in global or shared memory.

To count the number of positive elements in an array, each thread checks one element and atomically increments a shared counter:

```cpp
__global__ void countPositiveAtomic(const int* arr, int* counter, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && arr[i] > 0) {
    atomicAdd(counter, 1);
  }
}
```

This guarantees that every increment to `counter` is correct, even with thousands of threads running in parallel.

### CUDA Streams

CUDA Streams allow you to run multiple sequences of operations on the GPU. Operations within a single stream are executed in order, but different streams can run concurrently. This enables overlapping of computation and memory transfers, improving GPU utilization and overall performance.

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Launch a kernel in stream1
vector_add_gpu<<<num_blocks, BLOCK_SIZE, 0, stream1>>>(d_a, d_b, d_c, N);

// Perform a memory copy in stream2
cudaMemcpyAsync(d_x, h_x, size * sizeof(float), cudaMemcpyHostToDevice, stream2);
```

CUDA Events are used to mark specific points in the execution timeline. They enable synchronization between streams, so you can ensure that an operation in one stream starts only after an event (such as completion of a previous operation) in another stream has occurred. The following example shows how to overlap computation and data loading:

```cpp
cudaEvent_t event;
cudaEventCreate(&event);

// Launch a kernel in stream1
vector_add_gpu<<<num_blocks, BLOCK_SIZE, 0, stream1>>>(d_a, d_b, d_c, N);

// Perform a memory copy in stream2
cudaMemcpyAsync(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice, stream2);

// Record event on stream2 after the copy
cudaEventRecord(event, stream2);

// Make stream1 wait for the event (i.e., wait for the copy to finish)
cudaStreamWaitEvent(stream1, event, 0);

// Launch a kernel with new data in stream1
vector_add_gpu<<<num_blocks, BLOCK_SIZE, 0, stream1>>>(d_x, d_c, d_c, N);
```

You can also use Events to measure the time it takes to execute a kernel:

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(args);
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
```

You can synchronize a specific stream with `cudaStreamSynchronize(stream1)`, which blocks the host until all operations in that stream are complete. To synchronize all streams and the entire device, use `cudaDeviceSynchronize()`.

### Callbacks

Callbacks let you create a pipeline where the completion of a GPU operation automatically triggers a CPU task. This enables efficient coordination between GPU and CPU workloads.

```cpp
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    printf("GPU operation completed\n");
    // Trigger next batch of work
}
```

## CUDA API

NVIDIA provides libraries with highly optimized CUDA kernels for GPU computing.

### cuBLAS

cuBLAS is NVIDIA's industry-standard library for BLAS and GEMM routines, highly optimized for GPU performance. It supports operation fusion for greater efficiency.

```cpp
__half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
// Performs half-precision matrix multiplication: d_C_h = alpha_h * d_A_h * d_B_h + beta_h * d_C_h
cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_h, d_B_h, N, d_A_h, K, &beta_h, d_C_h, N);
```

**Parameter breakdown:**

- `handle`: cuBLAS library context
- `CUBLAS_OP_N`: No transpose for input matrices
- `N, M, K`: Matrix dimensions (output: N x M, input: N x K and K x M)
- `&alpha_h`: Scalar multiplier for the product
- `d_B_h`: Device pointer to matrix B
- `N`: Leading dimension of matrix B
- `d_A_h`: Device pointer to matrix A
- `K`: Leading dimension of matrix A
- `&beta_h`: Scalar multiplier for matrix C
- `d_C_h`: Device pointer to output matrix C
- `N`: Leading dimension of matrix C

#### cuBLASLt

Extension for more flexible APIs, mainly for deep learning workloads. Optimized for Tensor Cores, allows for more fine-grained control. Generally, is faster than cuBLAS.

cuBLASLt provides a more flexible and high-performance matrix multiplication API, especially suited for deep learning and Tensor Core acceleration. It allows fine-grained control over operation parameters and memory layouts.

```cpp
cublasLtHandle_t handle;
cublasLtCreate(&handle);

cublasLtMatmulDesc_t matmulDesc_fp32;
cublasLtMatmulDescCreate(&matmulDesc_fp32, CUBLAS_COMPUTE_32F, CUDA_R_32F);

// Create matrix layouts for fp32 (float) data
cublasLtMatrixLayout_t matA_fp32, matB_fp32, matC_fp32;
cublasLtMatrixLayoutCreate(&matA_fp32, CUDA_R_32F, K, M, K); // A: M x K
cublasLtMatrixLayoutCreate(&matB_fp32, CUDA_R_32F, N, K, N); // B: K x N
cublasLtMatrixLayoutCreate(&matC_fp32, CUDA_R_32F, N, M, N); // C: M x N

cublasLtMatmul(handle, matmulDesc_fp32, &alpha_h, d_B_fp32, matB_fp32, d_A_fp32, matA_fp32, &beta_h, d_C_fp32, matC_fp32, d_C_fp32, matC_fp32, nullptr, nullptr, 0, 0);
```

This example demonstrates cuBLASLt performing matrix multiplication in single precision (fp32). The matrix layouts, descriptors, and device pointers all use float (fp32) types for consistency.

**Parameter breakdown:**

- `handle`: cuBLASLt library context
- `matmulDesc_fp32`: Matmul operation descriptor (algorithm, compute type, etc.)
- `&alpha_h`: Scalar multiplier for the product
- `d_B_fp32`: Device pointer to matrix B (float)
- `matB_fp32`: Layout descriptor for matrix B (fp32)
- `d_A_fp32`: Device pointer to matrix A (float)
- `matA_fp32`: Layout descriptor for matrix A (fp32)
- `&beta_h`: Scalar multiplier for matrix C
- `d_C_fp32`: Device pointer to output matrix C (float)
- `matC_fp32`: Layout descriptor for matrix C (fp32)
- `d_C_fp32`, `matC_fp32`: Workspace/output buffer and its layout (can be same as above)
- `nullptr, nullptr`: Optional algorithm and workspace pointers (not used here)
- `0, 0`: Size parameters for workspace (set to zero if not used)

#### cuBLASXt

Multi-GPU BLAS support. Thread-safe. Host+GPU solving is slower due to memory bottlenecks, but the GH200 chip's fast interconnect may help.

The following example demonstrates how to use cuBLASXt to perform single-precision matrix multiplication (SGEMM) across multiple GPUs. cuBLASXt allows you to select devices and operate directly on host memory, automatically managing data transfers and computation distribution.

```cpp
cublasXtHandle_t handle;
CHECK_CUBLAS(cublasXtCreate(&handle));

int devices[1] = {0};
cublasXtDeviceSelect(handle, 1, devices); // Select GPU 0 for computation

cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B_host, N, A_host, K, &beta, C_host_gpu, N);
```

**Parameter breakdown:**

- `handle`: cuBLASXt library context
- `CUBLAS_OP_N`: No transpose for input matrices
- `N, M, K`: Matrix dimensions (output: N x M, input: N x K and K x M)
- `&alpha`: Scalar multiplier for the product
- `B_host`: Pointer to matrix B in host memory
- `N`: Leading dimension of matrix B
- `A_host`: Pointer to matrix A in host memory
- `K`: Leading dimension of matrix A
- `&beta`: Scalar multiplier for matrix C
- `C_host_gpu`: Pointer to output matrix C (host memory, result may be computed on GPU)
- `N`: Leading dimension of matrix C

#### cuBLASDx

Device-side BLAS API (preview) for fusing operations inside CUDA kernels, reducing latency and improving performance.

#### CUTLASS

cuBLAS and variants run on the host, and cuBLASDx is not yet fully optimized or documented. Matrix multiplication is key for deep learning, but cuBLAS is limited in fusing custom operations. CUTLASS is a template library for flexible, efficient fusion of linear algebra operations, ideal for deep learning and custom GPU workloads.

### cuDNN

NVIDIA cuDNN is a GPU-accelerated library providing highly optimized implementations of core deep learning operations, including matrix multiplication (GEMM), pooling, softmax, activation functions, tensor transformations, batch normalization, and more.

The following example demonstrates applying a tanh activation function to a 4D tensor using cuDNN:

```cpp
cudnnHandle_t cudnn;
cudnnCreate(&cudnn);

// Define input tensor: batch_size x channels x height x width
cudnnTensorDescriptor_t input_descriptor;
cudnnCreateTensorDescriptor(&input_descriptor);
cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                           batch_size, channels, height, width);

// Configure tanh activation
cudnnActivationDescriptor_t activation_descriptor;
cudnnCreateActivationDescriptor(&activation_descriptor);
cudnnSetActivationDescriptor(activation_descriptor, CUDNN_ACTIVATION_TANH,
                             CUDNN_PROPAGATE_NAN, 0.0);

// Apply activation: output = alpha * tanh(input) + beta * output
cudnnActivationForward(cudnn, activation_descriptor, &alpha, input_descriptor, d_input,
                       &beta, input_descriptor, d_output_cudnn);
```

cuDNN also supports advanced fusion patterns, combining multiple operations for greater efficiency. Instead of individual API calls, you can express computation as a graph, which cuDNN optimizes for maximum performance on your hardware.
