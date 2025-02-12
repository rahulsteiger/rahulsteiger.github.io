---
layout: distill
title: Running DeepSeek R1 locally
description: This blog post presents the steps required to run inference for DeepSeek R1 using llama.cpp on a single HPC node equipped with 4 A100 GPUs and 1 TB of memory.

tags: deepseek llama.cpp llms
giscus_comments: false
date: 2025-02-10
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
  - name: llama.cpp setup
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Downloading the model from Hugging Face
  - name: Expected Payoff
  - name: Running the model 

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

Running these LLMs requires significant computational resources. Fortunately, I have access to HPC hardware thanks to my previous participation in Team RACKlette during my bachelor's. The node I am using for this experiment has 2 `AMD EPYC 7773X 64-Core` CPUs, 4 `NVIDIA A100 80GB` GPUs, and `1 TB` of memory. This is the only reason I am able to play around with LLMs. 

## llama.cpp setup

The main goal of llama.cpp is to enable LLM inference with minimal setup and state-of-the-art performance on a wide range of hardware. More information can be found [here](https://github.com/ggerganov/llama.cpp).

Clone the github repository:
```bash
git clone git@github.com:ggerganov/llama.cpp.git && cd llama.cpp
```

Build llama.cpp:
```bash
cmake -B build -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined .
cmake --build build --config Release -j
```

**Note:** Make sure that the cuda compiler is available (you can check it with `nvcc --version`). More information on how to install CUDA can be found [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).

## Downloading the model from Hugging Face
 
As an initial step, we will be running a model with 1.58 quantization since it has the lowest hardware requirements. 

Install the Hugging Face CLI:
```bash
pip install -U "huggingface_hub[cli,hf_transfer]"
```

Download the GGUF model files from the unsloth/DeepSeek-R1-GGUF repository:
```bash
export MODEL="unsloth/DeepSeek-R1-GGUF"

export FILES=(
  "DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf"
  "DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00002-of-00003.gguf"
  "DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00003-of-00003.gguf"
)

export LOCAL_DIR="$HOME/models"

HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
  --repo-type "model" \
  --local-dir "$LOCAL_DIR" \
  $MODEL "${FILES[@]}"
```

## Running the model 

In the same terminal session as before, we can go to the go to the directory that contains the llama-server binary:
```bash
cd build/bin
```

We can then start the inference server with the following command:

```bash
./llama-server \
  --port 8192 \
  --model "$LOCAL_DIR/${FILES[0]}" \
  --n-gpu-layers 256 \
  --tensor-split 24,25,25,25 \
  --split-mode row \
  --flash-attn \
  --ctx-size 16384
```

Parameter explanation:
- `--model "$LOCAL_DIR/${FILES[0]}"`: Path to the first GGUF file.
- `--tensor-split 24,25,25,25`: Fraction of the model that is split across the GPUs (keep a slightly lower fraction for GPU 0 for metadata).
- `--n-gpu-layers 256`: Number of layers that are offloaded to the GPU (VRAM) for acceleration.
- `--split-mode row`: The tensors are split row-wise across GPUs.
- `--flash-attn`: Use [FlashAttention](https://github.com/Dao-AILab/flash-attention).
- `--ctx-size 16384`: Number of tokens the model can process in a single context window.

**Note:** Further information can be found [here](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md).

We can test the model with the following command:

```bash
curl --request POST \
     --url http://localhost:8192/completion \
     --header "Content-Type: application/json" \
     --data '{"prompt": "Why is the answer 42?"}'
```

The model gave me the following answer:

```bash
(Number Theory) #2 - Mathematics Stack Exmost recent 30 from math.stackexchange.com2024Doubtful2024-07-17T14:19:43Zhttps://math.stackexchange.com/feeds/question/4921410https://creativecommons.org/licenses/by-sa/4.0/rdfhttps://math.stackexchange.com/q/49214100Why is the answer 42? (Number Theory) #2Misha Parishhttps://math.stackexchange.com/users/13142522024-05-23T16:32:05Z2024Doubtful2024-05Doubtfulhttps://math.stackexchange.com/q/49214100Why is the answer 42? (Number Theory) #2Misha Parishhttps://math.stackexchange.com/users/13142522024-05-23T16:32:05Z2024-05Doubtful<p>So, I'm a student, and I'm learning math from the <a href="https://math.stackexchange.com/questions/4921410/why-is-the-answer-42-number-theory-2">Ground Up</a> series. I have a question from <a href="https://math.stackexchange.com/questions/4921410/why-is-the-answer-42-number-theory-2">Volume 1</a>, and I need help. The problem is: "If a four-digit number is made by combining the numbers 1, 2, 3, and 4. What is the sum of all the possible four-digit numbers that are formed?" The answer is given as 42, but I can't figure out why. I need an explanation. Here are my thoughts:</p> <p>I think that since there are 4 unique digits, there are 4! = 24 possible permutations. Each digit (1, 2, 3, 4) will appear in each place (thousands, hundreds, tens, ones) 6 times. So, for each digit, it appears 6 times in each position. The total sum for each position would be 6*(1+2+3+4) = 6*10 = 60. Then, for the total sum, it's 60*1000 + 60*100 + 60*10 + 60*1 = 60*(1000+100+10+1) = 60*1111 = 66660. But the answer is supposed to be 42. Where did I go wrong?
```

So the answer is not particularly useful. 

## Using a larger model

We change `FILES` environment variable in [download](#downloading-the-model-from-hugging-face) section. 

```bash
export FILES=(
  "DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00001-of-00015.gguf"
  "DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00002-of-00015.gguf"
  "DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00003-of-00015.gguf"
  "DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00004-of-00015.gguf"
  "DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00005-of-00015.gguf"
  "DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00006-of-00015.gguf"
  "DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00007-of-00015.gguf"
  "DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00008-of-00015.gguf"
  "DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00009-of-00015.gguf"
  "DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00010-of-00015.gguf"
  "DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00011-of-00015.gguf"
  "DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00012-of-00015.gguf"
  "DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00013-of-00015.gguf"
  "DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00014-of-00015.gguf"
  "DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00015-of-00015.gguf"
)

```bash
./llama-server \
  --port 8192 \
  --model "$LOCAL_DIR/${FILES[0]}" \
  --n-gpu-layers 20 \
  --tensor-split 24,25,25,25 \
  --split-mode row \
  --flash-attn \
  --ctx-size 16384
```

Running the same prompt as before, we get roughly of 2.0 tokens per second. 

## Geht es besser?
At this point, our model does not fit into VRAM, meaning that we will have to use CPU for some part of the inference. Consequently, we are bound by the performance of the CPU. As I learnt in my first semester lectures, you should always ask yourself: geht es besser? 

One could potentially increase the CPU performance if we compiled llama.cpp differently. We will be using the [BLIS](https://github.com/flame/blis/tree/master) libary to see if we can accelerate linear algebra operations on the CPU. 

Create a folder `dependencies` in the home directory and go into it:

```bash
mkdir dependencies && cd dependencies
```

We then clone the BLIS repository and go into it:

```bash
git clone git@github.com:amd/blis.git && cd blis
```

We configure and build BLIS as follows:

```bash
./configure --prefix=. --enable-cblas -t pthreads zen3 && make -j && make install
```

We export the following variables and rebuild llama.cpp in a different folder with some additional flags:

```bash
cd ~/llama.cpp

export BLIS_PREFIX="$HOME/dependencies/blis"
export BLAS_LIBRARIES="$BLIS_PREFIX/lib/zen3/libblis-mt.so"
export BLAS_INCLUDE_DIRS="$BLIS_PREFIX/include/zen3"
export LD_LIBRARY_PATH="$BLIS_PREFIX/lib/zen3:$LD_LIBRARY_PATH"

cmake -B blis_build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=FLAME -DBLAS_LIBRARIES=$BLAS_LIBRARIES -DBLAS_INCLUDE_DIRS=$BLAS_INCLUDE_DIRS -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined .
cmake --build blis_build --config Release -j 
``` 

We can run llama.cpp as before with:

```bash
cd blis_build/bin

./llama-server \
  --port 8192 \
  --model "$LOCAL_DIR/${FILES[0]}" \
  --n-gpu-layers 20 \
  --tensor-split 24,25,25,25 \
  --split-mode row \
  --flash-attn \
  --ctx-size 16384
``` 

Running the same prompt as before, we get roughly 2.0 tokens per second as well. I did not do proper benchmarking, but from my estimate, there does not seem to be a significant difference compared to the version without BLIS. After reading more into this and scouring GitHub for answers, I concluded that if raw inference speed is what you care about, you should probably look into other frameworks such as [TensorRT](https://github.com/NVIDIA/TensorRT). However, I still find the minimal setup and decent performance of llama.cpp to be what makes it great for experimentation.

## Larger?

Thanks to memory paging, we can run models that do not fit into our available RAM and VRAM by swapping out memory pages to disk. So, let's try running DeepSeek-R1-BF16. The weights on Hugging Face are roughly 1.5 TB, which requires us to download 30 GGUF files. 

```bash
export FILES=(
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00001-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00002-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00003-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00004-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00005-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00006-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00007-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00008-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00009-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00010-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00011-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00012-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00013-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00014-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00015-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00016-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00017-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00018-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00019-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00020-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00021-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00022-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00023-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00024-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00025-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00026-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00027-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00028-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00029-of-00030.gguf"
  "DeepSeek-R1-BF16/DeepSeek-R1.BF16-00030-of-00030.gguf"
)
```

Once the download is complete, I will try to run it with (TODO): 

```bash
cd blis_build/bin

./llama-server \
  --port 8192 \
  --model "$LOCAL_DIR/${FILES[0]}" \
  --n-gpu-layers 10 \
  --tensor-split 24,25,25,25 \
  --split-mode row \
  --flash-attn \
  --ctx-size 16384
``` 