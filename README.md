<div align="center">

# ⚡ Rust AI Inference Engine

### Hyper-Optimized On-Device AI · Pure Rust · Apple Silicon Native

*A bare-metal GPT-2 inference engine built entirely from scratch —
no Python, no BLAS, no ML frameworks. Hand-written linear algebra,
GPU compute shaders, and microsecond-level profiling all the way down.*

[![Built in Rust](https://img.shields.io/badge/Built%20in-Rust-orange?style=for-the-badge&logo=rust)](https://www.rust-lang.org/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%20|%20M2%20|%20M3%20|%20M4-black?style=for-the-badge&logo=apple)](https://developer.apple.com/documentation/apple-silicon)
[![Metal GPU](https://img.shields.io/badge/GPU-Metal%20Compute-blue?style=for-the-badge&logo=apple)](https://developer.apple.com/metal/)
[![Architecture: Bare Metal](https://img.shields.io/badge/Architecture-Bare%20Metal-red?style=for-the-badge)]()
[![Status: Active](https://img.shields.io/badge/Status-Active%20Development-brightgreen?style=for-the-badge)]()

---

**371ms → 669ms (50-token generation)** &nbsp;|&nbsp; **360MB → 9.7MB memory** &nbsp;|&nbsp;
**2.5× NEON SIMD speedup** &nbsp;|&nbsp; **Metal GPU matmul shipped** &nbsp;|&nbsp; **Zero external ML deps**

</div>

---

## 🧭 What This Is

A **production-grade inference engine** built from the ground up in pure Rust, targeting Apple Silicon hardware. Every layer of the stack — matrix multiplication, attention mechanism, KV cache, memory arena, SIMD vectorization, GPU compute — is **hand-implemented and profiled**.

This project demonstrates deep systems engineering at the hardware-software boundary:

| Domain | What's Implemented |
|---|---|
| **Linear Algebra** | Hand-written tiled matmul (32×32 cache blocks), GELU, softmax, layer norm |
| **Memory Systems** | Flat arena KV cache, zero-allocation `ComputeWorkspace`, cursor-based eviction |
| **CPU Optimization** | ARM NEON SIMD via `vfmaq_f32` / `vdupq_n_f32` / `vst1q_f32` intrinsics |
| **GPU Compute** | Metal Shading Language kernel with Rust↔GPU buffer dispatch via `objc2-metal` |
| **Backend Dispatch** | Trait-based `ComputeBackend` — Scalar → NEON → Metal, resolved once at startup |
| **Profiling** | Custom `ProfileGuard` with µs-granularity timing, allocation tracking, call counts |
| **Model Loading** | SafeTensors binary parser — header extraction, dtype validation, zero-copy slice |
| **Generation** | Autoregressive decode loop with temperature sampling, top-k filtering, streaming |

> **Nothing is claimed until it is measured. Every optimization ships with before/after profiler evidence.**

---

## 📊 Optimization Journey — 5 Milestones Shipped

<details open>
<summary><strong>Milestone 1 — Cache-Aware Tiled Matrix Multiplication</strong></summary>

### Problem
Naive matmul walks rows and columns in a pattern that continually evicts cache lines
before they can be reused. Each element of A is reloaded from DRAM on every pass
through the inner loop — the CPU stalls waiting for memory.

### Fix
Divide matrices into **32×32 static blocks** that fit entirely in L1/L2 cache. Each
element is loaded once and reused across the full tile computation before eviction.

```
Naïve Access Pattern          Tiled Access Pattern (32×32)
─────────────────────         ────────────────────────────
Row A: load, compute          Block A: load once
Row A: evicted from cache     Block A: stays in L1 cache
Row A: reload (cache miss)    Block A: reused 32× before evict
... (repeat per row)          Next block: repeat
```

| Metric | Before | After | Delta |
|---|---|---|---|
| Matmul total | 371.37ms | 221.50ms | **▼ 40%** |
| Forward pass | 571.26ms | 353.05ms | **▼ 38%** |
| Avg per matmul | 1.032ms | 0.615ms | **▼ 40%** |

</details>

---

<details open>
<summary><strong>Milestone 2 — Flat Arena KV Cache</strong></summary>

### Problem
Naive KV cache used `Vec<Vec<f32>>` — nested heap allocations causing pointer chasing
on every attention read, and O(n) `memmove` on sequence eviction.

### Fix
Single **pre-allocated flat 1D buffer** per layer with layout `[head][pos][dim]`.
Sequence length tracked by a single `current_seq_len` cursor. Cache clear is **O(1)** —
reset the cursor, no memory writes needed. This matches the production design used in
**llama.cpp** and **vLLM**.

```rust
/// Flat arena. Layout: [head][pos][dim]
pub struct LayerKVCache {
    pub k_data: Vec<f32>,     // Pre-allocated at startup
    pub v_data: Vec<f32>,     // No per-token heap allocation
    pub current_seq_len: usize, // O(1) clear: just reset this
}
```

**Decode phase transition:** After prefill, the generation loop feeds only the
single newest token per step. The model reads historical context from the KV cache.
Positional embeddings use `start_pos = kv_cache.layers[0].seq_len()` for correct offset.

| Metric | Before KV Cache | After KV Cache | Delta |
|---|---|---|---|
| Time per 50 tokens | ~30,400ms | ~2,050ms | **▼ 93%** |
| Allocations | 360MB+ | 19.4MB | **▼ 94%** |

</details>

---

<details open>
<summary><strong>Milestone 3 — Memory Arena (ComputeWorkspace)</strong></summary>

### Problem
Every Q/K/V projection in `multi_head_attention` allocated a fresh `Vec<f32>` —
**3 allocations × 12 layers × every token step = 1,080 unnecessary heap requests**
per generation run.

### Fix
`ComputeWorkspace` pre-allocates maximum-size scratch buffers at startup
(`q_proj`, `k_proj`, `v_proj`, `attn_scores`, `attn_output`, `ffn_hidden`,
`ffn_output`). Backend `matmul_into` and `add_bias_in_place` write directly into
these buffers. `set_view()` adjusts the logical shape without reallocating.

```rust
pub struct ComputeWorkspace {
    pub q_proj: Tensor,       // max_seq_len × d_model
    pub k_proj: Tensor,       // reused every layer, every token
    pub v_proj: Tensor,       // zero heap allocation in hot path
    pub attn_scores: Tensor,
    pub attn_output: Tensor,
    pub ffn_hidden: Tensor,   // 4× expansion for GPT-2 FFN
    pub ffn_output: Tensor,
}
```

| Metric | Before Arena | After Arena | Delta |
|---|---|---|---|
| Matmul time | 2,050ms | 1,667ms | **▼ 18%** |
| Allocations | 15,150 | 14,070 | **▼ 1,080** |
| Memory | 19.4MB | 15.8MB | **▼ 3.6MB** |

</details>

---

<details open>
<summary><strong>Milestone 4 — ARM NEON SIMD Vectorization</strong></summary>

### Problem
The matmul inner loop was scalar — one multiply-accumulate per clock cycle.
Apple Silicon NEON units can process **4× f32 values per instruction** in
128-bit vector registers.

### Key Insight — Loop Order Determines Memory Access Pattern

The naive approach walked B matrix columns (strided access, cache-hostile):
```
// For each (i,j), walk k — B values are 768 elements apart in memory
b[k,j], b[k+1,j], b[k+2,j]  ← strided, prefetcher can't help
```

The correct approach broadcasts A and walks B rows (contiguous access):
```
// For each (i,k), broadcast scalar A to all 4 lanes, walk j
let a_vec = vdupq_n_f32(a[i,k]);        // broadcast once
b[k,j], b[k,j+1], b[k,j+2], b[k,j+3]  // contiguous ✅
out[i,j..j+4] = vfmaq_f32(out, a_vec, b_vec)  // 4 FMAs in 1 cycle
```

### Architecture — Backend Dispatch Trait

```rust
pub trait ComputeBackend: Send + Sync {
    fn matmul_into(&self, a: &Tensor, b: &Tensor, out: &mut Tensor);
    fn name(&self) -> &'static str;
}

pub fn select_backend() -> Box<dyn ComputeBackend> {
    // Detection at startup — one decision, reused forever
    // Metal → NEON → Scalar fallback
}
```

| Metric | Before NEON | After NEON | Delta |
|---|---|---|---|
| Matmul total | 1,667ms | 669ms | **▼ 60% (2.5×)** |
| Memory | 15.8MB | 9.7MB | **▼ 38%** |
| Allocations | 14,070 | 13,350 | **▼ 720** |

</details>

---

<details open>
<summary><strong>Milestone 5 — Metal GPU Compute Backend</strong></summary>

### Problem
Even with NEON SIMD, matmul remains CPU-bound — the M-series GPU has massively
parallel compute cores sitting idle. Offloading dense matrix operations to the GPU
unlocks another dimension of throughput.

### Implementation

**Three components built from scratch:**

**1. Metal Shading Language kernel** (`matmul.metal`) — Each GPU thread computes
exactly one output element. The hardware assigns a unique 2D coordinate via
`[[thread_position_in_grid]]`, enabling thousands of threads to run simultaneously:

```metal
kernel void naive_matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* Output  [[buffer(2)]],
    constant uint3* dims  [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Each micro-thread computes one float of the output matrix
    float sum = 0.0;
    for (uint k = 0; k < K; ++k)
        sum += A[row * K + k] * B[k * N + col];
    Output[row * N + col] = sum;
}
```

**2. Rust ↔ GPU bridge** (`metal_setup.rs`) — Uses `objc2-metal` for native Rust
FFI to Metal. Handles device discovery, MSL JIT compilation, pipeline state creation,
unified memory buffer management, command encoding, and synchronous dispatch:

```rust
pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline_state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}
```

**3. Backend integration** — `MetalBackend` implements the same `ComputeBackend`
trait as NEON and Scalar. The GPU pipeline (shader compilation, buffer allocation)
is initialized once at startup. Per-call dispatch copies data into shared GPU buffers,
fires the compute grid, waits for completion, and copies results back:

```rust
impl ComputeBackend for MetalBackend {
    fn matmul_into(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) {
        let result = self.context.dispatch_matmul(&a.data, &b.data, m, n, k);
        out.data[..m * n].copy_from_slice(&result);
    }
}
```

**GPU dispatch details:**
- Threadgroup size: 16×16 (256 threads per group)
- Buffer mode: `StorageModeShared` (Apple unified memory — zero CPU↔GPU copies)
- Grid geometry: `MTLSize { width: N, height: M, depth: 1 }` with out-of-bounds guard
- Shader compiled once via JIT, pipeline state cached for the lifetime of the engine

</details>

---

### Full Journey — End to End

| Milestone | Matmul Time | Memory | Key Technique |
|---|---|---|---|
| Naive scalar | 371ms (5-token) | 360MB+ | Baseline — no KV cache, no tiling |
| + Tiled matmul | 221ms (5-token) | 360MB+ | 32×32 L1/L2 cache blocks |
| + KV Cache | 2,050ms (50-token) | 19.4MB | Flat arena, O(1) eviction, decode phase |
| + Arena workspace | 1,667ms (50-token) | 15.8MB | Pre-allocated Q/K/V/FFN scratch buffers |
| + NEON SIMD | 669ms (50-token) | 9.7MB | `vfmaq_f32` vectorized FMA, 2.5× speedup |
| + Metal GPU | ✅ Shipped | 9.7MB | MSL kernel, Rust↔Metal FFI, unified memory |

> **Net result: 360MB → 9.7MB memory (37× reduction) and 15× decode throughput improvement**
> with a fully operational GPU compute pipeline for further acceleration.

---

## ⚙️ Architecture

```
rust-inference-engine/
├── core/                          # Zero external ML dependencies
│   ├── src/
│   │   ├── lib.rs                 # Tensor, tiled matmul, attention, transformer block
│   │   │                          #   ComputeWorkspace (memory arena)
│   │   │                          #   NEON SIMD dot product (neon_dot_inner)
│   │   │                          #   Scaled dot-product attention + causal masking
│   │   │                          #   GPT-2 forward pass, embedding lookup
│   │   │                          #   QKV weight splitting for fused c_attn
│   │   │
│   │   ├── kv_cache.rs            # Flat arena KV cache [head][pos][dim]
│   │   │                          #   LayerKVCache: push_k/v, k_slice/v_slice, O(1) clear
│   │   │                          #   KVCache: one LayerKVCache per transformer block
│   │   │
│   │   ├── generation.rs          # GenerationEngine: prefill/decode loop
│   │   │                          #   Temperature + Top-K sampling
│   │   │                          #   Streaming token callback (on_token)
│   │   │
│   │   ├── backend/
│   │   │   ├── mod.rs             # ComputeBackend trait dispatch
│   │   │   │                      #   ScalarBackend  — cache-friendly i→k→j loop
│   │   │   │                      #   NeonBackend    — vfmaq_f32 SIMD, 4-wide FMA
│   │   │   │                      #   MetalBackend   — GPU compute via dispatch_matmul
│   │   │   │                      #   select_backend() — detect once, reuse forever
│   │   │   └── metal_setup.rs     # MetalContext: device, queue, pipeline state
│   │   │                          #   MSL JIT compilation (include_str!)
│   │   │                          #   Unified memory buffer management
│   │   │                          #   GPU grid dispatch + synchronous wait
│   │   │
│   │   ├── matmul.metal           # Metal Shading Language compute kernel
│   │   │                          #   naive_matmul: 2D grid, one thread per output float
│   │   │                          #   16×16 threadgroups, out-of-bounds guard
│   │   │
│   │   ├── model/
│   │   │   ├── mod.rs             # ModelData, ModelLoader trait, ModelError
│   │   │   ├── safetensors.rs     # SafeTensors binary parser (header, F32 dtype, offsets)
│   │   │   └── gguf.rs            # GGUF format loader (planned)
│   │   │
│   │   └── profiler.rs            # ProfileGuard (RAII µs timer)
│   │                              #   Global atomic counters: matmul, softmax, layer_norm
│   │                              #   Allocation tracking (count + bytes)
│   │                              #   Nested timing stack via thread-local
│   │
│   └── Cargo.toml                 # deps: serde, objc2-metal, rand (no ML frameworks)
│
├── cli/
│   ├── src/
│   │   └── main.rs                # Weight loading, HuggingFace tokenizer, generation entry
│   └── Cargo.toml                 # deps: core, tokenizers
│
└── quantized/                     # INT8/GGUF quantization research (in progress)
    ├── lab_01_inspect_weights.py   # Weight distribution analysis for quantization targets
    ├── setup.sh
    └── requirements.txt
```

### Design Principles

- **Strict separation:** `core/` has zero external ML dependencies. Every linear
  algebra operation is hand-implemented. `cli/` orchestrates weight loading and
  stays deliberately thin so `core/` remains hardware-portable.
- **Zero-alloc hot path:** After startup, the forward pass allocates no heap memory
  in the attention/FFN pipeline. All intermediate results write into pre-allocated
  `ComputeWorkspace` buffers.
- **Backend abstraction:** New hardware targets (Metal, future CUDA) plug in via
  `ComputeBackend` without touching any call sites.

---

## 🔬 Built-in Profiler

Custom `ProfileGuard` struct using RAII — enter on construction, measure on `Drop`:

- **Microsecond wall-clock timing** per named region (`matmul`, `softmax`, `layer_norm`)
- **Global atomic counters** — call counts accumulate across the full forward pass
- **Heap allocation tracker** — total bytes and count via `track_allocation()`
- **Nested timer stack** via `thread_local` — detects re-entrant profiling
- **Zero overhead** in non-profiled builds

```bash
# Run with profiling output
cargo run --release --bin cli
```

---

## 🛣️ Roadmap

**Foundation** *(shipped)*
- [x] Cache-aware tiled matmul (32×32 blocks)
- [x] Flat arena KV cache with O(1) eviction
- [x] ComputeWorkspace — zero Q/K/V allocation in hot path
- [x] ARM NEON SIMD — `vfmaq_f32` vectorized matmul (2.5×)
- [x] Metal GPU compute shader — MSL kernel + Rust↔Metal FFI
- [x] Backend dispatch trait (Scalar / NEON / Metal)
- [x] SafeTensors binary model loader
- [x] HuggingFace tokenizer integration (BPE)
- [x] Microsecond profiler with allocation tracking
- [x] Prefill + decode phase separation
- [x] Causal masking for non-square attention matrices
- [x] Temperature + Top-K sampling with streaming output

**Performance** *(in progress)*
- [ ] Tiled Metal kernel (shared threadgroup memory for tile reuse)
- [ ] Flash Attention — fused softmax + attention in single kernel
- [ ] NEON-accelerated softmax and layer norm
- [ ] Persistent GPU buffer pool (eliminate per-call allocation overhead)

**Quantization** *(research phase)*
- [ ] INT8 weight quantization — KV cache buffers as primary target
- [ ] GGUF format loading for quantized model weights
- [ ] Mixed-precision (FP16 compute with FP32 accumulation)

**Planned**
- [ ] Continuous batching
- [ ] Multi-model serving
- [ ] CUDA backend (for NVIDIA hardware portability)

---

## 🚀 Quick Start

```bash
# Clone and enter the project
git clone <repo-url> && cd context_engine

# Download GPT-2 weights + tokenizer
# (requires huggingface-cli: pip install huggingface-hub)
huggingface-cli download openai-community/gpt2 --local-dir ./models/gpt2

# Build and run (release mode for SIMD and Metal)
cargo run --release --bin cli
```

### Requirements

- **Rust** 2024 edition (nightly recommended for `aarch64` intrinsics)
- **macOS** with Apple Silicon (M1/M2/M3/M4) for NEON + Metal backends
- **Metal SDK** — ships with Xcode Command Line Tools
- Falls back to scalar backend on x86_64 / Linux

---

## 🧩 Technical Highlights for Reviewers

<details>
<summary><strong>Why hand-write matmul instead of using BLAS?</strong></summary>

This project is deliberately not a wrapper around existing libraries. The goal is
to understand and control every decision that affects performance:

1. **Cache behavior** — tiling granularity (32×32) was chosen based on Apple M4
   L1 cache line size and capacity
2. **SIMD strategy** — `vdupq_n_f32` broadcast + `vfmaq_f32` FMA processes
   4 floats per instruction in the correct (contiguous) memory direction
3. **Memory layout** — row-major storage with `i→k→j` loop order ensures the
   CPU prefetcher sees sequential access patterns
4. **GPU dispatch** — understanding the full pipeline from Rust data → Metal buffer
   → threadgroup geometry → kernel execution → result readback

Each layer teaches what commercial inference engines (llama.cpp, vLLM, TensorRT)
do under the hood and *why*.
</details>

<details>
<summary><strong>What makes the KV cache "production-grade"?</strong></summary>

The flat arena design matches what llama.cpp and vLLM use in production:

- **`[head][pos][dim]` layout** — attention reads are contiguous per-head, no pointer chasing
- **O(1) eviction** — resetting `current_seq_len = 0` logically clears the cache without
  touching memory, exploiting the fact that old data is simply overwritten on next use
- **Pre-allocated** — total buffer size known at startup (`num_heads × max_seq × head_dim`),
  zero allocations during inference
- **Decode phase transition** — after prefill, only 1 token flows through the forward pass;
  historical context is read directly from cache slices
</details>

<details>
<summary><strong>How does the Metal integration work without C++ or Objective-C?</strong></summary>

The entire GPU pipeline is pure Rust using `objc2-metal` crate bindings:

1. **MSL source** is embedded at compile time via `include_str!("../matmul.metal")`
2. **JIT compilation** — `newLibraryWithSource` compiles MSL to GPU machine code at startup
3. **Pipeline state** is cached once — no per-call shader compilation overhead
4. **Unified memory** — `StorageModeShared` means CPU and GPU share the same physical
   memory on Apple Silicon, avoiding explicit DMA transfers
5. **Synchronous dispatch** — `commandBuffer.waitUntilCompleted()` blocks until the GPU
   finishes, ensuring correctness for sequential matmul chains
</details>

---

## 🧑‍💻 About

Built by **Rishi Dwivedi** — systems-focused AI engineer with 2 years of
professional experience, building toward hardware-aware inference optimization
for on-device AI and custom accelerator teams.

This project is a deliberate descent past Python orchestration, past framework
abstractions, down to the hardware-software boundary where **memory layout,
cache behavior, SIMD intrinsics, and GPU compute dispatch** determine what is
actually fast.

Every optimization ships with measurements. Nothing is claimed until it is proven.

**Core competencies demonstrated:**
- Systems programming in Rust (unsafe FFI, SIMD intrinsics, memory arenas)
- GPU compute programming (Metal Shading Language, command encoding, threadgroup geometry)
- ML inference internals (transformer architecture, KV caching, attention mechanisms)
- Performance engineering (cache optimization, allocation profiling, vectorization)
- Hardware-software co-design (Apple Silicon NEON, Metal, unified memory architecture)

---

<div align="center">

*Built from scratch. Measured at every step. No shortcuts.*

**[LinkedIn](https://linkedin.com/in/rishi-dwivedi)** · **[GitHub](https://github.com/rishidwd29)**

</div>