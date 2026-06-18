<div align="center">

# ⚡ Rust AI Inference Engine
### Hyper-Optimized On-Device AI for Apple Silicon

*A bare-metal GPT-2 inference engine built entirely from scratch in pure Rust —
no Python, no BLAS, no ML frameworks, no compromises.*

[![Built in Rust](https://img.shields.io/badge/Built%20in-Rust-orange?style=for-the-badge&logo=rust)](https://www.rust-lang.org/)
[![Target: Apple Silicon](https://img.shields.io/badge/Target-Apple%20Silicon-black?style=for-the-badge&logo=apple)](https://developer.apple.com/documentation/apple-silicon)
[![Architecture: Bare Metal](https://img.shields.io/badge/Architecture-Bare%20Metal-red?style=for-the-badge)]()
[![Status: Active](https://img.shields.io/badge/Status-Active%20Development-brightgreen?style=for-the-badge)]()

---

**2050ms → 669ms matmul** &nbsp;|&nbsp; **2.5x NEON SIMD speedup** &nbsp;|&nbsp;
**19.4MB → 9.7MB allocations** &nbsp;|&nbsp; **Zero external ML deps**

</div>

---

## 🧭 What This Is

Most inference runtimes are built *on top of* Python, BLAS, and layers of
abstraction that obscure what the hardware is actually doing. This project
goes the other direction — straight to the metal.

Built entirely in **pure Rust**, this engine gives strict, explicit control over:
- **Memory layout** and heap allocation in the hot path
- **CPU cache behavior** via tiled and SIMD-vectorized matrix operations
- **KV cache architecture** matching production inference engine design
- **Hardware dispatch** via a backend trait (Scalar → NEON → Metal roadmap)
- **Profiling at microsecond granularity** — no guessing, just numbers

This is not a wrapper. This is a ground-up construction of an AI inference system,
optimized at each layer with measured before/after evidence.

---

## 📊 Optimization Journey

Every optimization ships with measurements. Nothing is claimed until it is proven.

### Milestone 1 — Cache-Aware Tiled Matrix Multiplication

**Root cause:** Naive matmul iterates rows and columns in a pattern that
continually evicts cache lines before they can be reused. Each element of A
is reloaded from DRAM on every pass through the inner loop.

**Fix:** Divide matrices into 32×32 static blocks that fit entirely in L1/L2
cache. Each element is loaded once and reused across the full tile computation
before eviction.

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

---

### Milestone 2 — Flat Arena KV Cache

**Root cause:** Naive KV cache used `Vec<Vec<f32>>` — nested heap allocations
causing pointer chasing on every attention read, and O(n) memmove on sequence
eviction.

**Fix:** Single pre-allocated flat 1D buffer per layer with layout
`[head][pos][dim]`. Sequence length tracked by a single `current_seq_len`
cursor. Cache clear is O(1) — reset the cursor, no memory writes needed.
Matches the production design used in llama.cpp.

**Decode phase transition:** After prefill, the generation loop feeds only
the single newest token per step. The model reads historical context from the
KV cache. Positional embeddings use `start_pos = kv_cache.layers[0].seq_len()`
for correct offset.

| Metric | Before KV Cache | After KV Cache | Delta |
|---|---|---|---|
| Time per 50 tokens | ~30,400ms | ~2,050ms | **▼ 93%** |
| Allocations | 360MB+ | 19.4MB | **▼ 94%** |

---

### Milestone 3 — Memory Arena (ComputeWorkspace)

**Root cause:** Every Q/K/V projection in `multi_head_attention` allocated a
new `Vec<f32>` — 3 allocations × 12 layers × every token step = 1,080
unnecessary heap requests per generation run.

**Fix:** `ComputeWorkspace` struct pre-allocates maximum-size scratch buffers
at startup (`q_proj`, `k_proj`, `v_proj`, `attn_scores`, `attn_output`,
`ffn_hidden`, `ffn_output`). `matmul_into` and `add_bias_in_place` write
directly into these buffers. `set_view()` adjusts the logical shape without
reallocating.

| Metric | Before Arena | After Arena | Delta |
|---|---|---|---|
| Matmul time | 2050ms | 1667ms | **▼ 18%** |
| Allocations | 15,150 | 14,070 | **▼ 1,080** |
| Memory | 19.4MB | 15.8MB | **▼ 3.6MB** |

---

### Milestone 4 — ARM NEON SIMD Vectorization

**Root cause:** The matmul inner loop was scalar — one multiply-accumulate per
clock cycle. Apple Silicon M4 NEON units can process 4× f32 values per
instruction in 128-bit vector registers.

**Key insight — loop order determines memory access pattern:**

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
out[i,j..j+4] = vfmaq_f32(out, a_vec, b_vec)  // 4 FMAs in 1 instruction
```

**Architecture:** Backend dispatch trait (`ComputeBackend`) selected once at
startup. `select_backend()` detection order: Metal (planned) → NEON (aarch64)
→ Scalar (fallback). Call sites never change when a new backend is added.

```rust
// Detection at startup — one decision, reused forever
pub fn select_backend() -> Box<dyn ComputeBackend> {
    #[cfg(target_arch = "aarch64")]
    return Box::new(NeonBackend);  // vfmaq_f32 + vdupq_n_f32 + vst1q_f32
    Box::new(ScalarBackend)
}
```

| Metric | Before NEON | After NEON | Delta |
|---|---|---|---|
| Matmul total | 1667ms | 669ms | **▼ 60% (2.5x)** |
| Memory | 15.8MB | 9.7MB | **▼ 38%** |
| Allocations | 14,070 | 13,350 | **▼ 720** |

---

### Full Journey — End to End

| Milestone | Matmul Time | Memory | Notes |
|---|---|---|---|
| Naive scalar | 371ms (5-token) | 360MB+ | No KV cache |
| Tiled matmul | 221ms (5-token) | 360MB+ | 32×32 cache blocks |
| KV Cache | 2,050ms (50-token) | 19.4MB | 15x decode speedup |
| Arena workspace | 1,667ms (50-token) | 15.8MB | Zero Q/K/V allocs |
| NEON SIMD | 669ms (50-token) | 9.7MB | 2.5x vectorized FMA |

---

## ⚙️ Architecture

```
rust-inference-engine/
├── core/
│   ├── src/
│   │   ├── lib.rs           # Tensor, matmul, attention, transformer block
│   │   ├── kv_cache.rs      # Flat arena KV cache [head][pos][dim]
│   │   ├── generation.rs    # GenerationEngine, prefill/decode loop, sampler
│   │   ├── backend/
│   │   │   └── mod.rs       # ComputeBackend trait, NEON/Scalar/Metal stubs
│   │   └── profiler.rs      # ProfileGuard, allocation tracking
│   └── Cargo.toml           # Zero ML dependencies
│
└── cli/
    ├── src/
    │   └── main.rs          # Weight loading, tokenizer, generation entry
    └── Cargo.toml
```

**Strict separation:** `core/` has zero external ML dependencies. Every linear
algebra operation is hand-implemented. `cli/` orchestrates weight loading and
stays deliberately thin so `core/` remains hardware-portable.

---

## 🔬 Built-in Profiler

Custom `ProfileGuard` struct that:
- Tracks **wall-clock microseconds** per named region
- Accumulates **call counts** across the forward pass
- Reports **total heap allocation bytes** per run
- Operates at **zero overhead** in non-profiled builds

```bash
cargo run --release --bin cli
```

---

## 🛣️ Roadmap

**Foundation** *(shipped)*
- [x] Cache-aware tiled matmul (32×32 blocks)
- [x] Flat arena KV cache with O(1) eviction
- [x] ComputeWorkspace — zero Q/K/V allocation in hot path
- [x] ARM NEON SIMD — vfmaq_f32 vectorized matmul (2.5x)
- [x] Backend dispatch trait (Scalar / NEON / Metal roadmap)
- [x] Microsecond profiler with allocation tracking
- [x] Prefill + decode phase separation
- [x] Causal masking for non-square attention matrices

**Hardware** *(in progress)*
- [ ] Metal GPU compute shader for matmul kernel
- [ ] Flash Attention — tiled SRAM implementation on Metal
- [ ] INT8/FP16 quantization — KV cache buffers as primary target

**Generation** *(planned)*
- [ ] BPE tokenizer integration
- [ ] Continuous batching

---

## 🧑‍💻 About

Built by **Rishi Dwivedi** — systems-focused AI engineer with 2 years of
professional experience, building toward hardware-aware inference optimization
for custom AI accelerator companies.

This project is a deliberate descent past Python orchestration, past framework
abstractions, down to the hardware-software boundary where memory layout,
cache behavior, and instruction-level parallelism determine what is actually fast.

Every optimization ships with measurements. Nothing is claimed until it is proven.

---

<div align="center">

*Built from scratch. Measured at every step. No shortcuts.*

</div>