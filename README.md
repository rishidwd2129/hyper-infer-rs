<div align="center">

# ⚡ Rust AI Inference Engine
### Hyper-Optimized On-Device AI for Apple Silicon

*A bare-metal inference engine built entirely from scratch in pure Rust — no Python, no bloat, no compromises.*

[![Built in Rust](https://img.shields.io/badge/Built%20in-Rust-orange?style=for-the-badge&logo=rust)](https://www.rust-lang.org/)
[![Target: Apple Silicon](https://img.shields.io/badge/Target-Apple%20Silicon-black?style=for-the-badge&logo=apple)](https://developer.apple.com/documentation/apple-silicon)
[![Architecture: Bare Metal](https://img.shields.io/badge/Architecture-Bare%20Metal-red?style=for-the-badge)]()
[![Status: Active](https://img.shields.io/badge/Status-Active%20Development-brightgreen?style=for-the-badge)]()

---

**571ms → 353ms forward pass** &nbsp;|&nbsp; **~40% faster matmul** &nbsp;|&nbsp; **~38% lower latency** &nbsp;|&nbsp; **Zero external ML deps**

</div>

---

## 🧭 What This Is

Most inference runtimes are built *on top of* Python, BLAS, and layers of abstraction that obscure what the hardware is actually doing. This project goes the other direction — straight to the metal.

Built entirely in **pure Rust**, this engine gives strict, explicit control over:
- **Memory layout** and heap allocation in the hot path
- **CPU cache behavior** via tiled matrix operations
- **Weight loading and transposition** without runtime overhead
- **Profiling at microsecond granularity** — no guessing, just numbers

This is not a wrapper. This is a ground-up construction of an AI inference system.

---

## 📊 Performance Results

The numbers below are real, measured on Apple Silicon using the engine's built-in profiler.

### Before — Naive Matrix Multiplication

```
========== PROFILE SUMMARY ==========
MATMUL:      360 calls   │  total 371.37ms  │  avg 1.032ms/call
SOFTMAX:     144 calls   │  total   0.05ms  │  avg 0.000ms/call
LAYER_NORM:   25 calls   │  total   0.19ms  │  avg 0.008ms/call
ALLOCATIONS: 504 allocations, 2.253 MB total
======================================
[SLOW] gpt2_forward_total took 571.26ms
```

### After — Cache-Aware Tiled Matrix Multiplication (32×32 blocks)

```
========== PROFILE SUMMARY ==========
MATMUL:      360 calls   │  total 221.50ms  │  avg 0.615ms/call  ▼ 40%
SOFTMAX:     144 calls   │  total   0.04ms  │  avg 0.000ms/call
LAYER_NORM:   25 calls   │  total   0.19ms  │  avg 0.008ms/call
ALLOCATIONS: 504 allocations, 2.253 MB total
======================================
[FAST] gpt2_forward_total took 353.05ms  ▼ 38%
```

| Metric | Before | After | Delta |
|---|---|---|---|
| Matmul total | 371.37ms | 221.50ms | **▼ 40.3%** |
| Forward pass | 571.26ms | 353.05ms | **▼ 38.2%** |
| Avg per matmul call | 1.032ms | 0.615ms | **▼ 40.4%** |
| Allocations | 504 / 2.25 MB | 504 / 2.25 MB | Next target |

---

## ⚙️ Optimization: Cache-Aware Tiled Matrix Multiplication

This is the first — and currently most impactful — systems optimization shipped in this engine.

Naive `matmul` iterates over rows and columns in a pattern that continually evicts cache lines before they can be reused. The tiled approach divides matrices into **32×32 static blocks** that fit entirely in L1/L2 cache, so each element is loaded once and reused across the entire tile computation before being evicted.

Every linear projection in the transformer (Q, K, V projections, output projections, MLP layers) is matmul-bound — so this single change compounds across every layer of the forward pass.

```
Naïve Access Pattern          Tiled Access Pattern (32×32)
─────────────────────         ────────────────────────────
Row A: load, compute          Block A: load once
Row A: evicted from cache     Block A: stays in L1 cache
Row A: reload (cache miss)    Block A: reused 32× before evict
... (repeat per row)          Next block: repeat
```

The result: **40% reduction in matmul time, 38% reduction in total forward pass latency** on Apple Silicon.

---

## 🏗️ Architecture

The workspace enforces a strict separation between *math* and *execution*:

```
rust-inference-engine/
├── core/                        ← Mathematical foundation
│   ├── src/
│   │   ├── tensor.rs            # Tensor struct, memory layout
│   │   ├── matmul.rs            # Tiled matrix multiplication
│   │   ├── attention.rs         # Causal self-attention + masking
│   │   ├── activations.rs       # GELU, Softmax
│   │   ├── norm.rs              # LayerNorm
│   │   └── profiler.rs          # Microsecond-level profiling
│   └── Cargo.toml               # Zero ML dependencies
│
└── cli/                         ← Execution & verification layer
    ├── src/
    │   ├── main.rs              # Weight loading, forward pass runner
    │   └── loader.rs            # Safetensors parser, QKV decoupling
    └── Cargo.toml
```

**`core/`** has zero external ML dependencies — only `serde` and a lightweight safetensors parser. Every linear algebra operation is hand-implemented.

**`cli/`** orchestrates weight loading and surfaces the profiling output. Deliberately kept thin so `core/` stays portable.

---

## 🔬 Built-in Profiler

The profiler is not a wrapper around `std::time`. It is a custom `ProfileGuard` struct that:

- Tracks **wall-clock microseconds** per named region
- Accumulates **call counts** across the forward pass
- Reports **total heap allocation bytes** per run
- Flags passes as `[SLOW]` or `[FAST]` against a configurable threshold

To run with profiling:

```bash
cargo run --release --bin cli
```

---

## 🛣️ Roadmap

Optimizations ship when they are measured and verified — not before.

**Foundation** *(shipped)*
- [x] Cache-Aware Tiled Matrix Multiplication (32×32 blocks, L1/L2 cache-aware)
- [x] Safetensors weight loading & fused QKV decoupling
- [x] Microsecond profiling with allocation tracking
- [x] Autoregressive causal masking for attention

**Generation** *(in progress)*
- [ ] BPE Tokenizer integration
- [ ] Interactive streaming generation loop
- [ ] KV Caching — O(1) attention state across decode steps

**Memory** *(planned)*
- [ ] Buffer ping-ponging via `matmul_into` — zero intermediate heap allocations
- [ ] Arena allocator — single slab for the entire forward pass

**Hardware** *(planned)*
- [ ] Apple Silicon / Metal GPU backend

---

## 🧑‍💻 About

Built by **Rishi Dwivedi** — AI Solutions Architect with 2 years of professional engineering experience, focused on on-device and hardware-aware AI inference systems.

This project is a deliberate descent into low-level systems programming: past Python orchestration, past framework abstractions, down to the hardware-software boundary where memory layout and cache behavior determine what is actually fast.

Every optimization ships with measurements. Nothing is claimed until it is proven.

---

<div align="center">

*Built from scratch. Measured at every step. No shortcuts.*

</div>
