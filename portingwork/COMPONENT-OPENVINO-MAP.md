# Inference Stack Component Map: CUDA/NVIDIA → OpenVINO GPU/CPU/NPU

## Target Architectures

| Target | Arch | Wave | Tensor Cores | int8 dot4 | bf16 | Notes |
|---|---|---|---|---|---|---|
| Old NVIDIA (Maxwell/Pascal, CC ≤ 5.x) | sm_50/sm_52/sm_53 | 32 | ❌ | ❌ | ❌ | No tensor cores, scalar integer only |
| Turing (RTX 20xx) | sm_75 | 32 | ✅ | ✅ | ❌ | First consumer dot4 (i8x4), no bf16 compute |
| Ampere (RTX 30xx) | sm_80 | 32 | ✅ | ✅ | ✅ | bf16 native, fp8 absent (in this registry) |
| AMD RDNA3 / MTL iGPU | gfx1100 class | 32 | ❌ | ✅ | ✅ | dot2/dot4, rocWMMA for WMMA, shared L3/system memory |
| **Intel Meteor Lake** | Xe-LPG | 16/32 | ❌ | ❌ (scalarized) | ❌ | 4x8BitPacked*Accelerated=0 → float FMA only |

## 1. KINDI (Vulkan Engine) — Component Map

| KINDI Component | CUDA/NVIDIA Pattern | OpenVINO GPU (Xe-LPG) | OpenVINO CPU (MTL P-cores) | OpenVINO NPU |
|---|---|---|---|---|
| `bytenr4` (PTQ1_0 5-trit GEMV) | Integer ops, shared mem tiling, no tensor cores | **GLSL arithmetic** — 5-trit mod-multiply as shader math; shared mem for tile; **NOT** int8 dot (Xe-LPG has no accelerated dot4) | AVX2/SSE integer ops (vbmi2 for trit decode) | **Unsupported** — 5-trit PTQ1_0 not in nGraph; fallback to f32 |
| `bytenrT` (T-token batched) | Warp-level FMA batch, no syncthreads per row | **`OpGroupAccumulateSharedMemory` + `OpFAdd`** per subgroup; 4-chain FMA; T tokens in registers | Thread-parallel: each thread handles T tokens, vector FMA (AVX-512) | T=1 only (single-batch NPU) — unroll T loop |
| `bwtest` (bandwidth probe) | N/A | n/a | n/a | n/a |
| `attn_sg` (sparse attention) | Warp-shuffle for KV reduction | Subgroup ops (16/32 lanes); `subgroupShuffleDown` for partial reduction | OpenMP simd reduction | Native `attention` op (no masking) |
| `gdn_conv/delta` (recurrent) | Shared memory stencil, `__syncthreads` | Workgroup shared memory + `barrier()` | Thread-level parallelism, no barriers | Sequential per-token (no parallelism) |
| `rope` (rotary pos) | `__fsqrt_rn` + sin/cos table | GLSL `sin`/`cos` or LUT texture; fma-chain | AVX `vrcp` + LUT | Native `rope` op |
| `argmax` | Warp reduction | `subgroupReduce` or single-thread loop (1 op) | Scalar loop or AVX reduce | Scalar loop |

**Kands-verified mappings:**
- `__shfl_xor_sync` → `__shfl_xor(var, lane_mask)` (manual — drop the mask arg)
- `__syncthreads` → `__syncthreads()` (auto — same name in SYCL/OpenVINO)
- `cudaMalloc` → `hipMalloc` (auto, but irrelevant for OpenVINO — use `ov::Tensor`)
- `cudaGraphLaunch` → `hipGraphLaunch` (auto, ROCm 6.0+; OpenVINO equivalent: `AsyncInferQueue`)
- `cp.async` / `ldmatrix` / `wmma::load_matrix_sync` → **NOT in reference table** — probe with `probe_isa`, likely needs manual GLSL shared-memory tiling on Xe-LPG

## 2. llama.cpp — Component Map

| llama.cpp Component | CUDA/NVIDIA Pattern | OpenVINO GPU (Xe-LPG) | OpenVINO CPU (MTL P-cores) | OpenVINO NPU |
|---|---|---|---|---|
| `ggml.c` (backend dispatch) | `ggml_cuda_compute` (CUDA kernel launch) | **OpenVINO handles dispatch automatically** — `core.compile_model("model.gguf", "GPU")` | Same API call, "CPU" device | Same API call, "NPU" device |
| `ggml_cuda_alloc` (pinned host mem) | `cudaHostAlloc` + DMA | OpenVINO: `ov::Tensor` with `host_cache_type` | Native memory, no pinning needed | NPU: compiled-in buffer management |
| `llama-model.cpp` (MoE routing) | Expert select + kernel dispatch | `intel_gpu::op::GroupedGEMM` (OpenVINO 2023.4+) | `ov::op::GroupedGEMM` → MKBatchedMatMul | Sequential dispatch (NPU: no GroupedGEMM) |
| `llama-moe.cpp` | `moe_align_block_size` kernel, expert FFN | **OpenVINO 2024.4+: native MoE support** via `ov::op::MoE` decomposition | Same as GPU (CPU handles via thread parallelism) | **NOT SUPPORTED** — fallback to dense or single expert |
| `llama-sampling.cpp` | CUDA: `blockReduceMaxOps` for top-p | GPU: `subgroupReduce` + `OpType=TOPK` | CPU: `parallel_for` + `std::nth_element` | NPU: native `topk` op |

**Key principle for old NVIDIA (pre-Turing, CC ≤ 5.x):**
- No `__shfl_sync` intrinsics (Maxwell) → use shared memory reduction
- No `cp.async` (Pascal) → use `ldg()` + shared memory loads
- No tensor cores → pure FP32 FMA for all GEMM
- `ggml_cuda` falls back to scalar integer ops for quant (matches bytenrT philosophy)

## 3. Strato / Speculative Framework — Component Map

| Strato Component | CUDA/NVIDIA Pattern | OpenVINO GPU (Xe-LPG) | OpenVINO CPU (MTL) | OpenVINO NPU |
|---|---|---|---|---|
| `draft_model` (n-gram/MTP) | Lightweight CUDA kernels | GPU: n-gram lookup in buffer, MTP as small GEMM | CPU: parallel string match, small matmul | NPU: sequential, single token draft |
| `verify_engine` | Single CUDA kernel (accept/reject per token) | GPU: single dispatch over T tokens (`bytenrT`-style) | CPU: sequential per token | NPU: sequential per token |
| `spec_buffer` (snapshot) | `cudaMemcpy` to pinned staging | OpenVINO: `ov::Tensor` copy or `gpu_mmap` | `std::memcpy` (fast) | NPU: host copy (slow) |

**Speculative decoding on OpenVINO:**
- GPU: Works if OpenVINO supports dynamic shapes (slow, but functional)
- CPU: Works via thread parallelism — best target
- NPU: **Single-batch only** — speculative drafting with T>1 not natively supported

## 4. NanoVLLM — Component Map

| NanoVLLM Component | CUDA/NVIDIA Pattern | OpenVINO GPU (Xe-LPG) | OpenVINO CPU (MTL) | OpenVINO NPU |
|---|---|---|---|---|
| `BMM` (Batched Masked MHA) | Custom CUDA kernel, warp-shuffle | `intel_gpu::op::ScaledDotProductAttention` (2024.1+) | `ov::op::SDPA` → parallel_for | Native `attention` op |
| `PagedAttention` | `paged_attention` custom kernel + pagemap | `intel_gpu::op::KVCache` + blocked layout (2024.2+) | Same via OpenVINO Runtime | NPU: streaming KV (no PagedAttention) |
| `SplitFusedAttention` | `fused_softmax` + warp-level | `intel_gpu::op::FusedSoftmax` | `ov::op::Softmax` + vectorization | `ov::op::Softmax` |
| `FusedSoftmax` | warp-shared reduction (no `__syncthreads`) | Subgroup `OpType=REDUCTION` | AVX-512 `vreduce` | Native `softmax` op |
| `QuantLinear` (W4A16) | `__hmul2` half-ops, `__ldg` | `int4_2d_scales` via `intel_gpu` layout | `ov::op::WeightOnly` decompression | **Unsupported** (NPU: fallback to f32) |
| `cudaGraph` (inference) | Graph capture + `cudaGraphLaunch` | OpenVINO: `AsyncInferQueue` (not true graphs) | Same | OpenVINO: compiled-in graph (best NPU perf) |
| `flash_attn` (unified) | `mbarrier` + `cp.async` + `ldmatrix` | **Not available** — Xe-LPG lacks `cp.async`/`ldmatrix` | Native: no GPU shared memory | NPU: native_attn (no flash) |

## 5. Critical Hardware Limitations

### Intel Meteor Lake Xe-LPG (this target)
1. **No accelerated int8 dot4** — `4x8BitPackedSignedAccelerated=0` (verified via KINDI_INFO dump). All int4/8 dot product operations **scalarize** in the driver → slower than float.
2. **No tensor cores / matrix cores** — all compute is FP32 FMA.
3. **Wave size 32** — pinned via `VK_EXT_subgroup_size_control` in code.
4. **Shared DDR5 memory** — 37 GB/s ceiling (verified by `bwtest.comp`).
5. **No `cp.async` or `ldmatrix`** analogs — memory pipelining must be done via GLSL shared memory + `barrier()`.

### Comparison: sm_75 vs Xe-LPG
| Feature | sm_75 (Turing) | Xe-LPG (Intel) | Implication |
|---|---|---|---|
| Tensor cores | ✅ (FP16/INT8) | ❌ | Kernel must use float FMA path |
| int8 dot4 | ✅ (accelerated) | ❌ (scalarized) | bytenrT (float FMA) is optimal |
| Wave size | 32 | 16/32 (pinned 32) | Same dispatch math |
| L1/shared mem | 64 KB | 48 KB (160→64 if SLM alloc) | Avoid SLM for weight streaming |
| Memory | GDDR6 (448 GB/s) | DDR5 shared (37 GB/s) | Bandwidth-bound, not compute-bound |

## 6. Porting Verdicts (from KandS tools + static analysis)

- `__shfl_xor_sync` → manual mapping (drop mask) — affects attention KV reduction
- `cudaMalloc` → auto (hipMalloc) — not relevant (OpenVINO manages memory)
- `cudaGraphLaunch` → auto (hipGraphLaunch) — OpenVINO `AsyncInferQueue` is the equivalent
- `wmma::load_matrix_sync` → **NOT in reference table** — probe with `probe_isa`; likely needs manual GLSL tiling
- `cp.async` → **NOT in reference table** — no equivalent on Xe-LPG; use GLSL shared memory + barrier
- `ldmatrix` → **NOT in reference table** — no equivalent on Xe-LPG

## 7. Summary: Where to focus optimization effort

Per the vulkan-opt capability ladder and demoscene analysis:
1. **Head row-tiling** (in progress) — Xe-LPG fabric ceiling means every byte must be reused
2. **QKV/gate fusion** — reduces dispatch count (27K→~10K ops/tok) which helps CPU/NPU more than GPU
3. **Branchless trit LUT** — removes subgroup divergence (demoscene: farbrausch)
4. **x `[col][T]` layout** — cuts memory traffic across all backends (ryg)
5. **OpenVINO baseline** — set `KINDI_QUEUE_PRIORITY=realtime` on Xe-LPG for EU dominance; use CPU backend for fair comparison (no memory contention)