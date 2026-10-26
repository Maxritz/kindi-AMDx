# HIP backend for the RX 6700 XT (Navi 22, gfx1031)

A new engine for RDNA2. The Vulkan engine stays where it is: it is the verified reference for
the RX 580 and the token-exactness oracle this backend is checked against. That is its only
role here.

## The framing

**This backend is not a port. It is an engine designed for this card.**

The Kepler sources and the Vulkan port are where the understanding comes from: the quant
formats, the model graph, the numerics needed for token exactness. That is the
*specification*, not the implementation. Every structural decision below is made for
gfx1031's actual shape, and where the existing engines' structure exists only to satisfy a
GCN or CUDA limitation, it is not carried over at all.

## Ground rules

1. **GGUF as it is.** No conversion step, no weight pack. Read the publisher's file directly.
2. **No dequantisation.** Weights are never materialised in a wider form in VRAM. A 1.75-bit
   tensor stays 1.75 bits; it is decoded in registers immediately before the MAC. This is what
   makes the models fit: a 27B at 8 bits is 27 GB on an 11.98 GB card.
3. **Token exactness.** Output must match the reference engine token for token. This rules out
   any lossy step on the hot path, and it is what decided against dp4a below.
4. **No shoehorning.** A method is adopted because it is right for this card, not because it
   exists in the sources.

## What the card is

Verified with `hipInfo` and by probing the compiler, not read off a spec sheet:

| property | value |
|---|---|
| arch | gfx1031 (Navi 22, RDNA2) |
| compute | 20 WGP = **40 CU** (2 CU per WGP, unlike GCN's 1) |
| wave | **32**, native |
| clock | 2.42 GHz |
| VRAM | 11.98 GB, 192-bit, **96 MB Infinity Cache** |
| LDS | **64 KB per workgroup**, 128 KB/WGP, 64 banks |
| VGPRs | 256 DWORDs/thread |
| L2 | 3 MB |
| concurrent kernels | 1 (no kernel overlap; parallelism must be inside kernels) |

## dp4a does not work here, and that is a measured result

The obvious idea for a ternary model on RDNA2 is `v_dot4c_i32_i8`: four int8 MACs in one
instruction, which gfx803 lacks entirely. I probed it, found it present, measured a dp4a GEMV
at 46.7 ms/token equivalent for a dense 27B, and wrote that down as the reason to build this
backend.

**That measurement was wrong**, and correcting it changed the design.

The probe compared dp4a against an *integer* reference with int8 activations, accumulating
over the whole row. Two problems:

1. **The model does not run in integers.** Activations are fp32. Quantising them to int8 is a
   lossy step the Vulkan engine never takes, and rule 3 forbids it.
2. **The accumulate was wrong.** int8 activations are quantised per block, so each 128-weight
   block must be dequantised with its own scale before it joins the fp32 total. A single
   accumulate over the whole row cannot be dequantised correctly at all.

Corrected and re-measured on a real layer shape (5120 x 6144, the 27B's `attn_output`):

| | |
|---|---|
| int8 vs exact fp32, mean relative error | **35%** |
| worst relative error | 239x |
| top-20 output rows identical | **1 of 20** |
| dp4a speed vs exact fp32 | **0.49x (slower)** |

The top-20 figure is decisive: if one of the twenty highest-scoring rows agrees, the ranking
is destroyed and the argmax token differs. And once each 128-weight block needs its own
shuffle reduction for dequantisation, dp4a is twice as slow as the exact fp32 arithmetic.

**So dp4a is not used.** It is exact only for an integer formulation the model does not use,
and slower even there. Probing it was still necessary: it ruled the idea out with evidence
instead of assumption, and it stops the next person repeating it.

The same argument rules out `v_dot2_f32_f16` (fp16 activations are not exact) and
`v_dot8_i32_i4` (same integer problem). `wmma` does not exist on RDNA2 at all; that is RDNA3.

## What actually makes this card faster

Not an instruction. Throughput and structure:

| | RX 580 (gfx803) | RX 6700 XT (gfx1031) | ratio |
|---|---|---|---|
| lanes | 36 CU x 64 = 2304 | 40 CU x 128 = **5120** | 2.2x |
| clock | 1.34 GHz | 2.42 GHz | 1.8x |
| lane-ops/s | 3.09 T | **12.4 T** | **4.0x** |
| LDS per workgroup | 32 KB | **64 KB** | 2x |
| wave | 64 native | 32 native | - |

Measured, exact fp32 ternary GEMV on this card: **701 GMAC/s** on that layer shape. The
Vulkan engine's equivalent path on the RX 580 manages roughly 14 GMAC/s. That is the real
gap, and it comes from lanes and clock.

Extrapolating the 27B (27 G weights, one MAC each per token): 27e9 / 701e9 = **38.5 ms/token
= 26 tok/s**, before DRAM effects. The memory bound for the same workload is 5.9 GB at about
384 GB/s = 15.4 ms/token, so the arithmetic figure is the tighter of the two and there is
headroom rather than a wall.

**Honest target: 20 to 30 tok/s on the 27B**, against 5.19 on the RX 580. A 4 to 6x
improvement, worth building, just not for the reason I first wrote down.

## Design decisions

### Exact fp32 ternary GEMV, decoded in registers

PTQ1_0 packs four 2-bit trits per byte, plus a fifth trit per byte in the higher part of the
block (five per byte overall, which is where 1.75 bits/weight comes from). The kernel decodes
a trit to a float and does one FMA, exactly as the Vulkan path does, because that is what
exactness requires.

What differs is everything around it: 5120 lanes instead of 2304, 2.42 GHz instead of
1.34 GHz, 64 KB of LDS instead of 32 KB, and wave32 instead of wave64.

The one instruction-level improvement worth testing here is the two-sum formulation: a ternary
dot is `sum(x where w=+1) - sum(x where w=-1)`, which replaces `decode, convert, multiply,
add` with `predicate, add` twice. It is exact in fp32 and it removes a convert per weight. It
gets measured before it is adopted.

### wave32 only, width from rows

Per the RDNA2 ISA section 2.1, wave64 is issued as two wave32 passes. It buys no width and
`-mwavefrontsize64` crashes the linker on this ROCm build (LLVM branch-relaxation, gfx1031),
so it is not even available. Wave32, with width from rows per wave: **RPW 4**, because a
32-wide wave needs four rows in flight to match one row of a 64-wide wave. That is arithmetic,
not a borrowed setting.

### One-pass FWHT

On gfx803 a 1024-point fold needed two stages and a separate `fwht_hi` dispatch, because it
did not fit in 32 KB of LDS. With 64 KB the whole fold is one workgroup and one pass. The
two-stage structure is a GCN limitation and does not belong here.

### Reductions

32-lane waves: `__shfl_down` over five steps, or `__reduce_add_sync` where it applies. No
CUDA-style warp emulation and no shared-memory tree.

### Fusion, for a different reason

The Vulkan port fused same-input projections because a compute-to-compute barrier cost 44 us
and could not be narrowed (measured: full memory barrier, `SHADER_WRITE`-only and
buffer-scoped all within noise). On HIP the cost is a kernel launch and a stream boundary,
which is smaller and different. Fusion is still worth it - q/k/v, qkv/gate, gate/up - but the
justification is launch overhead, and the size has to be measured rather than assumed.

### Infinity Cache is a design constraint

96 MB on a 192-bit bus. The 27B's weights are 5.9 GB and never fit, but the activation working
set and the per-token KV slice do. So achieved bandwidth is not one number: it depends on
access order and reuse distance, and it has to be measured per kernel rather than inferred
from a peak figure.

## rocblas: used in the three places it helps

rocblas operates on dense typed matrices, and our weights are packed at 1.75 to 6 bits, so a
rocblas call needs the operand unpacked first. That collides with rule 2, so the question is
where unpacking pays for itself.

**1. As the correctness oracle, in `--selftest`, always.** Runs our GEMV and
`rocblas_gemm_ex` on the same unpacked tile and compares them. An independent, heavily tested
implementation of the same arithmetic is the strongest check available, and it catches exactly
the class of bug that already bit this project twice (a wrong trit-to-int8 packing order
produced plausible numbers that only failed against a reference). Lands with the scaffold
because it gates everything after it.

**2. For the F32/BF16 tensors that are already dense in the GGUF** and so need no
dequantisation: the norms, `ssm_a`, `ssm_dt.bias`, `ssm_beta`, `ssm_alpha`, `ssm_conv1d`,
`attn_q_norm`, `attn_k_norm`. Only `ssm_conv1d` is genuinely matrix-shaped, but using the
library there removes a hand-written kernel for free.

**3. As an opt-in unpacked path for models that fit.** `--rocblas-int8` unpacks once to int8
and runs `rocblas_gemm_ex`. Possible on the 2B (2.0 GB unpacked), refused with a clear message
on the 27B (27 GB on an 11.98 GB card) rather than silently OOMing. Off by default, and it has
to beat the hand-written kernel in a measured A/B before being recommended, because rocblas is
tuned for large GEMMs and decode is M=1.

**Not for decode on the packed path.** A single-token GEMV is bandwidth bound at M=1;
unpacking 5.9 GB to feed a GEMM with one row of work is strictly worse.

Every rocblas number in the benchmarks is reported next to its hand-written equivalent, so the
comparison is visible rather than claimed.

## Additional quants and models

The Vulkan engine handles PTQ1_0 (5-trit), 4-trit, PQ2_0, Q8_0, Q4_0. Every new format
follows the same shape here: decode the packed block in registers, exact fp32 accumulate,
apply the block scale once per block.

| format | why | how |
|---|---|---|
| **Q4_K / Q5_K / Q6_K** | the mainstream llama.cpp quants; opens the GGUF model zoo | nibble / 5-bit / 6-bit decode in registers; superblock scale and min applied after the dot |
| **IQ2/IQ3/IQ4** | the Underdog-Saluki 27B needs these and the Vulkan engine cannot load it | codebook grid as a LUT in LDS; 64 KB makes the larger grids fit |
| **Q4_1 / Q5_0 / Q5_1** | completeness | Q4_0 plus a min term |
| **BF16** | native on RDNA2 | storage only, decoded to fp32 before use |

The i-quants matter most: they are what makes 27B-class models fit under 8 GB, and the Vulkan
port could not load that file at all.

## Milestones

Each gated on token-identical output against the Vulkan engine on the same GGUF, and each
ships with its instrumentation.

1. **Scaffold.** GGUF reader, HIP device init, buffers, tracing, and `--selftest` that checks
   the GEMV against both a host reference and `rocblas_gemm_ex`. Gate: 2B loads, one GEMV
   matches.
2. **Exact fp32 PTQ1_0 GEMV** at RPW 4, plus fused and batched variants. Gate: 27B
   token-exact against the Vulkan engine.
3. **Full 27B decode:** attention, GDN recurrence, norms, single-pass fold, head.
4. **Prefill:** chunked passes, column-major x.
5. **MTP speculative decode**, with the bit-identity invariant between single-token and
   batched kernels.
6. **Additional quants:** K-quants, then i-quants.
7. **rocblas paths:** oracle first (it lands with the scaffold), then the dense tensors, then
   the measured opt-in int8 path for models that fit.
8. **Benchmarks** in the shape of [BENCHMARKS.md](BENCHMARKS.md), on both cards.

## Instrumentation

Not optional: any change that can affect speed, memory or correctness lands with the
instrumentation to measure it.

- `--trace` per-component and per-shader tables: ops, device us, idle us, host us, bytes.
- `--trace-decode` to start the table at decode so prefill does not swamp it.
- `--trace-json` for baselines to diff.
- Traps: dispatch (launch count, grid, block), memory (bytes and **achieved** GB/s),
  synchronisation (stream waits, idle gaps), allocation, transfer, dependency, accuracy
  (token hash), regression (before/after).
- Device time from HIP events or `clock64()`; host time from `QueryPerformanceCounter`.
- The instrumentation floor measured at exit, so signal is separable from overhead.

The trap that matters most here is **achieved bandwidth against the 96 MB Infinity Cache**,
per kernel, because the 27B working set does not fit and cache behaviour decides whether a
kernel is bandwidth bound at all.

## Environment

ROCm 7.16 at `C:\Rocm10`. Three things that cost time to find, in [HIP-ENV.md](HIP-ENV.md):

1. `hipcc` needs `--rocm-device-lib-path=C:\Rocm10\lib\llvm\amdgcn\bitcode`, or it fails with
   "cannot find ROCm device library". `--rocm-path` does not fix it.
2. An exe outside `C:\Rocm10\bin` reports "no ROCm-capable device is detected" even with
   `C:\Rocm10\bin` on PATH: `amdhip64_7.dll` loads `rocm_kpack.dll` and `amd_comgr.dll` from
   the **exe directory**. Copy those three next to the exe, or launch from `C:\Rocm10\bin`.
3. `ROCM_PATH` must be `C:\Rocm10` for anything that reads `.kpack`.

## Reference

- RDNA2 ISA: `rdna2-shader-instruction-set-architecture.txt` (513 KB)
- Arch summary: `rdna2_arch_summary.txt`
- The Vulkan engine, as the correctness oracle and the source of the format knowledge:
  [README.md](README.md), [EXPERIMENTS-FINDINGS.md](EXPERIMENTS-FINDINGS.md),
  [PERFORMANCE.md](PERFORMANCE.md)
- Probes that produced the numbers above: `hip/exactness2.cpp` (the dp4a refutation),
  `hip/gate.cpp`, `hip/isa_probe.cpp`
