# HIP backend for the RX 6700 XT (Navi 22, gfx1031)

A new engine for RDNA2. The Vulkan engine stays where it is: it is the verified reference for
the RX 580 and the token-exactness oracle this backend is checked against. That is its only
role here.

## The framing, stated plainly

**This backend is not a port of anything. It is an engine designed for this card.**

The Kepler sources and the Vulkan port are where the understanding comes from: what the
quant formats are, what the model graph is, what the numerics have to be for token
exactness. That is the *specification*. It is not the implementation.

Every structural decision below is made for gfx1031's actual shape, and where the existing
engines' structure exists only to satisfy a GCN or CUDA limitation, it is not carried over at
all. RDNA2 does not inherit GCN's constraints, so it should not inherit GCN's workarounds.

## Ground rules

1. **GGUF as it is.** No conversion step, no weight pack. Read the publisher's file directly.
2. **No dequantisation.** Weights are never materialised in a wider form in VRAM. A 1.75-bit
   tensor stays 1.75 bits; it is expanded in registers immediately before the MAC. This is
   what makes the models fit: a 27B at 8 bits is 27 GB on an 11.98 GB card.
3. **No shoehorning.** A method is adopted because it is right for this card, not because it
   exists in the sources. Where the sources' method exists to work around gfx803 or CUDA, it
   is dropped.

## What the card actually is

Verified with `hipInfo` and by probing the compiler, not read off a spec sheet:

| property | value | what it means for the design |
|---|---|---|
| arch | gfx1031 (Navi 22, RDNA2) | |
| compute | 20 WGP = **40 CU** | 2 CU per WGP, unlike GCN's 1 |
| wave | **32**, native | wave64 is two wave32 passes (ISA 2.1), so never use it |
| clock | 2.42 GHz | |
| VRAM | 11.98 GB, 192-bit, **96 MB Infinity Cache** | the 27B fits, with room for KV |
| LDS | **64 KB per workgroup**, 128 KB/WGP, 64 banks | a whole 1024-point fold fits in one pass |
| VGPRs | 256 DWORDs/thread | |
| L2 | 3 MB | |
| concurrent kernels | 1 | no kernel overlap; parallelism must be inside kernels |

### Instructions that exist, probed not assumed

Compiled in isolation and disassembled with `llvm-objdump`:

| instruction | result |
|---|---|
| `v_dot4c_i32_i8` (dp4a) | **present** |
| `v_dot8_i32_i4` (signed nibble dot) | **present** |
| `v_dot2_f32_f16` | present |
| `wmma` (matrix cores) | **absent** - RDNA3, not RDNA2 |

### The one number that decides the design

The RX 580 has no integer dot product. `vulkaninfo` reports
`integerDotProduct8BitSignedAccelerated = false`, so the Vulkan engine had to decode every
ternary weight to a float and give it its own FMA: ~24 instructions per weight, which is what
capped the 27B at 5.19 tok/s.

Here, one instruction does four int8 MACs. Measured, same GEMV shape:

| | instructions/weight | dense 27B equivalent |
|---|---|---|
| Vulkan / gfx803 | ~24 | 192.7 ms/token (5.19 tok/s) |
| HIP / gfx1031 | **~3.25** | **46.7 ms/token (21.4 tok/s)** |

That single instruction is the reason this backend exists.

## Design decisions, each made for this card

### No dequant: expand in registers

PTQ1_0 packs four 2-bit trits per byte. `dp4a` wants four int8 lanes. So one 256-entry LUT
in constant memory turns a packed byte into four int8 trits, in a register:

```cpp
int w4 = c_lut[rp[i]];                          // byte -> four int8, 1 KB constant memory
int x4 = xq[i];                                 // four int8 activations
acc = __builtin_amdgcn_sdot4(w4, x4, acc, 0);   // four MACs, one instruction
```

The tensor stays at 1.75 bits in VRAM for the whole run. Verified bit-exact against an
integer reference: **max abs error 0.0**.

Compiled inner loop, 13 instructions per four weights, exactly one `v_dot4c`:

```
global_load_ubyte v0, v[11:12], off      ; packed trit byte
v_lshlrev_b32 v0, 2, v0                  ; LUT index
global_load_dword v11, v[7:8], off       ; four int8 activations
global_load_dword v0, v0, s[2:3]         ; LUT -> four int8 trits
v_dot4c_i32_i8 v4, v0, v11               ; four MACs
```

This generalises. Every quant format becomes "expand the packed block to int8 in registers,
one dp4a, apply the block scale after" - so the i-quants that the Vulkan engine could not
load at all need no separate dequant kernel, just a different expander.

### wave32 only, width from rows

The ISA says wave64 is issued as two wave32 passes. It buys nothing and
`-mwavefrontsize64` crashes the linker on this ROCm build. So: wave32, and get width by
putting several output rows in flight per wave.

The Vulkan port already parameterises this and found it the largest single win after the
decode rewrite (1.49x). It maps here as **RPW 4** rather than 2, because a 32-wide wave needs
four rows in flight to match what one row of lanes gave on a 64-wide wave. That is an
arithmetic consequence of the wave width, not a borrowed setting.

### One-pass FWHT

On gfx803 a 1024-point fold needed two stages with a separate `fwht_hi` dispatch, because
the whole thing did not fit in 32 KB of LDS. Here 64 KB per workgroup means the entire fold
is one workgroup and one pass. The two-stage structure is a GCN limitation; it does not
belong here.

### Reductions

32-lane waves, so `__shfl_down` over five steps, or `__reduce_add_sync` where it applies.
No CUDA-style warp emulation and no shared-memory tree.

### Fusion, for a different reason

The Vulkan port fused same-input projections because a compute-to-compute barrier cost 44 us
and could not be narrowed (measured: full memory barrier, `SHADER_WRITE`-only and
buffer-scoped all within noise). On HIP the cost is a kernel launch and a stream boundary,
which is a different and smaller thing. Fusion is still worth it - q/k/v, qkv/gate, gate/up -
but the justification is dispatch overhead, not a GCN drain, and the measurement has to
confirm the size before it is adopted.

### Infinity Cache is a first-class design constraint

96 MB of cache on a 192-bit bus. The 27B's weights are 5.9 GB, so they never fit, but the
activation working set and the per-token KV slice do. That means the achieved bandwidth is
not one number: it depends on access order and reuse distance, and it has to be measured per
kernel rather than inferred from a peak figure.

This is the main reason the new engine needs its own tracing rather than reusing Vulkan
numbers.

## rocblas: measured against the constraints, and it does not fit

The brief said to use rocblas where needed. Under the no-dequant rule it is not needed, and
it is worth writing down the arithmetic rather than inventing a use.

rocblas operates on dense typed matrices. Our weights are packed at 1.75 to 6 bits, so **any
rocblas call requires unpacking first**, which is the dequantisation the ground rules forbid:

| form | 27B | 2B |
|---|---|---|
| packed at 1.75 bits | 5.9 GB (fits) | 0.44 GB (fits) |
| unpacked to int8 | 27 GB (**does not fit**) | 2.0 GB (fits) |
| unpacked to fp16 | 54 GB (**does not fit**) | 4.0 GB (fits) |

For the 27B there is no rocblas path at all. For the 2B one exists but requires a dequantised
copy in VRAM, which is excluded. **So rocblas is not used.** If the constraint is ever relaxed
for small models, it becomes a legitimate experiment to measure against dp4a - not an
assumption that it wins.

## Additional quants and models

The Vulkan engine handles PTQ1_0 (5-trit), 4-trit, PQ2_0, Q8_0, Q4_0. Here every new format
follows the same shape - expand to int8 in registers, dp4a, apply scale - so the marginal cost
of a format is an expander, not a kernel:

| format | why | expander |
|---|---|---|
| **Q4_K / Q5_K / Q6_K** | the mainstream llama.cpp quants; opens the GGUF model zoo | nibble / 5-bit / 6-bit to int8; superblock scale and min applied after the dot |
| **IQ2/IQ3/IQ4** | the Underdog-Saluki 27B needs these and the Vulkan engine cannot load it | codebook grid as a LUT in LDS; 64 KB makes the larger grids fit |
| **Q4_1 / Q5_0 / Q5_1** | completeness | Q4_0 plus a min term |
| **BF16 activations** | native on RDNA2 | activation side only; weights stay packed |

The i-quants matter most: they are what makes 27B-class models fit under 8 GB, and with
in-register expansion there are no "dequant shaders" to write.

## Milestones

Each gated on token-identical output against the Vulkan engine on the same GGUF, and each
ships with its instrumentation.

1. **Scaffold.** GGUF reader, HIP device init, buffers, tracing, `--selftest` against the
   integer reference. Gate: 2B loads, one GEMV matches. *(The dp4a kernel is already verified
   exact; this milestone is the plumbing around it.)*
2. **PTQ1_0 dp4a GEMV** at RPW 4, plus the fused and batched variants. Gate: 27B token-exact.
3. **Full 27B decode:** attention, GDN recurrence, norms, single-pass fold, head.
4. **Prefill:** chunked passes, column-major x.
5. **MTP speculative decode**, with the bit-identity invariant between single-token and
   batched kernels.
6. **Additional quants:** K-quants, then i-quants.
7. **Benchmarks** in the shape of [BENCHMARKS.md](BENCHMARKS.md), on both cards.

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
per kernel, because the 27B working set does not fit and the cache behaviour decides whether
a kernel is bandwidth bound or not.

## Environment

ROCm 7.16 at `C:\Rocm10`. Three things that cost time to find, in [HIP-ENV.md](HIP-ENV.md):

1. `hipcc` needs `--rocm-device-lib-path=C:\Rocm10\lib\llvm\amdgcn\bitcode`, or it fails with
   "cannot find ROCm device library".
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
