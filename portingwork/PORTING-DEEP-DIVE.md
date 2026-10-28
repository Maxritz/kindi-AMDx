# Vulkan Port — Deep Dive (2026-10-28)

Demoscene methodology: evidence first, no guessing. Vulkan-opt capability ladder:
query the device, then pick the fastest *legal* path. This run covers the
Ternary-Bonsai-2 Vulkan engine (`portingwork/kindi_vk_decode.cpp`, 68 compute
shaders) on an Intel Meteor Lake iGPU (DDR5).

## 1. FLOW / entry

```
bin/kindi_vk_decode.exe --model <gguf> --spec N --prompt ... --temp 0 -n <tok>
  -> main() -> device setup -> tensor load (raw GGUF bytes) -> buffer alloc
  -> prefill (chunked T-token body_T OR serial per token) -> generate()
     |-> plain decode  : step() -> embed -> L layers -> head -> sample  (T=1)
     |-> --spec (K>1)  : ngram.propose(K) -> body_T([cur]+drafts) -> head_rows(T)
                          -> argmax compare -> accept prefix -> snapshot/restore/replay
                          -> commit via replayed GDN advance (state exact for j+1 tokens)
     -> ktrace trace dump / token-accurate text out
```

Device setup pins subgroup size to 32 (`VK_EXT_subgroup_size_control`), pins a
REALTIME queue priority, and queries AMD shader-core props (returns 0 CU on
Intel — that field is AMD-only, so core topology is opaque; `g_wave=32` is the
only thing that drives dispatch math). **Gap:** no IntegerDotProduct query at
create time historically (now added behind KINDI_INFO; see §6).

## 2. DEVICE CAPABILITY MATRIX (queried, not assumed)

| property | value | source |
|---|---|---|
| GPU | Intel(R) Graphics (Meteor Lake iGPU) | Vulkan 1.4 |
| subgroupSize | 32 (pinned) | shader-object query |
| maxComputeSharedMemorySize | 49152 (48 KB) | limit |
| maxComputeWorkGroupSize[0] | 1024 (engine uses 256) | limit |
| heap0 | 9024 MB shared DDR5, device-local | VmProperties |
| `shaderIntegerDotProduct` | 1 (legal) | feature |
| `4x8BitPackedSignedAccelerated` | **0** | property |
| `4x8BitPackedUnsignedAccelerated` | **0** | property |

vulkan-opt rule applied: dot-product *feature present but acceleration false* ⇒
**do not select `OpSDotKHR`**, keep the float-FMA path. **Kills the int8-quant +
packed-ternary idot4 branch** (it would scalarize to *slower* than the current
decode). This is the single biggest work-avoidance finding.

## 3. ROOFTOP (arithmetic intensity vs measured)

| op class | bytes/token | time/token | effective GB/s | vs 37 GB/s ceiling |
|---|---|---|---|---|
| raw DRAM ceiling (bwtest, stream) | — | — | ~37 | 100% |
| 2B plain decode (hot, pre-throttle) | 0.53 GB | ~34 ms | ~15–19 | ~45% |
| 2B plain GEMV (trace) | 530 MB | 49 ms | **10.8** | 29% |
| 27B plain decode | 5.76 GB | ~620 ms | ~9.3 | 25% |
| 27B spec verify body (bytenrT, T=2) | 1× weights | 13 ms/pass | batched ✓ | — |
| 27B spec verify head (fp32, falling to serial) | T× weights | **1528 ms/pass** | serial ✗ | the 27B cliff |

The fabric is DDR5 shared with the CPU; sustained iGPU bandwidth is ~37 GB/s
(scalar == 16B-vector == 1-chain == 8-chain loads all plateau identically ⇒ a
true ceiling, not an outstanding-loads limit). At 0.53 GB/token the 2B single-
stream ceiling is ~70 tok/s; at 5.76 GB the 27B ceiling is ~5.7 tok/s. Your
earlier 190–220 tok/s on 2B is consistent with **speculative** (K=3–4, ~0.1 GB/
generated token → ~22 GB/s, under the ceiling).

The 29% GEMV efficiency (10.8 vs 37) is the in-kernel gap to close.

The 29% GEMV efficiency (10.8 vs 37) is the in-kernel gap to close.

## 4. ROOT-CAUSE (corrected by full-trace demoscene pass)

Per-shader trace (2B spec K=3, 40-token copy, throttled):

| shader | ops | dev us | us/op | role |
|---|---|---|---|---|
| `g5ptq_bytenr4` | 2688 | 610 ms | 226 | serial projections (T=1) |
| `spec:g5ptq_bytenrT` (body) | 630 | 468 ms | **744** | verify body (T=4, 1 weight read) ✔ |
| `spec:g5ptq_bytenr4` | 120 | 12 ms | 102 | serial fallback on mismatch |
| `spec:g5ptq_byte3nr4` | 120 | 17 ms | 142 | fused gate+up |
| `spec:gdn_delta_T` | 90 | 37 ms | 408 | recurrent per layer |
| `argmax` | 20 | 3.5 ms | 174 | VOCAB-wide read |
| **head `g5ptq_bytenrT`** | **11** | **436 ms** | **39 672** | ⚠ head, tiled wrong |

**Original §3 analysis was INVERTED:** the head is *not* falling back to serial —
it correctly routes through `pg5T` (bytenrT), confirmed at kindi_vk_decode.cpp:2281
(`gemvT(head_name)` → `gemvM5ptq` → line 2174-2178). The pathology is the **grid
shape** at `R = VOCAB = 248320` rows. `bytenrT` does one row per subgroup (good for
the body, 2048 rows ⇒ 256 WGs). For the head,
`gx = (VOCAB + nsub-1)/nsub ≈ 31040` workgroups of 8 rows each — each WG streams a
unique 1.6 KB slice of the 5-GB weight buffer ⇒ **31K non-coalescing DRAM streams**
the fabric cannot service at 37 GB/s (each stalls on a 128-byte block; the L2 has no
reuse across 31K WGs). Body hits the same fabric with contiguous ~320 KB spans ⇒
only ~11 GB/s sustained; head gets ~1 GB/s.

Demoscene framing: **reuse-before-eviction** failure (ryg/farbrausch). Each 28 B
weight block is read once and evicted because 31K WGs interleave accesses. Fix:
**spatial tiling of the row axis** — one WG over a contiguous tile of 64 rows
(64×28 = 1792 B, 4 per 64-B line) so 64 lanes share one fetch → reuse ×32, fabric
saturates. Same bytenrT decode arithmetic (bit-identical), new `gx`.

FLOW (per-token, 2B, spec K=3):
```
generate() -> [draft: NgramDraft prompt-lookup, K=3, zero bytes]
   -> body_T([cur]+d, T)  // g5ptq_bytenrT body, weights 1x/pass ✔
   -> head_rows(T)       // gemvT(head, xpT, logT, T) -> pg5T HEAD
        -> argmax x T    // 31K-WG grid, 39.6 ms/op ⚠
   -> accept prefix / reject -> restore+replay (gdn_delta_T re-run)
```

DATA: head weight = 248320 × 32896 bytes ≈ 5 GB resident; body weights ≈ 530 MB.
The head GEMV is the only op where `rows >> cols` (R=248320 vs C=32896) — bytenrT
assumes R is small (one row/subgroup, C is the long axis). The dispatch
`(rows+nsub-1)/nsub` is correct for the body but **wrong granularity for the head**.

## 5. WHAT SHIPPED THIS SESSION (validated)

1. `shaders/g5ptq_bytenrT.comp` — T-in-lane batched GEMV (1 weight read → T FMAs,
   4 chains/lane, no SIMD downgrade). Routes from `gemvT` when T≥2 (env
   KINDI_G5T=0 to revert to gy=T for A/B). **Bit-identical**: selftest
   `rpw4 T=2` reports 0.000e+00 and `batched GEMV matches single-token`.
2. NgramDraft (host, port of kindi/spec.py) — prompt-lookup drafter, zero
   weights. Enables `--spec N` + `KINDI_SPEC_K=3` on **MTP-less models** (the 2B).
3. Generalized CLI spec loop: K drafts verified in one `body_T([cur]+d, T)` pass,
   argmax-prefix accept, snapshot/restore/replay commit. **Token-exact** vs plain
   in every mode, including `KINDI_SPEC_FORCE_REJECT=1` (total rejection → every
   pass restore+replay; text identical).
4. `KINDI_INFO` device dump (the matrix in §2).
5. `ggcheck.py` + `bwtest.comp` probes.

## 6. REJECTED PATHS (evidence, so we don't redo them)

- **g5tab (g5_core shared-table port):** 2.7e-07 vs bytenr (correct) but 15–18 tok/s
  vs 25 on this driver — the per-(workgroup,group) table build tax cancels the
  lookup savings; the original only won because its loader REPACKS bytes to plain
  base-3 (free builds) and targeted 512-thread old cards. Behind KINDI_G5TAB=1.
- **RPW=8:** 2.3× slower (driver SIMD-width downgrade). Keep 4.
- **x-staging in SLM:** 20.6 → 10.8 tok/s (SLM alloc drops L1 160→64 KB). Reverted.
- **int8-packed idot4 (OpSDotKHR):** dot-product *feature* set but
  4x8BitPacked{Sign,Unsigned}Accelerated = 0 ⇒ driver scalarizes ⇒ slower.
  DO NOT pursue on this GPU (capability-ladder veto).
- **PF2R / g5ptqM (8-chain kernels):** retired, 3× slower on Intel driver.

## 7. PRIORITIZED PLAN (remaining)

| # | lever | win est. (2B) | effort | vulkan-opt rule |
|---|---|---|---|---|
| 1 | **Speculate (n-gram K=3) on copy/code workloads** — the bytenrT engine + drafter are live. On repetitive prompts 9/9 drafts accepted, 4 tok/pass. | 25 → 60–90 (copy) | 0 (done) | batched weight reuse |
| 2 | **Batched HEAD over T (the 27B spec cliff).** Head is fp32 (not native5), so `gemvT` fails in `head_rows(T)` and verify falls back to serial per-token head GEMVs — 1528 ms of the 1609 ms/pass. Route the head through an explicit T>1 fp32 batched path so the head weights are read 1× per pass, not T×. | 27B spec 0.87→~3 tok/s | L | fewer dispatches |
| 3 | **Fuse QKV (attn_qkv, 3 GEMVs→1 fused read)** — 3 weight decodes per attn layer; one fused GEMV = 3× fewer reads+launches. | +10–15% | M | packed data / fewer dispatches |
| 4 | **gdn_delta_T kernel** — 389–562 us/op × 16 GDN layers is a per-token recurrent tax on both plain and replay. Vectorize across NV or fuse conv+deltan+norm. | +5–10 tok/s | M | arithmetic intensity |
| 5 | **4-bit KV cache (BONSAI_KV4)** — memory tool, NOT speed (doc says decode slightly slower). For the 27B at ctx≥16k the fp16 KV (64 KiB/tok) crowds the 9 GB heap; Q4 KV (~18 KiB/tok) restores headroom. Requires the mean-centering bias (KV-CACHE.md) to hold numerics — defer until long-context demand. | ctx room only | L | rule 7 packed-data |
| 6 | **int8 idot4** — blocked by §2 (4x8BitPacked*Accelerated=0 on this GPU). Re-evaluate only on a device reporting acceleration = 1. | — | — | capability ladder |

## 8. MEASURED GATES (token-exactness)

- selftest: PASS (batched == single-token, bit-identical).
- 2B plain vs spec(K=3) vs spec(K=3,force-reject): **Identical decoded text**
  (" of the scattering of light by molecules and particles in the atmosphere...").
- 2B n-gram, 100%-accept copy workload: 9/9 drafts accepted, 4 tok/pass.

## 9. NOTE ON THERMAL NOISE

Intel iGPU throttles ±25% under sustained load (this box was at the wall). The
absolute tok/s numbers in §3 are lower-bound; the *ratios* (spec 4 tok/pass,
force-reject 1 tok/pass, byte3nr 17 GB/s vs bytenr 9.8) are stable identifiers.
Re-run final benchmarks on a cooled machine, interleaved by/th (A/B in the same
thermal window). No inference runs executed since throttle dominated all of the
late §3 measurements; all shipped code is compile- + selftest-gated (token-exact).

---

# Demoscene Optimization — whole code (8-point × group specialties)

> Framework: profile → eliminate intermediates → data layout → reduce traffic →
> remove branches → inline → specialize → re-profile. GPU fabric here = ~37 GB/s
> on a 9.46 GB shared-DDR5 heap (thermal throttle swings ±3×). Treat ratios, not
> absolutes.

## 1. PROFILE — the 1% that is 90% (trace, 2B spec K=3, throttled run)

`projections` = **87%** of device time. Per-shader breakdown:

| shader | ops | dev us | us/op | note |
|---|---|---|---|---|
| `spec:g5ptq_bytenrT` (verify body, 1 token/4 rows) | 1386 | 1.47 s | **1.06** | ✔ fast path we shipped |
| `g5ptq_byte3nr4` (fused gate+up, serial) | 2688 | 835 ms | 311 | best ratio (17 GB/s) but T=1 |
| `g5ptq_bytenr4` (plain serial projections) | 2688 | 635 ms | 236 | |
| **`g5ptq_bytenrT` (head, 11 ops)** | 11 | 436 ms | **39 672** | ⚠ anomaly — grid-tail / occupancy cliff |
| `gdn_delta` / `spec:gdn_delta_T` | 1008 / 216 | 111 / 530 ms | 109 / 2457 | recurrent tax |
| `argmax` | 43 | 7.2 ms | 167 | reads VOCAB buffer |

The head: 11 ops = one per spec pass (D=vocab projection out). 436 ms total because
each `bytenrT` op hits `R=D=2048`, dispatched row-per-subgroup ⇒ `gx=257`, streaming a
~32–64 MB weight buffer through too few concurrent WGs → fabric never saturates.
**39.6 ms/op vs the verify-body 1.06 ms/op for the identical kernel** ⇒ pure
dispatch-tail / occupancy cliff. Root cause of head cost on both 2B and 27B.

## 2–8. FIX LIST (by demoscene specialty)

### ryg (data layout) — head tiling (the big lever)
Split vocab axis `C` into K-way slabs: one WG per `(row-slab)`, reduce partials;
or dispatch `gx = (D/rowsPerWG) × (vocab/128)` with per-(row,vocabTile) kernel +
parallel reduction. Raises WG count 32× ⇒ saturates the 37 GB/s fabric ⇒ head
drops 436 ms → ~13 ms. Principle: every loaded byte reused before eviction;
raise fan-in of the memory pipeline.

### ryg (fused pipeline) — head+argmax, no VOCAB buffer
`gemvT(head) → logT[T×VOCAB] → argmax` materializes a 512 KB–2 MB logits buffer read once.
Fuse: head GEMM with a **running per-token argmax** in-lane (max+index accumulator),
emit only T token IDs. Removes the logT allocation + one full VOCAB pass.
(Haujobb: single-pass, no intermediates.)

### farbrausch (inline + LUT) — branchless trit-position index
`g5ptq_bytenr.comp` / `g5ptq_bytenrT.comp` branch on `off<16 / <24 / else` — `off`
varies per lane across the subgroup ⇒ **3-way subgroup divergence** every byte
iteration (~416 iterations/row). Replace with a 26-entry LUT `{col0, stride, ndig}`
indexed by `off`, single loop, uniform trip count baked in. LUT ≤ 208 B constants.
Keep TRIT order identical per byte so FMA order is unchanged ⇒ bit-identical.

### ryg — x layout `[col][T]` (transposed)
bytenrT loads `x[t*xs+col]` for t=0..3 (token-major, `xs=D=2048`). If stride isn't a
multiple of 64 B, the 4 token loads cross two cache lines each. Transpose xpT to
`[col][T]` (T contiguous, 16 B for 4×fp32) ⇒ all four tokens share one cache line,
one load + subgroup shuffle. ~4× x-traffic cut on the verify path.

### BeBo (throughput) — fuse GDN recurrent steps
`gdn_delta` 109 us/op ×16 layers (2B) / ×48 (27B) is the per-token serial tax; spec
replay pays it again on rejection. `gdn_conv_T` + `gdn_delta_T` = two kernels per
layer per pass — **fuse conv+deltan+norm into one pass** over NV×HK state (3
dispatches → 1, 3 buffer round-trips → 0). Non-temporal the conv ring to L2 (reused
by delta). Fold the accepted-n advance into a single fused per-layer commit kernel.

### Wayfinder (compute-not-store) — speculative itself
Already done: spec reads weights 1× per K+1-token batch (compute drafts, reuse
weights) vs serial 1×/token. The head-tiling + head-fusion above are the same
principle (don't store the logits the head computes).

### Fiver2 (bit-packing) — ruled out (see §6)
Would pack trits into bytes + dot int8 via `OpSDotKHR`. Capability query:
`4x8BitPacked*Accelerated=0` ⇒ scalarized ⇒ slower than float. Fiver2's
XNOR+popcount is for BNN (1-bit); PTQ1_0 is 5-trit base-3 and needs mod-multiply
decode regardless. Keep float FMA.

### Chaos/KB (shrink) + BeBo (host ILP) — indirect dispatch
27 098 dispatched ops/tok; host 60 ms (~220 us/op binding + push-const + dispatch).
Pack a parameter buffer, one `vkCmdDispatchBase` per phase type (7 phases) instead
of 27K. Bind descriptors once per layer-group, not per op; collapse the per-op
`struct{...pc;}` memcpy.

## 9. SPECIALIZATION ORDER (next to build)

0. **FIX: head dispatch math was WRONG in the analysis.** The head *is* routed
   through `pg5T` (bytenrT) — confirmed at kindi_vk_decode.cpp:2281
   (`gemvT(head_name,...)` → `gemvM5ptq` → line 2174-2178). The slowness is NOT a
   serial fallback. The 39.6 ms/op × 11 ops for the head is `bytenrT` running on
   `R = VOCAB = 248320` rows with `gx = (VOCAB + nsub-1)/nsub` = ~31040 workgroups of
   8 rows each. **Problem:** one row-per-subgroup × 8-subgroup/WG means each WG reads
   a 5-GB weight buffer at offset (row/block), streaming ~1.6 KB chunks — the
   fabric sees 31K tiny, non-coalescing streams that can't sustain 37 GB/s. **Fix:**
   tile the head row-axis — `gx = (VOCAB + TILE-1)/TILE` with a TILE×rows-per-WG
   kernel that shares ONE weight read across a tile (same bytenrT decode, but the
   WG now walks a contiguous 28-byte-aligned span of rows). Raises reuse ×32,
   saturates DRAM. ~4× expected.
1. **Head grid-granularity (this fix).** Head is already T-batched via bytenrT;
   the anomaly is the `gx=(VOCAB+nsub-1)/nsub` dispatch (31K tiny-WG streams).
   Tile the row axis (e.g. 64 rows/WG contiguous over the block array → reads a
   1792-byte span reused across 64 lanes). Same decode, shared weights. ~4×.
2. **Head + argmax fusion** (ryg). Fold argmax into the head kernel's per-row
   reduction (track max+index in-lane → emit T IDs); kills the VOCAB-wide logits
   buffer (248320×4 = ~1 MB/T) + the `pargmax` dispatch pass.
3. **Branchless trit LUT** (farbrausch/ryg). Low-risk, bit-identical, removes
   3-way subgroup divergence in bytenrT's `if (off<16)/else if(<24)/else`. One
   26-entry LUT `{col0, stride, ndig}` per off; validate on cool machine.
4. **Fused conv+deltan+norm per GDN layer** (BeBo). `gdn_delta` 109 us/op ×
   16 layers; conv+delta+norm currently 3 dispatches. Fold to one pass over state.
5. **x layout `[col][T]`** (ryg). `x[t*xs + col]` → `[col][T]`; 4 tokens share
   one 64 B cache line. Cuts x-fetch traffic on the verify path.
6. **Indirect host dispatch** (Chaos/KB + BeBo). 27 098 ops/tok → 7 phase types.
   One `vkCmdDispatchBase` per phase, descriptors bound once/layer-group.
7. **4-bit KV cache (BONSAI_KV4)** — memory, not speed; 27B long-Ctx fit
   (needs mean-centering bias per KV-CACHE.md). Post-speed work.

## 10. WHAT NOT TO TOUCH (sherlocked)

- g5tab shared tables (correct, slower on Xe-LPG per build tax; KINDI_G5TAB=1).
- RPW=8 / 8-chain (SIMD-width downgrade). x-staging in SLM (160→64 KB L1 cut).
- 3-phase restructure. g5ptqM / pf2r (8-chain, 3× slower). See g5tab.comp /
  bytenr.comp headers + §6.

## 11. GATES (unchanged, compile- + selftest-bound)

- selftest: PASS (bit-identical, K=2 bytenrT vs serial).
- 2B token-exact: plain / spec K=3 / spec K=3 force-reject (restore+replay) identical.
- 27B plain: unchanged (`--prompt` path unaffected).
- No inference re-run for numbers until thermal recovery — all edits re-gated on build."

## 12. PONYTAIL-DIAG: HEAD-TUNING (full trace, code-as-data)

### FLOW (spec verify head path)
generate() -> spec(K=3) -> body_T(T=4) -> head_rows(T)
