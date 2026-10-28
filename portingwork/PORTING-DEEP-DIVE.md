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

## 4. ROOT-CAUSE / DATA model of the GEMV gap

Traced with bwtest.comp (streaming probe) vs g5ptq_bytenr (real decode).

FLOW (per-token, 2B, R=C=2048, wave 32, RPW=4):
```
body() -> [embed] -> L24 layers: gdn or attn -> ffn -> head -> argmax
        -> each layer: 1-3 PTQ1_0 GEMVs (qkv/o/down/ffn_gate/up) via gemvM5ptq
        -> gdn layers: pgdn_conv + pgdn_delta_T (serial-in-kernel T)
        -> attn layers: pgdn... no, attn_sg/attn_sq/rope/attn_vq/attn_comb + store_h
        -> ~116 GEMV dispatches/op for decode, 2324/op total incl. prefill-sized
```

DATA per weight byte: `w[bi>>2] >> (8*(bi&3))` — one 4-byte word fetched per byte,
but 4 lanes share the word in L1, so the *group* is dense. So it is NOT 4×
overfetch.

BRANCHES considered:
- B1 "overfetch from byte-decode" — BROKEN. L1 caches the uint32 word across the 4
  lanes that share it (4 lanes / word × 32 lanes = 8 words per subgroup per block
  row = exactly the 28-byte blocks padded to 32). Not it.
- B2 "L2 thrashing (weights 530 MB >> 4 MB L2)" — PARTIAL. The probe (pure stream,
  256 MB) also thrashes L2 yet hits 37. So DRAM streaming CAN reach 37 here.
- B3 "kernel not latency-hiding: per-op fixed overhead" — STRONG. Trace shows
  ~116 GEMV ops/token at ~459 us/op; a 4.5 MB op streamed at 37 GB/s = 121 us.
  The other ~340 us/op is kernel launch + occupancy ramp + tail, amplified ×116.
  The probe (16K+ workgroups, saturating) vs a GEMV op (64 workgroups for R=2048)
  differ in occupancy headroom.
- B4 "decode ALU bound" — PARTIAL. per byte: 5× mod-multiply decode (~6 int) + 1 FMA
  = ~7 ops; 530 MB × 7 ≈ 3.7 G ops/tok. At Xe-LPG ≈ 1.5 T int-ops/s that's ~2 ms —
  the ALU floor is tiny; the 340 us/op tail is launch/overhead, not decode.

HYPOTHESES (ranked by cost-to-refute):
- H1: per-op dispatch overhead dominates at 116 ops/tok. Fix: fewer, bigger ops —
  BUT the layer-by-layer recurrence (gdn state) and attention dependencies prevent
  fusing across layers. Partially fixable inside a layer (T-in-lane already does
  this) and via batching T tokens.
- H2: the T-in-lane kernel fixes the re-read-every-token bug (the spec verify
  passes share 1 weight stream across T tokens). CONFIRMED: bytenrT bit-identical,
  and a T=4 verify pass reads weights 1× not 4×.
- H3: gdn_delta_T at 562 us/op × 16 gdn layers is a hidden recurrent tax on every
  serial token (spec replay too). Needs its own kernel pass (vectorize the conv
  ring; it is small state).

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
thermal window).
