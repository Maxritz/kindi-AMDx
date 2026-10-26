# Kindi Vulkan  -  performance report

GPU: RX 580, 36 CU x 4 SIMD x 8 wavefronts = 1152 wave slots, wave size 64
Workload: `--prompt "The sky is blue because" --temp 0`
Correctness gate: 2B token ids `314 279 69377 314 3000`, 27B ` of Rayleigh scattering.`
(`x\kindi_cpu.exe` cross-checks the 2B identically)

## Headline

| stage | 2B tokens/s | change |
|---|---|---|
| original (per-dispatch fence, g4simple) | ~3.3 |  -  |
| + dispatch batching | 5.5 | 1.7x |
| + g4lut (256-entry LUT, 4 trits/byte) | 26.0 | 4.7x |
| + g4lut2 (4 rows/workgroup, subgroup reduce) | 34.5 | 1.3x |
| + gemv3 (3 tensors per launch) | **36.3** | 1.05x |

**~11x overall.** All steps verified token-identical.

## Where the time goes now (2B, n=40, --trace)

```
component       ops    %dev    dev us   idle  host us
projections     ...    73.3%  305947          2650
recurrent              9.9%   41180          1031
norms                  8.6%   35896          1818
ffn-activate           3.1%   13130           871
attention              2.1%    8912           795
embed                  1.6%    6643          1995
head+sample            0.9%    3940            22
residual               0.5%    1934           329
TOTAL         17852   100.0%  417582         12524

synchronisation: submits=40 fence_waits=40 (446.3 ops per submit)
queue wait     : 1282.77 ms in 40 flushes (32.07 ms each)
device+wait    : 1700.84 ms   (device 418.07 ms)
host total     : 12.44 ms
```

Per token: **device 10.4 ms, inter-dispatch drain 21.6 ms, host 0.3 ms.**
Wall marginal 27.5 ms/token.

## Fusing sibling projections: 1.5%, because only 6% of dispatches are siblings

The obvious next move after the barrier result was to fuse same-input projections into
one dispatch. `g5ptq_byte3` does that for up to three PTQ1_0 tensors: concatenated row
space, one 64-thread workgroup per row, same decode and same tree reduction as
`g5ptq_byte`, so results are bit-identical (`--selftest` asserts 0.000e+00 on all three
outputs). It fires on `attn_q`/`attn_k`/`attn_v` and on `attn_qkv`/`attn_gate`.

| | dispatches | device | wall | tok/s |
|---|---|---|---|---|
| separate launches | 26660 | 4707 ms | 6114 ms | 3.271 |
| fused, per-byte buffer select | 25060 | 4789 ms | 6082 ms | 3.288 |
| fused, branch hoisted | 25060 | 4728 ms | 6007 ms | 3.329 |

Two things to learn from this.

**The dynamic buffer select is expensive.** The first version picked the tensor per byte
with `(t==0) ? W0[i] : (t==1) ? W1[i] : W2[i]`, and measured 772 us/op against 563 for
the single-tensor shader, making fusion a net loss on device time. That is the same
mistake the abandoned `g4.comp` made. The row's tensor is uniform across the workgroup,
so the branch belongs outside the loop; hoisting it to three static arms took the fused
kernel to 722 us/op and recovered 61 ms.

**But the ceiling is low.** Fusion removes 1600 of 26660 dispatches, 6%. At the measured
44 us per dispatch that is ~70 ms, and the observed wall gain is 107 ms. Roughly
proportional, as the barrier experiment predicted.

The reason the ceiling is low is the shape of the workload, not the implementation:

| | dispatches/token | device ms/token | barrier cost at 44 us |
|---|---|---|---|
| `g5ptq_byte` (the GEMVs) | 401 | 225.7 | 17.6 ms |
| everything else | 932 | 9.7 | 41.0 ms |

The 932 non-GEMV dispatches do 9.7 ms of work and pay 41 ms of barrier. The worst
offender is `fwht_hi`: 257 dispatches per token at 3.6 us each, because `fold_hi` is
called from both `rmsfold` and `smfold` and is a separate dispatch each time. That is
11.3 ms/token of drain for 0.9 ms/token of work.

I first concluded from that table that the barrier cost was dominated by the small
shaders. **That was wrong, and the experiment that disproved it is below.** Barrier cost
scales with the work in flight at the barrier, not with the barrier's existence.

## The scale is one word, not two bytes: 1.04x

A PTQ1_0 block is 28 bytes, so it always starts at a multiple of 28, and 28 % 4 == 0.
Byte 26 of the block is therefore always the high half of word `(base/4 + 6)`. The
scale is one word load and a shift:

```glsl
int sw = ((row * bpr + blk) * 7) + 6;
float s = unpackHalf2x16(w[sw] >> 16).x;
```

instead of two byte loads, two shifts, an or and a mask.

| | us/op | gain |
|---|---|---|
| two byte loads (previous) | 535.3 | 1.00x |
| one word load | **516.0** | **1.04x** |

### A negative that matters: the FMA chain is not the bottleneck

All five digit FMAs accumulate into one scalar, so in principle each waits on the
previous result. FMA latency on GCN is ~4-6 cycles, which would be 20-30 cycles per
byte. Unrolling the digit loop and using four independent accumulators to break the
chain measured **0.98x**: slightly worse.

So the kernel is not latency-stalled. The compiler was already interleaving the loop
iterations, and the cost is instruction *issue throughput*. That is consistent with the
24 instructions per weight derived from the lane-ops bound, and it means there is no
free win hiding in the scheduling. Anything further has to remove instructions, not
hide them.

## A float LUT instead of integer decode: 1.09x

The inner loop turned a byte into five trits with, per trit, a shift, an and, an
int-to-float convert and a subtract before the FMA. Storing the already-converted
float in LDS removes all of it: the loop becomes one LDS load and one FMA per weight.

| variant | us/op | gain |
|---|---|---|
| integer decode, 5 packed trits in one uint | 581.9 | 1.00x |
| `float` LUT, 5 floats per byte (5 KB LDS) | **535.8** | **1.09x** |
| `float` LUT padded to 8, two vec4 loads | 595.4 | 0.98x |

The vec4 variant is slower because the padded stride hits LDS bank conflicts, so
scalar loads are kept. The arithmetic is identical, so tokens are unchanged, and the
same table is used by both the single-tensor and the fused kernel.

End to end: 3.586 -> **3.752 tok/s** serial, device 4340 -> 4053 ms.

Also tried and rejected: `bitfieldExtract` in place of shift+and, which measured
1.00x (no gain, and the naive form sign-extends when the packed value has bit 31 set).

## Tiling: 4 rows per workgroup, 1.07x, bit-identical

The GEMV launched one 64-thread workgroup per output row, so a 5120-row matrix meant
5120 workgroups. Putting four rows in one 256-thread workgroup, one wave per row,
cuts that 4x and amortises the byte-to-trit LUT build across four waves instead of one.

| kernel | 1 row/WG | 4 rows/WG | gain |
|---|---|---|---|
| `g5ptq_byte` (1 tensor) | 609.1 us/op | **568.5 us/op** | 1.07x |
| `g5ptq_byte3` (3 tensors fused) | 1121.5 us/op | **1045.5 us/op** | 1.07x |

Together with the ffn fusion: wall 5948 -> 5577 ms, **3.362 -> 3.586 tok/s**.

### The reduction was kept as a tree on purpose

An earlier 4-row variant used `subgroupAdd` and measured 1.09x, slightly better than
the 1.07x here. It is not used. A butterfly reduction sums the 64 lane partials in a
different order than the shared-memory tree, so it is not bit-identical to `g5ptqM`,
and the speculative verify pass compares trunk rows across those two kernels. A
near-tie in the logits could then accept a draft the serial path would have rejected.
0.02x is not worth that.

### A barrier hazard the first version had

`barrier()` must be reached by every invocation in the workgroup. The first version
wrote `if (row >= R) return;` before the reduction, which is a whole-wave exit: the
waves that returned would never reach the barrier that the surviving waves wait on,
which is undefined behaviour. It happened to work because every 27B row count is
divisible by 4. The shipped version clamps the row instead and suppresses only the
store, so all four waves always reach every barrier.

## Fusing ffn_gate/ffn_up, and why barrier count is the wrong metric

Two more fusions were tried, and they disagree, which is the useful part.

**ffn_gate + ffn_up, fused.** These are 128 dispatches per token, and they sit between
two large GEMVs (17408 rows each). `gemv3n` now covers this site:

| | dispatches | device | wall | tok/s |
|---|---|---|---|---|
| before | 25060 | 4728 ms | 6007 ms | 3.329 |
| ffn pair fused | 23780 | 4723 ms | 5948 ms | 3.362 |

**The second FWHT stage, folded inline. `KINDI_FOLD_INLINE=1`.** For HB > 512 (the 27B,
HB=1024) the fold is two stages: `rmsfold`/`smfold`/`fold_fwht` do one unscaled 512-chunk
FWHT, then `fwht_hi` is a *separate dispatch* that combines the chunks and scales. All
three shaders already carry a generic-HB path that does the whole FWHT in one workgroup,
so passing HB removes that dispatch entirely. It works, and it is token-identical:

| | dispatches | device | wall | tok/s |
|---|---|---|---|---|
| two-stage (default) | 25060 | 4743 ms | 6038 ms | 3.312 |
| inline, no fwht_hi | 19920 | 4802 ms | 6102 ms | 3.277 |

5140 dispatches removed, 20% of the total, and wall got **worse**. Tokens identical
either way, so it is a valid path, but it is not a win and it is opt-in.

### What that pair of results means

Removing 1280 barriers between large GEMVs bought 59 ms. Removing 5140 barriers after
3.6 us shaders bought nothing, and cost 59 ms in device time because the generic HB path
halves the workgroup count for the FWHT (5 workgroups instead of 10 for D=5120).

So the per-barrier cost is not a constant. A barrier after a kernel that barely fills a
36-CU card has nothing in flight to drain and is nearly free. A barrier between two
kernels that both saturate the card forces a real drain and restart. The 44 us figure
from `KINDI_NO_BARRIER` is the *average over all barriers* when every one is removed, and
it does not predict what removing a subset will do.

This also explains the sibling result: eliding two barriers per GDN layer bought 50 ms,
because those sat between the large qkv/gate GEMVs.

**Practical consequence: fuse between big kernels, not around small ones.** GEMV fusion
is now exhausted, because the three same-input groups (q/k/v, qkv/gate, gate/up) are all
fused and the remaining 129 GEMV dispatches per token are `ssm_out`, `attn_output`,
`ffn_down` and the head, each of which is the sole consumer of its input and cannot be
fused with anything.

## Barriers: 1180 ms of wall time, and why narrowing them does not help

Wall was 6100 ms against device 4707 ms, leaving a 1393 ms gap that the host code could
not account for (host total is 14 ms). The gap is pipeline barriers.

`batch_barrier()` runs after every dispatch, so the command stream is
`dispatch, barrier, dispatch, barrier, ...` for 26660 dispatches. The default path is a
full `VkMemoryBarrier`: `src SHADER_WRITE`, `dst SHADER_READ | SHADER_WRITE`, stage
COMPUTE to COMPUTE.

Removing them entirely (`KINDI_NO_BARRIER=1`, diagnostic only, results are wrong):

| | wall | device | gap | tok/s |
|---|---|---|---|---|
| barriers on | 6099.8 ms | 4707 ms | 1392 ms | 3.279 |
| no barrier | 4919.4 ms | 4655 ms | 265 ms | 4.066 |

1180 ms, 19% of wall, 44.3 us per dispatch. Device time barely moved, and `g5ptq_byte`
actually got 12 us/op *slower* (562.7 to 574.7), so the win is entirely from not
draining between dispatches.

### Narrowing the barrier does nothing

The engine already had two narrower modes. Both were measured on the same 20-token run:

| mode | wall | device | gap | tok/s | tokens |
|---|---|---|---|---|---|
| default full `VkMemoryBarrier` | 6156 ms | 4760 ms | 1396 ms | 3.249 | ok |
| `KINDI_BARRIER=readonly` (no WAR/WAW) | 6163 ms | 4764 ms | 1400 ms | 3.245 | ok |
| `KINDI_BARRIER=buffer` (scoped to bound buffers) | 6139 ms | 4769 ms | 1370 ms | 3.258 | ok |

All within noise. The cost is the **existence** of a COMPUTE to COMPUTE pipeline
barrier, not its access masks or buffer scope: the producer has to drain before the
consumer can start regardless of which buffers the barrier names. So "use a scoped
`VkBufferMemoryBarrier` instead of a full `VkMemoryBarrier`" is not a fix here.

### Removing individual barriers scales, but the ceiling is ~694 ms

The one place a barrier is provably unnecessary is the sibling group after a norm: in a
GDN layer, `attn_qkv`, `attn_gate` and the gate scalars all read the norm output and
write disjoint outputs, so one barrier after the group is enough. Eliding the other two
(`KINDI_SIB_BARRIER=1`, opt-in):

| | wall | tok/s | tokens |
|---|---|---|---|
| default | 6132 ms | 3.261 | ok |
| sibling elision | 6082 ms | 3.288 | ok |

50 ms, 0.8%. That is 1920 of 26660 barriers, 7.2% of them, so the effect is roughly
proportional and the linear ceiling for barrier removal is about 694 ms rather than
1180. The rest of the `NO_BARRIER` win is cross-dispatch overlap filling the machine,
which cannot be reclaimed one barrier at a time. It is opt-in because 0.8% does not
justify dropping a memory-ordering guarantee by default.

**Conclusion: the remaining barrier cost needs fewer dispatches, not cheaper barriers.**
That means fusing same-input projections into one dispatch. The 4-trit path already does
this in `g4lut3`, but `gemv3` requires `scalebufs` and excludes `native_names`, so it
never fires on the 27B's PTQ1_0 tensors and the 27B pays a drain per projection.

## The 27B 5-trit GEMV: 2643 -> 563 us/op, 4.7x

`g5ptq` (single token) and `g5ptqM` (batched) were 98% of a 27B decode step. Five
changes, each measured, each with the tokens checked afterwards:

| step | us/op | GB/s | vs original |
|---|---|---|---|
| original, per-trit decode | 2643.1 | 5.44 | 1.00x |
| widened scale buffer + `3^n` lookup | 1469.1 | 11.15 | 1.80x |
| byte-granular decode (5 weights per byte load) | 721.2 | 22.7 | 3.66x |
| + shared-memory byte-to-trits LUT | 526.9 | 31.1 | 5.02x |
| + scale read from the block (drops the buffer) | 563.3 | 29.1 | 4.69x |
| 4 rows per 256-thread workgroup, tree kept | 568.5 | 30.4 | 4.65x |

End to end on the 27B PTQ1_0 + MTP, 20 tokens, greedy, serial:
**0.848 -> 3.29 tok/s** (3.88x), tokens byte-identical.

### What the measurements ruled out

The kernel is **instruction-bound, not bandwidth-bound**. Two experiments prove it:

- *Cache-resident weights*: the same shader, same instructions, same MACs, but every
  weight read redirected to two cache-resident blocks. 727.9 us/op against 711.5
  shipped, i.e. **no change**. There is no memory stall to recover.
- *Ablation*: removing the weight load entirely saved 71%, but that also lets the
  compiler fold the whole decode chain, so that number is decode ALU, not traffic.
  Removing 4 of the 5 digits from each byte saved 52.8%, i.e. the MACs.

So cooperative loading, more rows per workgroup and wider loads cannot help much.
An earlier attempt at wide `uint` loads per 16 bytes was **4.4x slower** (3113.9
us/op): the scalar byte path was already coalescing.

### What actually helped, and why

1. **The fp16 scale was decoded per weight.** An ablation put this at 25% of the
   kernel. It is constant for all 128 weights in a block.
2. **`3^n` was a data-dependent multiply chain**, up to four dependent multiplies per
   weight. A 5-entry table removed it: 29% of the kernel.
3. **One byte carries five weights.** PTQ1_0 packs five base-3 digits per byte
   (3^5 = 243 < 256), so a lane can load a byte once and emit five MACs instead of
   one. That is the 2.03x step.
4. **The byte-to-trit decode is a 1 KB shared table.** One LDS load replaces 15 ALU
   ops per byte.
5. **`float sub[4]` was a mistake.** An array indexed by a runtime loop variable
   lands in scratch memory on GCN. It made the T=2 batched kernel 5.7x slower than
   it should be (2726 us/op against ~478 expected). Specialising T=2 with scalars
   took it to 1050 us/op and the spec pass from 1162 to 554 ms.

### The scale buffer was removed on purpose

Widening the fp16 block scale into a float buffer is 8% faster than reading it from
the block with `unpackHalf2x16`. It also costs **0.79 GB** of VRAM, which pushed the
27B past the 8 GB limit once the MTP state snapshot (149.6 MiB) was allocated, so
`--spec` stopped working entirely. The 8% is not worth losing speculative decode:
the block read wins on total throughput.

### Still not 8 tok/s

8 tok/s on the 27B needs 125 ms/token, i.e. `g5ptq` at ~288 us/op. It is at 563. The
remaining lever is not the kernel: it is **401 launches per token**, and dispatch
fusion is the untested idea. `g4lut3` already fuses up to three same-input tensors in
one launch on the 4-trit path; the 5-trit path does not do this yet.

## The remaining wall: inter-dispatch barriers

446 dispatches per token, each followed by a `VkMemoryBarrier`
(`SHADER_WRITE -> SHADER_READ|WRITE`, `COMPUTE -> COMPUTE`), because a later shader
reads what the previous one wrote.

Measured cost, controlled experiment (`KINDI_NO_BARRIER=1`, wrong results by design):

| | wall | device | queue wait |
|---|---|---|---|
| with barriers | 3,861 ms | 417,764 us | 1,282 ms |
| no barriers | 2,795 ms | 315,529 us | 332 ms |

So the barrier costs **~48 us each, 21.6 ms/token, 76% of runtime.** Confirmed it is
the drain itself, not the access-mask scope  -  three variants all cost the same:

| `KINDI_BARRIER` | wall | queue wait |
|---|---|---|
| `memory` (global) | 3,860 ms | 1,282 ms |
| `readonly` (dst=READ only) | 3,771 ms | 1,283 ms |
| `buffer` (VkBufferMemoryBarrier) | 3,753 ms | 1,268 ms |

**The only fix is fewer dispatches**, which needs stage fusion far beyond a shader
change (e.g. folding the elementwise ops into the GEMV epilogue). Not attempted.

## Fixes that landed

1. **Dispatch batching.** `dispatch()` allocated a command buffer, submitted and waited
   on a fence per launch (17,852 syncs for 40 tokens). Now one buffer per token.
   Correctness: COMPUTE->COMPUTE barrier after each dispatch, per-dispatch descriptor
   sets from a rotating pool (updating a set a pending buffer references is UB), pool
   reset only after the queue is idle, flush before every host readback.
   `KINDI_NO_BATCH=1` restores the old path for A/B: 7,293 vs 4,148 ms.

2. **g4lut.** `g4simple` decoded one weight per iteration with `(byte/div)%3`  -  a
   variable integer divide, which GCN has no hardware for  -  walked stride-32 (poor
   coalescing) and idled half the 64-wide wavegroup. Replaced with a 256-entry shared
   LUT and 4 trits decoded per byte. 5.4x device time.

3. **g4lut2.** One wave per output row, 4 rows per 256-thread workgroup, `subgroupAdd`
   reduction (no shared memory, no barriers), all 64 lanes loading. 2.02x device time.

4. **gemv3.** Up to three same-input tensors in one launch (as `engine.py gemv` does).
   q/k/v, attn_qkv/attn_gate, ffn_gate/ffn_up. Dispatches 17,852 -> 15,476.

## Ruled out by measurement

- **Queue priority is not a lever.** LOW vs REALTIME, 4 runs each: device time
  identical (863.4 vs 863.9 ms). The GPU is idle 35% of wall time, so it is not
  contending. REALTIME is requested anyway (harmless; helps under desktop load).
  Note: `VkQueueFamilyGlobalPriorityPropertiesKHR` must be queried through the
  **queue-family** chain; in the device-properties chain it silently returns
  `priorityCount = 0` and you stay on MEDIUM.
- **All 36 CUs are in use.** ~4,711 waves per dispatch against 1,152 wave slots =
  ~4 waves/slot. Not core-starved; it was stalling inside the shader.
- **Not host-bound.** Host time is 12.4 ms of a 1,700 ms run.

## Traps available (`--trace`, `--trace-json`)

Per component and per shader: ops, %dev, device us, idle us, host us, bound MB, waves,
waves/CU. Plus synchronisation (submits, fence waits, ops/submit), allocation,
transfer, accuracy (token count + FNV hash), queue wait, and a measured instrumentation
floor (0.09 us/op).

A/B switches: `KINDI_NO_BATCH`, `KINDI_NO_BARRIER` (diagnostic only, wrong results),
`KINDI_BARRIER=memory|readonly|buffer`, `KINDI_QUEUE_PRIORITY`, `KINDI_DEVICE`.

## MTP speculative decode (k=1), 27B PTQ1_0 + MTP head

`--spec N`: the MTP draft head proposes one token, one T=2 trunk pass verifies it, and
the matching prefix is accepted. Output is token-identical to the plain run.

| prompt (20 tokens, greedy) | plain | spec | accept | speedup |
|---|---|---|---|---|
| `The sky is blue because` | 0.850 tok/s | 1.25 tok/s | 100% | 1.47x |
| `The sky is blue because` (after the 5-trit rewrite) | 3.29 tok/s | 3.42 tok/s | 100% | 1.04x |
| `def fibonacci(n):` | 0.829 tok/s | 1.20 tok/s | 100% | 1.45x |
| `Once upon a time there was` | 0.871 tok/s | 1.00 tok/s | 53.8% | 1.15x |

Break-even is ~1.27 tokens per pass, i.e. ~27% acceptance.

Per pass the loop body is 1485 ms, and the split is the thing to watch:

```
verify 1351  +  head 82  +  draft 52  +  commit 0  +  snapshot 0
```

The verify pass is 1351 ms for 2 tokens against 1177 ms for 1 token serially: **1.74x on
the trunk**, close to the 2x ceiling of one weight read per two tokens. The remaining
overhead is 134 ms of head and draft plus the per-pass sync gap.

Measured against the alternatives: a larger `KINDI_BATCH_MAX` (3500) is *slower* per pass
(1885 vs 1767 ms), so 2048 stands.

Component split (`--trace`, 4 passes): projections 96.0%, mtp 3.0%, recurrent 0.5%,
norms 0.3%. The draft head is 3% of device time; it is not the cost, the trunk is.

Baseline: `trace-spec-k1.json`.

## Still open

- 27B PQ2_0 is correct but still uses the per-dispatch path for PQ2_0 tensors
  (`g2pq`), so it does not benefit from gemv3 (that needs the 4-trit path).
- MTP k=2 (two drafts per verify) would amortise the 134 ms of per-pass head+draft over
  3 tokens instead of 2. Needs T=3 in `body_T` and a two-draft acceptance rule.
- The head GEMV (248320 x 5120) is the biggest avoidable cost: 60 ms/pass batched in the
  trunk plus 50 ms for the draft.
- The draft's argmax is read back to feed `embed()`, which forces a fence per pass. A
  device-token embedding decoder would remove it (the reference's `LLAMA_MTP_CHAIN`).
- Q4 KV cache would cut KV from 64.2 to ~18.3 KiB/token, taking the 27B from ~14k to
  ~50k context on 8 GB. Measured ceiling today: ctx 14336 fits, 16384 OOMs.
- Prefix caching for the server: KV needs no copy (position-indexed, append-only); only
  the GDN state needs snapshots, at ~150 MiB each on the 27B.
