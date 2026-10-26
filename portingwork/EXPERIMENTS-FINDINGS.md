# Experiments and findings

Every measurement taken while porting Kindi to Vulkan on the RX 580, in the order it was
taken, including the ones that did not work. The negative results are the valuable part:
they are what stopped this from being a series of guesses.

Hardware and software: Radeon RX 580 (Polaris 10, gfx803), 36 CU x 4 SIMD x 8 wavefronts
= 1152 wave slots, wave size 64, 8 GB, 3.09 T lane-ops/s (2304 lanes at 1.34 GHz).
Windows, AMD proprietary driver 26.5.2, Vulkan 1.3.260. No ROCm, no containers.

## How the measurements were taken

`--trace` / `--trace-decode` / `--trace-json`, built into the engine. Device time comes
from a `vkCmdWriteTimestamp` pair around every dispatch (`timestampPeriod` 40 ns on this
card); host time from `QueryPerformanceCounter`; `idle = host - device`.

`--trace-decode` starts the table at the first decode step. Prefill and decode have
different shapes (one T-token pass against a chain of single-token passes), so a table
summing both answers neither question.

Two rules that mattered more than the tooling:

- **Never measure under `--trace` when the number is the point.** Prefill runs tens of
  thousands of dispatches; a timestamp pair each is enough overhead to distort what is
  being measured. All prefill figures come from the server, which reports prefill and
  decode separately with no tracing.
- **Never accept a kernel change without checking the tokens.** A faster wrong answer is
  worthless. Every experiment below compared emitted token ids, not text.

---

# 1. The 5-trit GEMV (27B)

`g5ptq` was 98% of a 27B decode step. Starting point: 2643.1 us/op, 5.44 GB/s, and the
whole model at 0.848 tok/s.

## 1.1 Ablation: where the time was

Each variant removed exactly one cost. Results wrong on purpose; this is attribution.

| variant | us/op | saved |
|---|---|---|
| baseline | 2614.7 | - |
| no fp16 decode in the loop | 1955.0 | 25% |
| + `3^n` lookup instead of a multiply chain | 1385.9 | 29% |
| + branchless accumulate | 1346.0 | 3% |

**Finding:** 54% of the kernel was two avoidable things. The fp16 block scale was decoded
once per weight when it is constant for all 128 weights in a block, and `3^n` was a
data-dependent chain of up to four dependent multiplies per weight.

## 1.2 Is it memory bound? No.

The decisive experiment. Same shader, same instructions, same MACs, but every weight read
redirected to two cache-resident blocks:

| | us/op |
|---|---|
| real weight reads | 711.5 |
| cache-resident reads | 727.9 |

**No change.** The kernel is entirely instruction-bound. This killed a whole class of
proposed fixes (cooperative loading, more bandwidth, prefetching) before any were written.

Wide `uint` loads (4 x 16 bytes per lane instead of 16 scalar byte loads) measured
**3113.9 us/op, 4.4x slower** - the scalar byte path was already coalescing.

## 1.3 What worked

| step | us/op | GB/s | vs original |
|---|---|---|---|
| widened scale buffer + `3^n` table | 1469.1 | 11.15 | 1.80x |
| byte-granular decode (5 weights per byte load) | 721.2 | 22.7 | 3.66x |
| + byte-to-trit LUT in shared memory | 526.9 | 31.1 | 5.02x |
| + scale read from the block (drops the buffer) | 563.3 | 29.1 | 4.69x |
| + 4 rows per 256-thread workgroup | 485.0 | 33.8 | 5.45x |

PTQ1_0 packs **five base-3 digits per byte** (3^5 = 243 < 256), so one byte load yields
five weights. That is the single biggest structural win, 2.03x.

The scale buffer was removed even though it was 8% faster: it cost 0.79 GB of VRAM, which
pushed the 27B past 8 GB once the MTP state snapshot (149.6 MiB) was allocated, so `--spec`
died with `Vulkan error -2`. Reading the fp16 scale from the block with `unpackHalf2x16`
is 0.93x and saves all of it.

## 1.4 The scale is one word, not two bytes

A block is 28 bytes and starts at a multiple of 28, and 28 % 4 == 0, so byte 26 of the
block is always the high half of word `(base/4 + 6)`:

```glsl
int sw = ((row * bpr + blk) * 7) + 6;
float s = unpackHalf2x16(w[sw] >> 16).x;
```

**1.04x.** Provable from the layout, not a heuristic.

## 1.5 The FMA chain is not the bottleneck

All five digit FMAs accumulate into one scalar, so in principle each waits on the
previous result and the chain is 20-30 cycles per byte. Unrolling and using four
independent accumulators measured **0.98x**, slightly worse.

**Finding:** the compiler was already interleaving loop iterations. The kernel is
issue-throughput bound, not latency-stalled. Nothing further can come from scheduling;
it has to be fewer instructions.

---

# 2. Barriers: 1180 ms of wall time

Wall was 6100 ms against device 4707 ms for 20 tokens. The 1393 ms gap was not host code
(host total 14 ms). It was the pipeline barrier after every dispatch - 26660 of them.

| | wall | device | gap | tok/s |
|---|---|---|---|---|
| barriers on | 6099.8 ms | 4707 ms | 1392 ms | 3.279 |
| no barrier (`KINDI_NO_BARRIER=1`, diagnostic) | 4919.4 ms | 4655 ms | 265 ms | 4.066 |

1180 ms, 19% of wall, 44.3 us per dispatch. Device time barely moved, and `g5ptq_byte` got
12 us/op **slower** - so the win is entirely from not draining between dispatches.

## 2.1 Narrowing the barrier does nothing

Three reviewers (three independent LLM agents) unanimously ranked "use a scoped
`VkBufferMemoryBarrier` instead of a full `VkMemoryBarrier`" as the first thing to try. It
was already implemented, so it was one line to test:

| mode | wall | gap | tok/s | tokens |
|---|---|---|---|---|
| default full `VkMemoryBarrier` | 6156 ms | 1396 ms | 3.249 | ok |
| `KINDI_BARRIER=readonly` (no WAR/WAW) | 6163 ms | 1400 ms | 3.245 | ok |
| `KINDI_BARRIER=buffer` (scoped to bound buffers) | 6139 ms | 1370 ms | 3.258 | ok |

**Zero, all within noise.** The cost is the *existence* of a COMPUTE-to-COMPUTE barrier,
not its access masks or buffer scope: the producer must drain before the consumer starts
regardless of which buffers the barrier names. This is a real correction to standard
advice, and it only shows up because it was measured.

## 2.2 Removing individual barriers scales, but the ceiling is low

In a GDN layer, `attn_qkv`, `attn_gate` and the gate scalars all read the norm output and
write disjoint buffers, so one barrier after the group suffices:

| | wall | tok/s |
|---|---|---|
| default | 6132 ms | 3.261 |
| sibling elision | 6082 ms | 3.288 |

0.8%, for 1920 of 26660 barriers (7.2% of them). The effect is proportional, so the linear
ceiling for barrier removal is ~694 ms, not 1180. The rest of the `NO_BARRIER` win is
cross-dispatch overlap filling the machine, which cannot be reclaimed one barrier at a
time. Shipped opt-in (`KINDI_SIB_BARRIER=1`).

## 2.3 Folding the second FWHT stage inline: no gain

For HB > 512 (the 27B, HB = 1024) the fold is two stages, and all three shaders already
have a generic-HB path that does the whole FWHT in one workgroup, so passing HB removes
the `fwht_hi` dispatch entirely:

| | dispatches | device | wall | tok/s |
|---|---|---|---|---|
| two-stage | 25060 | 4743 ms | 6038 ms | 3.312 |
| inline, no `fwht_hi` | 19920 | 4802 ms | 6102 ms | 3.277 |

5140 dispatches removed (20% of the total) and wall got **worse**. The generic path halves
the FWHT workgroup count, and that costs more than the dispatch saves. Opt-in
(`KINDI_FOLD_INLINE=1`).

**Finding, and it corrected an earlier wrong conclusion:** per-barrier cost is not a
constant. A barrier after a kernel that barely fills a 36-CU card has nothing in flight to
drain. The 44 us figure is the average over *all* barriers when every one is removed.

---

# 3. Fusion

## 3.1 Sibling projections

`g5ptq_byte3` / `g5ptq_byte3nr` compute up to three PTQ1_0 tensors sharing one input in
one dispatch. It fires on q/k/v and on qkv/gate.

| | dispatches | device | wall | tok/s |
|---|---|---|---|---|
| separate | 26660 | 4707 ms | 6114 ms | 3.271 |
| fused, per-byte buffer select | 25060 | 4789 ms | 6082 ms | 3.288 |
| fused, branch hoisted | 25060 | 4728 ms | 6007 ms | 3.329 |

**A mistake worth recording:** the first version selected the tensor per byte with
`(t==0) ? W0[i] : (t==1) ? W1[i] : W2[i]`, and measured **772 us/op against 563** - a net
loss on device time. That is the same mistake the abandoned `g4.comp` made. The row's
tensor is uniform across the workgroup, so the branch belongs outside the loop.

## 3.2 ffn_gate/ffn_up

| | dispatches | device | wall | tok/s |
|---|---|---|---|---|
| before | 25060 | 4728 ms | 6007 ms | 3.329 |
| ffn pair fused | 23780 | 4723 ms | 5948 ms | 3.362 |

---

# 4. Rows per wave (the largest win after the decode rewrite)

A wave64 is 64 lanes, so there is no half-wave to fill. What a wave needs is independent
work: with one row per wave every lane runs a single dependent accumulator and loads every
x element for that one row.

Two rows per wave share each x load:

```glsl
float xv = x[c];
accA += tlf[b5A + n] * xv;
accB += tlf[b5B + n] * xv;   // same xv
```

| kernel | 1 row/wave | 2 rows/wave | gain |
|---|---|---|---|
| `g5ptq_byte4t` -> `g5ptq_bytenr2` | 510.6 us/op | **343.2 us/op** | **1.49x** |
| `g5ptq_byte3t` -> `g5ptq_byte3nr2` | 918.3 us/op | **660.9 us/op** | **1.39x** |
| `g5ptqM` -> `g5ptqM2r2` (spec) | 1022.0 us/op | **449.2 us/op** | **2.28x** |

Selected from the device, not assumed. `vulkaninfo` reports `subgroupSize = 64` with
`minSubgroupSize = maxSubgroupSize = 64` on this card, so no width choice exists here, but
the shader is one source compiled at three widths and the engine picks from the wave size
it queries: wave128 -> 1 row, wave64 -> 2, wave32 -> 4.

**Not the same as the four-accumulator experiment (1.5), which measured 0.98x.** That
split one row's sum into four chains, doing the same MACs and only reordering them. This
does RPW times the MACs per x load, so work per instruction actually falls.

## 4.1 Two implementation traps

**Arrays indexed by a runtime loop variable land in scratch memory on GCN.** The first
version used `acc[RPW]`, and disassembly showed the same 9 float ops for RPW = 1, 2 and 4:
nothing had unrolled. Explicit scalars with `#if RPW >= n` gave 9, 18 and 36.

**GLSL bounds-checks a constant index even inside a preprocessor-dead branch.**
`shared float sh[NSUB][RPW][64]` at RPW = 1 is `[1][64]`, and the macro still names
`sh[sub][1]`, which fails to compile. Declared at a fixed width of 4 slots instead.

**Clamp, never return.** `barrier()` must be reached by every invocation in the workgroup.
The first version wrote `if (row >= R) return;` before the reduction - a whole-wave exit,
leaving the surviving waves waiting on a barrier the exited waves never reach. It worked
only because every 27B row count is divisible by the workgroup's row count.

## 4.2 Spec was left behind, then fixed

After the serial kernel got rows per wave, `--spec` became a **loss**: 3.74 tok/s against
serial's 5.19, because `g5ptqM` still ran one row per wave.

| | tok/s |
|---|---|
| serial, rows per wave | 5.19 |
| spec, before `g5ptqM2r2` | 3.74 |
| spec, after `g5ptqM2r2` | **6.31** |

---

# 5. Prefill

Prefill was **slower than decode** (3.8 against 5.1 tok/s), which is backwards: one weight
decode should feed several tokens. Three causes.

## 5.1 A correctness bug I introduced

`g5ptqM2r*` has exactly two token chains (it was written for the T=2 verify pass), and
chunked prefill passes T=4. Tokens 2 and 3 were never computed:

```
CHUNK=0  198 760 12386 19825 557 264 12401 11
CHUNK=4  561 10550 19825 557 66483 539 279 7413   <- wrong
```

I verified the spec path (T=2) and the serial path (T=1) but never that prefill (T=4) still
matched, and the selftest only exercised T=2 so it passed. Fixed by routing T>2 to a
kernel with a generic token loop.

## 5.2 Token-major x cost 3.16x

x is `[T][C]` with C = 6144, so the four token values for one weight are 24 KB apart: four
separate cache streams per weight byte. Measured by pointing all four chains at one
stream:

| | us/op | us/row-token |
|---|---|---|
| four streams | 1716.0 | 429.0 |
| one stream | **542.7** | **135.7** |

Fixed with `xpose.comp`, one transpose per norm into a column-major `[C][T]` buffer,
against five GEMVs per norm so it is cheap.

**Same mistake as 5.1 in a different form:** the transpose ran inside `body_T`, which the
speculative verify pass also uses with T = 2, but the T = 2 kernel reads token-major x.
Passing it the column-major buffer dropped draft acceptance to 5.3% and emitted garbage.
Both the transpose and the buffer choice are now gated on T == 4.

## 5.3 Sharing x across rows as well

Prefill shared x across tokens but not across rows. `g5ptq_pf2r2` does 2 rows x 4 tokens =
8 chains per lane, all fed by the same four x loads (column-major, so contiguous) and one
weight byte, trit lookup and scale.

| kernel | us/op | us/row-token |
|---|---|---|
| `g5ptq_pf` (1 row, 4 tokens) | 1234.6 | 308.7 |
| `g5ptq_pf2r` (2 rows, 4 tokens) | 727.8 | **90.9** |
| decode reference (2 rows, 1 token) | - | 122.2 |

Prefill is now more efficient per row-token than decode, which is the shape it should have
had from the start.

**Server prefill, 83-token prompt:** 5.5 tok/s serial, 3.8 chunked before this work,
**8.6 after**.

---

# 6. The 4-trit path (2B)

The 2B never had the treatment the 27B got. Three kernels were written.

| kernel | result |
|---|---|
| `g4lut2r` (single tensor, rows/wave, float LUT) | **1.52x on the kernel** (1574.2 -> 1033.6 us/op), but it is only 10% of 2B device time, so **0.7% end to end** |
| `g4lutMr` (4 tokens, column-major, rows/wave) | prefill 65.9 -> 65.0, **no gain** |
| `g4lut3r` (fused 3-tensor, rows/wave) | **DISABLED - produces wrong values** |

## 6.1 A binding-order bug

`g4lutM` declares `0=W 1=S 2=Y 3=X`. I had changed `gemvM`'s buffer vector to
`{codes, scales, x, out}` - the order the new kernel used - which fed x into the output
binding and out into the input binding. The 2B emitted garbage token ids. Both kernels now
use `0=W 1=S 2=Y 3=X` and one shared buffer order.

## 6.2 g4lut3r is wrong and stays disabled

The fused rows-per-wave 4-trit kernel compiles and runs, but the 2B emits garbage with it.
The row-space mapping is wrong somewhere and it was not found. The shader and pipeline stay
in the tree, gated off, rather than shipping a kernel that silently corrupts output.

**This is the biggest open item on the 2B:** `g4lut3` is 44% of its device time and still
one row per wave.

---

# 7. Vulkan instruction survey

What was tried, with verdicts.

| technique | result |
|---|---|
| `subgroupAdd` instead of the shared tree | **1.00x** - dead |
| `bitfieldExtract` instead of shift+and | **1.00x** - dead (and naive form sign-extends) |
| float LUT in LDS (no integer decode) | **1.09x - shipped** |
| `vec4` LDS loads, padded stride | **0.98x** - LDS bank conflicts |
| one-word scale load | **1.04x - shipped** |
| 4 independent accumulators | **0.98x** - not latency bound |
| wide `uint` loads per 16 B | **0.25x** - already coalescing |
| cache-resident weights | **1.02x** - proves not memory bound |
| `unpackHalf2x16` for the scale | already in use |

## 7.1 What the device actually offers

From `vulkaninfo` on this card:

| property | value | consequence |
|---|---|---|
| `subgroupSize` | 64, min = max = 64 | RPW 2 is correct; no width choice |
| `SUBGROUP_FEATURE_ARITHMETIC` | true | `subgroupAdd` available, but 1.00x |
| **`integerDotProduct8BitSignedAccelerated`** | **false** | **no dp4a on Polaris** - why the 4-trit path needs a LUT |
| `integerDotProduct16BitSignedAccelerated` | true | exists, but x is float; quantising breaks token-exactness |
| `storageBuffer8BitAccess` | true | direct byte indexing possible (untested) |
| `storageBuffer16BitAccess` | true | direct `uint16` scale load possible |
| `shaderFloat16` | false | no fp16 arithmetic; the LUT stays fp32 |
| `maxComputeSharedMemorySize` | 32768 | our 5 KB LUT is comfortable |

---

# 8. Ceilings, and why 2000 tok/s is not available

Every weight participates in one multiply-accumulate per token, so the arithmetic floor is
`weights / (lanes * clock)` = `weights / 3.09e12`.

| model | weights | FMA floor | absolute ceiling | measured prefill | % of ceiling |
|---|---|---|---|---|---|
| 2B PTQ1_0 | 2.0 G | 0.6 ms/token | **1544 tok/s** | 66.5 | 4.3% |
| 27B PTQ1_0 | 27.0 G | 8.7 ms/token | **114 tok/s** | 8.8 | 7.7% |

The gap is instruction count: our kernels do roughly 6-10 operations per weight where a
perfect one would do 1. A 2000 tok/s figure on a 2B would need about 30x this card's FP32
throughput, which is an RTX 4090 class GPU.

---

# 9. Final state

27B PTQ1_0 + MTP: **0.848 -> 5.19 tok/s** serial, **6.31** with speculative decode, prefill
1.7 -> 8.8. That is 6.1x on decode. Everything token-exact.

2B PTQ1_0: 36.95 decode, 66.5 prefill.

## Negative results, in one place

Things that were measured and did not work, so nobody repeats them:

1. Scoping the barrier to specific buffers - **0%**. The drain is structural.
2. Removing 20% of dispatches by folding the second FWHT stage inline - **0%**, and
   slightly worse.
3. `subgroupAdd` for the reduction - 1.00x.
4. `bitfieldExtract` for trit extraction - 1.00x.
5. vec4 LDS loads - 0.98x, bank conflicts.
6. Four accumulators to break the FMA chain - 0.98x, not latency bound.
7. Wide `uint` loads instead of scalar bytes - 0.25x.
8. Cooperative/cache-resident weight loading - 1.02x, not memory bound.
9. Per-byte dynamic buffer selection in the fused kernel - 0.68x (772 against 563 us/op).
10. `g4lut3r` on the 2B - wrong values, disabled.

## Open, in priority order

1. **27B PQ2_0 is 1.62 tok/s**, 3.2x slower than the PTQ1_0 file of the same model. Its
   `g2pq`/`g2pqM` never got the byte-granular decode, float LUT or rows per wave. Largest
   measured gap in the tree.
2. **`g4lut3r`** - 44% of 2B device time, still one row per wave, currently wrong.
3. **`g4lutM`** - 21% of 2B device time, no rows per wave.
4. `gdn_delta` on the 2B runs at 29 GB/s against 46 on the 27B, 6.7% of 2B device time.
5. 21 of 67 shaders are dead code.
