# Kindi Vulkan — performance report

GPU: RX 580, 36 CU x 4 SIMD x 8 wavefronts = 1152 wave slots, wave size 64
Workload: `--prompt "The sky is blue because" --temp 0`
Correctness gate: 2B token ids `314 279 69377 314 3000`, 27B ` of Rayleigh scattering.`
(`x\kindi_cpu.exe` cross-checks the 2B identically)

## Headline

| stage | 2B tokens/s | change |
|---|---|---|
| original (per-dispatch fence, g4simple) | ~3.3 | — |
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
the drain itself, not the access-mask scope — three variants all cost the same:

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

2. **g4lut.** `g4simple` decoded one weight per iteration with `(byte/div)%3` — a
   variable integer divide, which GCN has no hardware for — walked stride-32 (poor
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
