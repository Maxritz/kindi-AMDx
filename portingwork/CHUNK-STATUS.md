# Chunked prefill — complete

## Results

`KINDI_CHUNK=N` batches prefill over N tokens per pass. Off by default.

### Prefill, 144-token prompt, n=1

| model | CHUNK=0 | CHUNK=2 | CHUNK=4 | speedup |
|---|---|---|---|---|
| 2B (4-trit) | 6,122 ms | 5,191 ms | 4,538 ms | **1.35x** |
| 27B (PQ2_0) | 92,744 ms | 69,488 ms | 56,056 ms | **1.65x** |

### Correctness (token-exact at every chunk size)

```
2B  CHUNK=1/2/4 : 314 279 69377 314 3000
27B CHUNK=1/2/4 : 314 12870 60845 69377 13 1061 369 264
serial 2B       :  of the scattering of light
serial 27B      :  of Rayleigh scattering. This is a
27B PTQ1_0+MTP  :  of Rayleigh scattering. This is a
decode           : 36.3 tok/s (unchanged - decode is always T=1)
```

### Selftests

```
2B : selftest 4-trit T=2: max|y1-y2[0]|=0.000e+00  max|y1-y2[1]|=0.000e+00  PASS
27B: selftest PQ2_0 T=2: max|y1-y2[0]|=2.384e-07  max|y1-y2[1]|=2.384e-07  PASS
```

## Bugs found (all fixed)

### 1. Stale pipeline push-constant sizes — six of them
A `VkPushConstantRange` smaller than the shader's declared block silently supplies
garbage for the trailing fields, and it is invisible when the garbage happens to be
multiplied by zero. That is exactly why `CHUNK=1` passed and `CHUNK>1` failed.

| shader | needed | had |
|---|---|---|
| `rmsfold_T` | 24 | 16 |
| `smfold_T` | 32 | 24 |
| `gdn_delta_T` | 28 | 24 |
| `gdn_gates_T` | 12 | 8 |
| `l2w_T` | 20 | 12 |
| `g4lutM` | 16 | 12 |

**`mkpipe` now derives the push size from the SPIR-V** (`OpMemberDecorate ... Offset`),
so this class of bug cannot recur.

### 2. `g2pqM` destroyed coalescing (8x slowdown)
First version gave each lane a whole 128-weight block. Consecutive lanes then read
addresses 34 bytes apart — uncoalesced. Measured **10.4 ms/op vs 1.3 ms** for `g2pq`.
Rewritten to keep `g2pq`'s lane-over-C mapping and only add the token loop:
**1.70 ms for 2 tokens vs 1.30 ms for 1** = 1.54x better per token.

### 3. `body_T` used the wrong embedding decoder
It called `deq_emb3` unconditionally; the 27B's embedding is PQ2_0 and needs
`deq_emb2`. Wrong at every chunk size for the 27B, invisible on the 2B. Now branches
on format exactly like the serial `embed()`.

### 4. Two different token strides per kernel
`hT`/`xnT` are D-strided; `xpT` is work-strided (`work = max(D, NFF, NV*HK)`);
`zvT`/`outT` are NV*HK-strided; `ffgT`/`fo6T` are NFF-strided. One stride cannot serve
both input and output, so `rmsfold_T`/`smfold_T` take `stride` + `ostride`, `l2w_T`
takes `stride` + `ostride`, and `gemvM`/`gemvM2pq` take `xstride`.

### 5. `gdn_delta_T` wrote with the wrong output stride
Used the `CONV_CH` value stride into a `[T][NV*HK]` buffer, overrunning token 1.

### 6. Embed wrote to the wrong buffer
The dequantised row went into `hT` and was then overwritten by a copy of the stale `h`,
so the first token of every chunk was zero.

### 7. Unconditional residual swap
`rmsfold` only swaps `h`/`h2` when it added a residual (`pending>0`). `body_T` swapped
always, so layer 0 read an untouched buffer.

### 8. `pending` not reset after embed
First layer added a stale residual.

### 9. `gdn_gates_T` derived the head count from `gl_NumWorkGroups.x / T`
Now passed explicitly.

## Components

| piece | status |
|---|---|
| `g4lutM` — 4-trit batched GEMV | ✅ bit-identical at T=2 |
| `g2pqM` — PQ2_0 batched GEMV | ✅ 2.4e-07 at T=2 |
| `rmsfold_T`, `smfold_T`, `gdn_conv_T`, `gdn_delta_T`, `gdn_gates_T`, `l2w_T` | ✅ |
| `rmsw`, `rope` | no variant needed (row count x T) |
| `copyB` — in-batch device copy, no flush | ✅ |
| `body_T` + `KINDI_CHUNK` prefill loop with serial fallback | ✅ |
| `--selftest` — both formats, T=2 vs T=1 | ✅ PASS |
| attention | per token on the verified path (2% of device time) |

## Next

1. A T-row attention kernel (`att_flash16_T`) to remove the per-token copies. This is
   now the main limiter: chunked prefill still does attention serially per token.
2. MTP: draft + verify on the batched forward, needing `wb=0` state semantics in
   `gdn_delta_T` and a conv replay for the committed prefix.
