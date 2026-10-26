# MTP speculative decode — working, k=1

## Result

`--spec N` runs the MTP draft head one token ahead and verifies with one T=2 trunk pass.
Output is **token-identical to the plain run**; the speedup tracks acceptance.

| prompt (27B PTQ1_0 + MTP, 20 tokens, greedy) | plain | spec | accept | speedup |
|---|---|---|---|---|
| `The sky is blue because` | 0.850 tok/s | **1.25 tok/s** | 100% | **1.47x** |
| `def fibonacci(n):` | 0.829 tok/s | **1.20 tok/s** | 100% | **1.45x** |
| `Once upon a time there was` | 0.871 tok/s | **1.00 tok/s** | 53.8% | **1.15x** |

All three produced the same token ids as the plain run. Break-even is about 1.27
tokens per pass, so the draft needs roughly 27% acceptance to pay for itself; above
that it scales with acceptance.

`--mtp-accept N` remains as the acceptance oracle: it drafts against the serial trunk
and counts hits without changing the output path (17/20 = 85% on the sky prompt).

## How the verify pass works

Per pass, at position `p` with `last = x_p`:

1. snapshot the GDN state (`M` and `conv` for every recurrent layer) — 149.6 MiB on
   the 27B, ~1 ms
2. `body_T([last, draft], p, force)` — one T=2 pass, every GEMV reads its weights once
   for both tokens
3. `head_rows(2)` — residual add + final norm + fold for both rows, then **one** head
   GEMV over the 2 rows, then a per-row argmax, then one readback
4. `j = (r0 == draft)`; `ntok = 1 + j`
5. if `ntok < 2`, restore the snapshot and replay the first `ntok` tokens through the
   GDN layers only
6. draft the next token from the committed hidden state

The commit needs no new shader: `gdn_conv_T` and `gdn_delta_T` are already serial in
token order, so re-running them with `T = ntok` on the restored state is bit-identical
to having run the pass with `T = ntok`.

The KV cache needs no rollback. It is append-only and position-indexed, so a rejected
row is overwritten by the next pass before it is read.

## Where the time goes (27B, `--trace`, 4 passes)

```
component            ops   %dev       dev us
projections         3460  96.0%   9329111.68
mtp                  325   3.0%    291488.64
recurrent           1152   0.5%     47684.96
norms               5330   0.3%     32757.60
```

Per pass the loop body is 1485 ms, of which:

```
verify 1351  +  head 82  +  draft 52  +  commit 0  +  snapshot 0
```

The verify pass is the whole cost: 1351 ms for 2 tokens against 1177 ms for 1 token
serially, i.e. **1.74x on the trunk**, close to the 2x ceiling that one weight read per
two tokens allows. Everything else is overhead to keep small.

`spec:g5ptqM` (the batched trunk GEMVs) is 1116 ms of that 1351. `spec:g5ptq` is 144
ms/pass — the 16 attention layers, which still run q/k/v/output per token because
attention is per token by construction.

## The four bugs, and why the first two hid

Each produced plausible numbers. The `--mtp-accept` oracle was what made them visible,
and a force-reject probe (`KINDI_SPEC_FORCE_REJECT=1`) was what separated the verify
pass from the commit replay.

### 1. `block_q8_0` layout (this one also invalidated the old Q8 selftest)

GGML's `block_q8_0` is `{ ggml_half d; int8_t qs[32]; }` — **the scale comes first**.
The loader read bytes 0..31 as codes and bytes 32..33 as the scale, shifting every
weight by two bytes and turning the last two codes into a "scale", which can be a NaN
bit pattern.

The selftest could not catch it: it read the device code/scale buffers back and compared
against them, so it agreed with the loader by construction and reported
`max abs err 0.0000e+00` on a two-byte-shifted load. It now builds the reference from
the GGUF bytes and reports 1.77e-07.

### 2. `copyB` had no compute -> transfer barrier

`concat = [e_norm ; h_norm]` was two `copyB` calls reading buffers a dispatch had just
written, so they copied stale uninitialised device memory and `eh_proj` saw NaN. Fixed
by writing the two norm halves straight into the concat buffer at byte offsets, and
`copyB` now emits the barrier (also covering `TRANSFER_WRITE`, since it writes its
destination).

### 3. The head input and the head basis

- **h_input.** `t_h_nextn` is assigned in the trunk graph *after*
  `build_norm(cur, output_norm)` (`qwen35.cpp:287-290`), so the draft consumes
  `rms(h, output_norm)` and then applies `nextn.hnorm` on top. The comment in
  `llama-graph.h` ("before final output norm") is wrong; the code is the authority.
  Feeding the raw residual gave logits with the target token ranked 1894-68181 of
  248320.
- **The head is folded.** `output.weight` is one of the 401 entries in
  `prism.hadamard.weight_names`, so `build_lora_mm` folds the activation before the
  GEMV. `blk.64.*` are **not** in that list. The draft's head needs the fold the
  trunk's own head gets; the rest of the MTP block must not fold.

Confirmed offline before touching the engine: with both fixed, the reference argmax at
positions 2 and 4 equals the trunk's next token exactly (6105, 314). Position 3 misses —
an 85% proposer, not an oracle.

### 4. `body_T` overwrote its GDN scratch per layer

`body_T` used one set of `qkvT`/`csT`/`qkbufT`/`ggT`/`gbT` for every GDN layer, so
after a pass only the **last** layer's inputs survived. The commit replay then fed
layer 63's qkv/qk/cs to all 48 layers, and the recurrent state drifted after the first
rejected draft. `spec.cu` keeps a set per layer (`b["L"] = {il: {...}}`) for exactly
this reason. Now per-layer buffers, allocated only for `--spec` because the 27B PQ2_0
(7.2 GB of weights) has no room for anything it does not need.

That was invisible in a 100%-acceptance run, which never replays. The force-reject probe
is what exposed it: forcing every draft to be rejected must reproduce the plain run
exactly, and it did not.

## Other things that changed

- **`g5ptqM.comp`** — batched native PTQ1_0 GEMV. Mirrors `g5ptq` exactly (same 64-wide
  shared reduction), so T=1 matches it bit for bit and row t of a T>1 launch matches a
  separate T=1 launch. Selftest: `max|y1-y2[0]|=0.000e+00`. Without it the 27B's T path
  could not run at all.
- **`body_T` embedding** — goes through `embed()` per token, which removed the bail-out
  for native PTQ1_0. The 27B now uses chunked prefill instead of falling back to serial
  (`prefill: 1 chunked passes, 0 serial tokens`).
- **Attention-layer FFN** — `body_T` used to run gate/up/down per token for the 16
  attention layers while the 48 GDN layers batched theirs. Now it falls through to the
  shared batched FFN.
- **`head_rows`** — one batched head GEMV over the rows instead of one per row: 123 -> 82
  ms/pass. Logits go to a dedicated `logT` buffer because writing them into `yT`
  clobbered the residual that `row_h` reads.
- **Snapshot barriers** — 96 `copyB` calls each emitted a full memory barrier; batched
  into one compute->transfer and one transfer->compute.

## Instrumentation

`trace.h` gained a phase tag, so every draft dispatch is attributed to a separate `mtp`
component and to `mtp:<shader>` rows, and the verify pass to `spec:`. Without it the
draft's cost is invisible in the table used to decide what to optimise.

The spec loop reports its own phase breakdown per pass (verify / head / commit / draft /
snapshot) so the next change can be aimed at the phase that actually dominates.

`--trace` / `--trace-json` give the per-op table; `trace-spec-k1.json` is the committed
baseline for this configuration.

## Still open

- **k=2.** A second draft would amortise the per-pass overhead over 3 tokens, but it
  needs T=3 in `body_T` and a two-draft acceptance rule. Worth about 1.6x if the second
  draft is accepted often enough; the per-pass overhead (134 ms of head+draft) is what it
  would buy back.
- **The head GEMV is the single biggest avoidable cost**: 60 ms/pass batched in the
  trunk plus 50 ms for the draft, on a 248320 x 5120 matrix. A top-k draft head (the
  reference supports `nextn.draft_head`) would shrink the draft side.
- **Host-side token feedback.** The draft's argmax is read back to feed `embed()`, which
  forces a fence per pass. The reference's chained mode keeps the token on the device
  (`LLAMA_MTP_CHAIN`); the same needs a device-token embedding decoder here.
