# 27B status — verified findings (2026-10-26 05:2x)

## Fixed and verified

### 1. `xp` activation workspace was undersized (ROOT CAUSE of the "fold bug")
`kindi_vk_decode.cpp:588` allocated `xp=Z(D)`, but `smfold` writes `NV*HK` floats into it.
- 2B: `NV*HK = 16*128 = 2048 == D` → no overflow, which is why the 2B always worked.
- 27B: `NV*HK = 48*128 = 6144 > D = 5120` → 1024-float overrun. The last Hadamard block of
  **every GDN layer** (indices 5120..6143) was never written, holding garbage `-6.2598534e18`.
- Fix: `work = max(D, NFF, NV*HK)` (mirrors `engine.py`: `work = max(self.D, self.NFF, self.NV*self.HK)`).
- Verified: `dv_gdn_folded.bin` now matches an independent host FWHT to **max abs err 2e-8**
  across all 6144 elements (was 6.26e18). `verify_fold.py`.

### 2. Output head hardcoded to `token_embd.weight`
`step()` used `gemv("token_embd.weight", xp, y)` for every model.
- 2B: `prism.hadamard.tied_output = True`, **no `output.weight`** → tied head, correct.
- 27B: `tied_output` absent, **`output.weight` [5120, 248320] PQ2_0 exists** → separate head.
- Fix: `head_name = codebufs.count("output.weight") ? "output.weight" : "token_embd.weight"`.

## Ruled out (measured, not assumed)

| suspect | result |
|---|---|
| Embedding / `deq_emb2` fold | **correct** — `embed_h = 0.0178 0.0010 0.0039` (ref `input_embed` identical) |
| handoff's "-0.5706 / -sqrt(HB)" claim | **stale** — predates the push-size 12->16 fix. `--raw-embed` returns the *post-fold unnormalised* values because the fold loop runs before the raw check; that is what fooled the previous session |
| `deq_emb2` push block | correct — SPIR-V shows `HB` loaded from push constant, `InverseSqrt`, `sg*(x*s)`; PC struct is 4 ints = 16 B, pipeline range 16 B |
| generic GQA path (27B Grp=6) | **correct** — 2B `--force-gen` produces identical tokens `314 279 69377 314 3000` |
| GDN swiglu pairing / `perm_rep` | **must stay 1**. Reference: `ops.cpp:11045 ik1 = iv1 % nek1` (k-head = `v_head % 16`, interleaved — already done by `gdn_delta.comp`); `qwen35.cpp:332 ggml_swiglu_split(gate, normalized)` is elementwise. An experimental `perm_rep=NV/NK` was tried and **reverted** |
| GDN raw swiglu vs ref `node_61` | exact: `silu(z)*norm = 6.4e-6, -0.002671, -0.002791` vs ref `0.0000, -0.0027, -0.0028` |
| `ssm_out` GEMV + PQ2_0 weight layout | exact vs host `W @ folded` (max err 3e-7). `verify_ssmout.py` |
| other runtime buffers | audited — all correctly sized (`sbuf`, `apart`, `fo6`, `ffg`, `ffu`, `att`, `qr`, `qn`, `outkv`, `gout`, `zv`) |

## Still open

27B PQ2_0 emits garbage (`)/()/()/(`), 2B still exact.
`gdn_folded` is now provably correct, yet `ssm_out = -0.0320 0.0153 0.0067` does not match the
handoff's `linear_attn_out-0 = -0.0204 0.0285 -0.0247`. No fold convention (10 tried,
`verify_conv.py`) reproduces that number from the verified swiglu and verified weights.

=> **The handoff reference table is not trustworthy.** It was already wrong about the embedding.
   Get a fresh reference before chasing this further:
   `llama-eval-callback.exe` / the `cb()` node dump on the *same* PQ2_0 file, first token only.

Next suspects, in order:
1. Fresh reference values (the current table may be from PTQ1_0 or the MTP variant).
2. K/V cache precision — handoff claims the 27B uses q8_0 K/V; the decoder stores fp16 (`store_h`).
3. `NATT`/`kvi` mapping across 64 layers, and `M[il]` per-layer state indexing.
4. `--ctx` fit logic (`budget = 3.4e9`, or `per_pos*ctx`) never checks real free VRAM.

## Working tree state
- `x\kindi_vk_decode.exe` = current build (2B exact, 27B wrong).
- Debug tooling added: `--dumpvec` (full-vector dumps, first occurrence only),
  `emb2_truth.py`, `verify_fold.py`, `verify_ssmout.py`, `verify_conv.py`.
- `pack/`, `pack27/` deleted — the decoder is pure GGUF (proved by running with them hidden).
