# SHARDA_NOTE — g5ptq_byte3nr_silu: expected host wiring (for Shard C)

New shader: `shaders/g5ptq_byte3nr_silu.comp` (+ RPW variants 1/2/4, wired
into `build.ps1` `$variants`). Shard A did NOT touch `kindi_vk_decode.cpp`;
everything below is for Shard C to implement.

## Mode select

- `R2 == 0` → silu-fused path: paired gate+up rows, SwiGLU folded in, single
  output written to **Y0** (fo6 layout, NFF rows). Y1/Y2 untouched.
- `R2 != 0` → bit-identical copy of the `byte3nr` 3-tensor path.

## Suggested decode-loop wiring (replaces gemv3n + silu_mul for FFN)

Today (decode, T=1, fold-free path — `kindi_vk_decode.cpp` ~L1841 + L1991):

```
gemv3n(xp, ffn_gate, &ffg, ffn_up, &ffu, "", nullptr)   // 1 dispatch
... silu_mul(ffg, ffu -> fo6-or-mulbuf, NFF) ...        // fold-free MTP path
smfold(ffg, ffu, sFb -> fo6)                             // folded main path
```

With the new kernel (fold-free / MTP-style SwiGLU only — see caveat):

```
Pipe psilu3 = mkpipe("g5ptq_byte3nr_silu{RPW}.comp.spv", 7, 16);  // same (7,16) as pg53nr*
bufs = { gateCodes, fo6, upCodes, dummyBuf, dummyBuf, dummyBuf, xp }
//       W0         Y0   W1       Y1        W2        Y2        X
pc = { NFF, NFF, 0, C };   // R2==0 selects the fused path
groups = (NFF + RPW - 1) / RPW; wg = 256 / g_wave;
dispatch(psilu3, q0, cp, (groups + wg - 1) / wg, 1, 1, &pc, 16);
```

- `dummyBuf` must be a valid (small) storage buffer — it is bound but never
  read/written in R2==0 mode. Reuse any existing 4-byte device buffer.
- Gate/up codes come from the existing `codebufs[...ffn_gate.weight]` /
  `codebufs[...ffn_up.weight]` entries; both native 5-trit path (`native_names`).
- Dispatch-count math mirrors `gemv3n`'s RPW branch (groups per tensor, one
  tensor's worth of groups here) — same formula Shard C already uses.

## Important caveat (folded main path)

The main decode path feeds `smfold` (silu × up × signs, then fold) into
`ffn_down`. This kernel fuses **only** silu(gate)×up → fo6 (the MTP /
fold-free SwiGLU shape). It does NOT fold in `sF` signs or the FWHT fold, so
it cannot replace the `smfold` call on the 2B folded path without a further
variant. Natural first user: the MTP block (`mtp_ffg/mtp_ffu → mtp_mul`
+ `silu_mul`), which is exactly gate/up GEMV + fold-free silu_mul.

## Token-exactness argument

- Per-row FMA chain is statement-for-statement the `DO_TENSOR` chain:
  same `xv` load feeds both accumulators, same trit `n` order
  (0..4 / 0..4 / 0..3 per byte class), same `acc += s * p` order.
  Gate row r ≡ `DO_TENSOR(W0)` row r; up row r ≡ `DO_TENSOR(W1)` row r.
- Silu epilogue is char-identical to `silu_mul.comp`:
  `out = (x / (1.0 + exp(-x))) * u`.
- Interleaving uses disjoint accumulators (gA..gD / uA..uD), so neither
  row's rounding chain disturbs the other.

## Verification done in Shard A

- `glslangValidator -V --target-env vulkan1.1 [-DRPW={1,2,4}]` → all 4
  artifacts compile (see log below). `build.ps1` shader block re-run green.
- No C++ changed; no inference run (per constraints).

Compile log (exit 0 each; stderr noise is PowerShell NativeCommandError
echo of the filename, not a compiler diagnostic):

- default (RPW=2): `g5ptq_byte3nr_silu.comp.spv` 57548 bytes
- RPW=1: `g5ptq_byte3nr_silu1.comp.spv` 57536 bytes
- RPW=2: `g5ptq_byte3nr_silu2.comp.spv` 57548 bytes
- RPW=4: `g5ptq_byte3nr_silu4.comp.spv` 57548 bytes
