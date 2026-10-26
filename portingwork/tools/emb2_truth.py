"""Independent ground truth for the 27B PQ2_0 embedding fold.

Parses token_embd.weight straight out of the GGUF, dequantizes row 760 with the
PQ2_0 layout the shader uses, applies the normalized FWHT over HB blocks and the
explicit sign vector, and prints the first few values.

Compare:
  raw    -> ref node_0        = -0.0128  0.0000  -0.0128
  folded -> ref input_embed   =  0.0178  0.0010   0.0039
"""
import struct
import numpy as np

GGUF = r"C:\x\Ternary-Bonsai-2-27B-PQ2_0.gguf"
ROW = 760

def rd_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', 'replace')

def rd_val(f, t):
    if t == 0: return struct.unpack('<B', f.read(1))[0]
    if t == 1: return struct.unpack('<b', f.read(1))[0]
    if t == 2: return struct.unpack('<H', f.read(2))[0]
    if t == 3: return struct.unpack('<h', f.read(2))[0]
    if t == 4: return struct.unpack('<I', f.read(4))[0]
    if t == 5: return struct.unpack('<i', f.read(4))[0]
    if t == 6: return struct.unpack('<f', f.read(4))[0]
    if t == 7: return struct.unpack('<?', f.read(1))[0]
    if t == 8: return rd_str(f)
    if t == 9:
        et = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        return [rd_val(f, et) for _ in range(n)]
    if t == 10: return struct.unpack('<Q', f.read(8))[0]
    if t == 11: return struct.unpack('<q', f.read(8))[0]
    if t == 12: return struct.unpack('<d', f.read(8))[0]
    raise ValueError('kv type %d' % t)

f = open(GGUF, 'rb')
assert f.read(4) == b'GGUF'
ver = struct.unpack('<I', f.read(4))[0]
ntensors = struct.unpack('<Q', f.read(8))[0]
nkv = struct.unpack('<Q', f.read(8))[0]

kv = {}
for _ in range(nkv):
    k = rd_str(f)
    t = struct.unpack('<I', f.read(4))[0]
    kv[k] = rd_val(f, t)

tensors = {}
for _ in range(ntensors):
    name = rd_str(f)
    n_dims = struct.unpack('<I', f.read(4))[0]
    dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
    ty = struct.unpack('<I', f.read(4))[0]
    off = struct.unpack('<Q', f.read(8))[0]
    tensors[name] = (dims, ty, off)

kv_end = f.tell()
f.close()

import os
fsize = os.path.getsize(GGUF)

# data_start: same rule the engine uses (file size + last tensor), validated on output_norm
last = max(tensors.items(), key=lambda kv_: kv_[1][2])
def tsize(dims, ty, ne0):
    n = 1
    for d in dims: n *= d
    if ty == 0: return n * 4          # F32
    if ty == 30: return n * 2         # BF16
    if ty == 143: return (n // 128) * 28   # PTQ1_0
    if ty in (140, 141): return (n // 128) * 34  # PQ2_0 family
    return None

cands = set()
for ds in (kv_end, (kv_end + 31) // 32 * 32, fsize - tsize(*last[1], last[1][0][0]) if len(last[1][0]) == 2 else None):
    if ds is not None: cands.add(ds)

def sane(ds):
    dims, ty, off = tensors['output_norm.weight']
    n = dims[0]
    p = ds + off
    if p + 4 * n > fsize: return None
    with open(GGUF, 'rb') as g:
        g.seek(p)
        v = np.frombuffer(g.read(4 * n), dtype='<f4')
    if not np.all(np.isfinite(v)): return None
    return float(np.abs(v).max())

best = None
for ds in sorted(cands):
    m = sane(ds)
    if m is not None and m < 100:
        best = ds
        break
data_start = best
print('kv_end=%d  data_start=%d' % (kv_end, data_start))

dims, ty, off = tensors['token_embd.weight']
print('token_embd: dims=%s type=%d off=%d' % (dims, ty, off))
C = dims[0]
R = 1
for d in dims[1:]: R *= d
print('  C=%d R=%d' % (C, R))

bpr = C // 128
rowbytes = bpr * 34
with open(GGUF, 'rb') as g:
    g.seek(data_start + off + ROW * rowbytes)
    raw = g.read(rowbytes)

assert len(raw) == rowbytes, (len(raw), rowbytes)

w = np.empty(C, dtype=np.float32)
for b in range(bpr):
    blk = raw[b * 34:(b + 1) * 34]
    d = np.frombuffer(blk[0:2], dtype='<f2')[0].astype(np.float32)
    qs = np.frombuffer(blk[2:34], dtype=np.uint8)
    for j in range(128):
        q = (qs[j >> 2] >> (2 * (j & 3))) & 3
        w[b * 128 + j] = float(int(q) - 1) * float(d)

print('raw[0:8] =', np.round(w[0:8], 6))
print('  (ref node_0 expected -0.0128 0.0000 -0.0128)')

# normalized FWHT over HB blocks
HB = int(kv['prism.hadamard.block_size'])
print('HB =', HB)

def fwht(x):
    n = x.size
    y = x.astype(np.float64).copy()
    h = 1
    while h < n:
        y = y.reshape(-1, 2, h)
        a = y[:, 0, :].copy()
        b = y[:, 1, :].copy()
        y[:, 0, :] = a + b
        y[:, 1, :] = a - b
        y = y.reshape(-1)
        h *= 2
    return y

folded = np.empty(C, dtype=np.float64)
for c0 in range(0, C, HB):
    folded[c0:c0 + HB] = fwht(w[c0:c0 + HB]) / np.sqrt(HB)

sv = np.asarray(kv['prism.hadamard.sign_values'], dtype=np.float32)
widths = kv['prism.hadamard.sign_widths']
print('sign_widths =', widths, ' sign_values len =', sv.size)
sD = sv[0:widths[0]]

out = folded * sD
print()
print('=== HOST TRUTH (folded * signs) ===')
print('  [0:3] =', np.round(out[0:3], 6))
print('  ref input_embed expected 0.0178 0.0010 0.0039')
print()
print('=== variants (to identify the shader bug) ===')
print('  unnormalized * signs      [0:3] =', np.round(fwht(w[0:HB]) * sD[0:HB], 4)[0:3])
print('  normalized, no signs      [0:3] =', np.round(folded[0:3], 6))
print('  raw * signs               [0:3] =', np.round(w[0:3] * sD[0:3], 6))
print('  raw (no fold)             [0:3] =', np.round(w[0:3], 6))
