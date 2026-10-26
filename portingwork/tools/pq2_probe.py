import struct, numpy as np

PATH = r'C:\x\Ternary-Bonsai-2-27B-PQ2_0.gguf'
raw = open(PATH, 'rb').read()

p = 0
def u32():
    global p
    v = struct.unpack_from('<I', raw, p)[0]; p += 4; return v
def u64():
    global p
    v = struct.unpack_from('<Q', raw, p)[0]; p += 8; return v
def st():
    global p
    n = u64(); v = raw[p:p+n]; p += n; return v
assert u32() == 0x46554747
u32(); nt = u64(); nkv = u64()
def skip(t):
    global p
    if t == 9:
        et = u32(); n = u64()
        for _ in range(n): skip(et)
    elif t == 8: st()
    elif t in (0,1,7): p += 1
    elif t in (2,3): p += 2
    elif t in (4,5,6): p += 4
    elif t in (10,11,12): p += 8
align = 32
for _ in range(nkv):
    k = st().decode('utf-8','replace'); t = u32()
    if k == 'general.alignment': align = u32()
    else: skip(t)
ds = (p + align - 1)//align*align
tens = {}
for _ in range(nt):
    name = st().decode('utf-8','replace')
    nd = u32(); dims = [u64() for _ in range(nd)]
    ty = u32(); off = u64()
    tens[name] = (dims, ty, off)

# ---------- STEP 1: validate data_start with an F32 tensor ----------
print('=== STEP 1: F32 sanity (validates data_start) ===')
dims, ty, off = tens['output_norm.weight']
n = dims[0]
a = np.frombuffer(raw[ds+off : ds+off+n*4], dtype='<f4')
print('output_norm.weight  type=%d n=%d  min=%.4f max=%.4f mean=%.4f' % (ty, n, a.min(), a.max(), a.mean()))
print('  first 6:', np.round(a[:6], 5))
sane = bool(np.all(np.isfinite(a)) and a.max() < 100 and a.min() > -100)
print('  SANE:', sane)

# ---------- STEP 2: aligned scan of the embedding region ----------
print()
print('=== STEP 2: 34-byte-aligned scan for the reference block ===')
dims, ty, off = tens['token_embd.weight']
ne0, nrows = dims[0], dims[1]
bpr = ne0 // 128
start = ds + off
nblk = nrows * bpr
print('region %d .. %d  blocks=%d  bytes=%d' % (start, start + nblk*34, nblk, nblk*34))

region = np.frombuffer(raw, dtype=np.uint8, count=nblk*34, offset=start).reshape(nblk, 34)
d = region[:, 0:2].copy().view('<f2').reshape(-1).astype(np.float32)
b2 = region[:, 2]
q0 = (b2 >> 0) & 3
q1 = (b2 >> 2) & 3
q2 = (b2 >> 4) & 3

# reference node_0 row0 = -0.0128, 0.0000, -0.0128  -> q = (0,1,0), d = 0.0128
m = (d > 0.0120) & (d < 0.0136) & (q0 == 0) & (q1 == 1) & (q2 == 0)
hits = np.nonzero(m)[0]
print('aligned matches:', len(hits))
for h in hits[:8]:
    print('   block %d  row=%d  blkInRow=%d  d=%.6f' % (h, h//bpr, h%bpr, d[h]))
if len(hits):
    r0 = hits[0] // bpr
    print()
    print('>>> FIRST MATCH is row %d (expected row 760 if my stride were right)' % r0)
    print('>>> implied row stride = %d blocks = %d bytes' % (bpr, bpr*34))
