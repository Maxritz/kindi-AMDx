import struct, numpy as np, os

PATH = r'C:\x\Ternary-Bonsai-2-27B-PQ2_0.gguf'
raw = open(PATH, 'rb').read()
FS = len(raw)

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
ver = u32(); nt = u64(); nkv = u64()
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
pos0 = p
ds_parsed = (p + align - 1)//align*align
tens = {}
for _ in range(nt):
    name = st().decode('utf-8','replace')
    nd = u32(); dims = [u64() for _ in range(nd)]
    ty = u32(); off = u64()
    tens[name] = (dims, ty, off)

def tsize(dims, ty):
    n = 1
    for d in dims: n *= d
    if ty == 142: return (n // 128) * 34
    if ty == 143: return (n // 128) * 28
    if ty == 0:   return n * 4
    if ty == 30:  return n * 2
    if ty == 8:   return n + (n // 32) * 4
    return None

last_name, last = max(tens.items(), key=lambda kv: kv[1][2])
lsz = tsize(last[0], last[1])
ds_derived = FS - lsz - last[2]
ds_derived = (ds_derived // align) * align

print('file size        =', FS)
print('pos0 (kv end)    =', pos0)
print('ds_parsed        =', ds_parsed, ' (pos0 rounded up by %d)' % (ds_parsed - pos0))
print('last tensor      =', last_name, 'type', last[1], 'off', last[2], 'size', lsz)
print('ds_derived       =', ds_derived)
print('DIFFERENCE       =', ds_derived - ds_parsed)
print()

for label, ds in (('parsed', ds_parsed), ('derived', ds_derived)):
    dims, ty, off = tens['output_norm.weight']
    n = dims[0]
    a = np.frombuffer(raw[ds+off : ds+off+n*4], dtype='<f4')
    ok = bool(np.all(np.isfinite(a)) and a.max() < 100 and a.min() > -100)
    print('%-8s ds=%-10d output_norm: min=%9.4f max=%9.4f mean=%8.4f  first3=%s  SANE=%s'
          % (label, ds, a.min(), a.max(), a.mean(), np.round(a[:3],4), ok))

print()
dims, ty, off = tens['token_embd.weight']
ne0, nrows = dims[0], dims[1]
bpr = ne0//128
for label, ds in (('parsed', ds_parsed), ('derived', ds_derived)):
    base = ds + off + 760*bpr*34
    blk = raw[base:base+34]
    d = float(np.frombuffer(blk[0:2], dtype='<f2')[0])
    qs = blk[2:34]
    v = [(((qs[j//4] >> ((j%4)*2)) & 3) - 1) * d for j in range(4)]
    print('%-8s row760 d=%.6f  first4=%s' % (label, d, np.round(v,5)))
print('reference node_0 row0 = -0.0128  0.0000  -0.0128  ...  0.0140')
