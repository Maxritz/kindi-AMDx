"""Verify the 27B GDN fold: gdn_folded == sg * FWHT_HB(silu(z)*gdn_norm) / sqrt(HB)?"""
import struct, numpy as np

def fwht(x):
    y = x.astype(np.float64).copy()
    n = y.size
    h = 1
    while h < n:
        y = y.reshape(-1, 2, h)
        a = y[:, 0, :].copy(); b = y[:, 1, :].copy()
        y[:, 0, :] = a + b; y[:, 1, :] = a - b
        y = y.reshape(-1)
        h *= 2
    return y

z    = np.fromfile('dv_z.bin',        dtype=np.float32)
norm = np.fromfile('dv_gdn_norm.bin', dtype=np.float32)
got  = np.fromfile('dv_gdn_folded.bin', dtype=np.float32)
print('sizes', z.size, norm.size, got.size)

silu = z / (1.0 + np.exp(-z.astype(np.float64)))
swiglu = silu * norm.astype(np.float64)
print('swiglu[0:3] =', np.round(swiglu[0:3], 7))
print('  ref node_61 (swiglu-0) = 0.0000 -0.0027 -0.0028')

# sign vector: 27B sign_widths = [5120, 6144, 17408]; the 6144 block is the GDN one
GGUF = r"C:\x\Ternary-Bonsai-2-27B-PQ2_0.gguf"
f = open(GGUF, 'rb'); f.read(4)
struct.unpack('<I', f.read(4)); nt = struct.unpack('<Q', f.read(8))[0]
nkv = struct.unpack('<Q', f.read(8))[0]
def rd_str():
    n = struct.unpack('<Q', f.read(8))[0]; return f.read(n).decode('utf-8','replace')
def rd_val(t):
    if t==0: return struct.unpack('<B',f.read(1))[0]
    if t==1: return struct.unpack('<b',f.read(1))[0]
    if t==2: return struct.unpack('<H',f.read(2))[0]
    if t==3: return struct.unpack('<h',f.read(2))[0]
    if t==4: return struct.unpack('<I',f.read(4))[0]
    if t==5: return struct.unpack('<i',f.read(4))[0]
    if t==6: return struct.unpack('<f',f.read(4))[0]
    if t==7: return struct.unpack('<?',f.read(1))[0]
    if t==8: return rd_str()
    if t==9:
        et=struct.unpack('<I',f.read(4))[0]; n=struct.unpack('<Q',f.read(8))[0]
        return [rd_val(et) for _ in range(n)]
    if t==10: return struct.unpack('<Q',f.read(8))[0]
    if t==11: return struct.unpack('<q',f.read(8))[0]
    if t==12: return struct.unpack('<d',f.read(8))[0]
kv={}
for _ in range(nkv):
    k=rd_str(); t=struct.unpack('<I',f.read(4))[0]; kv[k]=rd_val(t)
f.close()
sv = np.asarray(kv['prism.hadamard.sign_values'], dtype=np.float32)
widths = kv['prism.hadamard.sign_widths']
off=0; signs={}
for w in widths:
    signs[int(w)] = sv[off:off+int(w)]; off+=int(w)
sg = signs[6144]
print('sg[0:3] =', sg[0:3])

HB = int(kv['prism.hadamard.block_size'])
C = z.size
print('HB =', HB, 'C =', C)

# expected: per HB block, sg*swiglu then FWHT then /sqrt(HB)
exp = np.empty(C, dtype=np.float64)
for c0 in range(0, C, HB):
    blk = (sg[c0:c0+HB].astype(np.float64) * swiglu[c0:c0+HB])
    exp[c0:c0+HB] = fwht(blk) / np.sqrt(HB)

print()
print('=== expected vs GPU ===')
print('exp[0:6] =', np.round(exp[0:6], 6))
print('got[0:6] =', np.round(got[0:6], 6))
err = np.abs(exp - got.astype(np.float64))
print('max abs err =', err.max())
print('mean abs err=', err.mean())
nz = np.abs(exp) > 1e-6
if nz.any():
    rel = err[nz] / np.abs(exp[nz])
    print('max rel err =', rel.max(), ' median rel =', np.median(rel))
# ratio test
m = np.abs(got) > 1e-6
if m.any():
    r = exp[m] / got[m].astype(np.float64)
    print('ratio exp/got: median=%.6f  min=%.4f max=%.4f' % (np.median(r), r.min(), r.max()))
print()
print('VERDICT:', 'FOLD CORRECT' if err.max() < 1e-4 else 'FOLD WRONG')
