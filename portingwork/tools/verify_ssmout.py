"""Isolate the 27B ssm_out defect: compute W @ folded on the host and compare with the GPU.

If the host result matches the GPU -> the GEMV is right and the defect is the weight layout.
If it matches the reference -> the GPU GEMV is wrong.
"""
import struct, numpy as np

GGUF = r"C:\x\Ternary-Bonsai-2-27B-PQ2_0.gguf"

def rd_str(f):
    n = struct.unpack('<Q', f.read(8))[0]; return f.read(n).decode('utf-8','replace')
def rd_val(f,t):
    if t==0: return struct.unpack('<B',f.read(1))[0]
    if t==1: return struct.unpack('<b',f.read(1))[0]
    if t==2: return struct.unpack('<H',f.read(2))[0]
    if t==3: return struct.unpack('<h',f.read(2))[0]
    if t==4: return struct.unpack('<I',f.read(4))[0]
    if t==5: return struct.unpack('<i',f.read(4))[0]
    if t==6: return struct.unpack('<f',f.read(4))[0]
    if t==7: return struct.unpack('<?',f.read(1))[0]
    if t==8: return rd_str(f)
    if t==9:
        et=struct.unpack('<I',f.read(4))[0]; n=struct.unpack('<Q',f.read(8))[0]
        return [rd_val(f,et) for _ in range(n)]
    if t==10: return struct.unpack('<Q',f.read(8))[0]
    if t==11: return struct.unpack('<q',f.read(8))[0]
    if t==12: return struct.unpack('<d',f.read(8))[0]

f=open(GGUF,'rb'); f.read(4); struct.unpack('<I',f.read(4))
nt=struct.unpack('<Q',f.read(8))[0]; nkv=struct.unpack('<Q',f.read(8))[0]
kv={}
for _ in range(nkv):
    k=rd_str(f); t=struct.unpack('<I',f.read(4))[0]; kv[k]=rd_val(f,t)
tensors={}
for _ in range(nt):
    name=rd_str(f); nd=struct.unpack('<I',f.read(4))[0]
    dims=[struct.unpack('<Q',f.read(8))[0] for _ in range(nd)]
    ty=struct.unpack('<I',f.read(4))[0]; off=struct.unpack('<Q',f.read(8))[0]
    tensors[name]=(dims,ty,off)
kv_end=f.tell(); f.close()
import os
fsize=os.path.getsize(GGUF)

def tsize(dims,ty):
    n=1
    for d in dims: n*=d
    if ty==0: return n*4
    if ty==30: return n*2
    if ty==143: return (n//128)*28
    if ty in (140,141,142): return (n//128)*34
    return None

# data_start (validated on output_norm)
def sane(ds):
    dims,ty,off=tensors['output_norm.weight']; n=dims[0]
    with open(GGUF,'rb') as g:
        g.seek(ds+off); v=np.frombuffer(g.read(4*n),dtype='<f4')
    return np.isfinite(v).all() and np.abs(v).max()<100

ds_cands=[kv_end,(kv_end+31)//32*32]
for c in ds_cands:
    if sane(c): data_start=c; break
else:
    raise SystemExit('no sane data_start')
print('data_start =', data_start)

name='blk.0.ssm_out.weight'
dims,ty,off=tensors[name]
print(name,'dims',dims,'type',ty)
C=dims[0]; R=1
for d in dims[1:]: R*=d
print('  C(input)=%d R(rows)=%d'%(C,R))

bpr=C//128; rowbytes=bpr*34
with open(GGUF,'rb') as g:
    g.seek(data_start+off); raw=np.frombuffer(g.read(rowbytes*R),dtype=np.uint8)
raw=raw.reshape(R,rowbytes)

# vectorized PQ2_0 dequant: per 128-weight block, 34 bytes = fp16 d | 32 bytes qs (2 bits each)
blk = raw.reshape(R, bpr, 34)
d   = blk[:, :, 0:2].copy().view('<f2').astype(np.float32).reshape(R, bpr)   # (R,bpr)
qs  = blk[:, :, 2:34]                                           # (R, bpr, 32)
j   = np.arange(128)
q   = (qs[:, :, j >> 2] >> (2 * (j & 3))) & 3                   # (R, bpr, 128)
w   = (q.astype(np.float32) - 1.0) * d[:, :, None]              # (R, bpr, 128)
w   = w.reshape(R, C)
print('  dequantised', w.shape, 'max|w|=', np.abs(w).max())

folded=np.fromfile('dv_gdn_folded.bin',dtype=np.float32)
got=np.fromfile('dv_ssm_out.bin',dtype=np.float32)
print('  folded',folded.shape,'got',got.shape)

host = w @ folded
print()
print('=== ssm_out[0:3] ===')
print('  host  =', np.round(host[0:3],6))
print('  gpu   =', np.round(got[0:3],6))
print('  ref linear_attn_out-0 = -0.0204  0.0285  -0.0247')
err=np.abs(host-got).max()
print('  max |host-gpu| =', err)
print()
print('VERDICT:', 'GEMV OK -> weight layout ok, bug elsewhere' if err<1e-3 else 'GEMV/WRONG-LAYOUT')
