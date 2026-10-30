/* gemma4_tc.cu - the Gemma 4 prompt GEMM on int8 tensor cores (mma.sync .s8, sm_80+), appended to gemma4.cu.

   The tile and the shared-memory decode are the dp4a GEMM's (g4_mmq): a block of 256 threads computes 64 weight
   rows x 64 tokens; per 256-column block it decodes the 16 runs of 16 int8 weights of each row (with the run's scale
   and minimum) and copies the int8 input. Warp w takes rows 16 (w & 3) .. +15 and tokens 32 (w >> 2) .. +31: four
   m16 x n8 tiles. Each run gives an exact int32 per tile element (m16n8k16); for the types whose two runs of 32
   columns share one scale (IQ4_XS, Q8_0, Q5_K) one m16n8k32 covers both runs. The int32 is scaled by the weight
   scale and the input scale and added in fp32, run after run, as in the dp4a GEMM, so both give the same numbers. */

#define G4_MMA16(c, a0, a1, b0) \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};" \
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(b0))
#define G4_MMA32(c, a0, a1, a2, a3, b0, b1) \
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

template<int TYPE> __device__ __forceinline__ void g4_tcmm(const unsigned char* w,const signed char* xq,const float* xs,
        float* y,int rows,int cols,int nt,int ldy) {
    const int MINS=(TYPE==10||TYPE==13);
    const int PAIR=(TYPE==23||TYPE==123||TYPE==8||TYPE==108||TYPE==13);
    __shared__ uint4 W[64][17];                    // [row][run] + one pad run: the fragment loads hit 32 banks
    __shared__ uint4 X[64][17];                    // [token][16-column group] + pad
    __shared__ float WS[64][16],WM[MINS?64:1][16],XS[64][8];
    __shared__ short XSUM[MINS?64:1][16];
    int tid=threadIdx.x,lane=tid&31,warp=tid>>5,gid=lane>>2,t4=lane&3;
    int row0=blockIdx.x*64,tok0=blockIdx.y*64,nb=cols>>8,mr=16*(warp&3),nc=32*(warp>>2);
    float acc[4][4];
#pragma unroll
    for(int j=0;j<4;j++)
#pragma unroll
        for(int i=0;i<4;i++)acc[j][i]=0.0f;
    for(int b=0;b<nb;b++) {
#pragma unroll
        for(int k=0;k<4;k++) {                     // weights: 1024 runs, four per thread
            int idx=tid+k*256,r=idx>>4,u=idx&15,row=min(row0+r,rows-1);
            uint4 q; float sc,mn;
            g4_load_run<TYPE>(w,rows,cols,nb,row,b,u,q,sc,mn);
            W[r][u]=q; WS[r][u]=sc; if(MINS)WM[r][u]=mn;
        }
#pragma unroll
        for(int k=0;k<4;k++) {
            int idx=tid+k*256,t=idx>>4,c=idx&15,tok=tok0+t;
            X[t][c]=tok<nt?*reinterpret_cast<const uint4*>(xq+(size_t)tok*cols+b*256+c*16):make_uint4(0u,0u,0u,0u);
        }
        for(int idx=tid;idx<512;idx+=256){int t=idx>>3,g=idx&7,tok=tok0+t;XS[t][g]=tok<nt?xs[((size_t)tok*cols+b*256)/32+g]:0.0f;}
        __syncthreads();
        if(MINS) {
#pragma unroll
            for(int k=0;k<4;k++){int idx=tid+k*256,t=idx>>4,u=idx&15;uint4 v=X[t][g4_run_col<TYPE>(u)>>4];
                XSUM[t][u]=(short)g4_dp4a((int)v.w,0x01010101,g4_dp4a((int)v.z,0x01010101,g4_dp4a((int)v.y,0x01010101,g4_dp4a((int)v.x,0x01010101,0))));}
            __syncthreads();
        }
        const int ra=mr+gid,rb=mr+gid+8;
#pragma unroll
        for(int u=0;u<16;u+=(PAIR?2:1)) {
            int c=g4_run_col<TYPE>(u)>>4,g=c>>1;
            unsigned int a0=g4_wk(W[ra][u],t4),a1=g4_wk(W[rb][u],t4),a2=0,a3=0;
            if(PAIR){a2=g4_wk(W[ra][u+1],t4);a3=g4_wk(W[rb][u+1],t4);}
            float wsa=WS[ra][u],wsb=WS[rb][u],wma=0.0f,wmb=0.0f;
            if(MINS){wma=WM[ra][u];wmb=WM[rb][u];}
#pragma unroll
            for(int j=0;j<4;j++) {                 // four n8 tiles: tokens nc + 8 j .. +7
                int tb=nc+8*j;
                unsigned int b0=g4_wk(X[tb+gid][c],t4),b1=PAIR?g4_wk(X[tb+gid][c+1],t4):0u;
                int cc[4]={0,0,0,0};
                if(PAIR)G4_MMA32(cc,a0,a1,a2,a3,b0,b1); else G4_MMA16(cc,a0,a1,b0);
                // c0,c1: row ra, tokens tb + 2 t4 (+1); c2,c3: row rb, the same tokens
                int ta=tb+2*t4,tc=ta+1;
                float xa=XS[ta][g],xc=XS[tc][g];
                float f0=(float)cc[0]*wsa,f1=(float)cc[1]*wsa,f2=(float)cc[2]*wsb,f3=(float)cc[3]*wsb;
                if(MINS) {
                    float sa=(float)XSUM[ta][u]+(PAIR?(float)XSUM[ta][u+1]:0.0f),sc2=(float)XSUM[tc][u]+(PAIR?(float)XSUM[tc][u+1]:0.0f);
                    f0-=sa*wma; f1-=sc2*wma; f2-=sa*wmb; f3-=sc2*wmb;
                }
                acc[j][0]=fmaf(f0,xa,acc[j][0]); acc[j][1]=fmaf(f1,xc,acc[j][1]);
                acc[j][2]=fmaf(f2,xa,acc[j][2]); acc[j][3]=fmaf(f3,xc,acc[j][3]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for(int j=0;j<4;j++) {
        int tb=nc+8*j,ta=tok0+tb+2*t4,ra=row0+mr+gid,rb=ra+8;
        if(ra<rows&&ta<nt)y[(size_t)ta*ldy+ra]=acc[j][0];
        if(ra<rows&&ta+1<nt)y[(size_t)(ta+1)*ldy+ra]=acc[j][1];
        if(rb<rows&&ta<nt)y[(size_t)ta*ldy+rb]=acc[j][2];
        if(rb<rows&&ta+1<nt)y[(size_t)(ta+1)*ldy+rb]=acc[j][3];
    }
}
#define G4_TCMM(T) extern "C" __global__ void __launch_bounds__(256) k_g4_tcmm_t##T(const unsigned char* w,const signed char* xq, \
    const float* xs,float* y,int rows,int cols,int nt,int ldy){ g4_tcmm<T>(w,xq,xs,y,rows,cols,nt,ldy); }
G4_TCMM(8) G4_TCMM(10) G4_TCMM(11) G4_TCMM(13) G4_TCMM(14) G4_TCMM(23) G4_TCMM(123) G4_TCMM(111) G4_TCMM(114) G4_TCMM(108)
