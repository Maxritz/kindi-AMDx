// pq2check.cpp — verify PQ2_0 / PTQ1_0 block decode using the SAME parse logic as
// kindi_vk_decode.cpp, but on the CPU (no Vulkan, no VRAM). Prints the raw embedding
// row for a token so it can be compared against llama-eval-callback's node_0.
#include <windows.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <functional>
#include <cmath>

static float h2f(uint16_t h){
    uint32_t sign=(h&0x8000)<<16, exp=(h>>10)&0x1F, mant=h&0x3FF, f;
    if(exp==0){ if(mant==0) f=sign; else { int e=1; while(!(mant&0x400)){mant<<=1;--e;} mant&=0x3FF; f=sign|((e+127-15)<<23)|(mant<<13);} }
    else if(exp==31) f=sign|0x7F800000|(mant<<13);
    else f=sign|((exp+127-15)<<23)|(mant<<13);
    float o; memcpy(&o,&f,4); return o;
}

struct GT{ std::string name; std::vector<uint64_t> dims; uint32_t type; uint64_t offset; };

int main(int argc,char**argv){
    const char* path=argv[1];
    int row=argc>2?atoi(argv[2]):760;
    HANDLE hf=CreateFileA(path,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);
    if(hf==INVALID_HANDLE_VALUE){ printf("cannot open\n"); return 1; }
    HANDLE hm=CreateFileMappingA(hf,NULL,PAGE_READONLY,0,0,NULL);
    const unsigned char* fb=(const unsigned char*)MapViewOfFile(hm,FILE_MAP_READ,0,0,0);
    const unsigned char* p=fb;
    auto u32=[&](){ uint32_t v; memcpy(&v,p,4); p+=4; return v; };
    auto u64=[&](){ uint64_t v; memcpy(&v,p,8); p+=8; return v; };
    auto str=[&](){ uint64_t n=u64(); std::string s((const char*)p,(size_t)n); p+=n; return s; };
    if(u32()!=0x46554747){ printf("not gguf\n"); return 1; }
    (void)u32();
    uint64_t nt=u64(), nkv=u64();
    uint32_t align=32;
    std::function<void(uint32_t)> skip=[&](uint32_t t){
        if(t==9){ uint32_t et=u32(); uint64_t n=u64(); for(uint64_t i=0;i<n;++i) skip(et); }
        else if(t==8){ str(); } else if(t==0||t==1||t==7){ p+=1; }
        else if(t==2||t==3){ p+=2; } else if(t==4||t==5||t==6){ p+=4; }
        else if(t==10||t==11||t==12){ p+=8; }
        else { printf("bad kv type %u\n",t); exit(1); }
    };
    for(uint64_t i=0;i<nkv;++i){
        std::string k=str(); uint32_t t=u32();
        if(k=="general.alignment"){ align=u32(); }
        else skip(t);
    }
    size_t pos0=(size_t)(p-fb);
    size_t ds=(pos0+align-1)/align*align;
    std::vector<GT> ts;
    for(uint64_t i=0;i<nt;++i){ GT g; g.name=str(); uint32_t nd=u32(); g.dims.resize(nd);
        for(uint32_t j=0;j<nd;++j) g.dims[j]=u64(); g.type=u32(); g.offset=u64(); ts.push_back(g); }
    printf("data_start=%zu align=%u tensors=%zu\n", ds, align, ts.size());

    const GT* emb=nullptr;
    for(auto&g:ts) if(g.name=="token_embd.weight") emb=&g;
    if(!emb){ printf("no token_embd\n"); return 1; }
    int c=(int)emb->dims[0];
    uint64_t r=1; for(size_t i=1;i<emb->dims.size();++i) r*=emb->dims[i];
    printf("token_embd type=%u ne0=%d rows=%llu off=%llu\n", emb->type, c, (unsigned long long)r, (unsigned long long)emb->offset);
    const unsigned char* data=fb+ds+emb->offset;

    std::vector<float> v(c);
    if(emb->type==142){
        int bpr=c/128;
        const unsigned char* base=data+(size_t)row*bpr*34;
        for(int b=0;b<bpr;++b){
            const unsigned char* blk=base+(size_t)b*34;
            uint16_t db; memcpy(&db,blk,2); float d=h2f(db);
            for(int j=0;j<128;++j){
                uint32_t q=(blk[2+(j>>2)]>>((j&3)*2))&3;
                v[b*128+j]=(float)((int)q-1)*d;
            }
        }
    } else if(emb->type==143){
        int bpr=c/128;
        uint32_t P3[5]={1,3,9,27,81};
        const unsigned char* base=data+(size_t)row*bpr*28;
        for(int b=0;b<bpr;++b){
            const unsigned char* blk=base+(size_t)b*28;
            int tr[128];
            for(int n=0;n<5;++n) for(int j=0;j<16;++j){ uint32_t q=((uint32_t)blk[j]*P3[n])&0xFF; tr[n*16+j]=(q*3)>>8; }
            for(int n=0;n<5;++n) for(int j=0;j<8;++j){ uint32_t q=((uint32_t)blk[16+j]*P3[n])&0xFF; tr[80+n*8+j]=(q*3)>>8; }
            for(int n=0;n<4;++n) for(int j=0;j<2;++j){ uint32_t q=((uint32_t)blk[24+j]*P3[n])&0xFF; tr[120+n*2+j]=(q*3)>>8; }
            uint16_t db; memcpy(&db,blk+26,2); float d=h2f(db);
            for(int j=0;j<128;++j) v[b*128+j]=(float)(tr[j]-1)*d;
        }
    } else { printf("unexpected type %u\n", emb->type); return 1; }

    printf("raw row[0:8] =");
    for(int i=0;i<8;++i) printf(" %.5f", v[i]);
    printf("\n");
    double ss=0; for(int i=0;i<c;++i) ss+=(double)v[i]*v[i];
    printf("rms=%.6f\n", sqrt(ss/c));
    return 0;
}
