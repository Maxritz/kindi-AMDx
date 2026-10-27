// kindi_cpu.cpp — CPU mirror of kindi_vk_decode.cpp, with layer-0 probes for llama diffing.
#include <windows.h>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <map>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include "tokenizer.h"

using F = std::vector<float>;

static float half_to_float(uint16_t h){
    uint32_t sign=(h&0x8000)<<16, exp=(h>>10)&0x1F, mant=h&0x3FF, f;
    if(exp==0){ if(mant==0) f=sign; else { int e=1; while(!(mant&0x400)){mant<<=1;--e;} mant&=0x3FF; f=sign|((e+127-15)<<23)|(mant<<13);} }
    else if(exp==31) f=sign|0x7F800000|(mant<<13);
    else f=sign|((exp+127-15)<<23)|(mant<<13);
    float o; memcpy(&o,&f,4); return o;
}
static uint16_t f32_to_half(float fv){
    uint32_t b; memcpy(&b,&fv,4);
    uint32_t sign=(b>>16)&0x8000;
    int32_t exp=(int32_t)((b>>23)&0xFF)-127+15;
    uint32_t mant=b&0x7FFFFF;
    if(exp<=0){ if(exp<-10) return (uint16_t)sign; mant|=(1u<<23); int s=14-exp; uint32_t m=(mant+(1u<<(s-1)))>>s; return (uint16_t)(sign|(m>>13)); }
    if(exp>=31) return (uint16_t)(sign|0x7C00);
    uint32_t m=(mant+0xFFF+((mant>>13)&1))>>13;
    if(m&0x400){ m=0; exp++; if(exp>=31) return (uint16_t)(sign|0x7C00); }
    return (uint16_t)(sign|(exp<<10)|m);
}

int main(int argc, char**argv){
    const char* model=nullptr; const char* prompt=nullptr;
    float temperature=0.0f; int topk=0; int dump=0;
    for(int i=1;i<argc;++i){
        if(!strcmp(argv[i],"--model")&&i+1<argc) model=argv[++i];
        else if(!strcmp(argv[i],"--prompt")&&i+1<argc) prompt=argv[++i];
        else if(!strcmp(argv[i],"--temp")&&i+1<argc) temperature=(float)atof(argv[++i]);
        else if(!strcmp(argv[i],"--top-k")&&i+1<argc) topk=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--dump")) dump=1;
    }
    if(!model){ fprintf(stderr,"usage: --model <path.gguf>\n"); return 1; }

    HANDLE hFile=CreateFileA(model,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);
    if(hFile==INVALID_HANDLE_VALUE){ fprintf(stderr,"cannot open %s\n",model); return 1; }
    LARGE_INTEGER fsz; GetFileSizeEx(hFile,&fsz);
    HANDLE hMap=CreateFileMappingA(hFile,NULL,PAGE_READONLY,0,0,NULL);
    const unsigned char* filebase=(const unsigned char*)MapViewOfFile(hMap,FILE_MAP_READ,0,0,0);

    const unsigned char* p=filebase;
    auto rd_u32=[&](){ uint32_t v; memcpy(&v,p,4); p+=4; return v; };
    auto rd_u64=[&](){ uint64_t v; memcpy(&v,p,8); p+=8; return v; };
    auto rd_f32=[&](){ float v; memcpy(&v,p,4); p+=4; return v; };
    auto rd_f64=[&](){ double v; memcpy(&v,p,8); p+=8; return v; };
    auto rd_str=[&](){ uint64_t n=rd_u64(); std::string s((const char*)p,(size_t)n); p+=n; return s; };
    if(rd_u32()!=0x46554747){ fprintf(stderr,"not GGUF\n"); return 1; }
    (void)rd_u32();
    uint64_t n_tensors=rd_u64(), n_kv=rd_u64();

    int D=0,NL=0,NFF=0,NH=0,NHKV=0,HD=0,NK=0,HK=0,NV=0,rope_dim=0,interval=0,HB=512,ctx=4096;
    int gdn_inner=0;
    std::vector<std::string> gguf_tokens, gguf_merges;
    std::vector<float> sign_values; std::vector<int> sign_widths;
    uint32_t align=32;
    auto read_scalar_int=[&](uint32_t t)->int{
        if(t==0){ uint8_t v=*p; p+=1; return v; } if(t==1){ int8_t v; memcpy(&v,p,1); p+=1; return v; }
        if(t==2){ uint16_t v; memcpy(&v,p,2); p+=2; return v; } if(t==3){ int16_t v; memcpy(&v,p,2); p+=2; return v; }
        if(t==4){ uint32_t v; memcpy(&v,p,4); p+=4; return (int)v; } if(t==5){ int32_t v; memcpy(&v,p,4); p+=4; return v; }
        if(t==6){ float v; memcpy(&v,p,4); p+=4; return (int)v; } if(t==7){ uint8_t v=*p; p+=1; return v; }
        if(t==10){ uint64_t v; memcpy(&v,p,8); p+=8; return (int)v; } if(t==11){ int64_t v; memcpy(&v,p,8); p+=8; return (int)v; }
        if(t==12){ double v; memcpy(&v,p,8); p+=8; return (int)v; }
        fprintf(stderr,"not int type %u\n",t); exit(1); return 0;
    };
    std::function<void(uint32_t)> skip_value=[&](uint32_t type){
        std::function<void(uint32_t)> sk=[&](uint32_t t){
            if(t==9){ uint32_t et=rd_u32(); uint64_t n=rd_u64(); for(uint64_t i=0;i<n;++i) sk(et); }
            else if(t==8){ rd_str(); } else if(t==0||t==1||t==7){ p+=1; }
            else if(t==2||t==3){ p+=2; } else if(t==4||t==5||t==6){ p+=4; }
            else if(t==10||t==11||t==12){ p+=8; } else { fprintf(stderr,"unknown type %u\n",t); exit(1); }
        };
        sk(type);
    };
    for(uint64_t ki=0;ki<n_kv;++ki){
        std::string key=rd_str(); uint32_t type=rd_u32();
        if(key=="qwen35.block_count") NL=read_scalar_int(type);
        else if(key=="qwen35.embedding_length") D=read_scalar_int(type);
        else if(key=="qwen35.feed_forward_length") NFF=read_scalar_int(type);
        else if(key=="qwen35.attention.head_count") NH=read_scalar_int(type);
        else if(key=="qwen35.attention.head_count_kv") NHKV=read_scalar_int(type);
        else if(key=="qwen35.attention.key_length") HD=read_scalar_int(type);
        else if(key=="qwen35.rope.dimension_count") rope_dim=read_scalar_int(type);
        else if(key=="qwen35.ssm.state_size") HK=read_scalar_int(type);
        else if(key=="qwen35.ssm.group_count") NK=read_scalar_int(type);
        else if(key=="qwen35.ssm.inner_size") gdn_inner=read_scalar_int(type);
        else if(key=="qwen35.full_attention_interval") interval=read_scalar_int(type);
        else if(key=="qwen35.context_length"){ (void)read_scalar_int(type); }
        else if(key=="qwen35.rope.freq_base"){ if(type==6){(void)rd_f32();} else if(type==12){(void)rd_f64();} else skip_value(type); }
        else if(key=="general.alignment") align=(uint32_t)read_scalar_int(type);
        else if(key=="prism.hadamard.block_size") HB=read_scalar_int(type);
        else if(key=="prism.hadamard.sign_values"){
            uint32_t et=rd_u32(); uint64_t n=rd_u64(); sign_values.resize((size_t)n);
            if(et==6) for(uint64_t i=0;i<n;++i) sign_values[(size_t)i]=rd_f32();
            else if(et==12) for(uint64_t i=0;i<n;++i) sign_values[(size_t)i]=(float)rd_f64();
            else if(et==5) for(uint64_t i=0;i<n;++i){ int32_t v; memcpy(&v,p,4); p+=4; sign_values[(size_t)i]=(float)v; }
            else if(et==1) for(uint64_t i=0;i<n;++i){ int8_t v; memcpy(&v,p,1); p+=1; sign_values[(size_t)i]=(float)v; }
            else if(et==4) for(uint64_t i=0;i<n;++i){ uint32_t v; memcpy(&v,p,4); p+=4; sign_values[(size_t)i]=(float)v; }
            else for(uint64_t i=0;i<n;++i) skip_value(et);
        }
        else if(key=="prism.hadamard.sign_widths"){ uint32_t et=rd_u32(); uint64_t n=rd_u64(); sign_widths.resize((size_t)n); for(uint64_t i=0;i<n;++i) sign_widths[(size_t)i]=read_scalar_int(et); }
        else if(key=="tokenizer.ggml.tokens"){ uint32_t et=rd_u32(); uint64_t n=rd_u64(); gguf_tokens.resize((size_t)n); for(uint64_t i=0;i<n;++i){ if(et==8) gguf_tokens[(size_t)i]=rd_str(); else { skip_value(et); gguf_tokens[(size_t)i].clear(); } } }
        else if(key=="tokenizer.ggml.merges"){ uint32_t et=rd_u32(); uint64_t n=rd_u64(); gguf_merges.resize((size_t)n); for(uint64_t i=0;i<n;++i){ if(et==8) gguf_merges[(size_t)i]=rd_str(); else { skip_value(et); gguf_merges[(size_t)i].clear(); } } }
        else skip_value(type);
    }
    NV=gdn_inner/HK; int VOCAB=248320; int NATT=NL/interval;
    fprintf(stderr,"dims: L=%d D=%d FFN=%d H=%d KV=%d HD=%d G=%d HK=%d NV=%d HB=%d\n",NL,D,NFF,NH,NHKV,HD,NK,HK,NV,HB);

    struct GT{ std::string name; std::vector<uint64_t> dims; uint32_t type; uint64_t offset; };
    std::vector<GT> gtensors;
    for(uint64_t ti=0;ti<n_tensors;++ti){ GT t; t.name=rd_str(); uint32_t nd=rd_u32(); t.dims.resize(nd); for(uint32_t i=0;i<nd;++i) t.dims[i]=rd_u64(); t.type=rd_u32(); t.offset=rd_u64(); gtensors.push_back(t); }
    size_t pos0=(size_t)(p-filebase);
    size_t data_start=(pos0+align-1)/align*align;

    uint32_t pow3[6]={1,3,9,27,81,243};
    std::map<std::string,F> W;
    std::map<std::string,std::pair<int,int>> tinfo;
    size_t n_tern=0, n_f32=0;
    for(auto& gt: gtensors){
        uint64_t ne0=gt.dims[0]; uint64_t rows=1;
        for(size_t i=1;i<gt.dims.size();++i) rows*=gt.dims[i];
        int r=(int)rows, c=(int)ne0;
        const unsigned char* data=filebase+data_start+gt.offset;
        if(gt.type==143){
            int bpr=c/128, nb=r*bpr;
            std::vector<unsigned char> trits((size_t)nb*128);
            for(int bi=0;bi<nb;++bi){
                const unsigned char* raw=data+(size_t)bi*28;
                unsigned char* out=trits.data()+(size_t)bi*128;
                for(int n=0;n<5;++n) for(int j=0;j<16;++j){ uint32_t q=((uint32_t)raw[j]*pow3[n])&0xFF; out[n*16+j]=(unsigned char)((q*3)>>8); }
                for(int n=0;n<5;++n) for(int j=0;j<8;++j){ uint32_t q=((uint32_t)raw[16+j]*pow3[n])&0xFF; out[80+n*8+j]=(unsigned char)((q*3)>>8); }
                for(int n=0;n<4;++n) for(int j=0;j<2;++j){ uint32_t q=((uint32_t)raw[24+j]*pow3[n])&0xFF; out[120+n*2+j]=(unsigned char)((q*3)>>8); }
            }
            F wf((size_t)r*c);
            std::vector<uint16_t> sc((size_t)r*bpr);
            for(int bi=0;bi<nb;++bi){
                const unsigned char* raw=data+(size_t)bi*28;
                const unsigned char* tr=trits.data()+(size_t)bi*128;
                int row=bi/bpr, ph=bi%bpr;
                uint16_t v; memcpy(&v,raw+26,2); sc[(size_t)row*bpr+ph]=v;
                float s=half_to_float(v);
                for(int j=0;j<128;++j) wf[(size_t)row*c+ph*128+j]=(float)((int)tr[j]-1)*s;
            }
            W[gt.name]=std::move(wf); tinfo[gt.name]={r,c}; n_tern++;
        } else if(gt.type==0){ size_t n=(size_t)r*c; F v(n); memcpy(v.data(),data,n*4); W[gt.name]=std::move(v); tinfo[gt.name]={r,c}; n_f32++; }
        else if(gt.type==30){ size_t n=(size_t)r*c; F v(n); const uint16_t* s=(const uint16_t*)data; for(size_t i=0;i<n;++i){ uint32_t b=((uint32_t)s[i])<<16; float f; memcpy(&f,&b,4); v[i]=f; } W[gt.name]=std::move(v); tinfo[gt.name]={r,c}; n_f32++; }
        else if(gt.type==1){ size_t n=(size_t)r*c; F v(n); const uint16_t* s=(const uint16_t*)data; for(size_t i=0;i<n;++i) v[i]=half_to_float(s[i]); W[gt.name]=std::move(v); tinfo[gt.name]={r,c}; n_f32++; }
        else { fprintf(stderr,"unsupported tensor type %u for %s\n",gt.type,gt.name.c_str()); return 1; }
    }
    fprintf(stderr,"weights uploaded: %zu ternary + %zu f32\n",n_tern,n_f32);

    auto Z=[&](int n){ return F((size_t)n,0.0f); };
    F y=Z(VOCAB), xp=Z(D), h=Z(D), xn=Z(D), zv=Z(NV*HK), h2=Z(D), part=Z(4096);
    F qkv=Z(6144), cs=Z(6144), qkbuf=Z(2*NK*HK), qkvbuf=Z(NHKV*HD);
    F kn=Z(NHKV*HD), qn=Z(NH*2*HD), qr=Z(NH*HD), kro=Z(NHKV*HD);
    F fo=Z(NH*HD), fo6=Z(NFF), outgdn=Z(NV*HK), att=Z(NH*HD), ffg=Z(NFF), ffu=Z(NFF);
    F gb=Z(NV), gg=Z(NV), gout=Z(NV*HK);
    F sbuf=Z(NH*ctx), apart=Z(((ctx+255)/256)*NH*HD);
    std::vector<F> M; for(int i=0;i<NL;++i) M.push_back(Z(NV*HK*HK));
    std::vector<F> convv; for(int i=0;i<NL;++i) convv.push_back(Z(3*6144));
    std::vector<std::vector<uint16_t>> Kc, Vc;
    for(int i=0;i<NATT;++i){ Kc.push_back(std::vector<uint16_t>((size_t)ctx*NHKV*HD,0)); Vc.push_back(std::vector<uint16_t>((size_t)ctx*NHKV*HD,0)); }

    std::map<int,F> signsmap; { size_t off=0; for(int w: sign_widths){ signsmap[w].assign(sign_values.begin()+off,sign_values.begin()+off+w); off+=w; } }
    F sD=signsmap.count(D)?signsmap[D]:F(D,1.0f); if((int)sD.size()<D) sD.resize(D,1.0f);
    F sF=signsmap.count(NFF)?signsmap[NFF]:F(NFF,1.0f); if((int)sF.size()<NFF) sF.resize(NFF,1.0f);
    F sV(NV*HK,1.0f), sA(NH*HD,1.0f);
    for(int i=0;i<NV*HK;++i) sV[i]=sD[i%D]; for(int i=0;i<NH*HD;++i) sA[i]=sD[i%D];

    F cosv((size_t)ctx*(rope_dim/2)), sinv((size_t)ctx*(rope_dim/2));
    std::vector<double> invfreq(rope_dim/2);
    for(int i=0;i<rope_dim/2;++i) invfreq[i]=pow(1e7,-2.0*i/rope_dim);
    for(int pp=0;pp<ctx;++pp) for(int i=0;i<rope_dim/2;++i){ double th=pp*invfreq[i]; cosv[(size_t)pp*(rope_dim/2)+i]=(float)cos(th); sinv[(size_t)pp*(rope_dim/2)+i]=(float)sin(th); }

    auto popc=[](uint32_t v){ v-=((v>>1)&0x55555555u); v=(v&0x33333333u)+((v>>2)&0x33333333u); return (((v+(v>>4))&0xF0F0F0Fu)*0x1010101u)>>24; };

    auto gemv=[&](const std::string& wn, const F& x, F& o){
        auto [rows,cols]=tinfo[wn]; const F& m=W[wn];
        for(int rr=0;rr<rows;++rr){ double acc=0; for(int cc=0;cc<cols;++cc) acc+=(double)m[(size_t)rr*cols+cc]*x[cc]; o[rr]=(float)acc; }
    };
    auto embed=[&](int tok){
        const F& ew=W["token_embd.weight"];
        static thread_local F w(6144);
        int row=tok, C=D;
        for(int e=0;e<C;++e) w[e]=ew[(size_t)row*C+e];
        for(int j2=0;j2<C;++j2){
            int c0=j2>>9, j=j2&511;
            float acc=0; for(int i=0;i<512;++i){ float ww=w[(c0<<9)+i]; acc+= (popc((uint32_t)(i&j))&1)? -ww : ww; }
            h[j2]=sD[j2]*(acc*0.044194173824159220878f);
        }
        { static int first_call=1; if(dump&&first_call){ fprintf(stderr,"P embed_h: %.4f %.4f %.4f\n",h[0],h[1],h[2]); first_call=0; } }
    };
    int pending=0;
    auto fwht512=[&](F& xs){
        for(int hh=1;hh<512;hh<<=1) for(int j=0;j<256;++j){ int i=(j/hh)*2*hh+(j%hh); float a=xs[i],b=xs[i+hh]; xs[i]=a+b; xs[i+hh]=a-b; }
    };
    auto rmsfold=[&](F& x, F& r, F& xw, F& xn_, F& xo, const std::string& wn, int C, int HBv){
        F& resbuf=(pending>1)?part:y;
        double acc=0;
        for(int e=0;e<C;++e){ float xf=x[e]; if(pending!=0){ float s=resbuf[e]; for(int i=1;i<pending;++i) s+=resbuf[(size_t)i*C+e]; xf=x[e]+s; } acc+=(double)xf*xf; }
        float bscale=1.0f/sqrtf((float)(acc/C)+1e-6f);
        const F& w=W[wn];
        int chunk=(HBv<0?512:HBv);
        for(int cc0=0;cc0<C;cc0+=chunk){
            F xs(512,0.0f);
            for(int m=0;m<chunk;++m){ int e=cc0+m; float xf=x[e]; if(pending!=0){ float rs=resbuf[e]; for(int i=1;i<pending;++i) rs+=resbuf[(size_t)i*C+e]; xf=x[e]+rs; xw[e]=xf; } float v=(xf*bscale)*w[e]; xn_[e]=v; xs[m]=sD[e]*v; }
            fwht512(xs);
            for(int m=0;m<chunk;++m) xo[cc0+m]=(HBv==512)? xs[m]*0.044194173824159220878f : xs[m];
        }
        if(pending>0) std::swap(h,h2);
        pending=0;
    };

    auto gdn=[&](int il, int pos){
        (void)pos;
        std::string p="blk."+std::to_string(il)+".";
        rmsfold(h,part,h2,xn,xp,p+"attn_norm.weight",D,HB);
        if(dump&&il==0){
            fprintf(stderr,"P attn_norm xn[0..2]: %.4f %.4f %.4f\n",xn[0],xn[1],xn[2]);
            fprintf(stderr,"P folded  xp[0..2]: %.6f %.6f %.6f\n",xp[0],xp[1],xp[2]);
            fprintf(stderr,"P hsum after swap h[0..2]: %.4f %.4f %.4f\n",h[0],h[1],h[2]);
        }
        gemv(p+"attn_qkv.weight",xp,qkv);
        gemv(p+"attn_gate.weight",xp,zv);
        if(dump&&il==0){
            fprintf(stderr,"P qkv[0..2]: %.4f %.4f %.4f\n",qkv[0],qkv[1],qkv[2]);
            fprintf(stderr,"P z[0..2]:   %.4f %.4f %.4f\n",zv[0],zv[1],zv[2]);
        }
        {
            const F& xn_=xn, &bw=W[p+"ssm_beta.weight"], &aw=W[p+"ssm_alpha.weight"], &dtb=W[p+"ssm_dt.bias"], &ssma=W[p+"ssm_a"];
            for(int hh2=0;hh2<NV;++hh2){
                double ab=0,aa=0; for(int i=0;i<D;++i){ ab+=(double)bw[(size_t)hh2*D+i]*xn_[i]; aa+=(double)aw[(size_t)hh2*D+i]*xn_[i]; }
                gb[hh2]=1.0f/(1.0f+expf(-(float)ab));
                float a=(float)aa+dtb[hh2]; float sp=(a>20.0f)?a:logf(1.0f+expf(a>20.0f?20.0f:a));
                gg[hh2]=sp*ssma[hh2];
            }
            if(dump&&il==0) fprintf(stderr,"P beta[0..2]: %.4f %.4f %.4f  gate[0..2]: %.4f %.4f %.4f\n",gb[0],gb[1],gb[2],gg[0],gg[1],gg[2]);
        }
        {
            F& conv=convv[il]; const F& cw=W[p+"ssm_conv1d.weight"];
            int C=6144;
            for(int c0=0;c0<C;++c0){
                float c0v=conv[c0],c1=conv[C+c0],c2=conv[2*C+c0],nw=qkv[c0];
                float o=c0v*cw[4*c0]+c1*cw[4*c0+1]+c2*cw[4*c0+2]+nw*cw[4*c0+3];
                conv[c0]=c1; conv[C+c0]=c2; conv[2*C+c0]=nw;
                cs[c0]=o/(1.0f+expf(-o));
            }
            if(dump&&il==0) fprintf(stderr,"P conv_silu[0..2]: %.4f %.4f %.4f\n",cs[0],cs[1],cs[2]);
        }
        {
            for(int rr=0;rr<2*NK;++rr){ double acc=0; for(int e=0;e<HK;++e) acc+=(double)cs[(size_t)rr*HK+e]*cs[(size_t)rr*HK+e];
                float sc=1.0f/fmaxf(sqrtf((float)acc),1e-6f);
                for(int e=0;e<HK;++e) qkbuf[(size_t)rr*HK+e]=cs[(size_t)rr*HK+e]*sc; }
        }
        {
            F& M_=M[il]; float scale=1.0f/sqrtf((float)HK);
            for(int hh2=0;hh2<NV;++hh2){
                int kh=hh2%NK;
                F ks(HK),qs(HK);
                for(int j=0;j<HK;++j){ ks[j]=qkbuf[(size_t)(NK+kh)*HK+j]; qs[j]=qkbuf[(size_t)kh*HK+j]; }
                float e=expf(gg[hh2]);
                for(int j=0;j<HK;++j){
                    double sk=0; for(int i=0;i<HK;++i) sk+=(double)(M_[(size_t)hh2*HK*HK+i*HK+j]*e)*ks[i];
                    float dd=(cs[(size_t)(2*NK+hh2)*HK+j]-(float)sk)*gb[hh2];
                    double o=0; for(int i=0;i<HK;++i){ float m=M_[(size_t)hh2*HK*HK+i*HK+j]*e+ks[i]*dd; M_[(size_t)hh2*HK*HK+i*HK+j]=m; o+=m*qs[i]; }
                    gout[(size_t)hh2*HK+j]=(float)o*scale;
                }
            }
            if(dump&&il==0) fprintf(stderr,"P delta_out[0..2]: %.4f %.4f %.4f\n",gout[0],gout[1],gout[2]);
        }
        {
            const F& w=W[p+"ssm_norm.weight"];
            for(int rr=0;rr<NV;++rr){ double acc=0; for(int e=0;e<HK;++e) acc+=(double)gout[(size_t)rr*HK+e]*gout[(size_t)rr*HK+e];
                float sc=1.0f/sqrtf((float)(acc/HK)+1e-6f);
                for(int e=0;e<HK;++e) outgdn[(size_t)rr*HK+e]=(gout[(size_t)rr*HK+e]*sc)*w[e]; }
            if(dump&&il==0) fprintf(stderr,"P gdn_norm[0..2]: %.4f %.4f %.4f\n",outgdn[0],outgdn[1],outgdn[2]);
        }
        {
            const F& a=zv, &b=outgdn; int prep=NV/NK;
            for(int cc0=0;cc0<NV*HK;cc0+=HB){
                F xs(512,0.0f);
                for(int q=0;q<HB;++q){ int e=cc0+q, src=e;
                    if(prep>1){ int hd=e%HK, grouped=e/HK, rep=grouped%prep, nk=grouped/prep; src=hd+HK*(nk+NK*rep); }
                    float xa=a[src]; xs[q]=sV[e]*((xa/(1.0f+expf(-xa)))*b[src]);
                }
                fwht512(xs);
                for(int q=0;q<HB;++q) xp[cc0+q]=xs[q]*0.044194173824159220878f;
            }
            if(dump&&il==0) fprintf(stderr,"P gdn_folded xp[0..2]: %.6f %.6f %.6f\n",xp[0],xp[1],xp[2]);
        }
        gemv(p+"ssm_out.weight",xp,y);
        if(dump&&il==0) fprintf(stderr,"P ssm_out y[0..2]: %.4f %.4f %.4f\n",y[0],y[1],y[2]);
        pending=1;
    };

    auto attn=[&](int il, int pos, int kvi){
        std::string p="blk."+std::to_string(il)+".";
        rmsfold(h,part,h2,xn,xp,p+"attn_norm.weight",D,HB);
        gemv(p+"attn_q.weight",xp,qn);
        gemv(p+"attn_k.weight",xp,kro);
        gemv(p+"attn_v.weight",xp,qkvbuf);
        for(int rr=0;rr<NH;++rr) for(int e=0;e<HD;++e) qr[(size_t)rr*HD+e]=qn[(size_t)rr*2*HD+e];
        { const F& w=W[p+"attn_q_norm.weight"]; for(int rr=0;rr<NH;++rr){ double acc=0; for(int e=0;e<HD;++e) acc+=(double)qr[(size_t)rr*HD+e]*qr[(size_t)rr*HD+e]; float sc=1.0f/sqrtf((float)(acc/HD)+1e-6f); for(int e=0;e<HD;++e) fo[(size_t)rr*HD+e]=(qr[(size_t)rr*HD+e]*sc)*w[e]; } }
        { for(int rr=0;rr<NH;++rr){ const float* cc=&cosv[(size_t)pos*(rope_dim/2)]; const float* ss=&sinv[(size_t)pos*(rope_dim/2)]; for(int j=0;j<32;++j){ float a=fo[(size_t)rr*HD+j],b=fo[(size_t)rr*HD+j+32]; qr[(size_t)rr*HD+j]=a*cc[j]-b*ss[j]; qr[(size_t)rr*HD+j+32]=a*ss[j]+b*cc[j]; } for(int j=64;j<HD;++j) qr[(size_t)rr*HD+j]=fo[(size_t)rr*HD+j]; } }
        { const F& w=W[p+"attn_k_norm.weight"]; for(int rr=0;rr<NHKV;++rr){ double acc=0; for(int e=0;e<HD;++e) acc+=(double)kro[(size_t)rr*HD+e]*kro[(size_t)rr*HD+e]; float sc=1.0f/sqrtf((float)(acc/HD)+1e-6f); for(int e=0;e<HD;++e) kn[(size_t)rr*HD+e]=(kro[(size_t)rr*HD+e]*sc)*w[e]; } }
        { for(int rr=0;rr<NHKV;++rr){ const float* cc=&cosv[(size_t)pos*(rope_dim/2)]; const float* ss=&sinv[(size_t)pos*(rope_dim/2)]; for(int j=0;j<32;++j){ float a=kn[(size_t)rr*HD+j],b=kn[(size_t)rr*HD+j+32]; kro[(size_t)rr*HD+j]=a*cc[j]-b*ss[j]; kro[(size_t)rr*HD+j+32]=a*ss[j]+b*cc[j]; } for(int j=64;j<HD;++j) kro[(size_t)rr*HD+j]=kn[(size_t)rr*HD+j]; } }
        for(int i=0;i<NHKV*HD;++i){ Kc[kvi][(size_t)pos*NHKV*HD+i]=f32_to_half(kro[i]); Vc[kvi][(size_t)pos*NHKV*HD+i]=f32_to_half(qkvbuf[i]); }
        int n=pos+1, G=NH/NHKV;
        for(int kv=0;kv<NHKV;++kv) for(int t=0;t<n;++t){
            F krow(HD); for(int e=0;e<HD;++e) krow[e]=half_to_float(Kc[kvi][(size_t)t*NHKV*HD+kv*HD+e]);
            for(int j=0;j<G;++j){ double a=0; for(int e=0;e<HD;++e) a+=(double)qr[(size_t)(kv*G+j)*HD+e]*krow[e]; sbuf[(size_t)(kv*G+j)*n+t]=(float)(a*0.0625); }
        }
        if(dump&&il==3&&pos<5) fprintf(stderr,"C scores_raw: %.4f %.4f %.4f\n",sbuf[0],sbuf[1],sbuf[2]);
        for(int hh=0;hh<NH;++hh){ float mx=-1e30f; for(int t=0;t<n;++t) mx=fmaxf(mx,sbuf[(size_t)hh*n+t]); double tot=0; for(int t=0;t<n;++t){ sbuf[(size_t)hh*n+t]=expf(sbuf[(size_t)hh*n+t]-mx); tot+=sbuf[(size_t)hh*n+t]; } for(int t=0;t<n;++t) sbuf[(size_t)hh*n+t]/=(float)tot; }
        int nch=(n+255)/256;
        for(int kv=0;kv<NHKV;++kv) for(int c0=0;c0<nch;++c0){
            int i0=c0*256, m=(256<(n-i0)?256:(n-i0));
            for(int f2=0;f2<HD;++f2){
                double a0=0,a1=0,a2=0,a3=0;
                for(int i=0;i<m;++i){ float v=half_to_float(Vc[kvi][(size_t)(i0+i)*NHKV*HD+kv*HD+f2]);
                    a0+=sbuf[(size_t)(kv*G+0)*n+i0+i]*v; a1+=sbuf[(size_t)(kv*G+1)*n+i0+i]*v; a2+=sbuf[(size_t)(kv*G+2)*n+i0+i]*v; a3+=sbuf[(size_t)(kv*G+3)*n+i0+i]*v; }
                int prow=(c0*NH+kv*G)*HD+f2;
                apart[prow]=(float)a0;
                if(G>1){ apart[prow+HD]=(float)a1; apart[prow+2*HD]=(float)a2; apart[prow+3*HD]=(float)a3; }
            }
        }
        for(int hh=0;hh<NH;++hh) for(int f=0;f<HD;++f){ double acc=apart[(size_t)hh*HD+f]; for(int c0=1;c0<nch;++c0) acc+=apart[(size_t)(c0*NH+hh)*HD+f]; float g=qn[(size_t)hh*2*HD+HD+f]; att[(size_t)hh*HD+f]=(float)(acc*(1.0/(1.0+exp(-g)))); }
        for(int cc0=0;cc0<NH*HD;cc0+=HB){ F xs(512,0.0f); for(int q=0;q<HB;++q) xs[q]=sA[cc0+q]*att[cc0+q]; fwht512(xs); for(int q=0;q<HB;++q) xp[cc0+q]=xs[q]*0.044194173824159220878f; }
        if(dump&&il==3&&pos<5){ fprintf(stderr,"C q_rope: %.4f %.4f %.4f\n",qr[0],qr[1],qr[2]);
            fprintf(stderr,"C k_rope: %.4f %.4f %.4f\n",kro[0],kro[1],kro[2]);
            fprintf(stderr,"C scores: %.4f %.4f %.4f\n",sbuf[0],sbuf[1],sbuf[2]);
            fprintf(stderr,"C comb: %.4f %.4f %.4f\n",att[0],att[1],att[2]);
            fprintf(stderr,"C att_fold xp: %.4f %.4f %.4f\n",xp[0],xp[1],xp[2]); }
        gemv(p+"attn_output.weight",xp,y);
        pending=1;
    };

    auto ffn=[&](int il){
        std::string p="blk."+std::to_string(il)+".";
        rmsfold(h,part,h2,xn,xp,p+"post_attention_norm.weight",D,HB);
        gemv(p+"ffn_gate.weight",xp,ffg);
        gemv(p+"ffn_up.weight",xp,ffu);
        for(int cc0=0;cc0<NFF;cc0+=HB){ F xs(512,0.0f); for(int q=0;q<HB;++q){ int e=cc0+q; float xa=ffg[e]; xs[q]=sF[e]*((xa/(1.0f+expf(-xa)))*ffu[e]); } fwht512(xs); for(int q=0;q<HB;++q) fo6[cc0+q]=xs[q]*0.044194173824159220878f; }
        gemv(p+"ffn_down.weight",fo6,y);
        if(dump&&il==0) fprintf(stderr,"P ffn_out y[0..2]: %.4f %.4f %.4f\n",y[0],y[1],y[2]);
        pending=1;
    };

    auto body=[&](int tok, int pos){ embed(tok); pending=0; int kvi=0; for(int il=0;il<NL;++il){ if((il+1)%interval!=0) gdn(il,pos); else { attn(il,pos,kvi); kvi++; } ffn(il); if(dump) fprintf(stderr,"L%02d res %.4f %.4f %.4f\n",il,h[0]+y[0],h[1]+y[1],h[2]+y[2]); } };

    auto step=[&](int tok, int pos)->int{
        body(tok,pos);
        rmsfold(h,part,h2,xn,xp,"output_norm.weight",D,HB);
        gemv("token_embd.weight",xp,y);
        if(dump){ fprintf(stderr,"P logits y[0..2]: %.4f %.4f %.4f\n",y[0],y[1],y[2]);
            float bv=-1e30f; int bi=0; for(int e=0;e<VOCAB;++e) if(y[e]>bv){bv=y[e];bi=e;}
            fprintf(stderr,"P argmax=%d val=%.4f\n",bi,bv); }
        if(temperature<=0.0f){ float bv=-1e30f; int bi=0; for(int e=0;e<VOCAB;++e) if(y[e]>bv){bv=y[e];bi=e;} return bi; }
        for(int i=0;i<VOCAB;++i) y[i]/=temperature;
        if(topk>0&&topk<VOCAB){ F tmp=y; std::nth_element(tmp.begin(),tmp.begin()+(VOCAB-topk),tmp.end()); float thr=tmp[VOCAB-topk]; for(int i=0;i<VOCAB;++i) if(y[i]<thr) y[i]=-1e30f; }
        float mx=-1e30f; for(int i=0;i<VOCAB;++i) if(y[i]>mx) mx=y[i];
        double sum=0; for(int i=0;i<VOCAB;++i){ y[i]=expf(y[i]-mx); sum+=y[i]; } for(int i=0;i<VOCAB;++i) y[i]/=(float)sum;
        double r=(double)rand()/RAND_MAX, acc=0; for(int i=0;i<VOCAB;++i){ acc+=y[i]; if(r<acc) return i; }
        return VOCAB-1;
    };

    kindi_tok::BpeTokenizer tok; std::string tokerr;
    if(!tok.load_from_strings(gguf_tokens,gguf_merges,&tokerr)){ fprintf(stderr,"tokenizer load failed: %s\n",tokerr.c_str()); return 1; }
    std::string promptstr=prompt?prompt:"The sky is blue because";
    std::vector<int> ids=tok.encode(promptstr);
    if(ids.empty()) ids.push_back(0);
    fprintf(stderr,"prompt ids (%zu):",ids.size()); for(int t: ids) fprintf(stderr," %d",t); fprintf(stderr,"\n");
    for(size_t i=0;i+1<ids.size();++i) body(ids[i],(int)i);
    int pos=(int)ids.size();
    std::vector<int> out; int next=step(ids.back(),pos-1); out.push_back(next); pos++;
    while((int)out.size()<5){ next=step(out.back(),pos); out.push_back(next); pos++; if(pos>=ctx) break; }
    fprintf(stderr,"\nGenerated %zu tokens: ",out.size()); for(int t: out) fprintf(stderr,"%d ",t);
    fprintf(stderr,"\nDecoded: %s\n",tok.decode(out).c_str());
    return 0;
}
