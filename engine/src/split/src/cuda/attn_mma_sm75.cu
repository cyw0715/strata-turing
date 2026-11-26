// Original paged KV bytes/scales retained. FP16 matrix operands, FP32 online softmax/accumulation.
#include "kernels.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
namespace fnx {namespace {
constexpr int SPLITS=16,QPK=NQH/NKVH,SEL=QSA_TOP_K+3,LD=HD+8;
static_assert(QPK<=16 && HD==256,"attention matrix shape");
struct Part {float m,z;};
__device__ __forceinline__ float wsum(float x) {
    for(int s=16;s;s>>=1)x+=__shfl_xor_sync(0xffffffffu,x,s);return x;
}
__device__ __forceinline__ float wmax(float x) {
    for(int s=16;s;s>>=1)x=fmaxf(x,__shfl_xor_sync(0xffffffffu,x,s));return x;
}
__device__ __forceinline__ unsigned bits(half2 x) {
    return unsigned(__half_as_ushort(__low2half(x))) | (unsigned(__half_as_ushort(__high2half(x)))<<16);
}
__device__ __forceinline__ void mma(float* d,unsigned a0,unsigned a1,unsigned b) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a0),"r"(a1),"r"(b));
}
__device__ __forceinline__ int owned(int p,int card) {
    const int b=p/4;return ((b+NCARD-1-card)/NCARD)*4+(kv_card(b)==card?p%4+1:0);
}
template<int CH> __global__ void __launch_bounds__(256) k_attn_mma(const float* qg,const half* kc,const half* vc,KvPages g,
    LaneRows lr,int T,float* output,Part* parts,const int* cells,const int* ncells,int card,const int* pos0) {
    const int tid=threadIdx.x,warp=tid/32,lane=tid&31,gg=lane/4,k4=lane%4;
    const int split=blockIdx.x,t=blockIdx.y,kv=blockIdx.z;
    g=row_pages(g,lr,t);
    const int* list=cells && ncells[t]>=0?cells+(size_t)t*SEL:nullptr;
    const int ctx=list?ncells[t]:owned(row_pos(lr,pos0,t),card);
    const int len=((ctx+SPLITS-1)/SPLITS+CH-1)/CH*CH,begin=split*len,end=min(ctx,begin+len);
    __shared__ __align__(16) half q[QPK][LD],qlow[CH==16?QPK:1][LD],kk[CH][LD],vv[CH][LD],prob[16][CH+8];
    __shared__ half kd[CH][HD/32];
    __shared__ float score[16][CH+1],scale[16],mm[16],zz[16];
    for(int i=tid;i<QPK*HD;i+=256) {
        const int h=i/HD,d=i%HD;
        const float value=qg[((size_t)t*NQH+kv*QPK+h)*2*HD+d];
        const half high=__float2half_rn(value);q[h][d]=high;
        if constexpr(CH==16)qlow[h][d]=__float2half_rn(value-__half2float(high));
    }
    if(tid<16){mm[tid]=-INFINITY;zz[tid]=0.f;}
    float accum[4][4]={};
    __syncthreads();
    for(int c0=begin;c0<end;c0+=CH) {
        const int np=min(CH,end-c0);
        for(int i=tid;i<CH*HD;i+=256) {
            const int p=i/HD,d=i%HD,pos=p<np?(list?list[c0+p]:c0+p):0;
            float k=0,v=0;
            if(d%32==0)kd[p][d/32]=__float2half(1.f);
            if(p<np) {
                if(g.q8) {
                    const size_t r=kvq_row(g,pos)+(size_t)kv*HD+d;
                    const size_t sc=kvq_scale(g,pos)+((size_t)kv*HD+d)/KV8_GROUP;
                    k=((const int8_t*)kc)[r];
                    if(d%32==0)kd[p][d/32]=kc[sc];
                    v=((const int8_t*)vc)[r]*__half2float(vc[sc]);
                } else {
                    const size_t r=kvp_row(g,pos)+(size_t)kv*HD+d;
                    k=__half2float(kc[r]);v=__half2float(vc[r]);
                }
            }
            kk[p][d]=__float2half_rn(k);vv[p][d]=__float2half_rn(v);
        }
        __syncthreads();
        if(warp<CH/8) {
            float dot[4]={};
#pragma unroll 1
            for(int group=0;group<HD;group+=32) {
                float block[4]={};
#pragma unroll
                for(int k=group;k<group+32;k+=8) {
                    const unsigned a0=*(const unsigned*)&q[gg][k+2*k4];
                    const unsigned a1=gg+8<QPK?*(const unsigned*)&q[gg+8][k+2*k4]:0;
                    const unsigned b=*(const unsigned*)&kk[warp*8+gg][k+2*k4];
                    unsigned l0,l1;
                    if constexpr(CH==16) {
                        l0=*(const unsigned*)&qlow[gg][k+2*k4];
                        l1=gg+8<QPK?*(const unsigned*)&qlow[gg+8][k+2*k4]:0;
                    } else {
                        const float2 x0=__ldg((const float2*)(qg+((size_t)t*NQH+kv*QPK+gg)*2*HD+k+2*k4));
                        const float2 x1=gg+8<QPK?__ldg((const float2*)(qg+((size_t)t*NQH+kv*QPK+gg+8)*2*HD+k+2*k4)):make_float2(0,0);
                        const half2 h0=*(const half2*)&q[gg][k+2*k4];
                        const half2 h1=gg+8<QPK?*(const half2*)&q[gg+8][k+2*k4]:__floats2half2_rn(0,0);
                        l0=bits(__floats2half2_rn(x0.x-__low2float(h0),x0.y-__high2float(h0)));
                        l1=bits(__floats2half2_rn(x1.x-__low2float(h1),x1.y-__high2float(h1)));
                    }
                    mma(block,a0,a1,b);mma(block,l0,l1,b);
                }
                const float s0=__half2float(kd[warp*8+2*k4][group/32]);
                const float s1=__half2float(kd[warp*8+2*k4+1][group/32]);
                dot[0]=fmaf(block[0],s0,dot[0]);dot[1]=fmaf(block[1],s1,dot[1]);
                dot[2]=fmaf(block[2],s0,dot[2]);dot[3]=fmaf(block[3],s1,dot[3]);
            }
            const int col=warp*8+2*k4;
            score[gg][col]=dot[0];score[gg][col+1]=dot[1];
            score[gg+8][col]=dot[2];score[gg+8][col+1]=dot[3];
        }
        __syncthreads();
        for(int h=warp;h<16;h+=8) {
            const float sv=lane<np?score[h][lane]*rsqrtf(float(HD)):-INFINITY;
            const float mn=fmaxf(mm[h],wmax(sv));
            const float w=sv==-INFINITY?0.f:__expf(sv-mn);
            const float sc=mm[h]==-INFINITY?0.f:__expf(mm[h]-mn);
            const float z=zz[h]*sc+wsum(w);
            if(lane<CH)prob[h][lane]=__float2half_rn(w);
            if(lane==0){mm[h]=mn;zz[h]=z;scale[h]=sc;}
        }
        __syncthreads();
#pragma unroll
        for(int tile=0;tile<4;++tile) {
            accum[tile][0]*=scale[gg];accum[tile][1]*=scale[gg];
            accum[tile][2]*=scale[gg+8];accum[tile][3]*=scale[gg+8];
            const int col=warp*32+tile*8+gg;
#pragma unroll
            for(int k=0;k<CH;k+=8) {
                const unsigned a0=*(const unsigned*)&prob[gg][k+2*k4];
                const unsigned a1=*(const unsigned*)&prob[gg+8][k+2*k4];
                const unsigned b=bits(__halves2half2(vv[k+2*k4][col],vv[k+2*k4+1][col]));
                mma(accum[tile],a0,a1,b);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for(int tile=0;tile<4;++tile) {
        const int col=warp*32+tile*8+2*k4;
        if(gg<QPK) {
            const size_t at=(((size_t)t*NQH+kv*QPK+gg)*SPLITS+split)*HD+col;
            output[at]=accum[tile][0];output[at+1]=accum[tile][1];
        }
        if(gg+8<QPK) {
            const size_t at=(((size_t)t*NQH+kv*QPK+gg+8)*SPLITS+split)*HD+col;
            output[at]=accum[tile][2];output[at+1]=accum[tile][3];
        }
    }
    for(int h=tid;h<QPK;h+=256)
        parts[((size_t)t*NQH+kv*QPK+h)*SPLITS+split]=Part{mm[h],zz[h]};
}
}
void attn_mma_run(const float* q,const uint16_t* k,const uint16_t* v,const KvPages& g,const LaneRows& lr,
                   int T,float* scratch,void* stream,const int* cells,const int* ncells,int card,const int* pos0) {
    Part* parts=(Part*)(scratch+(size_t)64*NQH*SPLITS*HD);
#define RUN(CH) k_attn_mma<CH><<<dim3(SPLITS,T,NKVH),256,0,(cudaStream_t)stream>>>(q,(const half*)k,(const half*)v,g,lr,T,scratch,parts,cells,ncells,card,pos0)
    if(T<=2){RUN(32);}else{RUN(16);}
#undef RUN
    const auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
void attn_mma_prepare() {
    cudaFuncAttributes a{};int blocks=0;
    cudaFuncGetAttributes(&a,k_attn_mma<16>);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k_attn_mma<16>,256,0);
    std::printf("prec attention CH16 regs=%d local=%zu shared=%zu blocks=%d\n",a.numRegs,a.localSizeBytes,a.sharedSizeBytes,blocks);
}
}
