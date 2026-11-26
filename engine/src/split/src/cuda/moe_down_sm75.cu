// Experimental Turing INT8 Tensor Core expert down projection. MIT.
// Quantization follows moe.cuh; MMA layout follows PTX and dense_i8_tc.cu.
#include "../kernels.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fnx {
namespace {
struct DownQ81 { half2 ds; int8_t qs[32]; };

__device__ __forceinline__ float down_warp_max(float v) {
#pragma unroll
    for(int o=16;o;o>>=1) v=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,o));
    return v;
}
__device__ __forceinline__ float down_warp_sum(float v) {
#pragma unroll
    for(int o=16;o;o>>=1) v+=__shfl_xor_sync(0xffffffffu,v,o);
    return v;
}
__device__ __forceinline__ unsigned q2_expand4(unsigned v) {
    const unsigned sel=(v&3)|((v&12)<<2)|((v&48)<<4)|((v&192)<<6);
    return __byte_perm(0x020100ffu,0x020100ffu,sel);
}
__device__ __forceinline__ uint2 iq4_expand8(unsigned v) {
    const unsigned sel=0x32103210u|((v&0x88888888u)>>1);
    const unsigned a=__byte_perm(0xbfad9881u,0xf6eaddcfu,v);
    const unsigned b=__byte_perm(0x26190d01u,0x71594535u,v);
    const unsigned c=__byte_perm(a,b,sel);
    const unsigned d=__byte_perm(0xbfad9881u,0xf6eaddcfu,v>>16);
    const unsigned e=__byte_perm(0x26190d01u,0x71594535u,v>>16);
    const unsigned f=__byte_perm(d,e,sel>>16);
    return make_uint2(__byte_perm(c,f,0x6420),__byte_perm(c,f,0x7531));
}
__device__ __forceinline__ int2 down_dot32(unsigned a0,unsigned a1,unsigned b0,unsigned b1) {
    int2 d=make_int2(0,0);
    asm volatile(
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%4}, {%5}, {%0,%1};\n"
        : "+r"(d.x),"+r"(d.y):"r"(a0),"r"(b0),"r"(a1),"r"(b1));
    return d;
}

template<int TYPE,int RPW,int WARPS>
__global__ void __launch_bounds__(WARPS*32) k_moe_down_sm75(
        DW w,size_t expert_stride,const Job* __restrict__ jobs,const int* njobs,
        const float* __restrict__ gu,float* __restrict__ out) {
    if((int)blockIdx.y>=*njobs) return;
    __shared__ Job jb;
    __shared__ DownQ81 h[MAXT][FF_CARD_MAX/32];
    for(int i=threadIdx.x;i<sizeof(Job)/sizeof(int);i+=WARPS*32)
        reinterpret_cast<int*>(&jb)[i]=reinterpret_cast<const int*>(jobs+blockIdx.y)[i];
    __syncthreads();
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,nb=w.n_in/32;
    for(int q=warp;q<jb.n*nb;q+=WARPS) {
        const int j=q/nb,b=q%nb;
        const float* g=gu+(size_t)jb.y[j]*2*w.n_in;
        const float gate=g[b*32+lane];
        const float v=(gate/(1.f+__expf(-gate)))*g[w.n_in+b*32+lane];
        const float am=down_warp_max(fabsf(v)),sum=down_warp_sum(v),d=am/127.f;
        h[j][b].qs[lane]=am==0?0:(int8_t)roundf(v/d);
        if(!lane) h[j][b].ds=__floats2half2_rn(d,sum);
    }
    __syncthreads();
    const int g=lane>>2,k4=lane&3,row=(blockIdx.x*WARPS+warp)*RPW+g;
    const char* base=w.ep?w.ep[jb.w]:(const char*)w.p+(size_t)jb.w*expert_stride;
    // Jobs usually hold 1-4 rows. Each 8-row tile needs just two int32
    // accumulators per lane; larger jobs reuse the quantized shared rows.
    for(int nt=0;nt<jb.n;nt+=8) {
        float acc[RPW/8][2]={};
        for(int b=0;b<nb;++b) {
            const DownQ81& xb=h[nt+min(g,jb.n-nt-1)][b];
            const unsigned* xp=reinterpret_cast<const unsigned*>(xb.qs);
            const unsigned bx0=xp[k4],bx1=xp[k4+4];
            const int t0=nt+2*k4,t1=t0+1;
            const float dx0=__low2float(h[min(t0,jb.n-1)][b].ds);
            const float dx1=__low2float(h[min(t1,jb.n-1)][b].ds);
#pragma unroll
            for(int r=0;r<RPW/8;++r) {
                const int rr=min(row+r*8,w.n_out-1);
                const uint8_t* wr=(const uint8_t*)base+(size_t)rr*w.row_bytes;
                unsigned a0,a1;
                float dw;
                if constexpr(TYPE==42) {
                    const uint8_t* wb=wr+(b/2)*18;
                    dw=__half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(wb)));
                    a0=q2_expand4(wb[2+(b&1)*8+k4]);
                    a1=q2_expand4(wb[2+(b&1)*8+4+k4]);
                } else {
                    const uint8_t* wb=wr+b*18;
                    dw=__half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(wb)));
                    // Q4 blocks are two-byte aligned, including their codes.
                    const uint16_t* p=(const uint16_t*)(wb+2+k4*4);
                    const uint2 a=iq4_expand8((unsigned)p[0]|((unsigned)p[1]<<16));
                    a0=a.x; a1=a.y;
                }
                const int2 d=down_dot32(a0,a1,bx0,bx1);
                acc[r][0]=fmaf(dw*dx0,(float)d.x,acc[r][0]);
                acc[r][1]=fmaf(dw*dx1,(float)d.y,acc[r][1]);
            }
        }
#pragma unroll
        for(int r=0;r<RPW/8;++r) {
            const int rr=row+r*8,t=nt+2*k4;
            if(rr<w.n_out && t<jb.n) out[(size_t)jb.out[t]*w.n_out+rr]=acc[r][0];
            if(rr<w.n_out && t+1<jb.n) out[(size_t)jb.out[t+1]*w.n_out+rr]=acc[r][1];
        }
    }
}

template<int TYPE> void down_launch(const DW& w,size_t stride,const Job* jobs,const int* njobs,
                                    int cap,const float* gu,float* out,cudaStream_t s,int variant) {
#define FNX_DOWN_TC(R,W) k_moe_down_sm75<TYPE,R,W><<<dim3((w.n_out+R*W-1)/(R*W),cap),W*32,0,s>>>(w,stride,jobs,njobs,gu,out)
    switch(variant) {
        case 0: FNX_DOWN_TC(8,4); break;
        case 1: FNX_DOWN_TC(16,4); break;
        case 2: FNX_DOWN_TC(8,8); break;
        case 3: FNX_DOWN_TC(16,8); break;
        default: throw std::runtime_error("moe_down_sm75: unknown variant");
    }
#undef FNX_DOWN_TC
}
} // namespace

void moe_down_tc_sm75(const DW& w,size_t stride,const Job* jobs,const int* njobs,int cap,
                       const float* gu,float* out,void* stream,int variant) {
    if(w.n_in<64 || w.n_in>FF_CARD_MAX || w.n_in%64 || w.n_out<1 || cap<1 || cap>MAXT*TOPK)
        throw std::runtime_error("moe_down_sm75: unsupported shape");
    if(w.type==42) down_launch<42>(w,stride,jobs,njobs,cap,gu,out,(cudaStream_t)stream,variant);
    else if(w.type==20) down_launch<20>(w,stride,jobs,njobs,cap,gu,out,(cudaStream_t)stream,variant);
    else throw std::runtime_error("moe_down_sm75: unsupported quantization");
    const cudaError_t e=cudaGetLastError();
    if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
} // namespace fnx
