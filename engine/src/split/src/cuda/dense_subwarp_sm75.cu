// Q8_0 x Q8_1 decode geometry for Turing. Arithmetic follows Strata/sFNX dense.cuh (MIT).
// Small-CTA/unroll-2 idea informed by zongxinqian/strata-ampere q8_sm86_fast.cuh,
// commit ef065d695d9525d1702eda5d375845ee9e7fba95 (MIT, Strata contributors).
// Adapted to the split engine's Q8 twins; no Ampere-only instructions.
#include "kernels.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

namespace fnx {
struct SwQ81 { half2 ds; int8_t qs[32]; };
struct Q8Twin { const char* q8; size_t bytes; int8_t* tq; float* td; };
std::vector<Q8Twin>& q8_twins();
namespace {
template<int T,int WARPS,bool SPLIT>
__global__ void __launch_bounds__(WARPS * 32, WARPS == 1 ? 16 : 32 / WARPS)
k_dense_subwarp(Q8S w,int rows,int cols,const SwQ81* __restrict__ y,
                const int8_t* __restrict__ tq,const float* __restrict__ td,float* out,int stride) {
    const int lane=threadIdx.x&31, row=blockIdx.x*WARPS+(threadIdx.x>>5), nb=cols/32;
    if(row>=rows) return;
    const int4* weights=reinterpret_cast<const int4*>(w.q+(size_t)row*cols);
    const uint16_t* scales=w.d+(size_t)row*nb;
    float acc[T] = {};
    auto add = [&](int half,int4 weight,float dw) {
#pragma unroll
        for(int t=0;t<T;++t) {
            int4 x;
            float dx;
            if constexpr(SPLIT) {
                x=__ldg(reinterpret_cast<const int4*>(tq+(size_t)t*cols)+half);
                dx=__ldg(td+(size_t)t*nb+(half>>1));
            } else {
                const SwQ81* block=y+(size_t)t*nb+(half>>1);
                const int* p=reinterpret_cast<const int*>(block->qs)+(half&1)*4;
                x=make_int4(__ldg(p),__ldg(p+1),__ldg(p+2),__ldg(p+3));
                dx=__low2float(block->ds);
            }
            int dot=__dp4a(weight.x,x.x,0);
            dot=__dp4a(weight.y,x.y,dot);
            dot=__dp4a(weight.z,x.z,dot);
            dot=__dp4a(weight.w,x.w,dot);
            acc[t]=fmaf(dw*dx,(float)dot,acc[t]);
        }
    };
    int h=lane;
    for(;h+32<cols/16;h+=64) {
        const int4 a=__ldg(weights+h), b=__ldg(weights+h+32);
        const float da=__half2float(__ushort_as_half(__ldg(scales+(h>>1))));
        const float db=__half2float(__ushort_as_half(__ldg(scales+((h+32)>>1))));
        add(h,a,da);
        add(h+32,b,db);
    }
    if(h<cols/16) add(h,__ldg(weights+h),__half2float(__ushort_as_half(__ldg(scales+(h>>1)))));
#pragma unroll
    for(int t=0;t<T;++t) {
        float v=acc[t];
#pragma unroll
        for(int offset=16;offset;offset>>=1) v+=__shfl_xor_sync(0xffffffffu,v,offset);
        if(lane==0) out[(size_t)t*stride+row]=v;
    }
}
template<int T,bool SPLIT> void sw_launch(Q8S w,int rows,int cols,const SwQ81* y,const int8_t* tq,const float* td,
                                        float* out,int stride,cudaStream_t s,int variant) {
#define SW_LAUNCH(W) k_dense_subwarp<T,W,SPLIT><<<(rows+W-1)/W,W*32,0,s>>>(w,rows,cols,y,tq,td,out,stride)
    switch(variant) {
        case 0: SW_LAUNCH(1); break;
        case 1: SW_LAUNCH(2); break;
        case 2: SW_LAUNCH(4); break;
        case 3: SW_LAUNCH(8); break;
    }
#undef SW_LAUNCH
}
template<bool SPLIT> void sw_run(Q8S w,int rows,int cols,const SwQ81* y,const int8_t* tq,const float* td,
                                 float* out,int stride,int T,cudaStream_t s,int variant) {
#define SW_T(T) case T: sw_launch<T,SPLIT>(w,rows,cols,y,tq,td,out,stride,s,variant); break
    switch(T) { SW_T(1); SW_T(2); SW_T(3); SW_T(4); SW_T(5); SW_T(6); SW_T(7); SW_T(8); }
#undef SW_T
}
} // namespace
void q8s_subwarp_sm75(Q8S w,int rows,int cols,const void* y,float* out,int stride,int T,void* stream,int variant) {
    if(rows<1 || cols<32 || cols%32 || stride<rows || T<1 || T>8 || variant<0 || variant>3)
        throw std::runtime_error("dense subwarp: unsupported shape or variant");
    const int8_t* tq=nullptr;
    const float* td=nullptr;
    for(const auto& twin:q8_twins()) if((const char*)y>=twin.q8 && (const char*)y<twin.q8+twin.bytes) {
        const size_t off=(const char*)y-twin.q8;
        if(off%q8_1_bytes(cols)) throw std::runtime_error("dense subwarp: unaligned activation row");
        tq=twin.tq+off/36*32;
        td=twin.td+off/36;
        break;
    }
    if(tq) sw_run<true>(w,rows,cols,(const SwQ81*)y,tq,td,out,stride,T,(cudaStream_t)stream,variant);
    else sw_run<false>(w,rows,cols,(const SwQ81*)y,nullptr,nullptr,out,stride,T,(cudaStream_t)stream,variant);
    const cudaError_t e=cudaGetLastError();
    if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

// Runtime dispatch only uses the candidate for the measured split-Q8 format.
bool q8s_subwarp_try_sm75(Q8S w,int rows,int cols,const void* y,float* out,int stride,int T,void* stream,int variant) {
    for(const auto& twin:q8_twins()) if((const char*)y>=twin.q8 && (const char*)y<twin.q8+twin.bytes) {
        q8s_subwarp_sm75(w,rows,cols,y,out,stride,T,stream,variant);
        return true;
    }
    return false;
}
} // namespace fnx
