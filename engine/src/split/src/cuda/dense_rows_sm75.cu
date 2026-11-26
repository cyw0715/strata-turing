// Turing row-group tuning of dense.cuh; arithmetic is the existing DP4A path (MIT).
#include "kernels.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>
namespace fnx {
struct block_q8_1 { half2 ds; int8_t qs[32]; };
struct Q8Twin { const char* q8; size_t bytes; int8_t* tq; float* td; };
std::vector<Q8Twin>& q8_twins();
namespace {
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for(int o=16;o;o>>=1) v+=__shfl_xor_sync(0xffffffffu,v,o);
    return v;
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for(int o=16;o;o>>=1) v=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,o));
    return v;
}
__device__ __forceinline__ float silu_(float x) { return x/(1.f+__expf(-x)); }
}
#include "dense.cuh"
namespace {

template<int T,bool SPLIT> void rows_launch(Q8Segs g,int rows,int cols,const block_q8_1* y,
                    const int8_t* tq,const float* td,cudaStream_t s) {
    k_q8s<T,2,SPLIT,false><<<(rows+15)/16,256,0,s>>>(g,rows,cols,y,tq,td);
}
template<bool SPLIT> void rows_run(Q8Segs g,int rows,int cols,const block_q8_1* y,
                    const int8_t* tq,const float* td,int T,cudaStream_t s) {
#define DENSE_RUN(T) case T: rows_launch<T,SPLIT>(g,rows,cols,y,tq,td,s); break
    switch(T) { DENSE_RUN(1); DENSE_RUN(2); DENSE_RUN(3); DENSE_RUN(4); DENSE_RUN(5); DENSE_RUN(6); DENSE_RUN(7); DENSE_RUN(8);
                DENSE_RUN(9); DENSE_RUN(10); DENSE_RUN(11); DENSE_RUN(12); DENSE_RUN(13); DENSE_RUN(14); DENSE_RUN(15); DENSE_RUN(16); }
#undef DENSE_RUN
}
} // namespace
void q8s_rows_sm75(Q8S w,int rows,int cols,const void* y,float* out,int stride,int T,void* stream) {
    if(rows<1 || cols%32 || stride<rows || T<1 || T>16) throw std::runtime_error("dense rows: unsupported shape");
    Q8Segs g{};
    for (int i=0;i<3;++i) g.q[i]=w.q,g.d[i]=w.d,g.out[i]=out,g.end[i]=rows,g.stride[i]=stride;
    const int8_t* tq=nullptr; const float* td=nullptr;
    for(const auto& twin:q8_twins()) if((char*)y>=twin.q8 && (char*)y<twin.q8+twin.bytes) {
        const size_t off=(char*)y-twin.q8;
        if(off%q8_1_bytes(cols)) throw std::runtime_error("dense rows: unaligned activation row");
        tq=twin.tq+off/36*32; td=twin.td+off/36; break;
    }
    if(tq) rows_run<true>(g,rows,cols,(const block_q8_1*)y,tq,td,T,(cudaStream_t)stream);
    else rows_run<false>(g,rows,cols,(const block_q8_1*)y,nullptr,nullptr,T,(cudaStream_t)stream);
    const cudaError_t e=cudaGetLastError();
    if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
} // namespace fnx
