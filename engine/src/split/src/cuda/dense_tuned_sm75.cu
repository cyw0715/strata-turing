// Same 16-value dot, FMA order, and XOR tree as dense.cuh; vary scheduling only.
#include "kernels.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
namespace fnx {
struct Q8Twin {const char* q8;size_t bytes;int8_t* tq;float* td;};
std::vector<Q8Twin>& q8_twins();
namespace {
struct Q81 {half2 ds;int8_t qs[32];};
template<int T,int R,int W,int U,bool PAIR,bool BOUND,int C>
__global__ void __launch_bounds__(W*32,BOUND?32/W:1) k_dense_tuned(Q8S w,int rows,const int8_t* x,
                                                               const float* dx,float* out,int stride) {
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,row0=(blockIdx.x*W+warp)*R;
    float acc[R][T]={};
#pragma unroll U
    for(int c=lane*16;c<C;c+=512) {
        int4 a[T];float da[T];
#pragma unroll
        for(int t=0;t<T;++t) {
            a[t]=__ldg((const int4*)(x+(size_t)t*C+c));
            da[t]=__ldg(dx+(size_t)t*(C/32)+c/32);
        }
#pragma unroll
        for(int r=0;r<R;++r) {
            if(row0+r>=rows) break;
            const int4 b=__ldg((const int4*)(w.q+(size_t)(row0+r)*C+c));
            const float db=__half2float(__ushort_as_half(__ldg(w.d+(size_t)(row0+r)*(C/32)+c/32)));
#pragma unroll
            for(int t=0;t<T;++t) {
                int d;
                if constexpr(PAIR) {
                    const int lo=__dp4a(b.y,a[t].y,__dp4a(b.x,a[t].x,0));
                    const int hi=__dp4a(b.w,a[t].w,__dp4a(b.z,a[t].z,0));
                    d=lo+hi;
                } else {
                    d=__dp4a(b.x,a[t].x,0);d=__dp4a(b.y,a[t].y,d);
                    d=__dp4a(b.z,a[t].z,d);d=__dp4a(b.w,a[t].w,d);
                }
                acc[r][t]=fmaf(db*da[t],(float)d,acc[r][t]);
            }
        }
    }
#pragma unroll
    for(int r=0;r<R;++r)
#pragma unroll
        for(int t=0;t<T;++t) {
            float v=acc[r][t];
#pragma unroll
            for(int o=16;o;o>>=1) v+=__shfl_xor_sync(0xffffffffu,v,o);
            if(!lane && row0+r<rows) out[(size_t)t*stride+row0+r]=v;
        }
}
template<int T,int C> void launch(Q8S w,int rows,const int8_t* x,const float* dx,float* out,int stride,cudaStream_t s,int v) {
#define RUN(R,W,U,P,B) k_dense_tuned<T,R,W,U,P,B,C><<<(rows+R*W-1)/(R*W),W*32,0,s>>>(w,rows,x,dx,out,stride)
    switch(v) {
        case 0:RUN(4,4,1,false,false);break;
        case 1:RUN(4,8,1,false,false);break;
        case 2:RUN(2,4,1,false,false);break;
        case 3:RUN(2,8,1,false,false);break;
        case 4:RUN(4,8,2,true,false);break;
        case 5:RUN(4,8,1,true,false);break;
        case 6:RUN(8,4,1,false,false);break;
        case 7:RUN(4,8,1,false,true);break;
        case 8:RUN(2,8,1,false,true);break;
        default:throw std::runtime_error("dense tune: invalid variant");
    }
#undef RUN
}
template<int C> void tokens(Q8S w,int rows,const int8_t* x,const float* dx,float* out,int stride,int T,cudaStream_t s,int v) {
#define RUN(T) case T:launch<T,C>(w,rows,x,dx,out,stride,s,v);break
    switch(T) {RUN(1);RUN(3);RUN(4);RUN(7);default:throw std::runtime_error("dense tune: unsupported token count");}
#undef RUN
}
}
void q8s_tuned_sm75(Q8S w,int rows,int cols,const void* q,float* out,int stride,int T,void* stream,int v) {
    const int8_t* x=nullptr;const float* dx=nullptr;
    for(const auto& twin:q8_twins()) if((const char*)q>=twin.q8 && (const char*)q<twin.q8+twin.bytes) {
        const size_t off=(const char*)q-twin.q8;
        if(off%q8_1_bytes(cols)) throw std::runtime_error("dense tune: activation offset");
        x=twin.tq+off/36*32;dx=twin.td+off/36;break;
    }
    if(!x || stride<rows || rows<1) throw std::runtime_error("dense tune: requires split Q8 activations");
    if(cols==2560) tokens<2560>(w,rows,x,dx,out,stride,T,(cudaStream_t)stream,v);
    else if(cols==3072) tokens<3072>(w,rows,x,dx,out,stride,T,(cudaStream_t)stream,v);
    else throw std::runtime_error("dense tune: unsupported columns");
    const cudaError_t e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
}

namespace fnx {
bool q8s_tuned_try_sm75(Q8S w,int rows,int cols,const void* q,float* out,int stride,int T,void* stream) {
    for(const auto& twin:q8_twins()) if((const char*)q>=twin.q8 && (const char*)q<twin.q8+twin.bytes) {
        q8s_tuned_sm75(w,rows,cols,q,out,stride,T,stream,2);return true;
    }
    return false;
}
}
