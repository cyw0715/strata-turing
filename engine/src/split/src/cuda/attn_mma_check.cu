#include "kernels.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
namespace fnx {
void attn_mma_run(const float*,const uint16_t*,const uint16_t*,const KvPages&,const LaneRows&,int,float*,void*,const int*,const int*,int,const int*);
void attn_warps_part(const float*,const uint16_t*,const uint16_t*,const KvPages&,const LaneRows&,int,float*,void*,const int*,const int*,int,int);
void attn_warps_single(const float*,const uint16_t*,const uint16_t*,const KvPages&,const int*,int,float*,void*,const int*,const int*,int);
namespace {
struct Stats {unsigned long long heads,failures;float relative,normalization;};
struct Check {float* scratch=nullptr;Stats* stats=nullptr;};
Check checks[16];
constexpr size_t MZ=(size_t)64*NQH*16*HD;
void ck(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__device__ float sum(float v) {for(int n=16;n;n>>=1)v+=__shfl_xor_sync(0xffffffffu,v,n);return v;}
__device__ float maximum(float v) {for(int n=16;n;n>>=1)v=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,n));return v;}
__global__ void compare(const float* reference,const float* candidate,Stats* stats) {
    const int h=blockIdx.x,i=threadIdx.x;
    __shared__ float weights[2][16],denom[2],normalization[2];
    __shared__ float errors[256],norms[256];
    __shared__ unsigned bad[256];
    if(i<32)for(int side=0;side<2;++side) {
        const float* at=side?candidate:reference;
        const float m=i<16?at[MZ+(h*16+i)*2]:-INFINITY;
        const float z=i<16?at[MZ+(h*16+i)*2+1]:0;
        const float mx=maximum(m),f=m==-INFINITY?0:__expf(m-mx),d=sum(z*f);
        if(i<16)weights[side][i]=f;
        if(i==0){denom[side]=d;normalization[side]=d>0?mx+logf(d):-INFINITY;}
    }
    __syncthreads();
    float a=0,b=0;
    for(int split=0;split<16;++split) {
        a+=reference[((size_t)h*16+split)*HD+i]*weights[0][split];
        b+=candidate[((size_t)h*16+split)*HD+i]*weights[1][split];
    }
    if(denom[0]>0)a/=denom[0];if(denom[1]>0)b/=denom[1];
    errors[i]=(a-b)*(a-b);norms[i]=a*a;
    bad[i]=!isfinite(a)||!isfinite(b)||(denom[0]>0)!=(denom[1]>0);
    __syncthreads();
    for(int n=128;n;n>>=1) {
        if(i<n){errors[i]+=errors[i+n];norms[i]+=norms[i+n];bad[i]|=bad[i+n];}
        __syncthreads();
    }
    if(i==0) {
        const float rel=sqrtf(errors[0]/fmaxf(norms[0],1.e-20f));
        const float dn=denom[0]>0 && denom[1]>0?fabsf(normalization[0]-normalization[1]):0;
        const bool fail=bad[0] || !isfinite(rel) || !isfinite(dn) || rel>1.e-3f || dn>1.e-2f;
        atomicAdd(&stats->heads,1ull);if(fail)atomicAdd(&stats->failures,1ull);
        atomicMax((unsigned*)&stats->relative,__float_as_uint(rel));
        atomicMax((unsigned*)&stats->normalization,__float_as_uint(dn));
    }
}
}
size_t attn_mma_check_bytes() {return attn_scratch_floats(NQH)*sizeof(float)+256;}
void attn_mma_check_bind(void* memory) {
    int dev=-1;ck(cudaGetDevice(&dev));if(dev<0 || dev>=16)throw std::runtime_error("attention check device");
    checks[dev]=Check{(float*)memory,(Stats*)((char*)memory+attn_scratch_floats(NQH)*sizeof(float))};
}
void attn_mma_dispatch(const float* q,const uint16_t* k,const uint16_t* v,const KvPages& g,const LaneRows& lr,
                       int T,float* scratch,void* stream,const int* cells,const int* ncells,int card,const int* pos0,bool check) {
    Check c;
    if(check) {
        int dev=-1;ck(cudaGetDevice(&dev));if(dev<0 || dev>=16 || !checks[dev].scratch)throw std::runtime_error("attention check buffers not bound");
        c=checks[dev];
        if(pos0)attn_warps_single(q,k,v,g,pos0,T,c.scratch,stream,cells,ncells,card);
        else attn_warps_part(q,k,v,g,lr,T,c.scratch,stream,cells,ncells,card,0);
    }
    attn_mma_run(q,k,v,g,lr,T,scratch,stream,cells,ncells,card,pos0);
    if(check) {
        compare<<<T*NQH,256,0,(cudaStream_t)stream>>>(c.scratch,scratch,c.stats);
        ck(cudaGetLastError());
    }
}
void attn_mma_check_report(int dev) {
    if(dev<0 || dev>=16 || !checks[dev].scratch)return;
    ck(cudaSetDevice(dev));Stats s{};ck(cudaMemcpy(&s,checks[dev].stats,sizeof(s),cudaMemcpyDeviceToHost));
    std::fprintf(stderr,"fnx: attention MMA shadow: dev=%d heads=%llu failures=%llu relative=%.8g lognorm=%.8g\n",
                 dev,s.heads,s.failures,s.relative,s.normalization);
    if(s.failures)throw std::runtime_error("attention MMA numerical validation failed");
}
}
