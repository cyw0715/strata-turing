// Turing INT8 expert gate/up candidate (MIT). Layouts and integer rounding
// follow qdot.cuh/native_quant.cuh; MMA fragments follow dense_i8_tc.cu.
#include "../kernels.hpp"
#include "../qdot.cuh"
#include "native_quant.cuh"

namespace fnx {
namespace {
__device__ __forceinline__ unsigned gu_ld32(const uint8_t* p) {
    const uint16_t* q=(const uint16_t*)p;
    return (unsigned)q[0]|((unsigned)q[1]<<16);
}
__device__ __forceinline__ unsigned gu_sign(unsigned v,unsigned signs) {
    const unsigned b=((signs&15)*0x00204081u)&0x01010101u,m=(b<<8)-b;
    return (v^m)+b;
}
__device__ __forceinline__ int2 gu_dot16(unsigned a,unsigned b) {
    int2 d=make_int2(0,0);
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(d.x),"+r"(d.y):"r"(a),"r"(b));
    return d;
}
template<int TYPE> __host__ __device__ constexpr int gu_grid_words() {
    return TYPE==16?512:TYPE==17?1024:TYPE==18?256:TYPE==21?512:2048;
}
template<int TYPE> __host__ __device__ constexpr int gu_block_bytes() {
    return TYPE==16?66:TYPE==17?74:TYPE==18?98:TYPE==21?110:82;
}
struct GuCodes { unsigned a0,a1; int s0,s1; float d; };
template<int TYPE> __device__ __forceinline__ GuCodes gu_decode(
        const uint8_t* wr,int b,int k4,const unsigned* grid,const uint8_t* sign_lut) {
    const int sb=b&7,j=k4>>1,half=k4&1;
    const uint8_t* w=wr+(b/8)*gu_block_bytes<TYPE>();
    GuCodes a{};
    a.d=__half2float(__ushort_as_half(*(const uint16_t*)w));
    if constexpr(TYPE==16) {
        const unsigned q=gu_ld32(w+2+8*sb),aux=gu_ld32(w+6+8*sb);
        a.a0=gu_sign(grid[2*((q>>(8*j))&255)+half],sign_lut[(aux>>(7*j))&127]>>(4*half));
        a.a1=gu_sign(grid[2*((q>>(8*(j+2)))&255)+half],sign_lut[(aux>>(7*(j+2)))&127]>>(4*half));
        a.s0=(aux>>27)|1;
    } else if constexpr(TYPE==17) {
        const uint16_t* q=(const uint16_t*)(w+2+8*sb);
        const unsigned q0=q[j],q1=q[j+2],s=w[66+sb];
        a.a0=gu_sign(grid[2*(q0&511)+half],sign_lut[q0>>9]>>(4*half));
        a.a1=gu_sign(grid[2*(q1&511)+half],sign_lut[q1>>9]>>(4*half));
        a.s0=s&15; a.s1=s>>4;
    } else if constexpr(TYPE==18) {
        const unsigned aux=gu_ld32(w+66+4*sb);
        a.a0=gu_sign(grid[w[2+8*sb+k4]],sign_lut[(aux>>(7*j))&127]>>(4*half));
        a.a1=gu_sign(grid[w[2+8*sb+4+k4]],sign_lut[(aux>>(7*(j+2)))&127]>>(4*half));
        a.s0=aux>>28;
    } else if constexpr(TYPE==21) {
        const unsigned qh=w[66+sb];
        a.a0=gu_sign(grid[w[2+8*sb+k4]|(((qh>>k4)&1)<<8)],w[74+4*sb+j]>>(4*half));
        a.a1=gu_sign(grid[w[2+8*sb+4+k4]|(((qh>>(4+k4))&1)<<8)],w[74+4*sb+2+j]>>(4*half));
        a.s0=1+2*((w[106+sb/2]>>(4*(sb&1)))&15);
    } else {
        const unsigned qh=w[66+sb],s=w[74+sb];
        const int q0=w[2+4*sb+j]|(((qh>>(2*j))&3)<<8);
        const int q1=w[2+4*sb+j+2]|(((qh>>(2*(j+2)))&3)<<8);
        a.a0=gu_sign(grid[2*q0+half],w[34+4*sb+j]>>(4*half));
        a.a1=gu_sign(grid[2*q1+half],w[34+4*sb+j+2]>>(4*half));
        a.s0=s&15; a.s1=s>>4;
    }
    return a;
}
template<int TYPE> __device__ __forceinline__ int gu_scale(int lo,int hi,int s0,int s1) {
    if constexpr(TYPE==16) return (lo+hi)*s0/8;
    else if constexpr(TYPE==17 || TYPE==22) return (lo*s0+hi*s1+(lo+hi)/2)/4;
    else if constexpr(TYPE==18) { const int sum=lo+hi; return (s0*sum+sum/2)/2; }
    else return (lo+hi)*s0;
}
template<int TYPE,int RPW,int WARPS> __device__ void gu_single(
        const DW& w,const uint8_t* base,const Job& jb,const block_q8_1* x,float* out) {
    constexpr int per_block=qtraits<(GgmlType)TYPE>::qi/2;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    for(int r=warp;r<RPW;r+=WARPS) {
        const int row=blockIdx.x*RPW+r;
        if(row>=w.n_out) continue;
        const void* wr=base+(size_t)row*w.row_bytes;
        float sum=0.f;
#pragma unroll 8
        for(int b=lane/per_block;b<w.n_in/256;b+=32/per_block) {
            const block_q8_1* y=x+(size_t)jb.y[0]*(w.n_in/32)+b*8;
            const int q=2*(lane%per_block);
            if constexpr(TYPE==16) sum+=vec_dot_iq2_xxs_q8_1(wr,y,b,q);
            else if constexpr(TYPE==17) sum+=vec_dot_iq2_xs_q8_1(wr,y,b,q);
            else if constexpr(TYPE==18) sum+=vec_dot_iq3_xxs_q8_1(wr,y,b,q);
            else if constexpr(TYPE==21) sum+=vec_dot_iq3_s_q8_1(wr,y,b,q);
            else sum+=vec_dot_iq2_s_q8_1(wr,y,b,q);
        }
#pragma unroll
        for(int o=16;o;o>>=1) sum+=__shfl_xor_sync(0xffffffffu,sum,o);
        if(!lane) out[(size_t)jb.out[0]*w.n_out+row]=sum;
    }
}
struct GuSharedDown {
    Q8S w{};
    const float* gu = nullptr;
    float* out = nullptr;
    int ffs = 0, T = 0;
};
__device__ __forceinline__ float gu_warp_max(float v) {
#pragma unroll
    for (int o=16;o;o>>=1) v=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,o));
    return v;
}
__device__ __forceinline__ float gu_warp_sum(float v) {
#pragma unroll
    for (int o=16;o;o>>=1) v+=__shfl_xor_sync(0xffffffffu,v,o);
    return v;
}
template<int WARPS> __device__ void gu_shared_down(const GuSharedDown& sd,int tile,block_q8_1* h) {
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,nb=sd.ffs/32;
    for (int q=warp;q<sd.T*nb;q+=WARPS) {
        const int t=q/nb,b=q%nb;
        const float* row=sd.gu+(size_t)t*2*sd.ffs;
        const float gate=row[b*32+lane];
        const float v=gate/(1.f+__expf(-gate))*row[sd.ffs+b*32+lane];
        const float am=gu_warp_max(fabsf(v)),sum=gu_warp_sum(v),d=am/127.f;
        h[q].qs[lane]=am==0.f?0:(int8_t)roundf(v/d);
        if (!lane) h[q].ds=__floats2half2_rn(d,sum);
    }
    __syncthreads();
    const int g=lane>>2,k4=lane&3,r=(tile*WARPS+warp)*8+g;
    const int safe=min(r,N-1);
    for (int nt=0;nt<sd.T;nt+=8) {
        const int t0=nt+2*k4,t1=t0+1;
        float v0=0.f,v1=0.f;
        for (int b=0;b<nb;++b) {
            const unsigned* w=(const unsigned*)(sd.w.q+(size_t)safe*sd.ffs+b*32);
            const unsigned* x=(const unsigned*)h[(nt+min(g,sd.T-nt-1))*nb+b].qs;
            const int2 lo=gu_dot16(w[k4],x[k4]),hi=gu_dot16(w[k4+4],x[k4+4]);
            const float dw=__half2float(__ushort_as_half(sd.w.d[(size_t)safe*nb+b]));
            v0=fmaf(dw*__low2float(h[min(t0,sd.T-1)*nb+b].ds),(float)(lo.x+hi.x),v0);
            v1=fmaf(dw*__low2float(h[min(t1,sd.T-1)*nb+b].ds),(float)(lo.y+hi.y),v1);
        }
        if (r<N && t0<sd.T) sd.out[(size_t)t0*N+r]=v0;
        if (r<N && t1<sd.T) sd.out[(size_t)t1*N+r]=v1;
    }
}
template<int TYPE,int RPW,int WARPS> struct GuScratch {
    Job jb;
    unsigned grid[gu_grid_words<TYPE>()];
    uint8_t signs[128];
    float partial[WARPS][RPW/8][32][2];
};
template<int TYPE,int RPW,int WARPS,bool FUSED=false>
__global__ void __launch_bounds__(WARPS*32) k_moe_gu_sm75(
        DW w,size_t stride,const Job* __restrict__ jobs,const int* njobs,
        const block_q8_1* __restrict__ x,float* __restrict__ out,GuSharedDown sd={},int sd_rows=0) {
    constexpr size_t core_bytes=sizeof(GuScratch<TYPE,RPW,WARPS>);
    constexpr size_t shared_bytes=MAXT*(SH_MAXF/32)*sizeof(block_q8_1);
    constexpr size_t bytes=FUSED && shared_bytes>core_bytes?shared_bytes:core_bytes;
    __shared__ __align__(16) char scratch[bytes];
    int ji=(int)blockIdx.y;
    if constexpr(FUSED) {
        if (ji<sd_rows) {
            gu_shared_down<WARPS>(sd,ji*gridDim.x+blockIdx.x,(block_q8_1*)scratch);
            return;
        }
        ji-=sd_rows;
    }
    if(ji>=*njobs) return;
    auto& core=*reinterpret_cast<GuScratch<TYPE,RPW,WARPS>*>(scratch);
    auto& jb=core.jb;
    auto& grid=core.grid;
    auto& signs=core.signs;
    auto& partial=core.partial;
    for(int i=threadIdx.x;i<sizeof(Job)/4;i+=WARPS*32)
        ((int*)&jb)[i]=((const int*)(jobs+ji))[i];
    __syncthreads();
    const uint8_t* base=(const uint8_t*)(w.ep?w.ep[jb.w]:(const char*)w.p+(size_t)jb.w*stride);
    if(jb.n==1) {
        gu_single<TYPE,RPW,WARPS>(w,base,jb,x,out);
        return;
    }
    for(int i=threadIdx.x;i<gu_grid_words<TYPE>();i+=WARPS*32) {
        if constexpr(TYPE==16) grid[i]=((const unsigned*)iq2xxs_grid)[i];
        else if constexpr(TYPE==17) grid[i]=((const unsigned*)iq2xs_grid)[i];
        else if constexpr(TYPE==18) grid[i]=iq3xxs_grid[i];
        else if constexpr(TYPE==21) grid[i]=iq3s_grid[i];
        else grid[i]=((const unsigned*)iq2s_grid)[i];
    }
    if(threadIdx.x<128) signs[threadIdx.x]=ksigns_iq2xs[threadIdx.x];
    __syncthreads();
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,k4=lane&3,g=lane>>2,nb=w.n_in/32;
    const int row=blockIdx.x*RPW+g;
    for(int nt=0;nt<jb.n;nt+=8) {
        float acc[RPW/8][2]={};
        for(int b=warp;b<nb;b+=WARPS) {
            const int t0=nt+2*k4,t1=t0+1;
            const block_q8_1& xb=x[(size_t)jb.y[nt+min(g,jb.n-nt-1)]*nb+b];
            const unsigned* p=(const unsigned*)xb.qs;
            const float dx0=__low2float(x[(size_t)jb.y[min(t0,jb.n-1)]*nb+b].ds);
            const float dx1=__low2float(x[(size_t)jb.y[min(t1,jb.n-1)]*nb+b].ds);
#pragma unroll
            for(int r=0;r<RPW/8;++r) {
                const GuCodes a=gu_decode<TYPE>(base+(size_t)min(row+r*8,w.n_out-1)*w.row_bytes,b,k4,grid,signs);
                const int2 lo=gu_dot16(a.a0,p[k4]),hi=gu_dot16(a.a1,p[k4+4]);
                // The integer divisions are intentional: these match decode's
                // IQ dot rounding, rather than silently changing to real-valued scales.
                const int z0=gu_scale<TYPE>(lo.x,hi.x,a.s0,a.s1),z1=gu_scale<TYPE>(lo.y,hi.y,a.s0,a.s1);
                acc[r][0]=fmaf(a.d*dx0,(float)z0,acc[r][0]);
                acc[r][1]=fmaf(a.d*dx1,(float)z1,acc[r][1]);
            }
        }
#pragma unroll
        for(int r=0;r<RPW/8;++r) {
            partial[warp][r][lane][0]=acc[r][0]; partial[warp][r][lane][1]=acc[r][1];
        }
        __syncthreads();
        if(!warp) {
#pragma unroll
            for(int r=0;r<RPW/8;++r) {
                float v0=0,v1=0;
#pragma unroll
                for(int k=0;k<WARPS;++k) { v0+=partial[k][r][lane][0]; v1+=partial[k][r][lane][1]; }
                const int rr=row+r*8,t=nt+2*k4;
                if(rr<w.n_out && t<jb.n) out[(size_t)jb.out[t]*w.n_out+rr]=v0;
                if(rr<w.n_out && t+1<jb.n) out[(size_t)jb.out[t+1]*w.n_out+rr]=v1;
            }
        }
        __syncthreads();
    }
}
template<int TYPE,bool FUSED=false> void gu_launch(const DW& w,size_t stride,const Job* jobs,const int* njobs,int cap,
                                  const void* x,float* out,cudaStream_t s,int variant,GuSharedDown sd={}) {
#define GU_TC(R,W) do { const int gx=(w.n_out+R-1)/R; const int sy=FUSED?(N+W*8*gx-1)/(W*8*gx):0; \
    k_moe_gu_sm75<TYPE,R,W,FUSED><<<dim3(gx,cap+sy),W*32,0,s>>>(w,stride,jobs,njobs,(const block_q8_1*)x,out,sd,sy); } while(0)
    switch(variant) { case 0: GU_TC(8,4); break; case 1: GU_TC(16,4); break;
                      case 2: GU_TC(8,8); break; case 3: GU_TC(16,8); break;
                      default: throw std::runtime_error("GU sm75: invalid variant"); }
#undef GU_TC
}
} // namespace
void moe_gu_fused_tc_sm75(const DW& w,size_t stride,const Job* jobs,const int* njobs,int cap,
                           const void* x,float* out,void* stream,int variant,const Q8S& shared_down,
                           int ffs,const float* shared_gu,float* shared_out) {
    if (w.n_in<256 || w.n_in%256 || w.n_out<1 || cap<1 || cap>MAXT*TOPK || cap%TOPK ||
        ffs<32 || ffs%32 || ffs>SH_MAXF || !shared_down.q || !shared_down.d || !shared_gu || !shared_out)
        throw std::runtime_error("GU sm75 fused: invalid shape");
    const GuSharedDown sd{shared_down,shared_gu,shared_out,ffs,cap/TOPK};
#define GU_FUSED(T) case T: gu_launch<T,true>(w,stride,jobs,njobs,cap,x,out,(cudaStream_t)stream,variant,sd); break
    switch (w.type) { GU_FUSED(16); GU_FUSED(17); GU_FUSED(18); GU_FUSED(21); GU_FUSED(22);
                     default: throw std::runtime_error("GU sm75 fused: invalid quantization"); }
#undef GU_FUSED
    const cudaError_t e=cudaGetLastError();
    if (e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void moe_gu_tc_sm75(const DW& w,size_t stride,const Job* jobs,const int* njobs,int cap,
                     const void* x,float* out,void* stream,int variant) {
    if(w.n_in%256 || w.n_out<1 || cap<1 || cap>MAXT*TOPK) throw std::runtime_error("GU sm75: invalid shape");
#define GU_TYPE(T) case T: gu_launch<T>(w,stride,jobs,njobs,cap,x,out,(cudaStream_t)stream,variant); break
    switch(w.type) { GU_TYPE(16); GU_TYPE(17); GU_TYPE(18); GU_TYPE(21); GU_TYPE(22);
                     default: throw std::runtime_error("GU sm75: invalid quantization"); }
#undef GU_TYPE
    const cudaError_t e=cudaGetLastError(); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
namespace {
__global__ void k_expert_cache_copy_exact(ExpertCache ec,int pool,const int* list,size_t gu_bytes,size_t dn_bytes) {
    const size_t ng = gu_bytes / 16, per = ng + dn_bytes / 16;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < (size_t) list[0] * per;
         i += (size_t) gridDim.x * blockDim.x) {
        const int g = list[1 + i / per];
        const size_t off = i % per;
        char* at = ec.pool[pool] + (size_t) (ec.slot_of[g] - ec.s0[pool]) * ec.slot_bytes[pool];
        if (off < ng) reinterpret_cast<uint4*>(at)[off] = reinterpret_cast<const uint4*>(ec.host_gu[g])[off];
        else reinterpret_cast<uint4*>(at + ec.dn_off[pool])[off - ng] = reinterpret_cast<const uint4*>(ec.host_dn[g])[off - ng];
    }
}
__global__ void k_expert_cache_count(const int* njobs,const int* list,size_t bytes,unsigned long long* stats) {
    stats[0] += (unsigned) njobs[0];
    stats[1] += (unsigned) list[0];
    stats[2] += (unsigned) list[0] * (unsigned long long) bytes;
    ++stats[3];
}
}
void expert_cache_copy_sm75(const ExpertCache& ec,int pool,const int* list,size_t gu_bytes,size_t dn_bytes,void* stream) {
    if (!gu_bytes || !dn_bytes || (gu_bytes | dn_bytes) % 16 || gu_bytes > ec.dn_off[pool] ||
        dn_bytes > ec.slot_bytes[pool] - ec.dn_off[pool])
        throw std::runtime_error("exact cache copy: invalid expert strides");
    k_expert_cache_copy_exact<<<160,256,0,(cudaStream_t)stream>>>(ec,pool,list,gu_bytes,dn_bytes);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void expert_cache_count_sm75(const int* njobs,const int* list,size_t bytes,unsigned long long* stats,void* stream) {
    k_expert_cache_count<<<1,1,0,(cudaStream_t)stream>>>(njobs,list,bytes,stats);
}
} // namespace fnx
