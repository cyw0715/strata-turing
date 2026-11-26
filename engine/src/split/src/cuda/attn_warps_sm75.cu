// Exact-score attention staging experiment; arithmetic from Strata kernels.cu (MIT).
#include "kernels.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
namespace fnx { namespace {
constexpr int ATT_SPLITS=16,ATT_CH=32,ATT_QPK=NQH/NKVH,ATT_MAXH=ATT_QPK,ATT_ROWS=64;
constexpr int QSA_MAXSEL=QSA_TOP_K+3;
struct AttPart {float m,z;};
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
__device__ __forceinline__ int kv_owned(int p,int card) {
    const int b=p/4,blocks=(b+NCARD-1-card)/NCARD;
    return blocks*4+(kv_card(b)==card?p%4+1:0);
}
template<int BT> __global__ void __launch_bounds__(BT) k_attn_warps(const float* qg, const __half* kc, const __half* vc, KvPages g, float* acc_out,
                                                  AttPart* mz_out, int nq, int q0, int nkv, int kv0, const int* pos0p,
                                                  const int* __restrict__ cells, const int* __restrict__ n_cells, int card, LaneRows lr) {
    // t: the window's row; its position and pages: its lane's (lr), or the sequence's *pos0p + t
    const int s = blockIdx.x, t = blockIdx.y, kvl = blockIdx.z, i = threadIdx.x, lane = i & 31, warp = i >> 5;
    g = row_pages(g, lr, t);
    // the attended positions (local indices on this card): the selection's list, or every owned one up to the row's
    const int* list = cells && n_cells[t] >= 0 ? cells + (size_t) t * QSA_MAXSEL : nullptr;
    const int ctx = list ? n_cells[t] : kv_owned(row_pos(lr, pos0p, t), card);
    // the local q heads of KV head kv0 + kvl: global heads [(kv0 + kvl) * QPK, + QPK) within [q0, q0 + nq)
    const int g0 = max(q0, (kv0 + kvl) * ATT_QPK), g1 = min(q0 + nq, (kv0 + kvl + 1) * ATT_QPK);
    const int h0 = g0 - q0, nh = g1 - g0;
    const int len = ((ctx + ATT_SPLITS - 1) / ATT_SPLITS + ATT_CH - 1) / ATT_CH * ATT_CH;
    const int p0 = s * len, p1 = min(ctx, p0 + len);
    __shared__ float q[ATT_MAXH][HD];
    __shared__ float kk[ATT_CH][HD + 1];
    __shared__ float e[ATT_MAXH][ATT_CH];
    __shared__ float scale[ATT_MAXH];
    for (int k = i; k < nh * HD; k += BT) q[k / HD][k % HD] = qg[((size_t) t * nq + h0 + k / HD) * 2 * HD + k % HD];
    constexpr int HPW = (ATT_MAXH + BT / 32 - 1) / (BT / 32);   // heads per warp: warp w keeps heads w, w + 8
    float m[HPW], z[HPW];
#pragma unroll
    for (int k = 0; k < HPW; ++k) {
        m[k] = -INFINITY;
        z[k] = 0.f;
    }
    float acc[ATT_MAXH] = {};
    __syncthreads();
#pragma unroll 1
    for (int c0 = p0; c0 < p1; c0 += ATT_CH) {
        const int np = min(ATT_CH, p1 - c0);
        // the chunk's K rows into shared memory and this thread's dimension of its V rows into registers
        float vr[ATT_CH];
        if(i<HD) {

#pragma unroll
        for (int pp = 0; pp < ATT_CH; ++pp) {
            const int pos = pp < np ? (list ? list[c0 + pp] : c0 + pp) : 0;
            if (g.q8) {
                const size_t r = kvq_row(g, pos) + (size_t) kvl * HD + i, sc = kvq_scale(g, pos) + ((size_t) kvl * HD + i) / KV8_GROUP;
                kk[pp][i] = pp < np ? reinterpret_cast<const int8_t*>(kc)[r] * __half2float(kc[sc]) : 0.f;
                vr[pp] = pp < np ? reinterpret_cast<const int8_t*>(vc)[r] * __half2float(vc[sc]) : 0.f;
                continue;
            }
            const size_t r = kvp_row(g, pos) + (size_t) kvl * HD + i;   // nkv == NKVH: a row holds both heads
            kk[pp][i] = pp < np ? __half2float(kc[r]) : 0.f;
            vr[pp] = pp < np ? __half2float(vc[r]) : 0.f;
        }
        }
        __syncthreads();
        // head warp + 8 k: lane = position; its score, then the head's online-softmax step
#pragma unroll
        for (int k = 0; k < HPW; ++k) {
            const int hh = warp + k * (BT / 32);
            if (hh >= nh) break;
            float sacc = 0.f;
#pragma unroll 8
            for (int d = 0; d < HD; ++d) sacc = fmaf(q[hh][d], kk[lane][d], sacc);
            const float sv = lane < np ? sacc * rsqrtf((float) HD) : -INFINITY;
            const float mn = fmaxf(m[k], wmax(sv));
            const float w = sv == -INFINITY ? 0.f : __expf(sv - mn);
            const float sc = m[k] == -INFINITY ? 0.f : __expf(m[k] - mn);
            z[k] = z[k] * sc + wsum(w);
            m[k] = mn;
            e[hh][lane] = w;
            if (lane == 0) scale[hh] = sc;
        }
        __syncthreads();
        if(i<HD) {
#pragma unroll
        for (int hh = 0; hh < ATT_MAXH; ++hh)
            if (hh < nh) acc[hh] *= scale[hh];
#pragma unroll
        for (int pp = 0; pp < ATT_CH; ++pp)
#pragma unroll
            for (int hh = 0; hh < ATT_MAXH; ++hh)
                if (hh < nh) acc[hh] = fmaf(e[hh][pp], vr[pp], acc[hh]);
        }
    }
#pragma unroll
    for (int hh = 0; hh < ATT_MAXH; ++hh) {
        if (hh >= nh) break;
        const size_t slot = ((size_t) t * nq + h0 + hh) * ATT_SPLITS + s;
        if(i<HD) acc_out[slot * HD + i] = acc[hh];
    }
#pragma unroll
    for (int k = 0; k < HPW; ++k) {
        const int hh = warp + k * (BT / 32);
        if (hh < nh && lane == 0) mz_out[((size_t) t * nq + h0 + hh) * ATT_SPLITS + s] = AttPart{m[k], z[k]};
    }
}


}
void attn_warps_run(const float* qg,const uint16_t* kc,const uint16_t* vc,const KvPages& g,
                    const LaneRows& lr,int T,float* scratch,void* stream,const int* cells,const int* ncells,int card,int mode,const int* pos0) {
    AttPart* mz=(AttPart*)(scratch+(size_t)ATT_ROWS*NQH*ATT_SPLITS*HD);
#define RUN(B) k_attn_warps<B><<<dim3(ATT_SPLITS,T,NKVH),B,0,(cudaStream_t)stream>>>(qg,(const __half*)kc,(const __half*)vc,g,scratch,mz,NQH,0,NKVH,0,pos0,cells,ncells,card,lr)
    if(mode==0) {RUN(384);} else {RUN(512);}
#undef RUN
    const cudaError_t e=cudaGetLastError();
    if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template<int B> void attn_warps_info() {
    cudaFuncAttributes at{};int blocks=0;
    cudaFuncGetAttributes(&at,k_attn_warps<B>);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k_attn_warps<B>,B,0);
    std::printf("warps threads=%d registers=%d local=%zu shared=%zu blocks=%d\n",B,at.numRegs,at.localSizeBytes,at.sharedSizeBytes,blocks);
}
void attn_warps_prepare() {attn_warps_info<384>();attn_warps_info<512>();}
}

namespace fnx {
void attn_warps_part(const float* qg,const uint16_t* kc,const uint16_t* vc,const KvPages& g,const LaneRows& lr,
                    int T,float* scratch,void* stream,const int* cells,const int* ncells,int card,int mode) {
    attn_warps_run(qg,kc,vc,g,lr,T,scratch,stream,cells,ncells,card,mode,nullptr);
}
void attn_warps_single(const float* qg,const uint16_t* kc,const uint16_t* vc,const KvPages& g,const int* pos0,
                      int T,float* scratch,void* stream,const int* cells,const int* ncells,int card) {
    attn_warps_run(qg,kc,vc,g,LaneRows{},T,scratch,stream,cells,ncells,card,0,pos0);
}
}
