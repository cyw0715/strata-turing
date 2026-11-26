// Staged HC candidate, arithmetic from hc.cuh (Strata/sFNX, MIT).
// Private translation unit: existing kernels and default dispatch are unchanged.
#include "kernels.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

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
__device__ __forceinline__ float sigm(float x) { return 1.f/(1.f+__expf(-x)); }
__device__ __forceinline__ float silu_(float x) { return x/(1.f+__expf(-x)); }
__device__ __forceinline__ float bf(uint16_t h) { return __uint_as_float((unsigned)h<<16); }
__device__ __forceinline__ float f16(uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ int2 hc_tc_dot32(unsigned a0,unsigned a1,unsigned b0,unsigned b1) {
    int2 d=make_int2(0,0);
    asm volatile(
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%4}, {%5}, {%0,%1};\n"
        : "+r"(d.x),"+r"(d.y):"r"(a0),"r"(b0),"r"(a1),"r"(b1));
    return d;
}
}
namespace {

// row groups of 96 rows: lo's quantization blocks (32 rows) never straddle two groups
constexpr int H_SL = 512, H_NS = D / H_SL, H_SPS = N / H_SL, H_RPG = 96;
constexpr int H_ROWS = LR + HC, H_RG = (H_ROWS + H_RPG - 1) / H_RPG;
static_assert(H_RPG % 32 == 0, "lo blocks inside a group");
constexpr int H_TPRD = 8, H_DT = (H_RPG * H_TPRD + 31) / 32 * 32, H_BLOCKS = H_NS * H_RG;
// up: 32 columns per block, H_TPRU lanes per column; lane b < LR / 32 takes lo's 32-value block b
constexpr int H_UPC = 32, H_TPRU = 16, H_UB = LR / 32;
static_assert(H_BLOCKS * H_UPC == N, "one up block per down block");
static_assert(H_UPC * H_TPRU <= H_DT && H_UB <= H_TPRU, "the up lanes");
// the grid barrier's counter runs modulo hc_bar_mod(g), a multiple of the grid size g, so every barrier's g arrivals
// stay in one generation [k * g, (k + 1) * g) across the wrap
__host__ __device__ constexpr unsigned hc_bar_mod(unsigned g) { return 0xffffffffu / g * g; }

struct HcArgs {
    float* R;
    const float* bo;
    const float* inj_prev;
    const float* w_norm;
    Q8S w_down;
    Q8S w_up;
    const uint16_t* w_inj;
    float eps;
    float* mixed;
    float* inject;
    float* part;   // [H_NS][T][H_ROWS]
    float* ssp;    // [T][H_NS]: the slices' square sums
    int8_t* loq;   // [T][LR]: lo as int8 per 32
    float* lod;    // [T][LR / 32]: their scales
    unsigned* bar;       // the grid barrier's counter (modulo hc_bar_mod(grid))
    unsigned* err;       // bit 4: a barrier timed out
    block_q8_1* q8;      // mixed as q8_1 too, or null
    int8_t* tq = nullptr;   // and its split twin, or null
    float* td = nullptr;
};

__device__ __forceinline__ float rprime(const HcArgs& a, int t, int c, int i) {
    float r = a.R[(size_t) t * D + c * N + i];
    if (a.bo) r = fmaf(a.bo[(size_t) t * N + i], 2.0f * sigm(a.inj_prev[t * HC + c] / (float) HC), r);
    return r;
}

// 8 int8 weights (uint2) . 8 int8 activations (two ints)
__device__ __forceinline__ int dot8i(const uint2 w, const int2 x) { return __dp4a((int) w.y, x.y, __dp4a((int) w.x, x.x, 0)); }

// one warp quantizes one 32-value block: lane i's value v -> q[i], the block's scale -> *d (lane 0)
__device__ __forceinline__ void q8_block(float v, int8_t* q, float* d) {
    const float amax = wmax(fabsf(v));
    const float sc = amax / 127.0f;
    q[threadIdx.x & 31] = amax == 0.0f ? 0 : (int8_t) roundf(v / sc);
    if ((threadIdx.x & 31) == 0) *d = sc;
}

// every block of the grid gets here before any goes on (the counter grows by the grid size g per barrier, modulo
// hc_bar_mod(g)). The block's last thread arrives and waits until the counter has left its barrier's generation: it is
// then 1 to 2 * g - 1 past the generation's start, as no block can pass the next barrier before this one does. A fence
// waits for the thread's own loads too, and that thread loads no up weights
__device__ __forceinline__ void grid_sync(unsigned* bar, unsigned* err, unsigned g) {
    __syncthreads();
    if (threadIdx.x == blockDim.x - 1) {
        __threadfence();
        const unsigned mod = hc_bar_mod(g);
        const unsigned old = atomicInc(bar, mod - 1u), base = old - old % g;
        const long long t0 = clock64();
        for (;;) {
            const unsigned cur = *(volatile unsigned*) bar;
            if ((cur >= base ? cur - base : cur + (mod - base)) >= g) break;
            if (clock64() - t0 > 2000000000LL) {
                atomicOr(err, 4u);
                break;
            }
            __nanosleep(64);
        }
        __threadfence();
    }
    __syncthreads();
}

// rs for token k, stream c: 1 / sqrt(mean of R'^2 over the stream + eps), from the slices' square sums
__device__ __forceinline__ float rs_of(const HcArgs& a, int k, int c) {
    float v = 0.f;
    for (int ss = 0; ss < H_SPS; ++ss) v += __ldcg(a.ssp + (size_t) k * H_NS + c * H_SPS + ss);
    return rsqrtf(v / (float) N + a.eps);
}

// the up part's arrays (after the second barrier) in the down part's xs (done before the first): 16 rows fit 48 KB
template <int T> struct HcUp {
    int8_t lq[T][LR];
    float ld[T][LR / 32];
    float rs[T][HC];
    float ms[T][H_UPC];
    float gs[T][HC][H_UPC];
};

// ONE: one grid barrier instead of two: after it every block makes the whole of lo itself (the same sums in the same
// order, so the same bits) from the slices' partials in L2, and block 0 the inject rows; ~T * 77 KB of L2 reads a block,
// so for small windows only
// WALK: the grid is short of H_BLOCKS and each block takes the shares vb = blockIdx.x + k * gridDim.x in turn; else
// vb = blockIdx.x alone, without the loops (the walk costs a register over the 79 of 80 SMs' form, and a stack slot at
// T = 15)

template<int T, int RPG> __global__ void __launch_bounds__(RPG * 8) k_hc_stage_down(HcArgs a) {
    constexpr int H_DT = RPG * 8, H_RPG = RPG, H_RG = (H_ROWS + RPG - 1) / RPG;
    __shared__ __align__(16) float xs[T][H_SL];
    __shared__ __align__(16) int8_t xq[T][H_SL];
    __shared__ float xd[T][H_SL / 32];
    __shared__ float red[T][H_SL / 32];
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    // ---------------- down, for each of the block's shares vb
    const int vb = blockIdx.x;
    const int s = vb / H_RG, grp = vb % H_RG;
    const int col0 = s * H_SL, c = col0 / N, i0 = col0 - c * N;
    // thread t < H_SL takes column t of every row, the rows' loads side by side (a row at a time, each cost a memory
    // latency: T of them on the group-0 blocks, which the others then waited for at the first barrier)
    for (int col = t; col < H_SL; col += H_DT) {
        float r[T];
        const float wn = a.w_norm[col0 + col];
#pragma unroll
        for (int k = 0; k < T; ++k) r[k] = rprime(a, k, c, i0 + col);
#pragma unroll
        for (int k = 0; k < T; ++k) xs[k][col] = r[k] * wn;
        if (grp == 0) {
#pragma unroll
            for (int k = 0; k < T; ++k) {
                const float v = wsum(r[k] * r[k]);
                if (lane == 0) red[k][col / 32] = v;
            }
        }
    }
    __syncthreads();
    if (grp == 0 && t < T) {
        float v = 0.f;
        for (int w = 0; w < H_SL / 32; ++w) v += red[t][w];
        a.ssp[(size_t) t * H_NS + s] = v;
    }
    for (int q = warp; q < T * (H_SL / 32); q += H_DT / 32) {
        const int k = q / (H_SL / 32), blk = q % (H_SL / 32);
        q8_block(xs[k][blk * 32 + lane], &xq[k][blk * 32], &xd[k][blk]);
    }
    __syncthreads();
    {
        constexpr int NQ = H_SL / 8 / H_TPRD;
        const int j = t % H_TPRD, row = grp * H_RPG + t / H_TPRD;
        const int nrows = a.w_inj ? H_ROWS : LR;
        const bool live = t / H_TPRD < H_RPG && row < nrows;
        const int rr = live ? row : 0;
        // the thread's weights to registers, then one pass per row t (rolled: unrolled over T the kernel is tens of KB
        // of code, fetched cold at every launch)
        if (rr < LR) {
            const uint2* w2 = reinterpret_cast<const uint2*>(a.w_down.q + (size_t) rr * D + col0);
            const uint16_t* wd = a.w_down.d + ((size_t) rr * D + col0) / 32;
            uint2 u[NQ];
            float sc[NQ];
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                u[q] = __ldg(w2 + q * H_TPRD + j);
                sc[q] = f16(__ldg(wd + (q * H_TPRD + j) / 4));
            }
#pragma unroll 1
            for (int k = 0; k < T; ++k) {
                float acc = 0.f;
#pragma unroll
                for (int q = 0; q < NQ; ++q) {
                    const int ch = q * H_TPRD + j;
                    acc = fmaf(sc[q] * xd[k][ch / 4], (float) dot8i(u[q], *reinterpret_cast<const int2*>(&xq[k][ch * 8])), acc);
                }
                for (int o = 1; o < H_TPRD; o <<= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
                if (live && j == 0) a.part[((size_t) s * T + k) * H_ROWS + row] = acc;
            }
        } else {
            const uint4* w4 = reinterpret_cast<const uint4*>(a.w_inj + (size_t) (rr - LR) * D + col0);
            uint4 u[NQ];
#pragma unroll
            for (int q = 0; q < NQ; ++q) u[q] = __ldg(w4 + q * H_TPRD + j);
#pragma unroll 1
            for (int k = 0; k < T; ++k) {
                float acc = 0.f;
#pragma unroll
                for (int q = 0; q < NQ; ++q) {
                    const int ch = q * H_TPRD + j;
                    const uint16_t* hh = reinterpret_cast<const uint16_t*>(&u[q]);
                    const float4 x0 = *reinterpret_cast<const float4*>(&xs[k][ch * 8]);
                    const float4 x1 = *reinterpret_cast<const float4*>(&xs[k][ch * 8 + 4]);
                    acc += bf(hh[0]) * x0.x + bf(hh[1]) * x0.y + bf(hh[2]) * x0.z + bf(hh[3]) * x0.w + bf(hh[4]) * x1.x +
                           bf(hh[5]) * x1.y + bf(hh[6]) * x1.z + bf(hh[7]) * x1.w;
                }
                for (int o = 1; o < H_TPRD; o <<= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
                if (live && j == 0) a.part[((size_t) s * T + k) * H_ROWS + row] = acc;
            }
        }
    }

}

template<int T> __global__ void k_hc_stage_reduce(HcArgs a) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    constexpr int H_DT = 256;
    const int grid = gridDim.x;
    // ---------------- reduce: item (token k, lo block b) per warp, then each token's inject rows
    {
        constexpr int ITEMS = T * (LR / 32) + T;
        for (int it = blockIdx.x * (H_DT / 32) + warp; it < ITEMS; it += grid * (H_DT / 32)) {
            const bool inj_item = it >= T * (LR / 32);
            const int k = inj_item ? it - T * (LR / 32) : it / (LR / 32);
            // every lane runs the same path (q8_block shuffles across the warp); only an inject item's first HC lanes
            // have a row
            const bool active = !inj_item || (a.w_inj && lane < HC);
            const int r = !active ? 0 : inj_item ? LR + lane : (it % (LR / 32)) * 32 + lane;
            float pv[H_NS];
#pragma unroll
            for (int q = 0; q < H_NS; ++q) pv[q] = __ldcg(a.part + ((size_t) q * T + k) * H_ROWS + r);
            float sum = 0.f;
#pragma unroll
            for (int cc = 0; cc < HC; ++cc) {
                float p = 0.f;
#pragma unroll
                for (int ss = 0; ss < H_SPS; ++ss) p += pv[cc * H_SPS + ss];
                sum = fmaf(rs_of(a, k, cc), p, sum);
            }
            if (inj_item) {
                if (active) a.inject[k * HC + lane] = sum;
            } else q8_block(silu_(sum / (float) HC), a.loq + k * LR + (r & ~31), a.lod + k * (LR / 32) + r / 32);
        }
    }

}

template<int T, bool REDUCE> __global__ void __launch_bounds__(512) k_hc_stage_up(HcArgs a) {
    __shared__ HcUp<T> up;
    auto& lq = up.lq;
    auto& ld = up.ld;
    auto& rs = up.rs;
    auto& ms = up.ms;
    auto& gs = up.gs;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31, vb = blockIdx.x;
    if constexpr (REDUCE) {
        // ---------------- reduce, in every block: rs, then lo per (token k, lo block b) a warp into lq / ld
        if (t < T * HC) rs[t / HC][t % HC] = rs_of(a, t / HC, t % HC);
        __syncthreads();
        for (int it = warp; it < T * (LR / 32); it += 512 / 32) {
            const int k = it / (LR / 32), r = (it % (LR / 32)) * 32 + lane;
            float pv[H_NS];
#pragma unroll
            for (int q = 0; q < H_NS; ++q) pv[q] = __ldcg(a.part + ((size_t) q * T + k) * H_ROWS + r);
            float sum = 0.f;
#pragma unroll
            for (int cc = 0; cc < HC; ++cc) {
                float p = 0.f;
#pragma unroll
                for (int ss = 0; ss < H_SPS; ++ss) p += pv[cc * H_SPS + ss];
                sum = fmaf(rs[k][cc], p, sum);
            }
            q8_block(silu_(sum / (float) HC), &lq[k][r & ~31], &ld[k][r / 32]);
        }
        if (blockIdx.x == 0 && a.w_inj && warp < T && lane < HC) {
            const int k = warp, r = LR + lane;
            float pv[H_NS];
#pragma unroll
            for (int q = 0; q < H_NS; ++q) pv[q] = __ldcg(a.part + ((size_t) q * T + k) * H_ROWS + r);
            float sum = 0.f;
#pragma unroll
            for (int cc = 0; cc < HC; ++cc) {
                float p = 0.f;
#pragma unroll
                for (int ss = 0; ss < H_SPS; ++ss) p += pv[cc * H_SPS + ss];
                sum = fmaf(rs[k][cc], p, sum);
            }
            a.inject[k * HC + lane] = sum;
        }
        __syncthreads();

    } else {
        if (t < T * HC) rs[t / HC][t % HC] = rs_of(a, t / HC, t % HC);
        for (int e = t; e < T * LR / 4; e += 512)
            reinterpret_cast<int*>(&lq[0][0])[e] = __ldcg(reinterpret_cast<const int*>(a.loq) + e);
        for (int e = t; e < T * (LR / 32); e += 512)
            ld[e / (LR / 32)][e % (LR / 32)] = __ldcg(a.lod + e);
        __syncthreads();
    }
    const int uj = t % H_TPRU, udd = t / H_TPRU;
    const bool live = uj < H_UB;
#pragma unroll
    for (int cc = 0; cc < HC; ++cc) {
        const size_t row = (size_t) cc * N + vb * H_UPC + udd;
        uint4 w0{}, w1{};
        float sc = 0.f;
        if (live) {
            const uint4* w = reinterpret_cast<const uint4*>(a.w_up.q + row * LR + uj * 32);
            w0 = __ldg(w); w1 = __ldg(w + 1);
            sc = f16(__ldg(a.w_up.d + row * (LR / 32) + uj));
        }
#pragma unroll 1
        for (int k = 0; k < T; ++k) {
            const int4* x = reinterpret_cast<const int4*>(&lq[k][(live ? uj : 0) * 32]);
            const int4 x0 = x[0], x1 = x[1];
            const float dl = ld[k][live ? uj : 0];
            int s = __dp4a((int) w0.x, x0.x, 0);
            s = __dp4a((int) w0.y, x0.y, s);
            s = __dp4a((int) w0.z, x0.z, s);
            s = __dp4a((int) w0.w, x0.w, s);
            s = __dp4a((int) w1.x, x1.x, s);
            s = __dp4a((int) w1.y, x1.y, s);
            s = __dp4a((int) w1.z, x1.z, s);
            s = __dp4a((int) w1.w, x1.w, s);
            float v = sc * dl * (float) s;
#pragma unroll
            for (int o = 1; o < H_TPRU; o <<= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (uj == 0) gs[k][cc][udd] = v;
        }
    }
    __syncthreads();
    // item (k, c, col): this block's 32 columns of stream c, row k, coalesced along the columns
    const int b = vb;
    for (int e = t; e < T * HC * H_UPC; e += 512) {
        const int col = e % H_UPC, cc = (e / H_UPC) % HC, k = e / (H_UPC * HC), dc = b * H_UPC + col;
        const float rv = rprime(a, k, cc, dc);
        if (a.bo) a.R[(size_t) k * D + cc * N + dc] = rv;
        gs[k][cc][col] = sigm(gs[k][cc][col]) * (rv * a.w_norm[cc * N + dc] * rs[k][cc]);
    }
    __syncthreads();
    for (int e = t; e < T * H_UPC; e += 512) {
        const int col = e % H_UPC, k = e / H_UPC;
        const float sum = (gs[k][0][col] + gs[k][1][col] + gs[k][2][col] + gs[k][3][col]) / (float) HC;
        a.mixed[(size_t) k * N + b * H_UPC + col] = sum;
        ms[k][col] = sum;
    }
    if (!a.q8) return;
    __syncthreads();
    static_assert(H_UPC == 32, "one q8_1 block per token per up block");
    if (warp < T) {
        const float xi = ms[warp][lane];
        const float amax = wmax(fabsf(xi)), sm = wsum(xi);
        const float dq = amax / 127.0f;
        block_q8_1& yq = a.q8[(size_t) warp * (N / 32) + b];
        const int8_t qv = amax == 0.0f ? 0 : (int8_t) roundf(xi / dq);
        yq.qs[lane] = qv;
        if (lane == 0) yq.ds = make_half2(__float2half(dq), __float2half(sm));
        if (a.tq) {
            a.tq[(size_t) warp * N + b * 32 + lane] = qv;
            if (lane == 0) a.td[(size_t) warp * (N / 32) + b] = __half2float(__float2half(dq));
        }
    }

}

template<int T> void staged_launch(HcArgs a, cudaStream_t s, int variant) {
    if (variant == 2 || variant == 4) k_hc_stage_down<T,64><<<H_NS * ((H_ROWS + 63) / 64),512,0,s>>>(a);
    else if (variant == 3) k_hc_stage_down<T,32><<<H_NS * ((H_ROWS + 31) / 32),256,0,s>>>(a);
    else k_hc_stage_down<T,96><<<H_BLOCKS,H_DT,0,s>>>(a);
    if (variant == 1 || variant == 4) k_hc_stage_up<T,true><<<H_BLOCKS,512,0,s>>>(a);
    else {
        constexpr int items = T * (LR / 32) + T;
        k_hc_stage_reduce<T><<<(items + 7) / 8,256,0,s>>>(a);
        k_hc_stage_up<T,false><<<H_BLOCKS,512,0,s>>>(a);
    }
}
} // namespace
void hc_staged_read(float* R,const float* bo,const float* inj_prev,const float* w_norm,const Q8S& down,
                    const Q8S& up,const uint16_t* wi,float eps,float* mixed,float* inject,
                    float* scratch,unsigned*,int T,void* stream,void* q8,int variant) {
    if (T < 1 || T > 16 || variant < 0 || variant > 4) throw std::runtime_error("staged HC: unsupported shape or variant");
    HcArgs a{R,bo,inj_prev,w_norm,down,up,wi,eps,mixed,inject,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,(block_q8_1*)q8};
    a.part = scratch;
    a.ssp = a.part + (size_t) H_NS * T * H_ROWS;
    a.loq = (int8_t*) (a.ssp + (size_t) T * H_NS);
    a.lod = a.ssp + (size_t) T * H_NS + (size_t) T * LR / 4;
    for (const auto& twin : q8_twins()) if (q8 && (char*)q8 >= twin.q8 && (char*)q8 < twin.q8 + twin.bytes) {
        const size_t off = ((char*)q8 - twin.q8) / 36;
        a.tq = twin.tq + off * 32; a.td = twin.td + off; break;
    }
#define HC_RUN(T) case T: staged_launch<T>(a,(cudaStream_t)stream,variant); break
    switch (T) { HC_RUN(1); HC_RUN(2); HC_RUN(3); HC_RUN(4); HC_RUN(5); HC_RUN(6); HC_RUN(7); HC_RUN(8);
                 HC_RUN(9); HC_RUN(10); HC_RUN(11); HC_RUN(12); HC_RUN(13); HC_RUN(14); HC_RUN(15); HC_RUN(16); }
#undef HC_RUN
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
} // namespace fnx
