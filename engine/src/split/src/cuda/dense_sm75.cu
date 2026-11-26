// Experimental small-window Q8_0 x Q8_1 kernel for Turing.
// Fragment layout follows NVIDIA PTX mma.m8n8k16 and the main engine's
// dense_i8_tc.cu (MIT). Split weight layout and per-32 scales are preserved.
#include "../kernels.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fnx {
namespace {
struct TcQ81 {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(TcQ81) == 36, "Q8_1 row layout");

__device__ __forceinline__ int2 tc_dot32(unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
    int2 d = make_int2(0, 0);
    asm volatile(
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%4}, {%5}, {%0,%1};\n"
        : "+r"(d.x), "+r"(d.y) : "r"(a0), "r"(b0), "r"(a1), "r"(b1));
    return d;
}

template<int NT, int RPW, int WARPS, bool SPLIT_K = false>
__global__ void __launch_bounds__(WARPS * 32) k_dense_sm75(
        Q8S w, int rows, int cols, const TcQ81* __restrict__ x,
        float* __restrict__ out, int stride, int T) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, k4 = lane & 3, nb = cols / 32;
    const int row = (blockIdx.x * (SPLIT_K ? 1 : WARPS) + (SPLIT_K ? 0 : warp)) * RPW + g;
    float acc[RPW / 8][NT][2] = {};
    // One int32 dot per 32-element quantization group. No new quantization,
    // intermediate FP16 conversion, or approximate rescaling is introduced.
    for (int b = SPLIT_K ? warp : 0; b < nb; b += SPLIT_K ? WARPS : 1) {
        unsigned ax[RPW / 8][2];
        float dw[RPW / 8];
#pragma unroll
        for (int r = 0; r < RPW / 8; ++r) {
            const int rr = row + r * 8;
            const int safe = min(rr, rows - 1);
            const unsigned* p = reinterpret_cast<const unsigned*>(w.q + (size_t) safe * cols + b * 32);
            ax[r][0] = __ldg(p + k4);
            ax[r][1] = __ldg(p + 4 + k4);
            dw[r] = __half2float(__ushort_as_half(__ldg(w.d + (size_t) safe * nb + b)));
        }
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int token = nt * 8 + g;
            const TcQ81* xb = x + (size_t) min(token, T - 1) * nb + b;
            const unsigned* p = reinterpret_cast<const unsigned*>(xb->qs);
            const unsigned bx0 = p[k4], bx1 = p[4 + k4];
            const int t0 = nt * 8 + 2 * k4, t1 = t0 + 1;
            const float dx0 = __low2float(x[(size_t) min(t0, T - 1) * nb + b].ds);
            const float dx1 = __low2float(x[(size_t) min(t1, T - 1) * nb + b].ds);
#pragma unroll
            for (int r = 0; r < RPW / 8; ++r) {
                const int2 d = tc_dot32(ax[r][0], ax[r][1], bx0, bx1);
                acc[r][nt][0] = fmaf(dw[r] * dx0, (float) d.x, acc[r][nt][0]);
                acc[r][nt][1] = fmaf(dw[r] * dx1, (float) d.y, acc[r][nt][1]);
            }
        }
    }
    if constexpr (SPLIT_K) {
        __shared__ float partial[WARPS][RPW / 8][NT][32][2];
#pragma unroll
        for (int r = 0; r < RPW / 8; ++r)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                partial[warp][r][nt][lane][0] = acc[r][nt][0];
                partial[warp][r][nt][lane][1] = acc[r][nt][1];
            }
        __syncthreads();
        if (warp) return;
#pragma unroll
        for (int r = 0; r < RPW / 8; ++r)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                acc[r][nt][0] = acc[r][nt][1] = 0.f;
#pragma unroll
                for (int w = 0; w < WARPS; ++w) {
                    acc[r][nt][0] += partial[w][r][nt][lane][0];
                    acc[r][nt][1] += partial[w][r][nt][lane][1];
                }
            }
    }
#pragma unroll
    for (int r = 0; r < RPW / 8; ++r) {
        const int rr = row + r * 8;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int t = nt * 8 + 2 * k4;
            if (rr < rows && t < T) out[(size_t) t * stride + rr] = acc[r][nt][0];
            if (rr < rows && t + 1 < T) out[(size_t) (t + 1) * stride + rr] = acc[r][nt][1];
        }
    }
}

template<int NT> void dense_launch(Q8S w, int rows, int cols, const void* x, float* out,
                                    int stride, int T, cudaStream_t s, int variant) {
#define FNX_DENSE_TC(R,W) k_dense_sm75<NT,R,W><<<(rows + R * W - 1) / (R * W), W * 32, 0, s>>>(w,rows,cols,(const TcQ81*)x,out,stride,T)
#define FNX_DENSE_SK(R,W) k_dense_sm75<NT,R,W,true><<<(rows + R - 1) / R, W * 32, 0, s>>>(w,rows,cols,(const TcQ81*)x,out,stride,T)
    switch (variant) {
        case 0: FNX_DENSE_TC(8,4); break;
        case 1: FNX_DENSE_TC(16,4); break;
        case 2: FNX_DENSE_TC(8,8); break;
        case 3: FNX_DENSE_TC(16,8); break;
        case 4: FNX_DENSE_SK(8,4); break;
        case 5: FNX_DENSE_SK(8,8); break;
        case 6: FNX_DENSE_SK(16,4); break;
        case 7: FNX_DENSE_SK(16,8); break;
        default: throw std::runtime_error("dense_sm75: unknown variant");
    }
#undef FNX_DENSE_TC
#undef FNX_DENSE_SK
}
} // namespace

void q8s_tc_sm75(Q8S w, int rows, int cols, const void* x, float* out,
                  int stride, int T, void* stream, int variant) {
    if (rows < 1 || cols < 32 || cols % 32 || stride < rows || T < 1 || T > 16)
        throw std::runtime_error("dense_sm75: unsupported shape");
    if (T <= 8) dense_launch<1>(w,rows,cols,x,out,stride,T,(cudaStream_t)stream,variant);
    else dense_launch<2>(w,rows,cols,x,out,stride,T,(cudaStream_t)stream,variant);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}
} // namespace fnx
