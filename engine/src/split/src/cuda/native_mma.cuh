// Native tiles for the mixed IQ3_XXS file. Codebooks and layouts match the main
// engine's iq_kernels.cu; each dequantized value is rounded once to f16.
#pragma once

template <bool XS> struct MmTileIQ2 {
    uint32_t dq, q[2], aux;
    bool live, item;
    __device__ __forceinline__ void fetch(const char* we, size_t row_a, int nrows, int row0, int k0, int kc) {
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR;
        item = u * 32 < kc;
        live = item && row0 + r < nrows;
        const int ib = (k0 % 256) / 32 + u;
        if (live) {
            const uint8_t* b = (const uint8_t*) we + (size_t) (row0 + r) * row_a + (size_t) (k0 / 256) * (XS ? 74 : 66);
            dq = mq_ld16(b);
            q[0] = mq_ld32(b + 2 + 8 * ib);
            q[1] = mq_ld32(b + 6 + 8 * ib);
            aux = XS ? b[66 + ib] : q[1];
        }
    }
    __device__ __forceinline__ void put(int, int, __half* sw) const {
        if (!item) return;
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR;
        if (!live) {
            mq_zero32(sw + r * MM_LD + u * 32);
            return;
        }
        float f[32];
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const unsigned code = XS ? (q[il / 2] >> (16 * (il % 2))) & 65535 : (q[0] >> (8 * il)) & 255;
            const uint64_t grid = XS ? iq2xs_grid[code & 511] : iq2xxs_grid[code];
            const unsigned signs = ksigns_iq2xs[XS ? code >> 9 : (aux >> (7 * il)) & 127];
            const float scale = XS ? 0.5f + (float) ((aux >> (4 * (il / 2))) & 15) : 0.5f + (float) (aux >> 28);
            const float d = mq_half(dq) * scale * 0.25f;
            mq_signed4(f + 8 * il, (uint32_t) grid, d, signs, 0);
            mq_signed4(f + 8 * il + 4, (uint32_t) (grid >> 32), d, signs, 4);
        }
        mq_put32(sw + r * MM_LD + u * 32, f);
    }
};
template <> struct MmTile<T_IQ2_XXS> : MmTileIQ2<false> {};
template <> struct MmTile<T_IQ2_XS> : MmTileIQ2<true> {};

template <> struct MmTile<T_Q2_0> {
    uint32_t dq, q[2];
    bool live, item;
    __device__ __forceinline__ void fetch(const char* we, size_t row_a, int nrows, int row0, int k0, int kc) {
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR;
        item = u * 32 < kc;
        live = item && row0 + r < nrows;
        if (live) {
            const int k = k0 + u * 32;
            const uint8_t* b = (const uint8_t*) we + (size_t) (row0 + r) * row_a + (size_t) (k / 64) * 18;
            dq = mq_ld16(b);
            q[0] = mq_ld32(b + 2 + (k % 64) / 4);
            q[1] = mq_ld32(b + 6 + (k % 64) / 4);
        }
    }
    __device__ __forceinline__ void put(int, int, __half* sw) const {
        if (!item) return;
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR;
        if (!live) {
            mq_zero32(sw + r * MM_LD + u * 32);
            return;
        }
        float f[32];
#pragma unroll
        for (int i = 0; i < 32; ++i) f[i] = mq_half(dq) * ((int) ((q[i / 16] >> (2 * (i % 16))) & 3) - 1);
        mq_put32(sw + r * MM_LD + u * 32, f);
    }
};
