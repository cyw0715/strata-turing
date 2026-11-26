// Native singleton table staging, copied from the current Strata dots (MIT).
// Layouts and integer rounding
// follow qdot.cuh/native_quant.cuh; MMA fragments follow dense_i8_tc.cu.
#include "kernels.hpp"
#include "qdot.cuh"
#include "cuda/native_quant.cuh"

namespace fnx {
void moe_gu_native_lut_sm75(const DW&,size_t,const Job*,const int*,int,const void*,float*,void*,int,const Q8S&,int,const float*,float*,int);
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
template<bool SHARED,bool FAST> __device__ __forceinline__ float native_iq2_xxs(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                      const int& kbx, const int& iqs, const unsigned* lut) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*) (SHARED ? (const uint64_t*) lut : iq2xxs_grid))[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = FAST ? gu_sign(grid_pos.x,signs) : __vsub4(grid_pos.x ^ signs0,signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi = qd_dp4a(grid0, u0, sumi);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = FAST ? gu_sign(grid_pos.y,signs>>4) : __vsub4(grid_pos.y ^ signs1,signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi = qd_dp4a(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}
template<bool SHARED,bool FAST> __device__ __forceinline__ float native_iq2_xs(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs, const unsigned* lut) {
    const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t* q2 = (const uint16_t*) &q2_packed;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*) (SHARED ? (const uint64_t*) lut : iq2xs_grid))[q2[l0 / 2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = FAST ? gu_sign(grid_pos.x,signs) : __vsub4(grid_pos.x ^ signs0,signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = FAST ? gu_sign(grid_pos.y,signs>>4) : __vsub4(grid_pos.y ^ signs1,signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = qd_dp4a(grid_l, u0, sumi0);
            sumi0 = qd_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = qd_dp4a(grid_l, u0, sumi1);
            sumi1 = qd_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}
template<bool SHARED,bool FAST> __device__ __forceinline__ float native_iq3_xxs(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, const unsigned* lut) {

    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;

    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs+1));
    const uint8_t * q3 = (const uint8_t *) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K/16 + iqs/2);

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2((SHARED ? (const uint32_t*) lut : iq3xxs_grid)[q3[l0 + 0]], (SHARED ? (const uint32_t*) lut : iq3xxs_grid)[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7*l0/2));

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = FAST ? gu_sign(grid_pos.x,signs) : __vsub4(grid_pos.x ^ signs0,signs0);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = FAST ? gu_sign(grid_pos.y,signs>>4) : __vsub4(grid_pos.y ^ signs1,signs1);

        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = qd_dp4a(grid_l, u0, sumi);
        sumi = qd_dp4a(grid_h, u1, sumi);
    }

    const int ls = aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}
template<bool SHARED,bool FAST> __device__ __forceinline__ float native_iq3_s(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, const unsigned* lut) {

    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;

    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq3->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            (SHARED ? (const uint32_t*) lut : iq3s_grid)[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
            (SHARED ? (const uint32_t*) lut : iq3s_grid)[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);

        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        const int grid_l = FAST ? gu_sign(grid_pos.x,signs_packed_8[l0/2]) : __vsub4(grid_pos.x ^ signs0,signs0);
        const int grid_h = FAST ? gu_sign(grid_pos.y,signs_packed_8[l0/2]>>4) : __vsub4(grid_pos.y ^ signs1,signs1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = qd_dp4a(grid_l, u0, sumi);
        sumi = qd_dp4a(grid_h, u1, sumi);
    }

    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);

    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}
template<bool SHARED,bool FAST> __device__ __forceinline__ float native_iq2_s(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, const unsigned* lut) {

    const block_iq2_s * bq2 = (const block_iq2_s *) vbq + kbx;

    const int       qs_packed = get_int_b2(bq2->qs, iqs/2);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq2->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq2->qs, QK_K/32 + iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int * grid_pos = (const int *)((SHARED ? (const uint64_t*) lut : iq2s_grid) + (qs[l0/2] | ((qh << (8-l0)) & 0x300)));

        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        const int grid_l = FAST ? gu_sign(grid_pos[0],signs_packed_8[l0/2]) : __vsub4(grid_pos[0] ^ signs0,signs0);
        const int grid_h = FAST ? gu_sign(grid_pos[1],signs_packed_8[l0/2]>>4) : __vsub4(grid_pos[1] ^ signs1,signs1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0 = qd_dp4a(grid_l, u0, sumi0);
            sumi0 = qd_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = qd_dp4a(grid_l, u0, sumi1);
            sumi1 = qd_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;

    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}
template<bool SHARED,bool FAST> __device__ __forceinline__ float native_iq2_xxs_acc(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                      const int& kbx, const int& iqs, const unsigned* lut,float acc) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*) (SHARED ? (const uint64_t*) lut : iq2xxs_grid))[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = FAST ? gu_sign(grid_pos.x,signs) : __vsub4(grid_pos.x ^ signs0,signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi = qd_dp4a(grid0, u0, sumi);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = FAST ? gu_sign(grid_pos.y,signs>>4) : __vsub4(grid_pos.y ^ signs1,signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi = qd_dp4a(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return fmaf(d,(float)sumi,acc);
}
template<bool SHARED,bool FAST> __device__ __forceinline__ float native_iq2_xs_acc(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs, const unsigned* lut,float acc) {
    const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t* q2 = (const uint16_t*) &q2_packed;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*) (SHARED ? (const uint64_t*) lut : iq2xs_grid))[q2[l0 / 2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = FAST ? gu_sign(grid_pos.x,signs) : __vsub4(grid_pos.x ^ signs0,signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = FAST ? gu_sign(grid_pos.y,signs>>4) : __vsub4(grid_pos.y ^ signs1,signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = qd_dp4a(grid_l, u0, sumi0);
            sumi0 = qd_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = qd_dp4a(grid_l, u0, sumi1);
            sumi1 = qd_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return fmaf(d,(float)sumi,acc);
}
template<bool SHARED,bool FAST> __device__ __forceinline__ float native_iq3_xxs_acc(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, const unsigned* lut,float acc) {

    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;

    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs+1));
    const uint8_t * q3 = (const uint8_t *) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K/16 + iqs/2);

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2((SHARED ? (const uint32_t*) lut : iq3xxs_grid)[q3[l0 + 0]], (SHARED ? (const uint32_t*) lut : iq3xxs_grid)[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7*l0/2));

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = FAST ? gu_sign(grid_pos.x,signs) : __vsub4(grid_pos.x ^ signs0,signs0);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = FAST ? gu_sign(grid_pos.y,signs>>4) : __vsub4(grid_pos.y ^ signs1,signs1);

        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = qd_dp4a(grid_l, u0, sumi);
        sumi = qd_dp4a(grid_h, u1, sumi);
    }

    const int ls = aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return fmaf(d,(float)sumi,acc);
}
template<bool SHARED,bool FAST> __device__ __forceinline__ float native_iq3_s_acc(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, const unsigned* lut,float acc) {

    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;

    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq3->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            (SHARED ? (const uint32_t*) lut : iq3s_grid)[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
            (SHARED ? (const uint32_t*) lut : iq3s_grid)[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);

        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        const int grid_l = FAST ? gu_sign(grid_pos.x,signs_packed_8[l0/2]) : __vsub4(grid_pos.x ^ signs0,signs0);
        const int grid_h = FAST ? gu_sign(grid_pos.y,signs_packed_8[l0/2]>>4) : __vsub4(grid_pos.y ^ signs1,signs1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = qd_dp4a(grid_l, u0, sumi);
        sumi = qd_dp4a(grid_h, u1, sumi);
    }

    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);

    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return fmaf(d,(float)sumi,acc);
}
template<bool SHARED,bool FAST> __device__ __forceinline__ float native_iq2_s_acc(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, const unsigned* lut,float acc) {

    const block_iq2_s * bq2 = (const block_iq2_s *) vbq + kbx;

    const int       qs_packed = get_int_b2(bq2->qs, iqs/2);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq2->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq2->qs, QK_K/32 + iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int * grid_pos = (const int *)((SHARED ? (const uint64_t*) lut : iq2s_grid) + (qs[l0/2] | ((qh << (8-l0)) & 0x300)));

        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        const int grid_l = FAST ? gu_sign(grid_pos[0],signs_packed_8[l0/2]) : __vsub4(grid_pos[0] ^ signs0,signs0);
        const int grid_h = FAST ? gu_sign(grid_pos[1],signs_packed_8[l0/2]>>4) : __vsub4(grid_pos[1] ^ signs1,signs1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0 = qd_dp4a(grid_l, u0, sumi0);
            sumi0 = qd_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = qd_dp4a(grid_l, u0, sumi1);
            sumi1 = qd_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;

    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return fmaf(d,(float)sumi,acc);
}
template<int TYPE,int RPW,int WARPS,bool SHARED,bool FAST> __device__ void gu_single(
        const DW& w,const uint8_t* base,const Job& jb,const block_q8_1* x,float* out,const unsigned* lut) {
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
            if constexpr(TYPE==16) sum+=native_iq2_xxs<SHARED,FAST>(wr,y,b,q,lut);
            else if constexpr(TYPE==17) sum+=native_iq2_xs<SHARED,FAST>(wr,y,b,q,lut);
            else if constexpr(TYPE==18) sum+=native_iq3_xxs<SHARED,FAST>(wr,y,b,q,lut);
            else if constexpr(TYPE==21) sum+=native_iq3_s<SHARED,FAST>(wr,y,b,q,lut);
            else sum+=native_iq2_s<SHARED,FAST>(wr,y,b,q,lut);
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

template<int TYPE,int ROWS,bool FAST,bool TOKEN_PARALLEL> __device__ void pair_body(
        const DW& w,const uint8_t* base,const Job& jb,const block_q8_1* x,float* out,const unsigned* grid) {
    const int lane=threadIdx.x&3,hardware_lane=threadIdx.x&31,group=threadIdx.x/4;
    const unsigned mask=15u<<(hardware_lane/4*4);
    constexpr int NT=TOKEN_PARALLEL?1:2;
    const int positions=TOKEN_PARALLEL?2*ROWS:ROWS;
    for(int position=group;position<positions;position+=32) {
        const int row=blockIdx.x*ROWS+(TOKEN_PARALLEL?position%ROWS:position);
        if(row>=w.n_out)continue;
        const int token=TOKEN_PARALLEL?position/ROWS:0;
        const uint8_t* weights=base+(size_t)row*w.row_bytes;
        float sums[NT]={};
#pragma unroll 4
        for(int b=lane;b<w.n_in/32;b+=4) {
            const int block=b/8,q=2*(b&7);
#pragma unroll
            for(int t=0;t<NT;++t) {
                const block_q8_1* y=x+(size_t)jb.y[token+t]*(w.n_in/32)+block*8;
                if constexpr(TYPE==16)sums[t]=native_iq2_xxs_acc<true,FAST>(weights,y,block,q,grid,sums[t]);
                else if constexpr(TYPE==17)sums[t]=native_iq2_xs_acc<true,FAST>(weights,y,block,q,grid,sums[t]);
                else if constexpr(TYPE==18)sums[t]=native_iq3_xxs_acc<true,FAST>(weights,y,block,q,grid,sums[t]);
                else if constexpr(TYPE==21)sums[t]=native_iq3_s_acc<true,FAST>(weights,y,block,q,grid,sums[t]);
                else sums[t]=native_iq2_s_acc<true,FAST>(weights,y,block,q,grid,sums[t]);
            }
        }
#pragma unroll
        for(int t=0;t<NT;++t) {
            float value=0;
#pragma unroll
            for(int p=0;p<4;++p)value+=__shfl_sync(mask,sums[t],p,4);
            if(lane==0)out[(size_t)jb.out[token+t]*w.n_out+row]=value;
        }
    }
}
template<int TYPE,int RPW,int WARPS,bool SHARED,bool FAST,bool TOKEN_PARALLEL>
__global__ void __launch_bounds__(WARPS*32) k_gu_pair_exact_fused(
        DW w,size_t stride,const Job* __restrict__ jobs,const int* njobs,
        const block_q8_1* __restrict__ x,float* __restrict__ out,GuSharedDown sd={},int sd_rows=0) {
    constexpr size_t core_bytes=sizeof(GuScratch<TYPE,RPW,WARPS>);
    constexpr size_t shared_bytes=MAXT*(SH_MAXF/32)*sizeof(block_q8_1);
    constexpr size_t bytes=shared_bytes>core_bytes?shared_bytes:core_bytes;
    __shared__ __align__(16) char scratch[bytes];
    int ji=(int)blockIdx.y;
    if constexpr(true) {
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
    if constexpr(!SHARED) {
        if(jb.n==1) {
            gu_single<TYPE,RPW,WARPS,false,FAST>(w,base,jb,x,out,nullptr);
            return;
        }
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
    if(jb.n==2) {
        pair_body<TYPE,RPW,FAST,TOKEN_PARALLEL>(w,base,jb,x,out,grid);
        return;
    }
    if constexpr(SHARED) {
        if(jb.n==1) {
            gu_single<TYPE,RPW,WARPS,true,FAST>(w,base,jb,x,out,grid);
            return;
        }
    }
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


template<int TYPE,int ROWS,bool TOKEN> void fused_tile(const DW& w,size_t stride,const Job* jobs,const int* njobs,int cap,
        const void* x,float* out,cudaStream_t stream,GuSharedDown shared) {
    constexpr bool FAST=TYPE==16 || TYPE==21;
    const int grid=(w.n_out+ROWS-1)/ROWS,shared_rows=(N+32*grid-1)/(32*grid);
    k_gu_pair_exact_fused<TYPE,ROWS,4,true,FAST,TOKEN><<<dim3(grid,cap+shared_rows),128,0,stream>>>(
        w,stride,jobs,njobs,(const block_q8_1*)x,out,shared,shared_rows);
}
template<int TYPE> void fused_type(const DW& w,size_t stride,const Job* jobs,const int* njobs,int cap,const void* x,
                                  float* out,cudaStream_t stream,GuSharedDown shared,int strategy) {
    if(strategy==1)fused_tile<TYPE,16,false>(w,stride,jobs,njobs,cap,x,out,stream,shared);
    else if(strategy==2)fused_tile<TYPE,16,true>(w,stride,jobs,njobs,cap,x,out,stream,shared);
    else if(strategy==3)fused_tile<TYPE,32,false>(w,stride,jobs,njobs,cap,x,out,stream,shared);
    else throw std::runtime_error("exact fused GU pair strategy");
}
} // namespace
void moe_gu_pair8_sm75(const DW& w,size_t stride,const Job* jobs,const int* njobs,int cap,const void* x,float* out,
        void* stream,int,const Q8S& shared_down,int ffs,const float* shared_gu,float* shared_out,int strategy) {
    if(w.n_in<256 || w.n_in%256 || w.n_out<1 || cap<1 || cap>MAXT*TOPK || cap%TOPK ||
       ffs<32 || ffs%32 || ffs>SH_MAXF || !shared_down.q || !shared_down.d || !shared_gu || !shared_out)
        throw std::runtime_error("exact fused GU pair shape");
    // T=1 cannot contain pairs. Keep the original resource footprint and arithmetic.
    if(cap==TOPK) {
        moe_gu_native_lut_sm75(w,stride,jobs,njobs,cap,x,out,stream,1,shared_down,ffs,shared_gu,shared_out,
                              (w.type==16 || w.type==21)?2:3);
        return;
    }
    if(strategy==0)strategy=cap/TOPK>=7?3:1;
    const GuSharedDown shared{shared_down,shared_gu,shared_out,ffs,cap/TOPK};
#define RUN(TYPE) case TYPE:fused_type<TYPE>(w,stride,jobs,njobs,cap,x,out,(cudaStream_t)stream,shared,strategy);break
    switch(w.type){RUN(16);RUN(17);RUN(18);RUN(21);RUN(22);default:throw std::runtime_error("exact fused GU pair type");}
#undef RUN
    const auto error=cudaGetLastError();
    if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}
} // namespace fnx
