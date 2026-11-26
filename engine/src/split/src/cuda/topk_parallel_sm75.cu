// Exact Turing Top64: Strata radix tiles plus warp merge of sorted lists (MIT).
#include "kernels.hpp"
#include "env.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
namespace fnx { namespace topk_parallel {

// The sampler's candidates (included by kernels.cu): per verify row, this card's slice of the head's logits with the
// row's penalties applied (llama.cpp's penalties stage, as Strata applies it: repeat multiplies a non-positive logit
// and divides a positive one, then frequency x count and presence once), then its TOPK_N highest, ordered by logit
// (descending) and token id (ascending) on ties. A block per row: a radix select of the TOPK_N-th highest key, then
// everything at or above it sorted. More than TOPK_CAP at or above it (a tie at the threshold that wide: a row of equal
// logits) takes a second pass that picks the same TOPK_N, so the result never depends on how many tie. A NaN or inf
// among the logits sets err bit 16.
namespace {

constexpr int TOPK_THREADS = 256, TOPK_CAP = 256;

__device__ __forceinline__ unsigned tk_key(float v) {   // order-preserving: a larger float, a larger key
    const unsigned u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__global__ void __launch_bounds__(TOPK_THREADS) k_topk_tiles(float* __restrict__ logits, int n, int v0, const int* __restrict__ pen_tok,
                                                           const int* __restrict__ pen_cnt, const int* __restrict__ pen_n, int pen_stride,
                                                           const float* __restrict__ pen_par, float* __restrict__ out_v,
                                                           int* __restrict__ out_i, unsigned* err) {
    const int row = blockIdx.y, tid = threadIdx.x, lane = tid & 31;
    const int full_n=n, start=(int)blockIdx.x*full_n/(int)gridDim.x;
    n=((int)blockIdx.x+1)*full_n/(int)gridDim.x-start;
    float* L = logits + (size_t) row * full_n + start;
    v0+=start;
    // a NaN or inf among the head's logits (before the penalties, which a request sets): err bit 16
    bool bad = false;
    for (int j = tid; j < n; j += TOPK_THREADS) bad = bad || !isfinite(L[j]);
    if (bad) atomicOr(err, 16u);
    __syncthreads();   // every logit checked before any penalty changes one
    // the row's penalties on this card's tokens (each token once in the list)
    const float repeat = pen_par[row * 3], freq = pen_par[row * 3 + 1], present = pen_par[row * 3 + 2];   // the row's penalty parameters
    for (int i = tid; i < pen_n[row]; i += TOPK_THREADS) {
        const int tok = pen_tok[row * pen_stride + i] - v0, cnt = pen_cnt[row * pen_stride + i];
        if (tok < 0 || tok >= n || cnt <= 0) continue;
        float l = L[tok];
        l = l <= 0.f ? l * repeat : l / repeat;
        L[tok] = l - (float) cnt * freq - present;
    }
    __syncthreads();
    // the TOPK_N-th highest key: four passes of 8 bits
    __shared__ unsigned hist[256];
    __shared__ unsigned prefix_s, want_s;
    unsigned prefix = 0, mask = 0, want = TOPK_N;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = tid; b < 256; b += TOPK_THREADS) hist[b] = 0;
        __syncthreads();
        for (int jb = 0; jb < n; jb += TOPK_THREADS) {
            const int j = jb + tid;
            const unsigned k = j < n ? tk_key(L[j]) : 0u;
            const bool in = j < n && (k & mask) == prefix;
            const unsigned bin = in ? (k >> shift) & 255u : 256u;
#if FNX_HIP
            const unsigned peers = (unsigned) __match_any_sync(0xffffffffull, bin);   // HIP wants a 64-bit mask (wave32: the high half is empty)
#else
            const unsigned peers = __match_any_sync(0xffffffffu, bin);
#endif
            if (in && lane == __ffs(peers) - 1) atomicAdd(&hist[bin], (unsigned) __popc(peers));
        }
        __syncthreads();
        if (tid == 0) {
            unsigned acc = 0;
            int b = 255;
            for (; b > 0; --b) {
                if (acc + hist[b] >= want) break;
                acc += hist[b];
            }
            prefix_s = prefix | ((unsigned) b << shift);
            want_s = want - acc;
        }
        __syncthreads();
        prefix = prefix_s;
        want = want_s;
        mask |= 255u << shift;
        __syncthreads();
    }
    // everything at or above the threshold, then sorted: (key desc, id asc) as one 64-bit key, descending
    __shared__ unsigned long long cand[TOPK_CAP];
    __shared__ unsigned n_cand;
    for (int i = tid; i < TOPK_CAP; i += TOPK_THREADS) cand[i] = 0ull;
    if (tid == 0) n_cand = 0;
    __syncthreads();
    for (int j = tid; j < n; j += TOPK_THREADS) {
        const unsigned k = tk_key(L[j]);
        if (k >= prefix) {
            const unsigned s = atomicAdd(&n_cand, 1u);
            if (s < TOPK_CAP) cand[s] = ((unsigned long long) k << 32) | (0xffffffffu - (unsigned) j);
        }
    }
    __syncthreads();
    const bool wide = n_cand > TOPK_CAP;
    __syncthreads();   // every thread has read n_cand before the wide pass resets it
    if (wide) {   // block-uniform. The first TOPK_N of the sorted set are every key above the threshold
        // (TOPK_N - want of them) and the `want` lowest ids at it: those exactly, the ties taken in id order a chunk at a
        // time (a block-wide rank from each warp's ballot), stopping once `want` are in
        __shared__ unsigned warp_n[TOPK_THREADS / 32];
        const int warp = tid >> 5;
        for (int i = tid; i < TOPK_CAP; i += TOPK_THREADS) cand[i] = 0ull;
        if (tid == 0) n_cand = 0;
        __syncthreads();
        for (int j = tid; j < n; j += TOPK_THREADS) {
            const unsigned k = tk_key(L[j]);
            if (k > prefix) cand[atomicAdd(&n_cand, 1u)] = ((unsigned long long) k << 32) | (0xffffffffu - (unsigned) j);
        }
        const unsigned base = TOPK_N - want;
        unsigned taken = 0;
        for (int jb = 0; jb < n && taken < want; jb += TOPK_THREADS) {
            const int j = jb + tid;
            const bool eq = j < n && tk_key(L[j]) == prefix;
            const unsigned bal = __ballot_sync(0xffffffffu, eq);
            if (lane == 0) warp_n[warp] = (unsigned) __popc(bal);
            __syncthreads();
            unsigned before = (unsigned) __popc(bal & ((1u << lane) - 1u)), total = 0;
            for (int w = 0; w < TOPK_THREADS / 32; ++w) {
                before += w < warp ? warp_n[w] : 0u;
                total += warp_n[w];
            }
            if (eq && taken + before < want) cand[base + taken + before] = ((unsigned long long) prefix << 32) | (0xffffffffu - (unsigned) j);
            taken += total;
            __syncthreads();   // warp_n read by every thread before the next chunk writes it
        }
        __syncthreads();
    }
    for (int size = 2; size <= TOPK_CAP; size <<= 1)
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if (tid < TOPK_CAP) {
                const int o = tid ^ stride;
                if (o > tid) {
                    const bool desc = (tid & size) == 0;
                    const unsigned long long a = cand[tid], b = cand[o];
                    if (desc ? a < b : a > b) {
                        cand[tid] = b;
                        cand[o] = a;
                    }
                }
            }
            __syncthreads();
        }
    if (tid < TOPK_N) {
        const unsigned long long c = cand[tid];
        const int j = (int) (0xffffffffu - (unsigned) (c & 0xffffffffu));
        out_i[(row * gridDim.x + blockIdx.x) * TOPK_N + tid] = v0 + j;
        out_v[(row * gridDim.x + blockIdx.x) * TOPK_N + tid] = L[j];
    }
}

}  // namespace

__device__ __forceinline__ unsigned ordered_key(float value) {
    const unsigned bits=__float_as_uint(value);
    return bits&0x80000000u?~bits:bits|0x80000000u;
}
__global__ void merge_sorted_tiles(const float* values,const int* indices,float* out_values,int* out_indices,int parts) {
    const int row=blockIdx.x,lane=threadIdx.x;
    const int base=(row*parts+lane)*TOPK_N;
    int position=0;
    unsigned long long head=lane<parts?((unsigned long long)ordered_key(values[base])<<32) |
        (0xffffffffu-(unsigned)indices[base]):0ull;
    // Each tile is already sorted. Only the winning list advances at each exact global pick.
#pragma unroll 1
    for(int pick=0;pick<TOPK_N;++pick) {
        unsigned long long best=head;
#pragma unroll
        for(int offset=16;offset;offset>>=1)
            best=max(best,__shfl_xor_sync(0xffffffffu,best,offset));
        if(lane==(pick&31)) {
            const unsigned key=best>>32,bits=key&0x80000000u?key&0x7fffffffu:~key;
            out_values[row*TOPK_N+pick]=__uint_as_float(bits);
            out_indices[row*TOPK_N+pick]=(int)(0xffffffffu-(unsigned)best);
        }
        if(head==best) {
            ++position;
            head=position<TOPK_N?((unsigned long long)ordered_key(values[base+position])<<32) |
                (0xffffffffu-(unsigned)indices[base+position]):0ull;
        }
    }
}

struct CheckStats {unsigned long long calls=0,failures=0;};
struct Workspace {
    float *partial_values=nullptr,*reference_logits=nullptr,*reference_values=nullptr;
    int *partial_indices=nullptr,*reference_indices=nullptr;
    unsigned* reference_error=nullptr;
    CheckStats* stats=nullptr;
    int max_n=0;
    bool check=false;
};
Workspace slots[16];
size_t align256(size_t bytes) {return (bytes+255)&~size_t(255);}
size_t workspace_bytes(int max_n,bool check) {
    const size_t partial=align256((size_t)MAXT*32*TOPK_N*4);
    return partial*2+(check?align256((size_t)MAXT*max_n*4)+2*align256((size_t)MAXT*TOPK_N*4)+512:0);
}
__global__ void compare(const float* logits,const float* ref_logits,int count,const float* values,const float* ref_values,
                        const int* indices,const int* ref_indices,int picks,const unsigned* error,const unsigned* ref_error,
                        CheckStats* stats) {
    unsigned long long bad=0;
    const int start=blockIdx.x*blockDim.x+threadIdx.x,step=gridDim.x*blockDim.x;
    for(int i=start;i<count;i+=step)bad+=__float_as_uint(logits[i])!=__float_as_uint(ref_logits[i]);
    for(int i=start;i<picks;i+=step)
        bad+=(indices[i]!=ref_indices[i]) || (__float_as_uint(values[i])!=__float_as_uint(ref_values[i]));
    if(start==0) {
        atomicAdd(&stats->calls,1ull);
        bad+=*error!=*ref_error;
    }
    if(bad)atomicAdd(&stats->failures,bad);
}
} // namespace topk_parallel

size_t topk_parallel_bytes_sm75(int max_n,bool check) {return topk_parallel::workspace_bytes(max_n,check);}
void topk_parallel_bind_sm75(void* memory,int max_n,bool check) {
    int device=-1;cudaGetDevice(&device);
    if(device<0 || device>=16 || max_n<2048)throw std::runtime_error("TopK workspace shape/device");
    auto& w=topk_parallel::slots[device];
    w=topk_parallel::Workspace{};
    char* cursor=(char*)memory;
    auto take=[&](size_t bytes) {char* p=cursor;cursor+=topk_parallel::align256(bytes);return p;};
    w.partial_values=(float*)take((size_t)MAXT*32*TOPK_N*4);
    w.partial_indices=(int*)take((size_t)MAXT*32*TOPK_N*4);
    w.max_n=max_n;w.check=check;
    if(check) {
        w.reference_logits=(float*)take((size_t)MAXT*max_n*4);
        w.reference_values=(float*)take((size_t)MAXT*TOPK_N*4);
        w.reference_indices=(int*)take((size_t)MAXT*TOPK_N*4);
        w.reference_error=(unsigned*)take(4);
        w.stats=(topk_parallel::CheckStats*)take(sizeof(topk_parallel::CheckStats));
        cudaMemset(w.stats,0,sizeof(topk_parallel::CheckStats));
    }
    if((size_t)(cursor-(char*)memory)>topk_parallel_bytes_sm75(max_n,check))throw std::runtime_error("TopK workspace overflow");
    const auto error=cudaGetLastError();
    if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}

void top_tokens_dispatch_sm75(float* logits,int n,int T,int v0,const int* pt,const int* pc,const int* pn,int stride,
                              const float* pp,float* values,int* indices,unsigned* error,void* stream) {
    static const bool enabled=env_int("FNX_TOPK_PARALLEL",0)!=0;
    if(!enabled || n<2048) {top_tokens(logits,n,T,v0,pt,pc,pn,stride,pp,values,indices,error,stream);return;}
    int device=-1;cudaGetDevice(&device);
    if(device<0 || device>=16)throw std::runtime_error("TopK device out of range");
    const auto& w=topk_parallel::slots[device];
    if(!w.partial_values || n>w.max_n || T<1 || T>MAXT)throw std::runtime_error("TopK workspace was not bound before capture");
    const cudaStream_t st=(cudaStream_t)stream;
    if(w.check) {
        cudaMemcpyAsync(w.reference_logits,logits,(size_t)T*n*4,cudaMemcpyDeviceToDevice,st);
        cudaMemsetAsync(w.reference_error,0,4,st);
        top_tokens(w.reference_logits,n,T,v0,pt,pc,pn,stride,pp,w.reference_values,w.reference_indices,w.reference_error,stream);
    }
    topk_parallel::k_topk_tiles<<<dim3(32,T),256,0,st>>>(logits,n,v0,pt,pc,pn,stride,pp,w.partial_values,w.partial_indices,error);
    topk_parallel::merge_sorted_tiles<<<T,32,0,st>>>(w.partial_values,w.partial_indices,values,indices,32);
    if(w.check)
        topk_parallel::compare<<<160,256,0,st>>>(logits,w.reference_logits,T*n,values,w.reference_values,indices,w.reference_indices,
            T*TOPK_N,error,w.reference_error,w.stats);
    const auto status=cudaGetLastError();
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}

void topk_parallel_check_report_sm75(int device) {
    if(device<0 || device>=16 || !topk_parallel::slots[device].stats)return;
    cudaSetDevice(device);
    topk_parallel::CheckStats stats;
    const auto error=cudaMemcpy(&stats,topk_parallel::slots[device].stats,sizeof(stats),cudaMemcpyDeviceToHost);
    if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
    std::fprintf(stderr,"fnx: TopK exact shadow: dev=%d calls=%llu failures=%llu\n",device,stats.calls,stats.failures);
    if(stats.failures)throw std::runtime_error("TopK numerical validation failed");
}
} // namespace fnx
