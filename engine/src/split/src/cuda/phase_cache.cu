#include "model.hpp"
#include "kernels.hpp"
#include "phase_cache.h"
#include "env.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

namespace fnx {
void phase_cache_fast_run(const ExpertCache&,PhaseCache*,int,const Job*,const int*,int*,std::size_t,std::size_t,void*);
namespace {
__device__ int slot_index(int slot, int base) {
    if (slot < base) return slot;
    return slot >= base + NL * NEXP ? slot - NL * NEXP : -1; // fixed hot experts have no CLOCK entry
}
__device__ unsigned char& phase_ref(const ExpertCache& ec, PhaseCache& pc, int i) {
    return i < ec.n_slots ? ec.ref[i] : pc.ref[i - ec.n_slots];
}
__device__ int& phase_owner(const ExpertCache& ec, PhaseCache& pc, int i) {
    return i < ec.n_slots ? ec.owner[i] : pc.owner[i - ec.n_slots];
}
__global__ void k_phase_cache_plan(ExpertCache ec, PhaseCache* state, int layer, const Job* jobs, const int* njobs,
                                  int* list, std::size_t gs, std::size_t ds) {
    const int lane = threadIdx.x, base = layer * NEXP, nj = *njobs;
    PhaseCache& pc = *state;
    const std::size_t span = (gs + ds + 255) & ~(std::size_t) 255;
    int cls = -1;
    for (int c = 0; c < pc.nclasses; ++c)
        if (pc.classes[c].span == span) cls = c;
    const int extra = pc.nclasses ? (cls >= 0 ? pc.classes[cls].count : 0) : pc.count;
    const int start = cls >= 0 ? pc.classes[cls].start : 0;
    const int capacity = ec.n_slots + extra;
    for (int j = lane; j < nj; j += 32) {
        const int s = ec.slot_of[base + jobs[j].w];
        if (s >= 0) {
            const int i = slot_index(s, ec.n_slots);
            if (i >= 0) phase_ref(ec, pc, i) = 2;
        }
    }
    __syncwarp();
    int hand = cls >= 0 ? pc.classes[cls].hand : pc.hand, copied = 0;
    if (hand < 0 || hand >= capacity) hand = 0;
    for (int j0 = 0; j0 < nj; j0 += 32) {
        unsigned miss = __ballot_sync(0xffffffffu, j0 + lane < nj && ec.slot_of[base + jobs[j0 + lane].w] < 0);
        while (miss) {
            const int g = base + jobs[j0 + __ffs(miss) - 1].w;
            miss &= miss - 1;
            int victim = -1;
            while (victim < 0) {
                const int index = (hand + lane) % capacity;
                const int i = index < ec.n_slots ? index : index + start;
                const unsigned char r = phase_ref(ec, pc, i);
                const unsigned avail = __ballot_sync(0xffffffffu, r == 0);
                const int f = avail ? __ffs(avail) - 1 : 32;
                if (lane < f && r == 1) phase_ref(ec, pc, i) = 0;
                if (avail) {
                    const int index = (hand + f) % capacity;
                    victim = index < ec.n_slots ? index : index + start;
                }
                hand = (hand + (avail ? f + 1 : 32)) % capacity;
            }
            if (lane == 0) {
                int& owner = phase_owner(ec, pc, victim);
                if (owner >= 0) {
                    ec.slot_of[owner] = -1;
                    ec.ep_gu[owner] = ec.host_gu[owner];
                    ec.ep_dn[owner] = ec.host_dn[owner];
                }
                owner = g;
                phase_ref(ec, pc, victim) = 2;
                if (victim < ec.n_slots) {
                    const int p = ec.pool_of[layer];
                    char* at = ec.pool[p] + (std::size_t) (victim - ec.s0[p]) * ec.slot_bytes[p];
                    ec.ep_gu[g] = at;
                    ec.ep_dn[g] = at + ec.dn_off[p];
                    ec.slot_of[g] = victim;
                } else {
                    ec.ep_gu[g] = pc.gu[victim - ec.n_slots];
                    ec.ep_dn[g] = pc.nclasses ? ec.ep_gu[g] + gs : pc.dn[victim - ec.n_slots];
                    ec.slot_of[g] = victim + NL * NEXP;
                }
                list[1 + copied] = g;
            }
            ++copied;
            __syncwarp();
        }
    }
    for (int j = lane; j < nj; j += 32) {
        const int i = slot_index(ec.slot_of[base + jobs[j].w], ec.n_slots);
        if (i >= 0) phase_ref(ec, pc, i) = 1;
    }
    if (lane == 0) {
        if (cls >= 0) pc.classes[cls].hand = hand;
        else pc.hand = hand;
        list[0] = copied;
    }
}
__global__ void k_phase_cache_copy(ExpertCache ec, const int* list, std::size_t gs, std::size_t ds) {
    const std::size_t ng = gs / 16, per = ng + ds / 16;
    for (std::size_t i = (std::size_t) blockIdx.x * blockDim.x + threadIdx.x; i < (std::size_t) list[0] * per;
         i += (std::size_t) gridDim.x * blockDim.x) {
        const int g = list[1 + i / per];
        const std::size_t off = i % per;
        if (off < ng) reinterpret_cast<uint4*>(const_cast<char*>(ec.ep_gu[g]))[off] = reinterpret_cast<const uint4*>(ec.host_gu[g])[off];
        else reinterpret_cast<uint4*>(const_cast<char*>(ec.ep_dn[g]))[off - ng] = reinterpret_cast<const uint4*>(ec.host_dn[g])[off - ng];
    }
}
}
void phase_cache_run(const ExpertCache& ec, PhaseCache* state, int layer, const Job* jobs, const int* njobs,
                     int* list, std::size_t gs, std::size_t ds, void* stream) {
    if (ec.n_pools != 1 || ec.n_slots < 32 || !gs || !ds || (gs | ds) % 16)
        throw std::runtime_error("phase cache: needs one static pool and aligned expert strides");
    static const bool fast=env_int("FNX_CACHE_FAST",0)!=0;
    if(fast) {phase_cache_fast_run(ec,state,layer,jobs,njobs,list,gs,ds,stream);return;}
    const auto s = (cudaStream_t) stream;
    k_phase_cache_plan<<<1,32,0,s>>>(ec,state,layer,jobs,njobs,list,gs,ds);
    k_phase_cache_copy<<<160,256,0,s>>>(ec,list,gs,ds);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
}
