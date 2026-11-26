// CUDA/NCCL port of ../hip/rccl.h. Load NCCL dynamically, create one communicator
// per participating device, and check an exact FP16 all-reduce before use.
// Shared RCCL/NCCL ABI types and policy stay in ../hip/rccl_dl.hpp.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "../hip/rccl_dl.hpp"

namespace fnx {

struct RcclState {
    RcclApi api;
    std::vector<void*> comm;   // [card]
    ~RcclState() {
        for (void* c : comm)
            if (c && api.CommDestroy) api.CommDestroy(c);
        // (the library stays loaded: RCCL's helper threads may still be winding down)
    }
};

namespace rccl_detail {
static __global__ void k_fill16(__half* p, int n, float scale) {
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += gridDim.x * blockDim.x) p[e] = __float2half((float) (e % 16) * scale);
}
static __global__ void k_check16(const __half* p, int n, float scale, unsigned* bad) {
    unsigned b = 0;
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += gridDim.x * blockDim.x) b += __half2float(p[e]) != (float) (e % 16) * scale;
    if (b) atomicAdd(bad, b);
}
}  // namespace rccl_detail

// devs: the cards' device ordinals. On success st holds the communicators; why is the library path (or the reason for the failure).
inline bool rccl_open(const std::vector<int>& devs, std::shared_ptr<RcclState>& st, std::string& why) {
    auto s = std::make_shared<RcclState>();
    if (!rccl_load(s->api, "", why)) return false;
    const std::string lib = why;
    const int n = (int) devs.size();
    s->comm.assign((size_t) n, nullptr);
    auto fail = [&](const char* what, int r) {
        why = std::string(what) + (s->api.ErrorString ? std::string(": ") + s->api.ErrorString(r) : std::string(" failed"));
        return false;
    };
    int r = s->api.CommInitAll(s->comm.data(), n, devs.data());
    if (r != RCCL_SUCCESS) {
        s->comm.assign((size_t) n, nullptr);
        return fail("ncclCommInitAll", r);
    }
    // the self-test: every card holds (e % 16) * (card + 1) as fp16, the sum must be (e % 16) * n (n + 1) / 2 on every card, in full
    const int cnt = 1 << 18;
    const float scale_sum = (float) (n * (n + 1) / 2);
    std::vector<__half*> buf((size_t) n, nullptr);
    std::vector<unsigned*> bad((size_t) n, nullptr);
    std::vector<cudaStream_t> str((size_t) n, nullptr);
    auto cleanup = [&] {
        for (int i = 0; i < n; ++i) {
            cudaSetDevice(devs[(size_t) i]);
            if (buf[(size_t) i]) cudaFree(buf[(size_t) i]);
            if (bad[(size_t) i]) cudaFree(bad[(size_t) i]);
            if (str[(size_t) i]) cudaStreamDestroy(str[(size_t) i]);
        }
    };
    for (int i = 0; i < n; ++i) {
        cudaSetDevice(devs[(size_t) i]);
        if (cudaMalloc((void**) &buf[(size_t) i], (size_t) cnt * 2) != cudaSuccess || cudaMalloc((void**) &bad[(size_t) i], 4) != cudaSuccess ||
            cudaStreamCreateWithFlags(&str[(size_t) i], cudaStreamNonBlocking) != cudaSuccess) {
            cudaGetLastError();
            cleanup();
            why = "the self-test buffers could not be allocated";
            return false;
        }
        cudaMemset(bad[(size_t) i], 0, 4);
        rccl_detail::k_fill16<<<256, 256, 0, str[(size_t) i]>>>(buf[(size_t) i], cnt, (float) (i + 1));
    }
    r = s->api.GroupStart();
    for (int i = 0; i < n && r == RCCL_SUCCESS; ++i) {
        cudaSetDevice(devs[(size_t) i]);
        r = s->api.AllReduce(buf[(size_t) i], buf[(size_t) i], (size_t) cnt, RCCL_FLOAT16, RCCL_SUM, s->comm[(size_t) i], (void*) str[(size_t) i]);
    }
    const int r2 = s->api.GroupEnd();
    if (r == RCCL_SUCCESS) r = r2;
    if (r != RCCL_SUCCESS) {
        cleanup();
        return fail("ncclAllReduce", r);
    }
    for (int i = 0; i < n; ++i) {
        cudaSetDevice(devs[(size_t) i]);
        rccl_detail::k_check16<<<256, 256, 0, str[(size_t) i]>>>(buf[(size_t) i], cnt, scale_sum, bad[(size_t) i]);
    }
    // a bounded wait (30 s): a collective that never completes must not hang the start
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
        cudaSetDevice(devs[(size_t) i]);
        cudaError_t status;
        while ((status = cudaStreamQuery(str[(size_t) i])) == cudaErrorNotReady)
            if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(30)) {
                why = "the self-test all-reduce did not finish in 30 s";
                return false;   // (buffers stay: the kernels may still be running)
            }
        if (status != cudaSuccess) {
            why = std::string("CUDA error during the NCCL self-test: ") + cudaGetErrorString(status);
            return false;
        }
    }
    unsigned total = 0;
    for (int i = 0; i < n; ++i) {
        cudaSetDevice(devs[(size_t) i]);
        unsigned b = 0;
        if (cudaMemcpy(&b, bad[(size_t) i], 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
            why = "CUDA error reading the NCCL self-test result";
            return false;
        }
        total += b;
    }
    cleanup();
    if (cudaGetLastError() != cudaSuccess) {
        why = "a CUDA error during the self-test";
        return false;
    }
    if (total) {
        why = "the self-test sum was wrong in " + std::to_string(total) + " values";
        return false;
    }
    st = s;
    why = lib;
    return true;
}

}  // namespace fnx
