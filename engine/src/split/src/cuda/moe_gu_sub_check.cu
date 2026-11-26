#include "kernels.hpp"
#include "env.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
namespace fnx {
void moe_gu_native_lut_sm75(const DW&,size_t,const Job*,const int*,int,const void*,float*,void*,int,const Q8S&,int,const float*,float*,int);
void moe_gu_sub8_sm75(const DW&,size_t,const Job*,const int*,int,const void*,float*,void*,int,const Q8S&,int,const float*,float*,int);
void moe_gu_pair8_sm75(const DW&,size_t,const Job*,const int*,int,const void*,float*,void*,int,const Q8S&,int,const float*,float*,int);
void moe_down_tc_sm75(const DW&,size_t,const Job*,const int*,int,const float*,float*,void*,int);
namespace {
constexpr size_t PAIRS=MAXT*TOPK,GU_FLOATS=PAIRS*2*FF_CARD_MAX,SHARED_FLOATS=MAXT*N,DN_FLOATS=PAIRS*N;
struct Stats {unsigned long long calls,failures;unsigned types;float gu_relative,down_relative,gu_absolute;};
struct Check {float *gu=nullptr,*shared=nullptr,*dn_ref=nullptr,*dn_new=nullptr;Stats* stats=nullptr;};
Check checks[16];
void ck(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__global__ void compare(Check ref,const float* actual,const float* shared,int count,int outputs,int T,int type) {
    const int tid=threadIdx.x;
    float ge=0,gr=0,de=0,dr=0,maximum=0,scale=0;unsigned bad=0;
    for(int i=tid;i<count*outputs;i+=256) {
        const float a=ref.gu[i],b=actual[i],e=a-b;
        bad|=!isfinite(a)||!isfinite(b);ge+=e*e;gr+=a*a;
        maximum=fmaxf(maximum,fabsf(e));scale=fmaxf(scale,fabsf(a));
    }
    for(int i=tid;i<count*N;i+=256) {
        const float a=ref.dn_ref[i],b=ref.dn_new[i],e=a-b;
        bad|=!isfinite(a)||!isfinite(b);de+=e*e;dr+=a*a;
    }
    for(int i=tid;i<T*N;i+=256)
        bad|=__float_as_uint(ref.shared[i])!=__float_as_uint(shared[i]);
    __shared__ float values[6][256];__shared__ unsigned errors[256];
    values[0][tid]=ge;values[1][tid]=gr;values[2][tid]=de;values[3][tid]=dr;
    values[4][tid]=maximum;values[5][tid]=scale;errors[tid]=bad;__syncthreads();
    for(int n=128;n;n>>=1) {
        if(tid<n) {
            for(int j=0;j<4;++j)values[j][tid]+=values[j][tid+n];
            for(int j=4;j<6;++j)values[j][tid]=fmaxf(values[j][tid],values[j][tid+n]);
            errors[tid]|=errors[tid+n];
        }
        __syncthreads();
    }
    if(tid==0) {
        const float g=sqrtf(values[0][0]/fmaxf(values[1][0],1.e-30f));
        const float d=sqrtf(values[2][0]/fmaxf(values[3][0],1.e-30f));
        const bool fail=errors[0] || !isfinite(g) || !isfinite(d) || g>2.e-6f || d>1.e-3f ||
                        values[4][0]>2.e-6f*fmaxf(1.f,values[5][0]);
        atomicAdd(&ref.stats->calls,1ull);if(fail)atomicAdd(&ref.stats->failures,1ull);
        atomicOr(&ref.stats->types,1u<<type);
        atomicMax((unsigned*)&ref.stats->gu_relative,__float_as_uint(g));
        atomicMax((unsigned*)&ref.stats->down_relative,__float_as_uint(d));
        atomicMax((unsigned*)&ref.stats->gu_absolute,__float_as_uint(values[4][0]));
    }
}
}
size_t moe_gu_sub_check_bytes() {return (GU_FLOATS+SHARED_FLOATS+2*DN_FLOATS)*sizeof(float)+256;}
void moe_gu_sub_check_bind(void* memory) {
    int dev=-1;ck(cudaGetDevice(&dev));if(dev<0 || dev>=16)throw std::runtime_error("GU check device");
    Check c;c.gu=(float*)memory;c.shared=c.gu+GU_FLOATS;c.dn_ref=c.shared+SHARED_FLOATS;c.dn_new=c.dn_ref+DN_FLOATS;
    c.stats=(Stats*)(c.dn_new+DN_FLOATS);checks[dev]=c;
}
void moe_gu_subwarp_run(const DW& gu,size_t stride,const Job* jobs,const int* njobs,int cap,const void* x,
                        float* out,void* stream,int variant,const Q8S& shared_down,int ffs,const float* shared_gu,
                        float* shared_out,int mode,const DW& down,size_t down_stride,const Job* down_jobs,bool check) {
    Check c;
    if(check) {
        int dev=-1;ck(cudaGetDevice(&dev));if(dev<0 || dev>=16 || !checks[dev].gu)throw std::runtime_error("GU check buffers not bound");
        c=checks[dev];
        moe_gu_native_lut_sm75(gu,stride,jobs,njobs,cap,x,c.gu,stream,variant,shared_down,ffs,shared_gu,c.shared,mode);
    }
    static const bool pair=env_int("FNX_GU_PAIR_EXACT",0)!=0;
    if(pair)moe_gu_pair8_sm75(gu,stride,jobs,njobs,cap,x,out,stream,1,shared_down,ffs,shared_gu,shared_out,0);
    else moe_gu_sub8_sm75(gu,stride,jobs,njobs,cap,x,out,stream,1,shared_down,ffs,shared_gu,shared_out,mode);
    if(check) {
        if(env_int("FNX_TC_DOWN",0) && (down.type==T_Q2_0 || down.type==T_IQ4_NL)) {
            moe_down_tc_sm75(down,down_stride,down_jobs,njobs,cap,c.gu,c.dn_ref,stream,3);
            moe_down_tc_sm75(down,down_stride,down_jobs,njobs,cap,out,c.dn_new,stream,3);
        } else {
            moe_down(down,down_stride,down_jobs,njobs,cap,c.gu,c.dn_ref,stream);
            moe_down(down,down_stride,down_jobs,njobs,cap,out,c.dn_new,stream);
        }
        compare<<<1,256,0,(cudaStream_t)stream>>>(c,out,shared_out,cap,gu.n_out,cap/TOPK,gu.type);
        ck(cudaGetLastError());
    }
}
void moe_gu_sub_check_report(int dev) {
    if(dev<0 || dev>=16 || !checks[dev].gu)return;
    ck(cudaSetDevice(dev));Stats s{};ck(cudaMemcpy(&s,checks[dev].stats,sizeof(s),cudaMemcpyDeviceToHost));
    std::fprintf(stderr,"fnx: %s: dev=%d calls=%llu failures=%llu types=%u GUrel=%.8g DNrel=%.8g maxabs=%.8g\n",
                 env_int("FNX_GU_PAIR_EXACT",0)?"GU exact pairs shadow":"GU subwarp shadow",dev,s.calls,s.failures,s.types,s.gu_relative,s.down_relative,s.gu_absolute);
    if(s.failures)throw std::runtime_error("GU subwarp numerical validation failed");
}
}
