#include "proposal_power.hpp"
#include <cstdio>
#include <numeric>
#include <random>
static double error(const std::vector<double>& p,const std::vector<double>& actual,const std::vector<double>& used) {
    const int n=p.size();std::vector<double> out(n),residual(n);double rejection=0,rest=0;
    for(int i=0;i<n;++i){residual[i]=std::max(0.0,p[i]-used[i]);rest+=residual[i];}
    for(int i=0;i<n;++i) {
        const double accept=used[i]>0?std::min(1.0,p[i]/used[i]):0;
        out[i]+=actual[i]*accept;rejection+=actual[i]*(1-accept);
    }
    for(int i=0;i<n;++i)out[i]+=rejection*(rest>1.e-12?residual[i]/rest:p[i]);
    double result=0;for(int i=0;i<n;++i)result=std::max(result,std::abs(out[i]-p[i]));return result;
}
int main() {
    std::mt19937_64 rng(912);int checked=0,negative=0;
    for(int round=0;round<1000;++round) {
        const int n=2+rng()%31;std::vector<double> p(n),q(n);
        for(int i=0;i<n;++i){p[i]=rng()%5?1+rng()%1000:0;q[i]=rng()%5?1+rng()%1000:0;}
        p[0]+=1;q[0]+=1;
        const double ps=std::accumulate(p.begin(),p.end(),0.0),qs=std::accumulate(q.begin(),q.end(),0.0);
        for(auto& x:p)x/=ps;for(auto& x:q)x/=qs;
        for(double power:{0.5,0.75,1.0,1.25,1.5,2.0}) {
            auto actual=q;fnx::temper_proposal(actual,power);
            if(std::abs(std::accumulate(actual.begin(),actual.end(),0.0)-1)>1.e-12 || error(p,actual,actual)>1.e-12)
                throw std::runtime_error("tempered proposal changed target law");
            if(power==1.0 && actual!=q)throw std::runtime_error("default proposal changed");
            if(power!=1.0 && error(p,actual,q)>1.e-5)++negative;
            ++checked;
        }
    }
    if(negative<3000)throw std::runtime_error("negative control did not detect stale proposal probabilities");
    std::printf("PASS %d normalized proposals: exact target-law identity; negative-control detections=%d\n",checked,negative);
}
