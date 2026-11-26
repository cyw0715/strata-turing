#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <vector>
namespace fnx {
// Proposal-only online softmax calibration. Observations affect future drafts;
// every already sampled draft retains the actual distribution used for its draw.
class ProposalCalibration {
    std::unordered_map<int,double> bias_;
    static double probability(const std::vector<int>& ids,const std::vector<double>& p,int id) {
        for(size_t i=0;i<ids.size();++i)if(ids[i]==id)return p[i];
        return 0;
    }
    void update(int id,double gradient,double rate) {
        double& b=bias_[id];b=std::clamp(b+rate*gradient,-2.0,2.0);
    }
public:
    void transform(const std::vector<int>& ids,std::vector<double>& p,double strength=1.0) const {
        if(ids.size()!=p.size() || p.empty() || !std::isfinite(strength) || strength<0 || strength>4)
            throw std::runtime_error("invalid draft calibration distribution");
        double sum=0;
        for(size_t i=0;i<p.size();++i) {
            const auto found=bias_.find(ids[i]);
            if(found!=bias_.end())p[i]*=std::exp(strength*found->second);
            sum+=p[i];
        }
        if(!(sum>0) || !std::isfinite(sum))throw std::runtime_error("invalid calibrated draft normalization");
        for(auto& x:p)x/=sum;
    }
    void observe(const std::vector<int>& target_ids,const std::vector<double>& target,
                 const std::vector<int>& proposal_ids,const std::vector<double>& proposal,double rate) {
        if(target_ids.size()!=target.size() || proposal_ids.size()!=proposal.size() ||
           !std::isfinite(rate) || rate<=0 || rate>1)
            throw std::runtime_error("invalid draft calibration observation");
        double mass=0;
        for(int id:proposal_ids)mass+=probability(target_ids,target,id);
        if(!(mass>0))return; // a bias cannot repair disjoint candidate supports
        for(size_t i=0;i<proposal_ids.size();++i)
            update(proposal_ids[i],probability(target_ids,target,proposal_ids[i])/mass-proposal[i],rate);
    }
};
}
