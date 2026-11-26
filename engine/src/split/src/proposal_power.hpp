#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace fnx {
// Change only the proposal law. The caller must retain these exact probabilities
// beside the sampled drafts and use them in p/q acceptance and residual sampling.
inline void temper_proposal(std::vector<double>& probabilities,double power) {
    // Public policy bounds belong to the caller. Relative powers used by the
    // read-only overlap probes can extend beyond those bounds.
    if(!std::isfinite(power) || power<=0.0)
        throw std::runtime_error("draft proposal power must be finite and positive");
    if(power==1.0)return; // preserve the original proposal and RNG mapping exactly
    if(probabilities.empty())throw std::runtime_error("empty draft proposal");
    const double maximum=*std::max_element(probabilities.begin(),probabilities.end());
    if(!(maximum>0.0) || !std::isfinite(maximum))throw std::runtime_error("invalid draft proposal");
    double sum=0.0;
    for(double& p:probabilities) {p=std::pow(p/maximum,power);sum+=p;}
    for(double& p:probabilities)p/=sum;
}
}
