#pragma once
#include <cstddef>
namespace fnx {
struct ExpertCache;
struct Job;
struct PhaseCacheClass {
    std::size_t span = 0;
    int start = 0, count = 0, hand = 0;
};
// Device descriptor. Base slots use ExpertCache's original ownership/ref arrays;
// extra slots borrow phase storage. Updating count at an idle boundary preserves graphs.
struct PhaseCache {
    int count = 0, hand = 0;
    char** gu = nullptr;
    char** dn = nullptr;
    int* owner = nullptr;
    unsigned char* ref = nullptr;
    // Optional exact-size domains. Base slots remain a shared fallback, so even a
    // class with no extra slots can accommodate a complete decode window.
    int nclasses = 0;
    PhaseCacheClass classes[48] = {};
};
void phase_cache_run(const ExpertCache&, PhaseCache*, int, const Job*, const int*, int*, std::size_t, std::size_t, void*);
}
