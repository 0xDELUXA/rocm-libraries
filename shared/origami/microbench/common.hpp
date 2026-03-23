#pragma once
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>

#define HIP_CHECK(expr)                                                     \
    do {                                                                    \
        hipError_t err = (expr);                                            \
        if (err != hipSuccess) {                                            \
            fprintf(stderr, "HIP error %d at %s:%d: %s\n",                 \
                    err, __FILE__, __LINE__, hipGetErrorString(err));        \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

struct GpuTimer {
    hipEvent_t start, stop;
    GpuTimer() {
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
    }
    ~GpuTimer() {
        (void)hipEventDestroy(start);
        (void)hipEventDestroy(stop);
    }
    void record_start(hipStream_t s = 0) { HIP_CHECK(hipEventRecord(start, s)); }
    void record_stop(hipStream_t s = 0)  { HIP_CHECK(hipEventRecord(stop, s)); }
    float elapsed_ms() {
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

inline void print_hw_info() {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    int sclk_khz, mclk_khz;
    HIP_CHECK(hipDeviceGetAttribute(&sclk_khz, hipDeviceAttributeClockRate, 0));
    HIP_CHECK(hipDeviceGetAttribute(&mclk_khz, hipDeviceAttributeMemoryClockRate, 0));
    printf("GPU: %s\n", prop.gcnArchName);
    printf("CUs: %d\n", prop.multiProcessorCount);
    printf("SCLK: %.0f MHz, MCLK: %.0f MHz\n", sclk_khz / 1000.0, mclk_khz / 1000.0);
    printf("L2 cache: %u KiB\n", prop.l2CacheSize / 1024);
    printf("LDS per CU: %zu KiB\n", prop.sharedMemPerBlock / 1024);
    printf("\n");
}

constexpr double SCLK_GHZ = 2.2;

inline double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}
