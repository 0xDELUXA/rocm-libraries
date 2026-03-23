/**
 * bench_latency: Global memory load latency at each cache level
 *
 * Uses pointer chasing with GPU event timing.
 * Buffer size determines which cache level is hit:
 * - < 32 KiB: L1
 * - 32 KiB - 4 MiB: L2
 * - 4 MiB - 256 MiB: MALL
 * - > 256 MiB: HBM
 *
 * Calibrates: load latency per level, H_PROLOGUE_PER_PGR
 */
#include "common.hpp"

__global__ void pointer_chase(const int* __restrict__ chain, int n, int iters,
                              int* out) {
    int idx = threadIdx.x % n;
    for (int i = 0; i < iters; ++i) {
        idx = chain[idx];
    }
    if (threadIdx.x == 0) out[0] = idx;
}

void create_chase_chain(int* h_chain, int n) {
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    srand(42);
    for (int i = n - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        std::swap(perm[i], perm[j]);
    }
    for (int i = 0; i < n - 1; ++i)
        h_chain[perm[i]] = perm[i + 1];
    h_chain[perm[n - 1]] = perm[0];
}

int main() {
    HIP_CHECK(hipSetDevice(0));
    print_hw_info();

    printf("# test,buffer_KiB,iters,time_us,latency_ns\n");

    int* d_out;
    HIP_CHECK(hipMalloc(&d_out, sizeof(int)));

    std::vector<size_t> sizes_kib = {
        4, 8, 16, 24, 32,
        64, 128, 256, 512, 1024,
        2048, 4096,
        8192, 16384, 32768, 65536,
        131072, 262144,
        524288,
    };

    for (size_t kib : sizes_kib) {
        size_t n = kib * 1024 / sizeof(int);
        int iters = std::max(1000, (int)(50000000 / n));

        int* h_chain = new int[n];
        create_chase_chain(h_chain, n);

        int* d_chain;
        HIP_CHECK(hipMalloc(&d_chain, n * sizeof(int)));
        HIP_CHECK(hipMemcpy(d_chain, h_chain, n * sizeof(int), hipMemcpyHostToDevice));

        // Warmup
        pointer_chase<<<1, 1>>>(d_chain, n, iters, d_out);
        HIP_CHECK(hipDeviceSynchronize());

        GpuTimer timer;
        const int TRIALS = 5;
        std::vector<double> latencies;
        for (int t = 0; t < TRIALS; ++t) {
            timer.record_start();
            pointer_chase<<<1, 1>>>(d_chain, n, iters, d_out);
            timer.record_stop();
            double us = timer.elapsed_ms() * 1000.0;
            double ns_per_load = us * 1000.0 / iters;
            latencies.push_back(ns_per_load);
        }

        double med_ns = median(latencies);
        double med_cycles = med_ns * SCLK_GHZ; // ns * GHz = cycles
        printf("ptr_chase,%zu,%d,%.2f,%.1f,%.1f_cycles\n",
               kib, iters, med_ns * iters / 1000.0, med_ns, med_cycles);

        delete[] h_chain;
        HIP_CHECK(hipFree(d_chain));
    }

    HIP_CHECK(hipFree(d_out));
    return 0;
}
