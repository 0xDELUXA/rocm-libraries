/**
 * bench_lds: LDS bandwidth, bank conflicts, and barrier latency
 *
 * Uses GPU event timing (wall-clock) with many iterations for accuracy.
 * Each kernel runs many iterations internally so the per-iteration overhead
 * can be computed by dividing total time by iteration count.
 *
 * Calibrates: LDS BW, bank conflict model, barrier_latency, per-iter overhead
 */
#include "common.hpp"

// ============================================================================
// LDS read, stride-1 (conflict-free: each thread reads consecutive dword)
// ============================================================================
__global__ void lds_read_stride1(int iters, float* out) {
    __shared__ float lds[64 * 64]; // 16 KiB
    const int N = 64 * 64;
    for (int i = threadIdx.x; i < N; i += blockDim.x) lds[i] = 1.0f;
    __syncthreads();

    float acc = 0;
    for (int it = 0; it < iters; ++it) {
        int idx = (threadIdx.x + it * blockDim.x) % N;
        acc += lds[idx];
    }
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

// ============================================================================
// LDS read with variable stride (controls bank conflict degree)
// ============================================================================
template <int STRIDE>
__global__ void lds_read_strided(int iters, float* out) {
    __shared__ float lds[64 * 64];
    const int N = 64 * 64;
    for (int i = threadIdx.x; i < N; i += blockDim.x) lds[i] = 1.0f;
    __syncthreads();

    float acc = 0;
    for (int it = 0; it < iters; ++it) {
        int idx = (threadIdx.x * STRIDE + it) % N;
        acc += lds[idx];
    }
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

// ============================================================================
// Barrier only
// ============================================================================
__global__ void barrier_only(int iters, float* out) {
    float acc = 0;
    for (int it = 0; it < iters; ++it) {
        __syncthreads();
        acc += 1.0f;  // prevent loop elimination
    }
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

// ============================================================================
// LDS write + barrier + read + barrier (full GEMM-like round-trip)
// ============================================================================
__global__ void lds_roundtrip(int iters, float* out) {
    __shared__ float lds[64 * 64];
    const int N = 64 * 64;

    float acc = 0;
    for (int it = 0; it < iters; ++it) {
        int w_idx = (threadIdx.x + it * blockDim.x) % N;
        lds[w_idx] = (float)(threadIdx.x + it);
        __syncthreads();

        int r_idx = (threadIdx.x + it * blockDim.x + 128) % N;
        acc += lds[r_idx];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

// ============================================================================
// LDS read with ds_read_b128 (4 dwords per read, simulates LRVW=8 for bf16)
// ============================================================================
__global__ void lds_read_vec4(int iters, float* out) {
    __shared__ float lds[64 * 64];
    const int N = 64 * 64;
    for (int i = threadIdx.x; i < N; i += blockDim.x) lds[i] = 1.0f;
    __syncthreads();

    float4 acc = {0, 0, 0, 0};
    for (int it = 0; it < iters; ++it) {
        int idx = ((threadIdx.x * 4 + it * blockDim.x * 4) % N) & ~3;
        float4 val = *reinterpret_cast<float4*>(&lds[idx]);
        acc.x += val.x;
    }
    if (threadIdx.x == 0) out[blockIdx.x] = acc.x;
}

// ============================================================================
int main() {
    HIP_CHECK(hipSetDevice(0));
    print_hw_info();

    float* d_out;
    HIP_CHECK(hipMalloc(&d_out, 256 * sizeof(float)));

    const int ITERS = 100000;

    printf("# test,threads,iters,time_us,cycles_per_iter\n");

    auto run = [&](const char* name, auto kernel, int threads) {
        // Warmup
        kernel<<<1, threads>>>(ITERS, d_out);
        HIP_CHECK(hipDeviceSynchronize());

        GpuTimer timer;
        const int TRIALS = 10;
        std::vector<double> times;
        for (int t = 0; t < TRIALS; ++t) {
            timer.record_start();
            kernel<<<1, threads>>>(ITERS, d_out);
            timer.record_stop();
            times.push_back(timer.elapsed_ms() * 1000.0); // us
        }
        double med_us = median(times);
        double cycles_per_iter = med_us * SCLK_GHZ * 1000.0 / ITERS;
        printf("%s,%d,%d,%.2f,%.2f\n", name, threads, ITERS, med_us, cycles_per_iter);
    };

    // 1. LDS read bandwidth (conflict-free)
    printf("# --- LDS Read Bandwidth (conflict-free) ---\n");
    run("lds_read_stride1_64t", lds_read_stride1, 64);
    run("lds_read_stride1_128t", lds_read_stride1, 128);
    run("lds_read_stride1_256t", lds_read_stride1, 256);

    // 2. LDS bank conflicts (stride controls conflict degree)
    printf("# --- LDS Bank Conflict Sweep ---\n");
    run("lds_stride_1", lds_read_strided<1>, 64);    // no conflict
    run("lds_stride_2", lds_read_strided<2>, 64);    // 2-way
    run("lds_stride_4", lds_read_strided<4>, 64);    // 4-way
    run("lds_stride_8", lds_read_strided<8>, 64);    // 8-way
    run("lds_stride_16", lds_read_strided<16>, 64);  // 16-way
    run("lds_stride_32", lds_read_strided<32>, 64);  // 32-way (max)

    // 3. LDS vector read (ds_read_b128)
    printf("# --- LDS Vector Read (float4) ---\n");
    run("lds_read_vec4_64t", lds_read_vec4, 64);
    run("lds_read_vec4_256t", lds_read_vec4, 256);

    // 4. Barrier latency
    printf("# --- Barrier Latency ---\n");
    run("barrier_64t", barrier_only, 64);
    run("barrier_128t", barrier_only, 128);
    run("barrier_256t", barrier_only, 256);

    // 5. Full round-trip: write + barrier + read + barrier
    printf("# --- LDS Round-trip ---\n");
    run("roundtrip_64t", lds_roundtrip, 64);
    run("roundtrip_128t", lds_roundtrip, 128);
    run("roundtrip_256t", lds_roundtrip, 256);

    HIP_CHECK(hipFree(d_out));
    return 0;
}
