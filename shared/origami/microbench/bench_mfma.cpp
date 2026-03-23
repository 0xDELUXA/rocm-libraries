/**
 * bench_mfma: MFMA instruction throughput and latency for gfx950
 *
 * Uses gfx950-specific builtins with native __bf16 type.
 * Measures throughput (multiple accumulators) vs latency (RAW chain).
 *
 * Calibrates: MFMA_TABLE entries, issue_interval, parallel_mi_per_cu
 */
#include "common.hpp"

// Native vector types for MFMA builtins
typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float  f32x4  __attribute__((ext_vector_type(4)));
typedef float  f32x16 __attribute__((ext_vector_type(16)));

// ============================================================================
// BF16 16x16x32 throughput — 4 independent accumulators (no RAW)
// ============================================================================
__global__ __launch_bounds__(64)
void mfma_tp_bf16_16x16(f32x4* out, int iters) {
    bf16x8 a = {}, b = {};
    f32x4  c0 = {}, c1 = {}, c2 = {}, c3 = {};

    for (int i = 0; i < iters; ++i) {
        c0 = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c0, 0, 0, 0);
        c1 = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c1, 0, 0, 0);
        c2 = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c2, 0, 0, 0);
        c3 = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c3, 0, 0, 0);
    }
    if (threadIdx.x == 0) { out[0]=c0; out[1]=c1; out[2]=c2; out[3]=c3; }
}

// BF16 16x16x32 latency — 1 accumulator (RAW chain)
__global__ __launch_bounds__(64)
void mfma_lat_bf16_16x16(f32x4* out, int iters) {
    bf16x8 a = {}, b = {};
    f32x4  c = {};
    for (int i = 0; i < iters; ++i)
        c = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0);
    if (threadIdx.x == 0) out[0] = c;
}

// ============================================================================
// BF16 32x32x16 throughput — 4 independent accumulators
// ============================================================================
__global__ __launch_bounds__(64)
void mfma_tp_bf16_32x32(f32x16* out, int iters) {
    bf16x8 a = {}, b = {};
    f32x16 c0 = {}, c1 = {}, c2 = {}, c3 = {};

    for (int i = 0; i < iters; ++i) {
        c0 = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a, b, c0, 0, 0, 0);
        c1 = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a, b, c1, 0, 0, 0);
        c2 = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a, b, c2, 0, 0, 0);
        c3 = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a, b, c3, 0, 0, 0);
    }
    if (threadIdx.x == 0) { out[0]=c0; out[1]=c1; out[2]=c2; out[3]=c3; }
}

// BF16 32x32x16 latency — 1 accumulator
__global__ __launch_bounds__(64)
void mfma_lat_bf16_32x32(f32x16* out, int iters) {
    bf16x8 a = {}, b = {};
    f32x16 c = {};
    for (int i = 0; i < iters; ++i)
        c = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a, b, c, 0, 0, 0);
    if (threadIdx.x == 0) out[0] = c;
}

// ============================================================================
// FP32 16x16x4 throughput — 4 independent accumulators
// ============================================================================
__global__ __launch_bounds__(64)
void mfma_tp_f32_16x16(f32x4* out, int iters) {
    float a = 0, b = 0;
    f32x4 c0 = {}, c1 = {}, c2 = {}, c3 = {};

    for (int i = 0; i < iters; ++i) {
        c0 = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c0, 0, 0, 0);
        c1 = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c1, 0, 0, 0);
        c2 = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c2, 0, 0, 0);
        c3 = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c3, 0, 0, 0);
    }
    if (threadIdx.x == 0) { out[0]=c0; out[1]=c1; out[2]=c2; out[3]=c3; }
}

// FP32 16x16x4 latency
__global__ __launch_bounds__(64)
void mfma_lat_f32_16x16(f32x4* out, int iters) {
    float a = 0, b = 0;
    f32x4 c = {};
    for (int i = 0; i < iters; ++i)
        c = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c, 0, 0, 0);
    if (threadIdx.x == 0) out[0] = c;
}

// ============================================================================
// Multi-wave BF16 16x16 (each wave has its own accumulator)
// ============================================================================
__global__
void mfma_multiwave(f32x4* out, int iters) {
    bf16x8 a = {}, b = {};
    f32x4  c = {};
    for (int i = 0; i < iters; ++i)
        c = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0);
    int wave_id = threadIdx.x / 64;
    if (threadIdx.x % 64 == 0) out[wave_id] = c;
}

// ============================================================================
// Helper: run a kernel, measure wall-clock time, compute cycles/mfma
// ============================================================================
template <typename KernelFunc, typename OutType>
void bench(const char* name, KernelFunc kernel, int blocks, int threads,
           int iters, int mfma_per_iter, OutType* d_out) {
    // Warmup
    kernel<<<blocks, threads>>>(d_out, iters);
    hipDeviceSynchronize();

    GpuTimer timer;
    const int TRIALS = 10;
    std::vector<double> times;
    for (int t = 0; t < TRIALS; ++t) {
        timer.record_start();
        kernel<<<blocks, threads>>>(d_out, iters);
        timer.record_stop();
        times.push_back(timer.elapsed_ms() * 1000.0); // us
    }
    double med_us = median(times);
    int total_mfma = iters * mfma_per_iter;
    int waves = blocks * threads / 64;
    double cycles = med_us * SCLK_GHZ * 1000;
    double per_mfma = cycles / total_mfma;
    printf("%s,%d,%d,%.2f,%.2f\n", name, total_mfma, waves, med_us, per_mfma);
}

// ============================================================================
int main() {
    HIP_CHECK(hipSetDevice(0));
    print_hw_info();

    // Allocate output buffer (large enough for f32x16)
    void* d_out;
    HIP_CHECK(hipMalloc(&d_out, 256 * 16 * sizeof(float)));

    const int ITERS = 10000;

    printf("# test,total_mfma,num_waves,time_us,cycles_per_mfma\n");

    // BF16 16x16x32
    printf("# === BF16 16x16x32 ===\n");
    bench("throughput_bf16_16x16", mfma_tp_bf16_16x16, 1, 64, ITERS, 4, (f32x4*)d_out);
    bench("latency_bf16_16x16",   mfma_lat_bf16_16x16, 1, 64, ITERS, 1, (f32x4*)d_out);

    // BF16 32x32x16
    printf("# === BF16 32x32x16 ===\n");
    bench("throughput_bf16_32x32", mfma_tp_bf16_32x32, 1, 64, ITERS, 4, (f32x16*)d_out);
    bench("latency_bf16_32x32",   mfma_lat_bf16_32x32, 1, 64, ITERS, 1, (f32x16*)d_out);

    // FP32 16x16x4
    printf("# === FP32 16x16x4 ===\n");
    bench("throughput_f32_16x16", mfma_tp_f32_16x16, 1, 64, ITERS, 4, (f32x4*)d_out);
    bench("latency_f32_16x16",   mfma_lat_f32_16x16, 1, 64, ITERS, 1, (f32x4*)d_out);

    // Multi-wave (1, 2, 4 waves on 1 CU, BF16 16x16)
    printf("# === Multi-wave BF16 16x16x32 (RAW chain per wave) ===\n");
    for (int waves : {1, 2, 4}) {
        char name[64];
        snprintf(name, sizeof(name), "multiwave_%dwaves", waves);
        bench(name, mfma_multiwave, 1, 64 * waves, ITERS, waves, (f32x4*)d_out);
    }

    HIP_CHECK(hipFree(d_out));
    return 0;
}
