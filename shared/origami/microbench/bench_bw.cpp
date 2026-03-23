/**
 * bench_bw: Memory bandwidth vs workgroup count and vector width
 *
 * Measures:
 * 1. Read BW at different CU counts (1, 2, 4, 8, 16, 32, 64, 128, 256)
 * 2. Read BW at different vector widths (1, 2, 4, 8 elements × bpe)
 * 3. Write BW at different CU counts
 * 4. L2-resident BW (buffer fits in L2)
 *
 * Output: CSV with columns: test,num_wgs,vw,bytes,time_us,bw_GBs,bw_frac
 *
 * This directly calibrates: H_BW_A/B/C, H_VW_BASE/SLOPE, L2_BW, STREAM_READ_GBS
 */
#include "common.hpp"

// Simple read kernel: each thread reads `iters` elements with stride
template <typename T, int VW>
__global__ void read_kernel(const T* __restrict__ src, T* __restrict__ dst,
                            size_t n, int iters) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    T acc = 0;
    for (int it = 0; it < iters; ++it) {
        size_t idx = (tid * VW + it * stride * VW) % n;
        if constexpr (VW == 1) {
            acc += src[idx];
        } else if constexpr (VW == 2) {
            auto v = reinterpret_cast<const __attribute__((ext_vector_type(2))) T*>(src + idx);
            auto val = *v;
            acc += val[0];
        } else if constexpr (VW == 4) {
            auto v = reinterpret_cast<const __attribute__((ext_vector_type(4))) T*>(src + idx);
            auto val = *v;
            acc += val[0];
        } else if constexpr (VW == 8) {
            auto v = reinterpret_cast<const __attribute__((ext_vector_type(8))) T*>(src + idx);
            auto val = *v;
            acc += val[0];
        }
    }
    if (acc != 0) dst[tid] = acc; // prevent optimization
}

// Write kernel
template <typename T, int VW>
__global__ void write_kernel(T* __restrict__ dst, size_t n, int iters) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (int it = 0; it < iters; ++it) {
        size_t idx = (tid * VW + it * stride * VW) % n;
        if constexpr (VW == 1) {
            dst[idx] = T(tid);
        } else if constexpr (VW == 4) {
            auto v = reinterpret_cast<__attribute__((ext_vector_type(4))) T*>(dst + idx);
            *v = {T(tid), T(tid), T(tid), T(tid)};
        }
    }
}

template <typename T, int VW>
void run_read_bw(const char* type_name, int vw, size_t buf_bytes, int num_wgs) {
    size_t n = buf_bytes / sizeof(T);
    n = (n / (VW * 256)) * (VW * 256); // align

    T *d_src, *d_dst;
    HIP_CHECK(hipMalloc(&d_src, n * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_dst, num_wgs * 256 * sizeof(T)));
    HIP_CHECK(hipMemset(d_src, 1, n * sizeof(T)));

    int threads = 256;
    int blocks = num_wgs;
    int iters_per_thread = std::max(1, (int)(n / (blocks * threads * VW)));
    size_t total_bytes = (size_t)blocks * threads * iters_per_thread * VW * sizeof(T);

    // Warmup
    read_kernel<T, VW><<<blocks, threads>>>(d_src, d_dst, n, iters_per_thread);
    HIP_CHECK(hipDeviceSynchronize());

    // Measure
    const int TRIALS = 20;
    std::vector<double> times;
    GpuTimer timer;
    for (int t = 0; t < TRIALS; ++t) {
        timer.record_start();
        read_kernel<T, VW><<<blocks, threads>>>(d_src, d_dst, n, iters_per_thread);
        timer.record_stop();
        times.push_back(timer.elapsed_ms());
    }

    double med_ms = median(times);
    double bw_gbs = total_bytes / (med_ms * 1e-3) / 1e9;

    printf("read,%s,vw=%d,%d,%zu,%.4f,%.1f\n", type_name, vw, num_wgs, total_bytes, med_ms * 1000, bw_gbs);

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));
}

int main() {
    HIP_CHECK(hipSetDevice(0));
    print_hw_info();

    // Large buffer (> MALL, forces HBM)
    size_t hbm_buf = 512ULL * 1024 * 1024; // 512 MiB
    // L2 buffer (fits in one XCD's L2 = 4 MiB)
    size_t l2_buf = 2ULL * 1024 * 1024;    // 2 MiB

    printf("# test,type,vw,num_wgs,total_bytes,time_us,bw_GBs\n");

    // 1. Read BW vs CU count (VW=4, BF16 = short)
    printf("# --- Read BW vs WG count (VW=4, bf16, HBM buffer) ---\n");
    for (int wgs : {1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256}) {
        run_read_bw<short, 4>("bf16", 4, hbm_buf, wgs);
    }

    // 2. Read BW vs VW (all CUs, BF16)
    printf("# --- Read BW vs VW (256 WGs, bf16, HBM buffer) ---\n");
    run_read_bw<short, 1>("bf16", 1, hbm_buf, 256);
    run_read_bw<short, 2>("bf16", 2, hbm_buf, 256);
    run_read_bw<short, 4>("bf16", 4, hbm_buf, 256);
    run_read_bw<short, 8>("bf16", 8, hbm_buf, 256);

    // 3. Read BW vs VW at LOW CU count (1 WG)
    printf("# --- Read BW vs VW (1 WG, bf16, HBM buffer) ---\n");
    run_read_bw<short, 1>("bf16", 1, hbm_buf, 1);
    run_read_bw<short, 2>("bf16", 2, hbm_buf, 1);
    run_read_bw<short, 4>("bf16", 4, hbm_buf, 1);
    run_read_bw<short, 8>("bf16", 8, hbm_buf, 1);

    // 4. L2-resident read BW (small buffer, should hit L2)
    printf("# --- L2-resident Read BW (VW=4, bf16) ---\n");
    for (int wgs : {1, 4, 16, 32, 64, 128, 256}) {
        run_read_bw<short, 4>("bf16_l2", 4, l2_buf, wgs);
    }

    // 5. FP32 read BW vs CU count
    printf("# --- Read BW vs WG count (VW=4, fp32, HBM buffer) ---\n");
    for (int wgs : {1, 4, 16, 64, 128, 256}) {
        run_read_bw<float, 4>("fp32", 4, hbm_buf, wgs);
    }

    return 0;
}
