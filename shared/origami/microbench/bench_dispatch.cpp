/**
 * bench_dispatch: Kernel dispatch overhead and minimum kernel latency
 *
 * Measures:
 * 1. Empty kernel dispatch time (pure overhead)
 * 2. Minimal compute kernel (1 MFMA)
 * 3. Kernel with varying WG counts (dispatch scaling)
 * 4. Back-to-back kernel launches (pipeline effect)
 *
 * Calibrates: H_DISPATCH, minimum kernel latency floor
 */
#include "common.hpp"

__global__ void empty_kernel() {}

__global__ void one_mfma_kernel(float* out) {
    typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
    typedef float  f32x4  __attribute__((ext_vector_type(4)));
    bf16x8 a = {}, b = {};
    f32x4  c = {};
    c = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0);
    if (threadIdx.x == 0) out[blockIdx.x] = c[0];
}

__global__ void small_compute_kernel(float* out, int iters) {
    float acc = 0;
    for (int i = 0; i < iters; ++i)
        acc += 1.0f;
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

int main() {
    HIP_CHECK(hipSetDevice(0));
    print_hw_info();

    float* d_out;
    HIP_CHECK(hipMalloc(&d_out, 256 * sizeof(float)));

    printf("# test,num_wgs,threads,trials,median_us\n");

    // 1. Empty kernel dispatch
    {
        const int TRIALS = 200;
        std::vector<double> times;
        GpuTimer timer;
        for (int t = 0; t < TRIALS; ++t) {
            timer.record_start();
            empty_kernel<<<1, 64>>>();
            timer.record_stop();
            times.push_back(timer.elapsed_ms() * 1000); // us
        }
        double med = median(times);
        printf("empty_kernel,1,64,%d,%.3f\n", TRIALS, med);
    }

    // 2. One MFMA kernel
    {
        const int TRIALS = 200;
        for (int wgs : {1, 4, 16, 64, 256}) {
            std::vector<double> times;
            GpuTimer timer;
            for (int t = 0; t < TRIALS; ++t) {
                timer.record_start();
                one_mfma_kernel<<<wgs, 64>>>(d_out);
                timer.record_stop();
                times.push_back(timer.elapsed_ms() * 1000);
            }
            double med = median(times);
            printf("one_mfma,%d,64,%d,%.3f\n", wgs, TRIALS, med);
        }
    }

    // 3. Small compute with varying iterations
    {
        const int TRIALS = 100;
        for (int iters : {1, 10, 100, 1000}) {
            std::vector<double> times;
            GpuTimer timer;
            for (int t = 0; t < TRIALS; ++t) {
                timer.record_start();
                small_compute_kernel<<<256, 256>>>(d_out, iters);
                timer.record_stop();
                times.push_back(timer.elapsed_ms() * 1000);
            }
            double med = median(times);
            printf("small_compute,256,256,%d,%.3f,iters=%d\n", TRIALS, med, iters);
        }
    }

    // 4. Back-to-back empty kernel pipeline
    {
        const int TRIALS = 50;
        for (int pipeline : {1, 2, 4, 8, 16}) {
            std::vector<double> times;
            GpuTimer timer;
            for (int t = 0; t < TRIALS; ++t) {
                timer.record_start();
                for (int p = 0; p < pipeline; ++p)
                    empty_kernel<<<1, 64>>>();
                timer.record_stop();
                times.push_back(timer.elapsed_ms() * 1000 / pipeline); // per kernel
            }
            double med = median(times);
            printf("pipeline_empty,1,64,%d,%.3f,depth=%d\n", TRIALS, med, pipeline);
        }
    }

    HIP_CHECK(hipFree(d_out));
    return 0;
}
