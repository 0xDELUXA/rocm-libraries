# Origami Microbenchmark Suite

Microbenchmarks to calibrate the GEMM cycle prediction model parameters.

## Build

```bash
cd shared/origami/microbench
cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++
cmake --build build
```

## Run

```bash
cd build
./bench_dispatch 2>&1 | tee ../results/dispatch.csv
./bench_latency  2>&1 | tee ../results/latency.csv
./bench_mfma     2>&1 | tee ../results/mfma.csv
./bench_lds      2>&1 | tee ../results/lds.csv
./bench_bw       2>&1 | tee ../results/bw.csv
```

## Benchmarks

| Benchmark | Measures | Calibrates |
|-----------|----------|------------|
| bench_bw | Read/write BW vs CU count and vector width | BW scaling curve, VW efficiency |
| bench_mfma | MFMA throughput and latency | Compute model, issue interval |
| bench_lds | LDS bandwidth, bank conflicts, barrier | Per-iteration overhead |
| bench_latency | Load latency at each cache level | Prologue cost |
| bench_dispatch | Kernel launch overhead | Dispatch constant |
