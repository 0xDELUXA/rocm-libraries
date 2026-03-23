cmake_minimum_required(VERSION 3.21)
project(origami_microbench LANGUAGES CXX)

find_package(hip REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# All benchmarks link against HIP
function(add_bench NAME)
    add_executable(${NAME} ${NAME}.cpp)
    target_link_libraries(${NAME} PRIVATE hip::device)
    target_compile_options(${NAME} PRIVATE -Wall -O2)
endfunction()

add_bench(bench_bw)
add_bench(bench_mfma)
add_bench(bench_lds)
add_bench(bench_latency)
add_bench(bench_dispatch)
