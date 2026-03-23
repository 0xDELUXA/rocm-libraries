// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>

#include "origami/cycle_model.hpp"
#include "origami/types.hpp"
#include "origami/hardware.hpp"

using namespace origami;
using namespace origami::cycle_model;
using Catch::Approx;

static kernel_params_t make_kp(size_t mt_m, size_t mt_n, size_t du,
                               size_t mi_m, size_t mi_n,
                               size_t miwt_m, size_t miwt_n,
                               size_t grvw_a, size_t grvw_b,
                               int pgr = 2, int sk = 3,
                               size_t wg_k = 1,
                               int dtla = 0, int dtlb = 0) {
    kernel_params_t kp;
    kp.mt_m = mt_m; kp.mt_n = mt_n; kp.du = du;
    kp.mi_m = mi_m; kp.mi_n = mi_n;
    kp.miwt_m = miwt_m; kp.miwt_n = miwt_n;
    kp.grvw_a = grvw_a; kp.grvw_b = grvw_b;
    kp.pgr = pgr; kp.sk = sk; kp.wg_k = wg_k;
    kp.dtla = dtla; kp.dtlb = dtlb;
    kp.wave_num = std::max((size_t)1, (mt_m / (mi_m * miwt_m)) *
                                       (mt_n / (mi_n * miwt_n)) * wg_k);
    return kp;
}

static problem_t make_problem(size_t M, size_t N, size_t K,
                               data_type_t dtype = data_type_t::BFloat16,
                               transpose_t tA = transpose_t::N,
                               transpose_t tB = transpose_t::T) {
    problem_t p;
    p.size = {M, N, K}; p.batch = 1;
    p.a_dtype = p.b_dtype = p.c_dtype = p.d_dtype = dtype;
    p.mi_dtype = dtype;
    p.a_transpose = tA; p.b_transpose = tB;
    return p;
}

TEST_CASE("Cycle model: predict_cycles returns positive values", "[cycle_model]") {
    auto hw = hardware_t::get_hardware_for_arch(
        hardware_t::architecture_t::gfx950, 256, 163840, 4096, 2200000);
    model_params_t mp;
    auto prob = make_problem(1024, 1024, 1024);
    auto kp = make_kp(128, 128, 64, 16, 16, 4, 4, 8, 8);

    double cycles = predict_cycles(prob, hw, kp, mp);
    REQUIRE(cycles > 0);
}

TEST_CASE("Cycle model: larger K gives more cycles", "[cycle_model]") {
    auto hw = hardware_t::get_hardware_for_arch(
        hardware_t::architecture_t::gfx950, 256, 163840, 4096, 2200000);
    model_params_t mp;
    auto kp = make_kp(128, 128, 64, 16, 16, 4, 4, 8, 8);

    auto p1 = make_problem(1024, 1024, 512);
    auto p2 = make_problem(1024, 1024, 2048);

    double c1 = predict_cycles(p1, hw, kp, mp);
    double c2 = predict_cycles(p2, hw, kp, mp);
    REQUIRE(c2 > c1);
}

TEST_CASE("Cycle model: larger tile gives different prediction", "[cycle_model]") {
    auto hw = hardware_t::get_hardware_for_arch(
        hardware_t::architecture_t::gfx950, 256, 163840, 4096, 2200000);
    model_params_t mp;
    auto prob = make_problem(2048, 2048, 1024);

    auto kp1 = make_kp(64, 64, 64, 16, 16, 2, 2, 4, 4);
    auto kp2 = make_kp(256, 256, 64, 16, 16, 8, 8, 8, 8);

    double c1 = predict_cycles(prob, hw, kp1, mp);
    double c2 = predict_cycles(prob, hw, kp2, mp);
    REQUIRE(c1 != c2);
}

TEST_CASE("Cycle model: compute-bound vs memory-bound distinction", "[cycle_model]") {
    auto hw = hardware_t::get_hardware_for_arch(
        hardware_t::architecture_t::gfx950, 256, 163840, 4096, 2200000);
    model_params_t mp;

    // Large square = compute-bound
    auto p_compute = make_problem(4096, 4096, 4096);
    auto kp_large = make_kp(256, 256, 32, 16, 16, 8, 8, 8, 8);
    double c_compute = predict_cycles(p_compute, hw, kp_large, mp);

    // Skinny = memory-bound
    auto p_memory = make_problem(32, 32768, 256);
    auto kp_small = make_kp(32, 256, 64, 16, 16, 1, 8, 4, 4);
    double c_memory = predict_cycles(p_memory, hw, kp_small, mp);

    // Both should be positive
    REQUIRE(c_compute > 0);
    REQUIRE(c_memory > 0);
}

TEST_CASE("Cycle model: BF16 vs FP32", "[cycle_model]") {
    auto hw = hardware_t::get_hardware_for_arch(
        hardware_t::architecture_t::gfx950, 256, 163840, 4096, 2200000);
    model_params_t mp;

    auto kp = make_kp(128, 128, 64, 16, 16, 4, 4, 4, 4);

    auto p_bf16 = make_problem(2048, 2048, 2048, data_type_t::BFloat16);
    auto p_f32 = make_problem(2048, 2048, 2048, data_type_t::Float);

    double c_bf16 = predict_cycles(p_bf16, hw, kp, mp);
    double c_f32 = predict_cycles(p_f32, hw, kp, mp);

    // FP32 should be slower (more bytes, slower MFMA)
    REQUIRE(c_f32 > c_bf16);
}

TEST_CASE("Cycle model: StreamK splits K for small tile count", "[cycle_model]") {
    auto hw = hardware_t::get_hardware_for_arch(
        hardware_t::architecture_t::gfx950, 256, 163840, 4096, 2200000);
    model_params_t mp;

    auto prob = make_problem(64, 64, 16384);

    auto kp_sk = make_kp(64, 64, 128, 16, 16, 2, 2, 8, 8, 2, 3);
    auto kp_dp = make_kp(64, 64, 128, 16, 16, 2, 2, 8, 8, 2, 0);

    double c_sk = predict_cycles(prob, hw, kp_sk, mp);
    double c_dp = predict_cycles(prob, hw, kp_dp, mp);

    // StreamK should be faster (more CUs utilized)
    REQUIRE(c_sk < c_dp);
}

TEST_CASE("Cycle model: predicted us is reasonable for known case", "[cycle_model]") {
    auto hw = hardware_t::get_hardware_for_arch(
        hardware_t::architecture_t::gfx950, 256, 163840, 4096, 2200000);
    model_params_t mp;

    // Large GEMM with many tiles: 2048x2048x1024 BF16, MT128x128x64
    // Measured: ~12 us. With 256 tiles at 256 CUs, well-predicted regime.
    auto prob = make_problem(2048, 2048, 1024);
    auto kp = make_kp(128, 128, 64, 16, 16, 4, 4, 8, 8);

    double pred = predict_cycles(prob, hw, kp, mp);
    double pred_us = pred / (hw.compute_clock_ghz * 1000);

    REQUIRE(pred_us > 1.0);
    REQUIRE(pred_us < 100.0);

    // Verify ranking: larger K should predict higher latency
    auto prob2 = make_problem(2048, 2048, 4096);
    double pred2_us = predict_cycles(prob2, hw, kp, mp) / (hw.compute_clock_ghz * 1000);
    REQUIRE(pred2_us > pred_us);
}
