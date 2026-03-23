// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "origami/types.hpp"
#include "origami/hardware.hpp"

namespace origami {
namespace cycle_model {

struct model_params_t {
    // BW scaling: BW/BW_peak = A*sqrt(WGs) + B*WGs + C
    double bw_a = 0.0757;
    double bw_b = -0.00131;
    double bw_c = 0.024;
    double bw_clamp_lo = 0.02;
    double bw_clamp_hi = 0.95;

    // Vector width BW efficiency
    double vw_base = 0.65;
    double vw_slope = 0.05;
    double vw_floor = 0.65;
    double vw_ceil = 1.05;

    // L2 reuse
    double l2_efficiency = 0.20;
    double nt_l2_reduction = 1.0;

    // DirectToLds
    double dtl_iter_discount = 0.5;

    // Overheads (cycles)
    double dispatch = 2000;
    double prologue_per_pgr = 220;
    double epilogue_base = 300;
    double per_iter_overhead = 20;
    double streamk_reduce = 2500;

    // Memory clock domain ratio
    double mem_clock_ratio = 0.789;

    // Stream benchmark BW (GB/s) — ground truth from microbenchmark
    double stream_read_gbs = 6476.0;
    double stream_write_gbs = 4759.0;

    // L2 BW (TB/s)
    double l2_bw_tbs = 17.0;
    double l2_capacity_per_partition = 4 * 1024 * 1024;
};

struct kernel_params_t {
    size_t mt_m = 0, mt_n = 0, du = 0;
    size_t mi_m = 16, mi_n = 16;
    size_t miwt_m = 1, miwt_n = 1;
    size_t wave_num = 4;
    int pgr = 2;
    int gsu = 0;
    int sk = 0;
    size_t grvw_a = 1, grvw_b = 1;
    int dtla = 0, dtlb = 0;
    int nta = 0, ntb = 0;
    size_t wg_k = 1; // LocalSplitU
};

inline double bw_scale(double num_wgs, const model_params_t& p) {
    double x = std::max(1.0, std::min(num_wgs, 256.0));
    double f = p.bw_a * std::sqrt(x) + p.bw_b * x + p.bw_c;
    return std::max(p.bw_clamp_lo, std::min(f, p.bw_clamp_hi));
}

inline size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

inline size_t compute_streamk_grid(size_t tiles, size_t k_iters, size_t cu_count) {
    if (tiles >= cu_count) return tiles;
    static const size_t fracs[] = {16, 12, 8, 6, 4, 3, 2, 1};
    for (auto f : fracs) {
        if (tiles * f <= cu_count && k_iters / f >= 8)
            return tiles * f;
    }
    return tiles;
}

struct mfma_info_t {
    double latency;
    size_t hw_mi_k;
    double issue_interval;
};

inline mfma_info_t get_mfma_info(size_t mi_m, size_t mi_n, size_t bpe_bits) {
    if (bpe_bits == 16) {
        if (mi_m == 16 && mi_n == 16) return {36, 32, 14};
        if (mi_m == 32 && mi_n == 32) return {36, 16, 26};
    } else if (bpe_bits == 32) {
        if (mi_m == 16 && mi_n == 16) return {45, 4, 26};
        if (mi_m == 32 && mi_n == 32) return {45, 2, 26};
    }
    return {36, 32, 14};
}

inline double estimate_l2_hit(const problem_t& prob, const kernel_params_t& kp,
                               size_t num_active_cus, size_t bpe,
                               const model_params_t& mp) {
    size_t gm = ceil_div(prob.size.m, kp.mt_m);
    size_t gn = ceil_div(prob.size.n, kp.mt_n);
    if (gm <= 1 && gn <= 1) return 0.0;

    size_t cus_per_part = 256 / 8;
    size_t wgm = std::max((size_t)1, (size_t)std::sqrt((double)cus_per_part));
    size_t eff = std::max((size_t)1, ceil_div(num_active_cus, (size_t)8));
    size_t sn = std::min(wgm, gn);
    size_t sm = std::max((size_t)1, std::min(ceil_div(eff, sn), gm));

    double uA = sm * kp.mt_m * kp.du * bpe;
    double uB = sn * kp.mt_n * kp.du * bpe;
    double tA = uA * sn, tB = uB * sm;
    double tr = std::max(tA + tB, 1.0);
    double h = std::max(0.0, (tr - (uA + uB)) / tr);

    if (uA + uB > mp.l2_capacity_per_partition)
        h *= mp.l2_capacity_per_partition / (uA + uB);
    if (kp.ntb > 0) h *= mp.nt_l2_reduction;
    return std::min(h, 0.95);
}

/**
 * @brief Predict the number of sclk cycles for a GEMM kernel.
 *
 * @param prob GEMM problem description
 * @param hw Hardware characteristics
 * @param kp Kernel parameters
 * @param mp Model tuning parameters
 * @return Predicted cycles at peak SCLK
 */
inline double predict_cycles(const problem_t& prob,
                             const hardware_t& hw,
                             const kernel_params_t& kp,
                             const model_params_t& mp = model_params_t{}) {
    size_t bpe = datatype_to_bits(prob.a_dtype) / 8;
    size_t bpe_d = datatype_to_bits(prob.d_dtype) / 8;
    size_t bpe_bits = bpe * 8;
    double sclk = hw.compute_clock_ghz;

    auto mi = get_mfma_info(kp.mi_m, kp.mi_n, bpe_bits);

    size_t gm = ceil_div(prob.size.m, kp.mt_m);
    size_t gn = ceil_div(prob.size.n, kp.mt_n);
    size_t tiles = gm * gn * prob.batch;
    size_t kpt = ceil_div(prob.size.k, kp.du);

    size_t sf, ac, ts;
    if (tiles >= hw.N_CU) {
        sf = 1; ac = hw.N_CU; ts = ceil_div(tiles, hw.N_CU);
    } else if (kp.sk > 0 && kpt >= 8) {
        size_t sg = compute_streamk_grid(tiles, kpt, hw.N_CU);
        sf = std::max((size_t)1, ceil_div(sg, tiles));
        size_t nw = tiles * sf;
        ac = std::min(nw, hw.N_CU); ts = ceil_div(nw, hw.N_CU);
    } else {
        sf = 1; ac = std::min(tiles, hw.N_CU); ts = ceil_div(tiles, hw.N_CU);
    }

    size_t Kps = ceil_div(prob.size.k, sf);
    size_t ki = ceil_div(Kps, kp.du);

    // Compute
    size_t DUw = kp.wg_k > 1 ? kp.du / kp.wg_k : kp.du;
    size_t mk = ceil_div(DUw, mi.hw_mi_k);
    size_t ns = kp.miwt_m * kp.miwt_n;
    size_t mpw = ns * mk;

    double Lc;
    double rg = ns * mi.issue_interval;
    if (rg >= mi.latency) {
        Lc = mpw * mi.issue_interval;
    } else {
        double st = mi.latency - rg;
        Lc = mk * (ns * mi.issue_interval) + std::max((size_t)0, mk - 1) * st;
    }

    // Memory
    double bpi = (kp.mt_m * kp.du + kp.mt_n * kp.du) * bpe;
    double stream_bpc = mp.stream_read_gbs * 1e9 / (sclk * 1e9);
    double l2_bpc = mp.l2_bw_tbs * 1e12 / (sclk * 1e9);
    double sc = bw_scale(ac, mp);
    double base_bw = stream_bpc * sc / ac;

    double avg_grvw = (kp.grvw_a + kp.grvw_b) / 2.0;
    double vw = std::max(mp.vw_floor, std::min(mp.vw_ceil, mp.vw_base + mp.vw_slope * avg_grvw));
    double l2h = estimate_l2_hit(prob, kp, ac, bpe, mp);
    double l2b = l2h * std::max(0.0, l2_bpc / ac - base_bw) * mp.l2_efficiency;
    double eff_bw = base_bw * vw + l2b;
    double Lm = eff_bw > 0 ? bpi / eff_bw : 0;

    // Iteration
    double Li = kp.pgr >= 1 ? std::max(Lc, Lm) : Lc + Lm;
    bool dtl = kp.dtla || kp.dtlb;
    double pio = mp.per_iter_overhead * (dtl ? mp.dtl_iter_discount : 1.0);
    double Lb = ki * (Li + pio);

    // Prologue / Epilogue
    double Lp = mp.prologue_per_pgr * kp.pgr;
    size_t sa = std::min(tiles, ac);
    double swb = mp.stream_write_gbs * 1e9 / (sclk * 1e9) * bw_scale(sa, mp) / sa;
    double Ls = swb > 0 ? kp.mt_m * kp.mt_n * bpe_d / swb : 0;
    double kr = Kps % kp.du;
    double Lt = (kr > 0 && kp.du > 1 && ki > 0) ? Li * (kr / (double)kp.du) * 1.5 : 0;
    double Le = Ls + Lt + mp.epilogue_base;
    double Llsu = kp.wg_k > 1 ? kp.mt_m * kp.mt_n * 4.0 / 256 + 200 * kp.wg_k : 0;

    double Ltile = Lp + Lb + Le + Llsu;
    double total = Ltile * ts + mp.dispatch;

    if (sf > 1) {
        double rb = kp.mt_m * kp.mt_n * 4.0 * sf;
        total += rb / 256.0 * ceil_div(tiles, ac) + mp.streamk_reduce;
    }

    return total;
}

} // namespace cycle_model
} // namespace origami
