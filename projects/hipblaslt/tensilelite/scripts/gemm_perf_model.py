#!/usr/bin/env python3
"""
GEMM Kernel Performance Model for TensileLite

Analytical model that estimates cycle counts and instruction counts for
TensileLite GEMM kernels based on kernel parameters, GEMM shape, data type,
and layout.

This model is derived from analysis of the TensileLite codebase:
  - cycle.cpp: instruction-level cycle simulation
  - formocast_simulator: cache hierarchy modeling
  - KernelWriterAssembly.py: assembly code generation
  - SIA.py: scheduling iteration algorithm
  - ValidParameters.py: parameter definitions
  - mfma.hpp / mem.hpp: instruction latencies
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DataType(Enum):
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    FP64 = "fp64"
    INT8 = "int8"
    FP8 = "fp8"
    BF8 = "bf8"
    XF32 = "xf32"


class Layout(Enum):
    TN = "TN"  # A transposed, B non-transposed (most common for GEMM)
    NN = "NN"
    NT = "NT"
    TT = "TT"


DTYPE_BPE = {
    DataType.FP16: 2,
    DataType.BF16: 2,
    DataType.FP32: 4,
    DataType.FP64: 8,
    DataType.INT8: 1,
    DataType.FP8: 1,
    DataType.BF8: 1,
    DataType.XF32: 4,
}

# MFMA issue latency divisor: how many quad-cycles per MFMA
# For GFX9 (gfx940/942/950) with MatrixInstB == 1
DTYPE_MI_DIVISOR = {
    DataType.FP16: 4,
    DataType.BF16: 4,
    DataType.INT8: 4,
    DataType.FP8: 2,  # gfx950: F8 takes 2x cycles
    DataType.BF8: 2,
    DataType.FP32: 2,
    DataType.FP64: 2,
    DataType.XF32: 4,  # XF32 uses sparse-like divisor
}


@dataclass
class GEMMShape:
    M: int
    N: int
    K: int
    batch: int = 1


@dataclass
class KernelParams:
    # Matrix Instruction [M, N, K, B]
    MatrixInstM: int = 16
    MatrixInstN: int = 16
    MatrixInstK: int = 16
    MatrixInstB: int = 1

    # Wave tile (MFMAs per wave per direction)
    MIWaveTile0: int = 4
    MIWaveTile1: int = 4

    # Wave group (waves per workgroup per direction)
    MIWaveGroup0: int = 2
    MIWaveGroup1: int = 2

    # Derived tile sizes (will be computed if not set)
    MacroTile0: Optional[int] = None
    MacroTile1: Optional[int] = None

    # Unroll parameters
    DepthU: int = 32
    InnerUnroll: int = 1

    # Memory parameters
    GlobalReadVectorWidthA: int = 4
    GlobalReadVectorWidthB: int = 4
    LocalReadVectorWidth: int = 4
    VectorWidthA: int = 4
    VectorWidthB: int = 4
    StoreVectorWidth: int = 4

    # Scheduling
    PrefetchGlobalRead: int = 2
    PrefetchLocalRead: int = 1
    ScheduleIterAlg: int = 3
    GlobalReadPerMfma: float = 1.0
    LocalWritePerMfma: float = -1.0

    # Split parameters
    GlobalSplitU: int = 1
    LocalSplitU: int = 1

    # Direct load/store flags
    DirectToVgprA: bool = False
    DirectToVgprB: bool = False
    DirectToLds: int = 0

    # Layout
    NumLoadsCoalescedA: int = 1
    NumLoadsCoalescedB: int = 1

    # LDS
    LdsPadA: int = 0
    LdsPadB: int = 0

    # Other
    WavefrontSize: int = 64
    StoreRemapVectorWidth: int = 0
    WorkGroupMapping: int = 8
    MaxOccupancy: int = 40

    def __post_init__(self):
        MatrixInstBM = max(1, self.MatrixInstB)
        MatrixInstBN = 1 if self.MatrixInstB > 1 else 1
        if self.MacroTile0 is None:
            self.MacroTile0 = (self.MatrixInstM * MatrixInstBM
                               * self.MIWaveGroup0 * self.MIWaveTile0)
        if self.MacroTile1 is None:
            self.MacroTile1 = (self.MatrixInstN * MatrixInstBN
                               * self.MIWaveGroup1 * self.MIWaveTile1)

    @property
    def num_threads(self) -> int:
        return (self.MIWaveGroup0 * self.MIWaveGroup1
                * self.LocalSplitU * self.WavefrontSize)

    @property
    def num_waves(self) -> int:
        return self.MIWaveGroup0 * self.MIWaveGroup1 * self.LocalSplitU


@dataclass
class Architecture:
    name: str = "gfx942"
    num_cus: int = 304
    num_xcds: int = 8
    wavefront_size: int = 64
    l1_cache_kb: float = 32.0
    l2_cache_mb: float = 256.0
    l3_cache_mb: float = 256.0
    hbm_bandwidth_tbs: float = 5.3
    math_freq_ghz: float = 1.5
    mem_freq_ghz: float = 1.6
    l1_bus_width_bytes: int = 64
    l2_bus_width_bytes: int = 64
    is_gfx9: bool = True


GFX942 = Architecture(
    name="gfx942", num_cus=304, num_xcds=8,
    hbm_bandwidth_tbs=5.3, math_freq_ghz=1.5
)
GFX950 = Architecture(
    name="gfx950", num_cus=304, num_xcds=8,
    hbm_bandwidth_tbs=6.0, math_freq_ghz=1.5
)


@dataclass
class InstructionCounts:
    """Instruction counts per full unroll loop iteration."""
    mfma_per_iter: int = 0
    mfma_total: int = 0
    global_reads_A: int = 0
    global_reads_B: int = 0
    global_reads_total: int = 0
    local_writes_A: int = 0
    local_writes_B: int = 0
    local_writes_total: int = 0
    local_reads_per_iter_A: int = 0
    local_reads_per_iter_B: int = 0
    local_reads_total: int = 0
    loop_iters: int = 0
    barriers: int = 1
    waitcnts: int = 2


@dataclass
class CycleEstimate:
    """Cycle count estimates for different kernel phases."""
    mfma_issue_latency: int = 0
    compute_cycles_per_unroll: int = 0
    compute_cycles_shader: int = 0
    memory_cycles_per_unroll: int = 0
    inner_loop_cycles: int = 0
    loop_count: int = 0
    total_loop_cycles: int = 0
    prefetch_cycles: int = 0
    tail_cycles: int = 0
    store_cycles: int = 0
    gsu_overhead_cycles: int = 0
    lsu_overhead_cycles: int = 0
    total_cycles: int = 0
    estimated_us: float = 0.0


@dataclass
class PerformanceReport:
    shape: GEMMShape
    params: KernelParams
    dtype: DataType
    layout: Layout
    arch: Architecture
    instructions: InstructionCounts
    cycles: CycleEstimate
    flops: float = 0.0
    estimated_tflops: float = 0.0


def compute_instruction_counts(
    shape: GEMMShape,
    params: KernelParams,
    dtype: DataType,
) -> InstructionCounts:
    """Compute instruction counts from kernel parameters."""
    counts = InstructionCounts()
    bpe = DTYPE_BPE[dtype]

    counts.mfma_per_iter = params.MIWaveTile0 * params.MIWaveTile1 * params.InnerUnroll
    if dtype == DataType.XF32:
        counts.mfma_per_iter *= 3

    counts.loop_iters = params.DepthU // params.MatrixInstK
    counts.mfma_total = counts.mfma_per_iter * counts.loop_iters

    num_threads = params.num_threads

    if num_threads > 0 and params.GlobalReadVectorWidthA > 0:
        counts.global_reads_A = max(1, (params.MacroTile0 * params.DepthU)
                                    // (params.GlobalReadVectorWidthA * num_threads))
    if num_threads > 0 and params.GlobalReadVectorWidthB > 0:
        counts.global_reads_B = max(1, (params.MacroTile1 * params.DepthU)
                                    // (params.GlobalReadVectorWidthB * num_threads))
    counts.global_reads_total = counts.global_reads_A + counts.global_reads_B

    if params.DirectToLds in (1, 2):
        counts.local_writes_A = 0
    else:
        counts.local_writes_A = counts.global_reads_A

    if params.DirectToLds in (1, 3):
        counts.local_writes_B = 0
    else:
        counts.local_writes_B = counts.global_reads_B

    counts.local_writes_total = counts.local_writes_A + counts.local_writes_B

    mi_input_per_thread = params.MatrixInstK
    local_read_block_width_bytes = params.LocalReadVectorWidth * bpe

    if local_read_block_width_bytes >= 4:
        reads_per_unroll_A = max(1, math.ceil(
            bpe * mi_input_per_thread / local_read_block_width_bytes))
    else:
        reads_per_unroll_A = mi_input_per_thread

    if local_read_block_width_bytes >= 4:
        reads_per_unroll_B = max(1, math.ceil(
            bpe * mi_input_per_thread / local_read_block_width_bytes))
    else:
        reads_per_unroll_B = mi_input_per_thread

    if not params.DirectToVgprA:
        counts.local_reads_per_iter_A = (
            params.InnerUnroll * params.MIWaveTile0 * reads_per_unroll_A)
    if not params.DirectToVgprB:
        counts.local_reads_per_iter_B = (
            params.InnerUnroll * params.MIWaveTile1 * reads_per_unroll_B)

    counts.local_reads_total = (
        (counts.local_reads_per_iter_A + counts.local_reads_per_iter_B)
        * counts.loop_iters)

    return counts


def compute_mfma_issue_latency(params: KernelParams, dtype: DataType) -> int:
    """Compute MFMA issue latency in quad-cycles."""
    divisor = DTYPE_MI_DIVISOR.get(dtype, 2)
    return params.MatrixInstM // divisor


def estimate_cycles(
    shape: GEMMShape,
    params: KernelParams,
    dtype: DataType,
    arch: Architecture,
    instructions: InstructionCounts,
) -> CycleEstimate:
    """Estimate cycle counts for the kernel."""
    est = CycleEstimate()
    bpe = DTYPE_BPE[dtype]

    est.mfma_issue_latency = compute_mfma_issue_latency(params, dtype)

    # Compute cycles: back-to-back MFMA chain with scheduling gaps
    mfma_cycles = instructions.mfma_per_iter * est.mfma_issue_latency

    # Local read issue cycles (interleaved with MFMA)
    lr_per_iter = (instructions.local_reads_per_iter_A
                   + instructions.local_reads_per_iter_B)
    lr_bytes = params.LocalReadVectorWidth * bpe
    if lr_bytes >= 16:
        lr_issue = 2
    else:
        lr_issue = 1
    lr_cycles = lr_per_iter * lr_issue

    # In well-scheduled kernels (SIA=3), local reads are hidden behind MFMA
    compute_per_iter = max(mfma_cycles, mfma_cycles + max(0, lr_cycles - mfma_cycles))

    # Local write issue cycles
    grvw_bytes_a = params.GlobalReadVectorWidthA * bpe
    grvw_bytes_b = params.GlobalReadVectorWidthB * bpe
    if grvw_bytes_a >= 16:
        lw_issue_a = 5
    elif grvw_bytes_a >= 8:
        lw_issue_a = 3
    else:
        lw_issue_a = 2
    if grvw_bytes_b >= 16:
        lw_issue_b = 5
    elif grvw_bytes_b >= 8:
        lw_issue_b = 3
    else:
        lw_issue_b = 2

    lw_cycles = (instructions.local_writes_A * lw_issue_a
                 + instructions.local_writes_B * lw_issue_b)

    # Global read issue cycles (1 quad-cycle each)
    gr_cycles = instructions.global_reads_total

    # Barrier + waitcnt overhead
    sync_cycles = 2 + 1  # s_barrier + s_waitcnt
    jump_cycles = 6  # branch at loop end

    # Total per unroll iteration (in quad-cycles)
    # In a well-pipelined kernel, GR and LW are scheduled during MFMA gaps
    non_mfma = lw_cycles + gr_cycles + sync_cycles + jump_cycles
    est.compute_cycles_per_unroll = compute_per_iter * instructions.loop_iters

    # The actual cycle count depends on how well instructions overlap
    # Best case: max(compute, memory_instructions)
    # Typical case: compute + partial overlap overhead
    overlap_efficiency = 0.85 if params.ScheduleIterAlg == 3 else 0.7
    total_quad = (est.compute_cycles_per_unroll
                  + int(non_mfma * (1 - overlap_efficiency)))

    if arch.is_gfx9:
        est.compute_cycles_shader = total_quad * 4
    else:
        est.compute_cycles_shader = total_quad

    # Memory bandwidth estimate
    bytes_A = params.MacroTile0 * params.DepthU * bpe
    bytes_B = params.MacroTile1 * params.DepthU * bpe
    total_bytes = bytes_A + bytes_B
    # Rough bandwidth: L1 bus width * frequency
    bandwidth_per_cu_per_cycle = arch.l1_bus_width_bytes
    est.memory_cycles_per_unroll = int(total_bytes / bandwidth_per_cu_per_cycle)
    if arch.is_gfx9:
        est.memory_cycles_per_unroll *= 4

    est.inner_loop_cycles = max(est.compute_cycles_shader,
                                est.memory_cycles_per_unroll)

    # Total loops
    k_eff = math.ceil(shape.K / max(1, params.GlobalSplitU))
    est.loop_count = k_eff // params.DepthU
    k_tail = k_eff % params.DepthU

    # Main loop
    if params.PrefetchGlobalRead > 0 and est.loop_count > 1:
        est.total_loop_cycles = (est.inner_loop_cycles * (est.loop_count - 1)
                                 + est.compute_cycles_shader)
    else:
        est.total_loop_cycles = est.inner_loop_cycles * est.loop_count

    # Prefetch cost (loading first tile before loop starts)
    est.prefetch_cycles = gr_cycles * 4 + 1024 if params.PrefetchGlobalRead > 0 else 0

    # Tail loop (handles K_tail remainder)
    if k_tail > 0:
        tail_fraction = k_tail / params.DepthU
        est.tail_cycles = int(est.inner_loop_cycles * tail_fraction)
    else:
        est.tail_cycles = 0

    # Store cycles (write results back to global memory)
    store_elements = params.MacroTile0 * params.MacroTile1
    bpe_out = bpe  # output type typically matches compute type
    store_bytes = store_elements * bpe_out
    est.store_cycles = int(store_bytes / arch.l1_bus_width_bytes)
    if arch.is_gfx9:
        est.store_cycles *= 4

    # GSU overhead
    if params.GlobalSplitU > 1:
        gsu_store_elements = params.MacroTile0 * params.MacroTile1
        est.gsu_overhead_cycles = int(gsu_store_elements * bpe_out
                                      / arch.l1_bus_width_bytes * 4)

    # LSU overhead
    if params.LocalSplitU > 1:
        lsu_elements = params.MacroTile0 * params.MacroTile1
        est.lsu_overhead_cycles = int(lsu_elements * bpe_out * 3
                                      / arch.l1_bus_width_bytes * 4)

    # Total cycles for one workgroup
    est.total_cycles = (est.prefetch_cycles + est.total_loop_cycles
                        + est.tail_cycles + est.store_cycles
                        + est.gsu_overhead_cycles + est.lsu_overhead_cycles)

    # Estimate time in microseconds
    m_tiles = math.ceil(shape.M / params.MacroTile0)
    n_tiles = math.ceil(shape.N / params.MacroTile1)
    total_wgs = m_tiles * n_tiles * shape.batch * max(1, params.GlobalSplitU)
    waves_per_cu = math.ceil(total_wgs / arch.num_cus)

    total_all_cycles = est.total_cycles * waves_per_cu
    freq_hz = arch.math_freq_ghz * 1e9
    est.estimated_us = total_all_cycles / freq_hz * 1e6

    return est


def analyze_gemm(
    shape: GEMMShape,
    params: KernelParams,
    dtype: DataType,
    layout: Layout = Layout.TN,
    arch: Architecture = GFX942,
) -> PerformanceReport:
    """Full performance analysis for a GEMM kernel configuration."""
    instructions = compute_instruction_counts(shape, params, dtype)
    cycles = estimate_cycles(shape, params, dtype, arch, instructions)

    flops = 2.0 * shape.M * shape.N * shape.K * shape.batch
    if cycles.estimated_us > 0:
        tflops = flops / (cycles.estimated_us * 1e-6) / 1e12
    else:
        tflops = 0.0

    return PerformanceReport(
        shape=shape, params=params, dtype=dtype, layout=layout,
        arch=arch, instructions=instructions, cycles=cycles,
        flops=flops, estimated_tflops=tflops,
    )


def print_report(report: PerformanceReport):
    """Print a formatted performance report."""
    s = report.shape
    p = report.params
    i = report.instructions
    c = report.cycles

    print("=" * 80)
    print("GEMM Kernel Performance Model Report")
    print("=" * 80)

    print(f"\n--- Problem ---")
    print(f"  Shape:    M={s.M}, N={s.N}, K={s.K}, Batch={s.batch}")
    print(f"  DataType: {report.dtype.value}")
    print(f"  Layout:   {report.layout.value}")
    print(f"  FLOPs:    {report.flops:.2e}")

    print(f"\n--- Kernel Parameters ---")
    print(f"  MacroTile:          {p.MacroTile0} x {p.MacroTile1}")
    print(f"  MatrixInstruction:  [{p.MatrixInstM}, {p.MatrixInstN}, {p.MatrixInstK}, {p.MatrixInstB}]")
    print(f"  MIWaveTile:         [{p.MIWaveTile0}, {p.MIWaveTile1}]")
    print(f"  MIWaveGroup:        [{p.MIWaveGroup0}, {p.MIWaveGroup1}]")
    print(f"  DepthU:             {p.DepthU}")
    print(f"  NumThreads:         {p.num_threads}")
    print(f"  NumWaves:           {p.num_waves}")
    print(f"  PrefetchGlobalRead: {p.PrefetchGlobalRead}")
    print(f"  PrefetchLocalRead:  {p.PrefetchLocalRead}")
    print(f"  ScheduleIterAlg:    {p.ScheduleIterAlg}")
    print(f"  GlobalSplitU:       {p.GlobalSplitU}")
    print(f"  DirectToVgprA/B:    {p.DirectToVgprA}/{p.DirectToVgprB}")
    print(f"  DirectToLds:        {p.DirectToLds}")
    print(f"  GRVW A/B:           {p.GlobalReadVectorWidthA}/{p.GlobalReadVectorWidthB}")

    print(f"\n--- Instruction Counts (per full unroll) ---")
    print(f"  Loop iterations:         {i.loop_iters}")
    print(f"  MFMA per sub-iter:       {i.mfma_per_iter}")
    print(f"  MFMA total:              {i.mfma_total}")
    print(f"  Global reads A:          {i.global_reads_A}")
    print(f"  Global reads B:          {i.global_reads_B}")
    print(f"  Global reads total:      {i.global_reads_total}")
    print(f"  Local writes A:          {i.local_writes_A}")
    print(f"  Local writes B:          {i.local_writes_B}")
    print(f"  Local writes total:      {i.local_writes_total}")
    print(f"  Local reads/iter A:      {i.local_reads_per_iter_A}")
    print(f"  Local reads/iter B:      {i.local_reads_per_iter_B}")
    print(f"  Local reads total:       {i.local_reads_total}")

    print(f"\n--- Cycle Estimates ---")
    print(f"  MFMA issue latency:      {c.mfma_issue_latency} quad-cycles")
    print(f"  Compute/unroll (shader): {c.compute_cycles_shader} cycles")
    print(f"  Memory/unroll (shader):  {c.memory_cycles_per_unroll} cycles")
    print(f"  Inner loop cycles:       {c.inner_loop_cycles} cycles")
    print(f"  Loop count (K-dim):      {c.loop_count}")
    print(f"  Total loop cycles:       {c.total_loop_cycles} cycles")
    print(f"  Prefetch cycles:         {c.prefetch_cycles} cycles")
    print(f"  Tail cycles:             {c.tail_cycles} cycles")
    print(f"  Store cycles:            {c.store_cycles} cycles")
    print(f"  GSU overhead:            {c.gsu_overhead_cycles} cycles")
    print(f"  LSU overhead:            {c.lsu_overhead_cycles} cycles")
    print(f"  Total WG cycles:         {c.total_cycles} cycles")

    print(f"\n--- Performance Estimate ---")
    print(f"  Architecture:       {report.arch.name}")
    print(f"  Estimated time:     {c.estimated_us:.2f} us")
    print(f"  Estimated TFLOPS:   {report.estimated_tflops:.2f}")

    m_tiles = math.ceil(s.M / p.MacroTile0)
    n_tiles = math.ceil(s.N / p.MacroTile1)
    total_wgs = m_tiles * n_tiles * s.batch * max(1, p.GlobalSplitU)
    print(f"  Tile grid:          {m_tiles} x {n_tiles}")
    print(f"  Total workgroups:   {total_wgs}")
    cu_util = min(1.0, total_wgs / report.arch.num_cus)
    print(f"  CU utilization:     {cu_util*100:.1f}%")
    print("=" * 80)


def compare_configs(
    shape: GEMMShape,
    configs: list[tuple[str, KernelParams, DataType]],
    arch: Architecture = GFX942,
    layout: Layout = Layout.TN,
):
    """Compare multiple kernel configurations for the same problem."""
    print(f"\n{'='*100}")
    print(f"Comparing {len(configs)} configurations for "
          f"M={shape.M}, N={shape.N}, K={shape.K}")
    print(f"{'='*100}")

    header = (f"{'Config':<30} {'MT0xMT1':>10} {'DU':>5} {'MFMAs':>6} "
              f"{'GReads':>7} {'LReads':>7} {'LWrites':>8} "
              f"{'Comp_cy':>10} {'Mem_cy':>10} {'Total_us':>10} {'TFLOPS':>8}")
    print(header)
    print("-" * 100)

    reports = []
    for name, params, dtype in configs:
        report = analyze_gemm(shape, params, dtype, layout, arch)
        reports.append((name, report))

        inst = report.instructions
        cyc = report.cycles
        print(f"{name:<30} "
              f"{params.MacroTile0}x{params.MacroTile1:>4} "
              f"{params.DepthU:>5} "
              f"{inst.mfma_total:>6} "
              f"{inst.global_reads_total:>7} "
              f"{inst.local_reads_total:>7} "
              f"{inst.local_writes_total:>8} "
              f"{cyc.compute_cycles_shader:>10} "
              f"{cyc.memory_cycles_per_unroll:>10} "
              f"{cyc.estimated_us:>10.2f} "
              f"{report.estimated_tflops:>8.2f}")

    print("-" * 100)
    best = min(reports, key=lambda x: x[1].cycles.estimated_us)
    print(f"Best: {best[0]} ({best[1].cycles.estimated_us:.2f} us, "
          f"{best[1].estimated_tflops:.2f} TFLOPS)")


# ──────────────────────────────────────────────────────────────────
# Example usage and demonstration
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Example 1: Large square GEMM (M=4096, N=4096, K=4096) with FP16")
    print("=" * 80)

    shape = GEMMShape(M=4096, N=4096, K=4096)
    params = KernelParams(
        MatrixInstM=16, MatrixInstN=16, MatrixInstK=16, MatrixInstB=1,
        MIWaveTile0=4, MIWaveTile1=4,
        MIWaveGroup0=2, MIWaveGroup1=2,
        DepthU=64,
        GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=8,
        LocalReadVectorWidth=8,
        PrefetchGlobalRead=2, ScheduleIterAlg=3,
    )
    report = analyze_gemm(shape, params, DataType.FP16, arch=GFX942)
    print_report(report)

    print("\n" + "=" * 80)
    print("Example 2: Comparing tile sizes for M=2048, N=2048, K=8192 with BF16")
    print("=" * 80)

    shape2 = GEMMShape(M=2048, N=2048, K=8192)
    configs = [
        ("MT128x128_DU32", KernelParams(
            MIWaveTile0=2, MIWaveTile1=2, MIWaveGroup0=2, MIWaveGroup1=2,
            DepthU=32, GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=8,
        ), DataType.BF16),
        ("MT256x128_DU32", KernelParams(
            MIWaveTile0=4, MIWaveTile1=2, MIWaveGroup0=2, MIWaveGroup1=2,
            DepthU=32, GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=8,
        ), DataType.BF16),
        ("MT256x256_DU32", KernelParams(
            MIWaveTile0=4, MIWaveTile1=4, MIWaveGroup0=2, MIWaveGroup1=2,
            DepthU=32, GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=8,
        ), DataType.BF16),
        ("MT256x256_DU64", KernelParams(
            MIWaveTile0=4, MIWaveTile1=4, MIWaveGroup0=2, MIWaveGroup1=2,
            DepthU=64, GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=8,
        ), DataType.BF16),
        ("MT128x128_DU128", KernelParams(
            MIWaveTile0=2, MIWaveTile1=2, MIWaveGroup0=2, MIWaveGroup1=2,
            DepthU=128, GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=8,
        ), DataType.BF16),
    ]
    compare_configs(shape2, configs, arch=GFX942)

    print("\n" + "=" * 80)
    print("Example 3: Data type comparison for M=4096, N=4096, K=4096")
    print("=" * 80)

    shape3 = GEMMShape(M=4096, N=4096, K=4096)
    dtype_configs = [
        ("FP16_MT256x256_DU64", KernelParams(
            MIWaveTile0=4, MIWaveTile1=4, MIWaveGroup0=2, MIWaveGroup1=2,
            DepthU=64, GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=8,
        ), DataType.FP16),
        ("BF16_MT256x256_DU64", KernelParams(
            MIWaveTile0=4, MIWaveTile1=4, MIWaveGroup0=2, MIWaveGroup1=2,
            DepthU=64, GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=8,
        ), DataType.BF16),
        ("FP32_MT128x128_DU32", KernelParams(
            MatrixInstK=1,
            MIWaveTile0=2, MIWaveTile1=2, MIWaveGroup0=2, MIWaveGroup1=2,
            DepthU=32, GlobalReadVectorWidthA=4, GlobalReadVectorWidthB=4,
            LocalReadVectorWidth=4,
        ), DataType.FP32),
        ("INT8_MT256x256_DU128", KernelParams(
            MIWaveTile0=4, MIWaveTile1=4, MIWaveGroup0=2, MIWaveGroup1=2,
            DepthU=128, GlobalReadVectorWidthA=16, GlobalReadVectorWidthB=16,
            LocalReadVectorWidth=16,
        ), DataType.INT8),
        ("FP8_MT256x256_DU128", KernelParams(
            MatrixInstK=32,
            MIWaveTile0=4, MIWaveTile1=4, MIWaveGroup0=2, MIWaveGroup1=2,
            DepthU=128, GlobalReadVectorWidthA=16, GlobalReadVectorWidthB=16,
            LocalReadVectorWidth=16,
        ), DataType.FP8),
    ]
    compare_configs(shape3, dtype_configs, arch=GFX942)

    print("\n" + "=" * 80)
    print("Example 4: Effect of GlobalSplitU on small-K problem")
    print("=" * 80)

    shape4 = GEMMShape(M=8192, N=8192, K=256)
    gsu_configs = [
        ("GSU1", KernelParams(
            MIWaveTile0=4, MIWaveTile1=4, MIWaveGroup0=2, MIWaveGroup1=2,
            DepthU=64, GlobalSplitU=1,
            GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=8,
        ), DataType.FP16),
        ("GSU2", KernelParams(
            MIWaveTile0=4, MIWaveTile1=4, MIWaveGroup0=2, MIWaveGroup1=2,
            DepthU=64, GlobalSplitU=2,
            GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=8,
        ), DataType.FP16),
        ("GSU4", KernelParams(
            MIWaveTile0=4, MIWaveTile1=4, MIWaveGroup0=2, MIWaveGroup1=2,
            DepthU=64, GlobalSplitU=4,
            GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=8,
        ), DataType.FP16),
    ]
    compare_configs(shape4, gsu_configs, arch=GFX942)

    print("\n" + "=" * 80)
    print("Example 5: Skinny matrix (M=16384, N=64, K=4096) - typical LLM decode")
    print("=" * 80)

    shape5 = GEMMShape(M=16384, N=64, K=4096)
    skinny_configs = [
        ("MT256x64_DU32", KernelParams(
            MIWaveTile0=4, MIWaveTile1=1, MIWaveGroup0=2, MIWaveGroup1=2,
            DepthU=32, GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=8,
        ), DataType.FP16),
        ("MT128x64_DU64", KernelParams(
            MIWaveTile0=2, MIWaveTile1=1, MIWaveGroup0=2, MIWaveGroup1=2,
            DepthU=64, GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=8,
        ), DataType.FP16),
        ("MT128x32_DU128", KernelParams(
            MIWaveTile0=2, MIWaveTile1=1, MIWaveGroup0=2, MIWaveGroup1=1,
            DepthU=128, GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=8,
        ), DataType.FP16),
    ]
    compare_configs(shape5, skinny_configs, arch=GFX942)
