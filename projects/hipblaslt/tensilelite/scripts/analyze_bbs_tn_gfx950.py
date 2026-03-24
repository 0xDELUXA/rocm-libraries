#!/usr/bin/env python3
"""
Concrete example: Instruction count analysis for the first solution
in the gfx950 BBS TN Equality library logic YAML.

Kernel: Custom_Cijk_Alik_Bljk_BBS_BH_MT256x256x64_MI16x16x1_UserArgs_shortname0_gfx950
GEMM:   M=1024, N=1024, K=1024
Type:   BF16 input, BF16 output, FP32 compute (HPA)
Layout: TN (A transposed, B non-transposed)
"""

import math

print("=" * 90)
print("GEMM Kernel Instruction Count Analysis")
print("Kernel: Custom_Cijk_Alik_Bljk_BBS_BH_MT256x256x64_MI16x16x1")
print("Target: gfx950 | GEMM: [1024, 1024, 1024] | BF16 TN layout")
print("=" * 90)

# ─────────────────────────────────────────────────────────────────────────────
# Parameters extracted from YAML (first solution in the Equality logic file)
# ─────────────────────────────────────────────────────────────────────────────
M, N, K = 1024, 1024, 1024
batch = 1
bpe = 2  # BF16 = 2 bytes

MT0 = 256           # MacroTile0
MT1 = 256           # MacroTile1
DU = 64             # DepthU

MI_M = 16           # MatrixInstM
MI_N = 16           # MatrixInstN
MI_K = 32           # MatrixInstK
MI_B = 1            # MatrixInstB

WT0 = 8             # MIWaveTile0
WT1 = 8             # MIWaveTile1
WG0 = 2             # MIWaveGroup0
WG1 = 2             # MIWaveGroup1

GRVW_A = 8          # GlobalReadVectorWidthA
GRVW_B = 8          # GlobalReadVectorWidthB
LRVW = 8            # LocalReadVectorWidth
VW_A = 8            # VectorWidthA
VW_B = 8            # VectorWidthB

NumThreads = 256    # WorkGroup = [32, 8, 1] -> 32*8*1 = 256 threads = 4 waves
NumWaves = 4
WavefrontSize = 64
InnerUnroll = 1
PGR = 2             # PrefetchGlobalRead
PLR = 1             # PrefetchLocalRead
SIA = 3             # ScheduleIterAlg

DTL_A = True        # DirectToLdsA
DTL_B = True        # DirectToLdsB
DTV_A = False       # DirectToVgprA
DTV_B = False       # DirectToVgprB

GSU = 1             # GlobalSplitU
LSU = 1             # LocalSplitU

MIInputPerThread = 8     # derived from MI_K * bpe / register_width
SubGroup0 = 8       # = MT0 / ThreadTile0 (or derived from MFMA)
SubGroup1 = 32      # = MT1 / ThreadTile1
ThreadTile0 = 32    # = MT0 / SubGroup0
ThreadTile1 = 8     # = MT1 / SubGroup1

print("\n--- Kernel Parameters (from YAML) ---")
print(f"  MacroTile:         {MT0} x {MT1}")
print(f"  DepthU:            {DU}")
print(f"  MatrixInstruction: [{MI_M}, {MI_N}, {MI_K}, {MI_B}]")
print(f"  MIWaveTile:        [{WT0}, {WT1}]")
print(f"  MIWaveGroup:       [{WG0}, {WG1}]")
print(f"  ThreadTile:        [{ThreadTile0}, {ThreadTile1}]")
print(f"  SubGroup:          [{SubGroup0}, {SubGroup1}]")
print(f"  NumThreads:        {NumThreads} ({NumWaves} waves)")
print(f"  GRVW A/B:          {GRVW_A}/{GRVW_B}")
print(f"  LocalReadVW:       {LRVW}")
print(f"  DirectToLds A/B:   {DTL_A}/{DTL_B}")
print(f"  PGR={PGR}, PLR={PLR}, SIA={SIA}")
print(f"  GlobalSplitU:      {GSU}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Compute instruction counts from formulas
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("STEP 1: Analytical Instruction Counts (from formulas)")
print("=" * 90)

# MFMA
numMfmaPerIter = WT0 * WT1 * InnerUnroll
LoopIters = DU // MI_K
totalMfma = numMfmaPerIter * LoopIters

print(f"\n  MFMA instructions:")
print(f"    numMfmaPerIter = MIWaveTile0 * MIWaveTile1 * InnerUnroll")
print(f"                   = {WT0} * {WT1} * {InnerUnroll} = {numMfmaPerIter}")
print(f"    LoopIters      = DepthU / MatrixInstK = {DU} / {MI_K} = {LoopIters}")
print(f"    totalMfma/loop = {numMfmaPerIter} * {LoopIters} = {totalMfma}")

# Global Reads (DirectToLds = data goes global -> LDS directly, via buffer_load ... lds)
numGR_A = (MT0 * DU) // (GRVW_A * NumThreads)
numGR_B = (MT1 * DU) // (GRVW_B * NumThreads)
numGR_total = numGR_A + numGR_B

print(f"\n  Global Read instructions (DirectToLds loads):")
print(f"    numGR_A = (MT0 * DU) / (GRVW_A * NumThreads)")
print(f"            = ({MT0} * {DU}) / ({GRVW_A} * {NumThreads}) = {numGR_A}")
print(f"    numGR_B = (MT1 * DU) / (GRVW_B * NumThreads)")
print(f"            = ({MT1} * {DU}) / ({GRVW_B} * {NumThreads}) = {numGR_B}")
print(f"    total   = {numGR_total}")

# Local Writes (DirectToLds -> NO local writes needed!)
numLW_A = 0 if DTL_A else numGR_A
numLW_B = 0 if DTL_B else numGR_B
numLW_total = numLW_A + numLW_B

print(f"\n  Local Write instructions:")
print(f"    DirectToLdsA={DTL_A}, DirectToLdsB={DTL_B}")
print(f"    numLW_A = {numLW_A} (DTL bypasses local writes)")
print(f"    numLW_B = {numLW_B}")
print(f"    total   = {numLW_total}")

# Local Reads
# With UnrollMajorLDS and bf16, MIInputPerThread = MI_K(=32) * bpe(=2) / 4 = 16 bytes = 4 dwords per unroll
# But the YAML says MIInputPerThread=8 (elements), and each ds_read_b128 reads 16 bytes = 8 bf16 elements
# numReadsPerUnroll = ceil(bpe * MIInputPerThread / (LRVW * bpe))
# = ceil(2 * 8 / (8 * 2)) = ceil(1) = 1  (one ds_read_b128 per unroll per wave-tile)
numReadsPerUnrollA = max(1, math.ceil(bpe * MIInputPerThread / (LRVW * bpe)))
numReadsPerUnrollB = max(1, math.ceil(bpe * MIInputPerThread / (LRVW * bpe)))

numLR_per_iter_A = InnerUnroll * WT0 * numReadsPerUnrollA
numLR_per_iter_B = InnerUnroll * WT1 * numReadsPerUnrollB
numLR_per_iter_total = numLR_per_iter_A + numLR_per_iter_B
numLR_total = numLR_per_iter_total * LoopIters

print(f"\n  Local Read instructions:")
print(f"    readsPerUnrollA = ceil(bpe * MIInputPerThread / (LRVW * bpe))")
print(f"                    = ceil({bpe} * {MIInputPerThread} / ({LRVW} * {bpe})) = {numReadsPerUnrollA}")
print(f"    numLR/iter A    = InnerUnroll * WT0 * readsPerUnroll")
print(f"                    = {InnerUnroll} * {WT0} * {numReadsPerUnrollA} = {numLR_per_iter_A}")
print(f"    numLR/iter B    = {InnerUnroll} * {WT1} * {numReadsPerUnrollB} = {numLR_per_iter_B}")
print(f"    numLR/iter total = {numLR_per_iter_total}")
print(f"    numLR total/loop = {numLR_per_iter_total} * {LoopIters} = {numLR_total}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Actual counts from assembly file
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("STEP 2: Actual Instruction Counts (from assembly)")
print("=" * 90)

actual_loop = {
    "MFMA (v_mfma_f32_16x16x32_bf16)": 128,
    "Global Read (buffer_load_dwordx4 w/ lds)": 16,
    "Local Read A (ds_read_b128 for A)": 16,
    "Local Read B (ds_read_b128 for B)": 16,
    "Local Write": 0,
    "s_barrier": 3,
    "s_waitcnt": 4,
    "Branch (s_cbranch)": 1,
    "v_xor_b32 (LR addr swap)": 2,
    "SALU (s_* misc)": 46,
    "s_nop": 0,
}

print("\n  Main unrolled loop body (label_LoopBeginL to label_LoopEndL):")
for name, count in actual_loop.items():
    print(f"    {name:48s}: {count:>4}")

actual_total = {
    "MFMA": 527,
    "Global Read (buffer_load_)": 368,
    "  - DirectToLds (buffer_load ... lds)": 48,
    "  - Regular buffer_load": 320,
    "Local Read (ds_read_b128)": 112,
    "Local Write (ds_store/write)": 0,
    "Global Write (buffer_store_)": 1056,
    "s_barrier": 6,
    "s_waitcnt": 55,
    "Branch": 64,
    "s_nop": 66,
    "v_accvgpr_read": 2560,
    "v_accvgpr_write": 16,
}

print("\n  Full kernel totals:")
for name, count in actual_total.items():
    print(f"    {name:48s}: {count:>5}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Compare analytical vs actual
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("STEP 3: Analytical vs Actual (Main Loop Body)")
print("=" * 90)

comparisons = [
    ("MFMA per loop",           totalMfma,   128),
    ("Global Reads per loop",   numGR_total, 16),
    ("Local Reads A per loop",  numLR_per_iter_A * LoopIters, 16),
    ("Local Reads B per loop",  numLR_per_iter_B * LoopIters, 16),
    ("Local Writes per loop",   numLW_total, 0),
]

print(f"\n  {'Metric':<35s} {'Analytical':>12s} {'Actual':>10s} {'Match':>8s}")
print(f"  {'-'*35} {'-'*12} {'-'*10} {'-'*8}")
for name, analytical, actual in comparisons:
    match = "YES" if analytical == actual else "NO"
    print(f"  {name:<35s} {analytical:>12} {actual:>10} {match:>8s}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Explanation of discrepancies
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("STEP 4: Understanding the Differences")
print("=" * 90)

print("""
  MFMA: Analytical=128, Actual=128  -> EXACT MATCH
    Formula: MIWaveTile0(8) * MIWaveTile1(8) * InnerUnroll(1) * LoopIters(2) = 128

  Global Reads: Analytical=16, Actual=16  -> EXACT MATCH
    Formula: (MT0*DU)/(GRVW_A*NT) + (MT1*DU)/(GRVW_B*NT)
           = (256*64)/(8*256) + (256*64)/(8*256) = 8 + 8 = 16
    All 16 are buffer_load_dwordx4 with 'lds' modifier (DirectToLds).

  Local Reads: Analytical=32, Actual=32  -> EXACT MATCH
    Formula: (WT0 * 1 + WT1 * 1) * LoopIters = (8+8)*2 = 32
    All are ds_read_b128 (128-bit = 16 bytes = 8 BF16 elements).
    Split: 16 for A, 16 for B.

  Local Writes: Analytical=0, Actual=0  -> EXACT MATCH
    DirectToLdsA=True, DirectToLdsB=True -> no local writes needed.

  The analytical formulas EXACTLY match the actual assembly instruction counts
  for all major instruction categories in the main loop body.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Full GEMM [1024, 1024, 1024] execution analysis
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 90)
print("STEP 5: GEMM [1024, 1024, 1024] Execution Analysis")
print("=" * 90)

K_eff = math.ceil(K / GSU)
loopCount = K_eff // DU
K_tail = K_eff % DU
M_tiles = math.ceil(M / MT0)
N_tiles = math.ceil(N / MT1)
total_WGs = M_tiles * N_tiles * batch * GSU

print(f"\n  Problem decomposition:")
print(f"    K_effective     = K / GSU = {K} / {GSU} = {K_eff}")
print(f"    loopCount       = K_eff / DepthU = {K_eff} / {DU} = {loopCount}")
print(f"    K_tail          = K_eff % DepthU = {K_eff} % {DU} = {K_tail}")
print(f"    M_tiles         = ceil(M/MT0) = ceil({M}/{MT0}) = {M_tiles}")
print(f"    N_tiles         = ceil(N/MT1) = ceil({N}/{MT1}) = {N_tiles}")
print(f"    Total workgroups = {M_tiles} * {N_tiles} * {batch} * {GSU} = {total_WGs}")

print(f"\n  Instructions per workgroup (main loop):")
total_mfma_per_wg = totalMfma * loopCount
total_gr_per_wg = numGR_total * loopCount
total_lr_per_wg = numLR_total * loopCount
print(f"    MFMA instructions  = {totalMfma}/loop * {loopCount} loops = {total_mfma_per_wg}")
print(f"    Global reads       = {numGR_total}/loop * {loopCount} loops = {total_gr_per_wg}")
print(f"    Local reads        = {numLR_total}/loop * {loopCount} loops = {total_lr_per_wg}")

# For gfx950 with BF16: MFMA issue latency = MI_M / 4 = 16/4 = 4 quad-cycles
mi_issue_latency = MI_M // 4
print(f"\n  MFMA issue latency:")
print(f"    BF16 on gfx950: MatrixInstM / 4 = {MI_M} / 4 = {mi_issue_latency} quad-cycles")

# Compute-bound cycle estimate for main loop body
mfma_cycles_per_loop = totalMfma * mi_issue_latency  # quad-cycles
shader_cycles_per_loop = mfma_cycles_per_loop * 4    # GFX9: multiply by 4

print(f"\n  Compute-bound estimate (pure MFMA chain):")
print(f"    MFMA cycles/loop  = {totalMfma} * {mi_issue_latency} = {mfma_cycles_per_loop} quad-cycles")
print(f"    Shader cycles     = {mfma_cycles_per_loop} * 4 = {shader_cycles_per_loop} shader-cycles")

total_compute_cycles = shader_cycles_per_loop * loopCount
print(f"    Total compute     = {shader_cycles_per_loop} * {loopCount} = {total_compute_cycles} shader-cycles")

# FLOPs
flops = 2 * M * N * K * batch
flops_per_mfma = 2 * MI_M * MI_N * MI_K  # each MFMA does MxNxK MACs = 2*M*N*K FLOPs
total_mfma_all_wgs = total_mfma_per_wg * total_WGs

flops_per_mfma_wavefront = flops_per_mfma * WavefrontSize
total_mfma_wavefront = total_mfma_all_wgs * NumWaves
computed_flops = total_mfma_all_wgs * flops_per_mfma * NumWaves * WavefrontSize // NumWaves

print(f"\n  FLOPs analysis:")
print(f"    Total FLOPs = 2*M*N*K = 2*{M}*{N}*{K} = {flops:,.0f}")
print(f"    FLOPs/MFMA  = 2*{MI_M}*{MI_N}*{MI_K} = {flops_per_mfma:,}")
print(f"    Each MFMA operates on {WavefrontSize} threads, producing {MI_M}*{MI_N}={MI_M*MI_N} output elements")
print(f"    Total MFMAs across all WGs = {total_mfma_per_wg} * {total_WGs} = {total_mfma_all_wgs:,}")
print(f"    Cross-check: {total_mfma_all_wgs:,} MFMAs * {flops_per_mfma:,} FLOPs/MFMA = {total_mfma_all_wgs * flops_per_mfma:,.0f}")
print(f"    Note: Total FLOPs = MFMAs * FLOPs/MFMA * WavesPerWG")
print(f"         = {total_mfma_all_wgs:,} * {flops_per_mfma:,} * {NumWaves} = {total_mfma_all_wgs * flops_per_mfma * NumWaves:,.0f}")
print(f"    Expected: {flops:,.0f}")

# Summary table
print(f"\n" + "=" * 90)
print("SUMMARY: Instruction Counts for GEMM [1024, 1024, 1024] BF16 TN on gfx950")
print("=" * 90)
print(f"""
  Kernel:  Custom_Cijk_Alik_Bljk_BBS_BH_MT256x256x64_MI16x16x1_..._gfx950
  Config:  MT=256x256, DU=64, MI=16x16x32x1, WT=8x8, WG=2x2, DTL=A+B

  ┌─────────────────────────────────────────────────────────────────────┐
  │ Metric                          │ Per Loop │ Per WG  │  Formula    │
  ├─────────────────────────────────┼──────────┼─────────┼─────────────┤
  │ LoopIters (sub-iterations)      │     2    │    -    │  DU/MI_K    │
  │ LoopCount (outer loops)         │     -    │   {loopCount:>3}    │  K/DU       │
  │ MFMA instructions               │   {totalMfma:>3}    │ {total_mfma_per_wg:>5}  │  WT0*WT1*LI │
  │ Global reads (DTL)              │    {numGR_total:>2}    │   {total_gr_per_wg:>3}  │  MT*DU/GV/N │
  │ Local reads (ds_read_b128)      │    {numLR_total:>2}    │   {total_lr_per_wg:>3}  │  WT*LI*rpu  │
  │ Local writes                    │     0    │     0   │  DTL->none  │
  │ MFMA issue latency              │     -    │    -    │  {mi_issue_latency} q-cyc    │
  │ Compute cycles (shader)         │ {shader_cycles_per_loop:>5}  │ {total_compute_cycles:>5}  │             │
  │ Workgroups                      │     -    │    -    │  {total_WGs} WGs     │
  │ Total FLOPs                     │     -    │    -    │  {flops:>11,} │
  └─────────────────────────────────┴──────────┴─────────┴─────────────┘

  All analytical instruction counts match the actual assembly exactly.
""")
