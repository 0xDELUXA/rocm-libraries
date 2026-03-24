# GEMM Kernel Assembly Analysis Report

**Kernel**: `Custom_Cijk_Alik_Bljk_BBS_BH_MT256x256x64_MI16x16x1_UserArgs_shortname0_gfx950`
**Target**: `amdgcn-amd-amdhsa--gfx950`
**File**: `projects/hipblaslt/tensilelite/Tensile/CustomKernels/Custom_Cijk_Alik_Bljk_BBS_BH_MT256x256x64_MI16x16x1_UserArgs_shortname0_gfx950.s`
**Total lines**: 19,059

---

## 1. Kernel Header & Configuration

```
Target:          gfx950
Operation:       GEMM (BF16 inputs, BF16 output, FP32 compute)
HPA:             True (High Precision Accumulate)
TransposeA:      1 (column-major / transposed)
TransposeB:      0 (row-major / non-transposed)
Batched:         True
UseBeta:         True
Wavefront Size:  64
Workgroup Size:  256 (max_flat_workgroup_size)
```

### Resource Usage
| Resource    | Count  |
|-------------|--------|
| VGPRs       | 248    |
| AccVGPRs    | 256    |
| SGPRs       | 88     |
| LDS bytes   | 133,120 (130 KB) |

### Tile & Depth Parameters (from `.set` directives)
| Parameter    | Value |
|-------------|-------|
| MT0 (MacroTile0) | 256 |
| MT1 (MacroTile1) | 256 |
| DepthU       | 64    |
| ThreadTile   | 32 x 8 |
| SubGroup     | 8 x 32 |
| VectorWidthA | 8     |
| VectorWidthB | 8     |
| BpeA (bytes/element A) | 2 (BF16) |
| BpeB (bytes/element B) | 2 (BF16) |
| DirectToLdsA | True  |
| DirectToLdsB | True  |
| SrdShiftLeftA | 8    |
| SrdShiftLeftB | 8    |

---

## 2. Total Instruction Counts (Entire Kernel)

| Instruction Type | Count | Notes |
|-----------------|-------|-------|
| **MFMA** (`v_mfma_`) | **527** | 512 x `v_mfma_f32_16x16x32_bf16` + 15 x `v_mfma_f32_32x32x16_bf16` |
| **Global Read** (`buffer_load_`) | **368** | 112 x `buffer_load_dwordx4` + 256 x `buffer_load_short_d16` |
| **Local Read** (`ds_read_`) | **112** | All `ds_read_b128` |
| **Local Write** (`ds_store_`/`ds_write_`) | **0** | None! Uses DirectToLDS (buffer_load with `, lds` suffix) |
| **Global Write** (`buffer_store_`) | **1,056** | 512 x `buffer_store_short` + 288 x `buffer_store_dwordx4` + 256 x `buffer_store_dword` |
| **s_barrier** | **6** | |
| **s_waitcnt** | **55** | |
| **Branch** (`s_cbranch_`/`s_branch`) | **64** | |
| **VALU** (`v_` excl. mfma/accvgpr) | **9,137** | General vector ALU |
| **SALU** (`s_` excl. barrier/waitcnt/branch/endpgm/nop) | **1,875** | General scalar ALU |
| **s_nop** | **66** | |
| **v_accvgpr_write** | **16** | |
| **v_accvgpr_read** | **2,560** | |
| **s_endpgm** | **2** | |

### DirectToLDS Breakdown
- `buffer_load_dwordx4 ... , lds`: **48** (global reads that go directly to LDS)
- `buffer_load_dwordx4` (to VGPR): 64 (= 112 - 48)
- `buffer_load_short_d16` (to VGPR): 256

---

## 3. Main Inner Loop Body

### Loop Labels
- **Loop Begin**: `label_LoopBeginL` (line 1499)
- **Loop End**: `label_LoopEndL` (line 1940)
- **Loop Open**: `label_openLoopL` (line 1493)
- Loop span: **441 lines** (lines 1499-1940)

The loop iterates with `sgprLoopCounterL` decremented by 1 each iteration, branching back to `label_LoopBeginL` when `sgprLoopCounterL != 2` (PGR=2, so the last 2 iterations are peeled off into NoGlobalLoad loops).

### Instruction Counts in Main Loop Body

| Instruction Type | Loop Count | Notes |
|-----------------|------------|-------|
| **MFMA** (`v_mfma_`) | **128** | All `v_mfma_f32_16x16x32_bf16` (mfmaIndex 0-127) |
| **Global Read** (`buffer_load_`) | **16** | All `buffer_load_dwordx4 ... , lds` (DirectToLDS) |
| **Local Read** (`ds_read_`) | **32** | All `ds_read_b128` |
| **Local Write** (`ds_store_`/`ds_write_`) | **0** | DirectToLDS used |
| **Global Write** (`buffer_store_`) | **0** | No stores in compute loop |
| **s_barrier** | **3** | |
| **s_waitcnt** | **4** | (lgkmcnt and vmcnt waits) |
| **Branch** (`s_cbranch_`) | **1** | `s_cbranch_scc0 label_LoopBeginL` |
| **VALU** (non-mfma, non-accvgpr) | **2** | `v_xor_b32` for local read swap (A and B) |
| **SALU** (non-barrier/waitcnt/branch/nop) | **38** | Address updates, SRD increments, limit calculations, m0 updates, swap offsets, loop counter decrement |
| **s_nop** | **0** | |
| **v_accvgpr_** | **0** | |

### Loop Body Structure

The loop body processes **2 unrolled iterations** per loop pass (DepthU=64, with 2 sub-iterations of 32 elements each):

- **Iteration 0** (mfmaIndex 0-63): Uses `ValuA_X0/ValuB_X0` operands, interleaved with:
  - 8 ds_read_b128 for A (next iteration's X1 values)
  - 8 ds_read_b128 for B (next iteration's X1 values)
  - 8 buffer_load_dwordx4 for A (DirectToLDS, next DepthU tile)
  - Global read increment logic (A and B SRD updates)
  - 2 s_barrier (at mfmaIndex 21 and 51)
  - 2 s_waitcnt lgkmcnt(0) (before barriers)

- **Iteration 1** (mfmaIndex 64-127): Uses `ValuA_X1/ValuB_X1` operands, interleaved with:
  - 8 ds_read_b128 for A (next loop's X0 values)
  - 8 ds_read_b128 for B (next loop's X0 values)
  - 8 buffer_load_dwordx4 for B (DirectToLDS, next DepthU tile)
  - Local read/write swap offsets
  - 1 s_barrier (at mfmaIndex 92)
  - 1 s_waitcnt vmcnt(13) (wait for previous global reads)
  - 1 s_waitcnt lgkmcnt(0) (at end)
  - Loop counter decrement and branch

Each sub-iteration computes 64 MFMAs = 8 (B-tiles) x 8 (A-tiles) using `v_mfma_f32_16x16x32_bf16`.

### Total instructions in loop body: ~226
(128 MFMA + 16 buffer_load + 32 ds_read + 3 s_barrier + 4 s_waitcnt + 1 branch + 2 VALU + 38 SALU + 0 s_nop = 224 actual instructions excluding comments and labels)

---

## 4. Key `.set` Directives

### VGPR Assignments
```
vgprValuC = 0
vgprBase = 4
vgprGlobalReadOffsetA = 0
vgprGlobalReadOffsetB = 1
vgprLocalReadAddrA = 2
vgprLocalReadAddrB = 3
vgprLocalReadSwapAddrA = 132
vgprLocalReadSwapAddrB = 133
vgprSerial = 134
vgprValuA_X0_I0_BASE = vgprBase+0 (= 4)
vgprValuB_X0_I0_BASE = vgprBase+64 (= 68)
```

### SGPR Assignments
```
sgprKernArgAddress = 0
sgprWorkGroup0 = 2, sgprWorkGroup1 = 3, sgprWorkGroup2 = 4
sgprLoopCounterL = 12
sgprOrigLoopCounter = 13
sgprSrdA = 52, sgprSrdB = 56
sgprShadowLimitA = 60, sgprShadowLimitB = 62
sgprGlobalReadIncsA = 68, sgprGlobalReadIncsB = 69
sgprAlpha = 44, sgprBeta = 45
sgprLocalWriteAddrA = 46, sgprLocalWriteAddrB = 47
sgprSwapA = 48, sgprSwapB = 49
sgprGSU = 50
```

### Architecture Parameters
```
MT0 = 256, MT1 = 256
DepthU = 64
BpeA = 2, BpeB = 2 (BF16)
SrdShiftLeftA = 8, SrdShiftLeftB = 8
BufferLimit = 0xffffffff
BufferOOB = 0x80000000
Srd127_96 = 0x20000
```

---

## 5. File Structure Overview

| Section | Lines | Description |
|---------|-------|-------------|
| Header & metadata | 1-198 | AMDHSA kernel descriptor, metadata YAML |
| Kernel entry & macros | 199-342 | Macros, VGPR/SGPR assignments, `.set` directives |
| Arg loading & setup | 343-724 | Load kernel arguments, workgroup mapping |
| Tile setup | 725-1334 | Address calculations, stagger, SRD setup, initial global reads & local reads |
| **Main unrolled loop** | **1490-1940** | `label_openLoopL` to `label_LoopEndL` |
| NoGlobalLoad loops | 1941-3338 | Drain remaining iterations (Ord. NGL, Opt. NLL) |
| Global write sections | 3339-19057 | Multiple store paths (B0_E0, B0_E1_N, B0_E1_M, B1_E0, B1_E1_N, B1_E1_M) with edge handling |
| Kernel end | 19057-19059 | `label_KernelEnd`, `s_endpgm` |

---

## 6. Performance-Relevant Observations

1. **MFMA-dense loop**: 128 MFMAs out of ~224 total instructions = ~57% MFMA density in the main loop
2. **Prefetch depth = 2** (PGR=2): Two iterations are peeled off for software pipelining
3. **DirectToLDS**: Both A and B matrices use direct-to-LDS global reads, eliminating separate LDS write instructions
4. **Double buffering**: Local read addresses are swapped each iteration via `v_xor_b32` and `s_xor_b32`
5. **128 MFMAs per loop iteration**: 8x8 tile of `v_mfma_f32_16x16x32_bf16` = 256 output elements (acc[0:255])
6. **Large store section**: The global write section (lines 3339-19057) dominates the file by line count due to multiple edge-handling paths with `v_accvgpr_read` and `buffer_store` instructions
7. **Zero s_nop in loop**: No explicit NOPs needed in the main loop, suggesting good instruction scheduling
