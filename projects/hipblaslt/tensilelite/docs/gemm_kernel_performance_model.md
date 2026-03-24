# GEMM Kernel Performance Modeling: TensileLite Parameter-to-Cycle Analysis

## Overview

This document provides a comprehensive analysis of how TensileLite GEMM kernel parameters
map to instruction counts, cycle counts, and ultimately execution time. By understanding
these relationships, we can formulate the number of cycles consumed by a kernel as a
function of GEMM shape (M, N, K), data type, layout, and tuning parameters.

TensileLite already has a multi-layered performance prediction system embedded in its
codebase. This document extracts, documents, and formalizes these relationships.

---

## 1. Architecture of the Performance Model

The existing TensileLite performance model operates at three levels:

```
Level 1: Instruction-Level Cycle Simulation (cycle.cpp)
  |-- Walks the kernel IR instruction-by-instruction
  |-- Models MFMA pipeline, LDS FIFOs, global read queues
  |-- Returns: MathClocksUnrolledLoop (cycles per inner-loop iteration)
  |
Level 2: Memory Hierarchy Model (formocast_simulator.cpp)
  |-- Models L1/L2/L3/HBM cache hit rates and bandwidth
  |-- Computes memory access costs per loop iteration
  |-- Returns: overall execution time in microseconds
  |
Level 3: Granularity Model (ContractionSolution.cpp)
  |-- Models tile, CU, and wave granularity losses
  |-- Combines with ideal performance lookup
  |-- Returns: predicted GFlops
```

---

## 2. Key Kernel Parameters and Their Roles

### 2.1 Tile Geometry Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| **MacroTile0 (MT0)** | Tile size in M dimension | 16-256 |
| **MacroTile1 (MT1)** | Tile size in N dimension | 16-256 |
| **DepthU (DU)** | Unroll depth in K dimension | 8-512 |
| **ThreadTile0 (TT0)** | Elements per thread in M | 1-16 |
| **ThreadTile1 (TT1)** | Elements per thread in N | 1-16 |
| **WorkGroup (WG)** | [SG0, SG1, LSU] dimensions | e.g., [4, 4, 1] |

**Derivations:**
```
MacroTile0 = SubGroup0 * ThreadTile0
MacroTile1 = SubGroup1 * ThreadTile1
NumThreads = SubGroup0 * SubGroup1 * LocalSplitU * WavefrontSize (typically 256)
```

### 2.2 Matrix Instruction (MFMA) Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| **MatrixInstruction** | [M, N, K, B] or 9-tuple | [16, 16, 16, 1] |
| **MatrixInstM** | MFMA output rows | 16 or 32 |
| **MatrixInstN** | MFMA output columns | 16 or 32 |
| **MatrixInstK** | K elements per MFMA | varies by datatype |
| **MatrixInstB** | Block count | 1, 2, or 4 |
| **MIWaveTile** | [WT0, WT1] MFMAs per wave | e.g., [4, 2] |
| **MIWaveGroup** | [WG0, WG1] waves per workgroup | e.g., [2, 2] |

**For MFMA kernels:**
```
MacroTile0 = MatrixInstM * MatrixInstBM * MIWaveGroup[0] * MIWaveTile[0]
MacroTile1 = MatrixInstN * MatrixInstBN * MIWaveGroup[1] * MIWaveTile[1]
```

### 2.3 Memory Access Parameters

| Parameter | Description | Values |
|-----------|-------------|--------|
| **GlobalReadVectorWidthA/B** | Elements per global load | 1-16 |
| **LocalReadVectorWidth** | Elements per LDS load | 1-16 |
| **VectorWidthA/B** | Store vector width | 1-8 |
| **NumLoadsCoalescedA/B** | Coalesced load count | 1-64 |
| **DirectToVgprA/B** | Bypass LDS for loads | True/False |
| **DirectToLds** | Load directly to LDS | 0-3 |
| **BufferLoad/BufferStore** | Use buffer instructions | True/False |

### 2.4 Scheduling & Prefetch Parameters

| Parameter | Description | Values |
|-----------|-------------|--------|
| **PrefetchGlobalRead (PGR)** | Global read prefetch depth | 0-16 |
| **PrefetchLocalRead (PLR)** | Local read prefetch iterations | 0-128 |
| **ScheduleIterAlg (SIA)** | Scheduling algorithm | 0-3 |
| **GlobalReadPerMfma** | GR instructions per MFMA | 0.01-32 |
| **LocalWritePerMfma** | LW instructions per MFMA | 0.01-32 or -1 |
| **ScheduleGlobalRead** | Schedule GR into LR iterations | 0/1 |
| **ScheduleLocalWrite** | Schedule LW into LR iterations | 0/1 |

### 2.5 Split & Mapping Parameters

| Parameter | Description | Values |
|-----------|-------------|--------|
| **GlobalSplitU (GSU)** | K-dimension split factor | 0-1024 |
| **LocalSplitU (LSU)** | Intra-WG K split | 1-256 (WorkGroup[2]) |
| **WorkGroupMapping (WGM)** | Tile-to-CU mapping | -1024 to 1024 |
| **StreamK** | StreamK variant | 0-3 |
| **StaggerU** | Stagger summation start | 0-64 |

---

## 3. Instruction Count Formulas

### 3.1 MFMA Instructions

**Per sub-iteration:**
```
numMfmaPerIter = MIWaveTile[0] * MIWaveTile[1] * InnerUnroll
```

**Multipliers for special data types:**
- Complex (C/Z): `numMfmaPerIter *= 4`
- TF32 emulation (XF32): `numMfmaPerIter *= 3`

**Number of sub-iterations per full unroll:**
```
LoopIters = DepthU / MatrixInstK
```

**Total MFMAs per full unroll loop:**
```
totalMfmaPerUnroll = numMfmaPerIter * LoopIters
```

**Example:** MT=256x128, MI=16x16x16x1, WT=[4,2], DU=64
- `numMfmaPerIter = 4 * 2 = 8`
- `LoopIters = 64 / 16 = 4`
- `totalMfmaPerUnroll = 8 * 4 = 32`

### 3.2 Global Read Instructions

```
numGlobalReadsA = (MacroTile0 * DepthU) / (GlobalReadVectorWidthA * NumThreads)
numGlobalReadsB = (MacroTile1 * DepthU) / (GlobalReadVectorWidthB * NumThreads)
totalGlobalReads = numGlobalReadsA + numGlobalReadsB
```

Where `NumThreads = WorkGroup[0] * WorkGroup[1] * WorkGroup[2] * WavefrontSize` (typically 256).

**Example:** MT0=256, MT1=128, DU=64, GRVW=4, NumThreads=256
- `numGlobalReadsA = (256 * 64) / (4 * 256) = 16`
- `numGlobalReadsB = (128 * 64) / (4 * 256) = 8`
- `totalGlobalReads = 24`

### 3.3 Local Write Instructions

One local write per global read (data flows global -> VGPR -> LDS):
```
numLocalWritesA = numGlobalReadsA
numLocalWritesB = numGlobalReadsB
totalLocalWrites = totalGlobalReads
```

Exception: DirectToLds bypasses VGPRs (data goes global -> LDS directly), eliminating
local write instructions but requiring specific alignment and vector width constraints.

### 3.4 Local Read Instructions

```
numReadsPerUnrollA = ceil(bpeA * MIInputPerThreadA / (LocalReadBlockWidth * 4))
numLocalReadsPerIterA = InnerUnroll * MIWaveTile[0] * numReadsPerUnrollA

numReadsPerUnrollB = ceil(bpeB * MIInputPerThreadB / (LocalReadBlockWidth * 4))
numLocalReadsPerIterB = InnerUnroll * MIWaveTile[1] * numReadsPerUnrollB

totalLocalReadsPerIter = numLocalReadsPerIterA + numLocalReadsPerIterB
totalLocalReadsPerUnroll = totalLocalReadsPerIter * LoopIters
```

### 3.5 Barrier and Synchronization Instructions

Per unroll loop iteration:
- 1 `s_barrier` (for LDS synchronization between global write and local read phases)
- 1-2 `s_waitcnt` instructions (to wait for local reads / global reads to complete)
- Branch instruction at loop end

---

## 4. Cycle Count Model

### 4.1 MFMA Issue Latency

The MFMA issue latency determines how many cycles must elapse before the next MFMA
can be issued:

```
For gfx940/gfx942/gfx950 with MatrixInstB == 1:
  Half/BF16/Int8/F8 types:
    issueLatency = MatrixInstM / 4   (e.g., 16/4 = 4 quad-cycles for MI16)
    miIssueLatency = 1
  Other types (FP32, FP64):
    issueLatency = MatrixInstM / 2   (e.g., 16/2 = 8 quad-cycles for MI16)
    miIssueLatency = 2

For gfx950 + F8 types:
    issueLatency = MatrixInstM / 2   (F8 MFMA takes 2x cycles but computes 4xK)

For sparse MFMA or XFloat32:
    issueLatency = MatrixInstM / 4
```

**GFX9 scaling:** All cycle counts are multiplied by 4 (quad-cycle to shader-clock
conversion) for ISA version 9.x.x architectures.

### 4.2 LDS (Local Data Share) Instruction Latencies

| Instruction | Issue Latency (quad-cycles) |
|-------------|---------------------------|
| ds_read_b32 | 1 |
| ds_read_b64 | 1 |
| ds_read_b128 | 2 |
| ds_write_b8 | 1 |
| ds_write_b16 | 2 |
| ds_write_b32 | 2 |
| ds_write_b64 | 3 |
| ds_write_b128 | 5 |
| ds_write_b256 | 10 |

**Bank conflict penalty:** LDS has 32 banks with 4-byte width. When multiple threads
access the same bank, conflicts cause additional latency:
```
effective_latency = base_latency + (bankConflictRatio - 1.0) * conflictMultiplier
```
Where `bankConflictRatio = maxBankUsage / avgBankUsage` (1.0 = no conflicts).

### 4.3 Global Read Latencies

The global read FIFO has a depth of 16 entries. When the FIFO is full, the pipeline
stalls. Multi-wave configurations interleave access to shared resources:
```
Stall when FIFO full:
  Large reads (>= 8 bytes): 4 quad-cycles stall
  Small reads (< 8 bytes): 1 quad-cycle stall
```

### 4.4 Inner Loop Cycle Formula

The instruction-level simulator (`cycle.cpp`) models the inner loop as:

```
For each instruction in the unrolled loop body:
  if MFMA:
    if (cycles - lastMFMA >= issueLatency - 1):
      cycles += 1           // No stall, issue immediately
    else:
      cycles = lastMFMA + issueLatency  // Stall until previous MFMA completes
    lastMFMA = cycles

  if DS_Load (local read):
    Check LR FIFO capacity (depth = 16 / numWaves)
    Add bank conflict penalty
    Add issue latency (1 or 2 quad-cycles)
    gfx950: +1 cycle if immediately after MFMA
    Two-wave sharing: +1 extra issue cycle for consecutive reads

  if Buffer_Load (global read):
    Check GR FIFO capacity (depth 16)
    Add stall cycles if FIFO full
    +1 cycle for SGPR offset

  if DS_Store (local write):
    Check LW queue capacity
    Add issue latency (2-10 quad-cycles depending on width)

  if s_waitcnt:
    Wait for local reads to complete (LGKM counter)

  if branch:
    Add jump overhead (6 quad-cycles)
    Must wait at least 4 cycles after last MFMA

  if s_barrier:
    Add 2 quad-cycles

  other VALU/SALU:
    Add 1 quad-cycle

Final: cycles_total = cycles * 4  (for GFX9 architectures)
```

---

## 5. End-to-End Performance Formula

### 5.1 Problem Decomposition

```
K_after_GSU = ceil(K / GlobalSplitU)
M_tiles = ceil(M / MacroTile0)
N_tiles = ceil(N / MacroTile1)
loopCount = floor(K_after_GSU / DepthU)
K_tail = K_after_GSU % DepthU
numWorkgroups = M_tiles * N_tiles * NumBatches * GlobalSplitU
```

### 5.2 Compute Cost

```
math_clocks = MathClocksUnrolledLoop  (from cycle.cpp simulation)
math_time = math_clocks / math_frequency
```

### 5.3 Memory Cost

Cache hit rates determine effective bandwidth:
```
A_L1_req = f(MT0, DU, bpeA, transA, GRVWA, NLCA, DTVA)
B_L1_req = f(MT1, DU, bpeB, transB, GRVWB, NLCB, DTVB)

A_L2_req = A_L1_req * (1 - L1_hit_A) / coalescing_factor
B_L2_req = B_L1_req * (1 - L1_hit_B) / coalescing_factor

A_L3_req = A_L2_req * (1 - L2_hit_A)
B_L3_req = B_L2_req * (1 - L2_hit_B)

A_hbm_req = A_L3_req * (1 - L3_hit_A)
B_hbm_req = B_L3_req * (1 - L3_hit_B)

mem_cost = (L1_cost + L2_cost + L3_cost + HBM_cost) / mem_frequency
```

Where each level's cost is:
```
L1_cost = (A_L1_req * A_L1_hit + B_L1_req * B_L1_hit) * CacheLineSize / L1BusWidth
L2_cost = (A_L2_req + B_L2_req) * CacheLineSize / min(L2Bandwidth, L2BusWidth)
L3_cost = (A_L3_req + B_L3_req) * CacheLineSize / L3Bandwidth
HBM_cost = (A_hbm_req + B_hbm_req) * elementSize / HBMBandwidth
```

### 5.4 Loop Performance

```
if PrefetchGlobalRead > 1:
    loop_time = max(math_time, mem_time) * (loopCount - 1) + math_time
else:
    loop_time = max(math_time, mem_time) * loopCount
```

### 5.5 Overhead Components

```
prefetch_cost = f(numGR_instructions, DU, frequency)
tail_cost = mem_time * (K_tail / DU) + math_time + 2 * prefetch_cost
store_cost = f(MT0, MT1, GWVW, bpeD, cache_hierarchy)
gsu_overhead = f(GSU, method, M, N, K, bpeD)  // for GSU > 1
lsu_overhead = f(MT0, MT1, LSU, SVW, bpeD)     // for LSU > 1
```

### 5.6 Final Performance

```
total_time = prefetch_cost + loop_time + store_cost + tail_cost + lsu_overhead
total_time = resolveOccupancy(total_time, num_tiles, CU_occupancy)
total_time += gsu_overhead
```

---

## 6. Parameter Impact Summary

### 6.1 Parameters that Affect Compute (MFMA) Cycles

| Parameter | Effect on Compute |
|-----------|-------------------|
| **MIWaveTile** | Linear: more MFMAs per iteration |
| **DepthU / MatrixInstK** | Linear: more loop iterations |
| **InnerUnroll** | Linear: more MFMAs per sub-iteration |
| **MatrixInstM** | Determines MFMA issue latency |
| **Data type** | Determines MFMA latency divisor (2 or 4) |

### 6.2 Parameters that Affect Memory Cycles

| Parameter | Effect on Memory |
|-----------|-----------------|
| **MacroTile0/1** | Larger tiles = more data to load, but better reuse |
| **DepthU** | More depth = more data per loop, fewer loops |
| **GlobalReadVectorWidth** | Wider reads = fewer instructions, better bandwidth |
| **NumLoadsCoalesced** | Affects coalescing efficiency |
| **DirectToVgpr/DirectToLds** | Bypasses LDS intermediary, reduces latency |
| **NonTemporal flags** | Affects cache bypass behavior |
| **StaggerU** | Reduces bank conflicts in DRAM |

### 6.3 Parameters that Affect Scheduling Efficiency

| Parameter | Effect on Scheduling |
|-----------|---------------------|
| **PrefetchGlobalRead** | Hides global read latency behind compute |
| **PrefetchLocalRead** | Hides LDS read latency behind MFMA |
| **ScheduleIterAlg** | Determines instruction interleaving strategy |
| **GlobalReadPerMfma** | Controls GR density (too high -> VMEM stall) |
| **LocalWritePerMfma** | Controls LW density |
| **ClusterLocalRead** | Dedicates VGPR buffers per iteration |
| **OptNoLoadLoop** | Interleaves stores with final MACs |

### 6.4 Parameters that Affect Occupancy & Overhead

| Parameter | Effect |
|-----------|--------|
| **WorkGroup size** | Affects wave count, occupancy |
| **GlobalSplitU** | Adds accumulation overhead, increases parallelism |
| **LocalSplitU** | Adds LDS reduction overhead |
| **WorkGroupMapping** | Affects L2 cache hit rate |
| **StreamK** | Distributes work more evenly across CUs |
| **StoreRemapVectorWidth** | Optimizes store pattern via LDS remap |
| **LdsPad** | Reduces LDS bank conflicts at cost of LDS space |
| **MaxOccupancy** | Limits waves per CU to reduce cache pressure |

---

## 7. Simplified Analytical Model

For a quick cycle estimate without full simulation, use this simplified model:

### 7.1 Compute Bound Estimate

```python
def compute_cycles(MI_M, MI_K, MIWaveTile, DepthU, InnerUnroll, datatype):
    numMfmaPerIter = MIWaveTile[0] * MIWaveTile[1] * InnerUnroll
    loopIters = DepthU // MI_K

    if datatype in ['half', 'bf16', 'int8', 'f8']:
        issue_latency = MI_M // 4
    else:
        issue_latency = MI_M // 2

    cycles_per_iter = numMfmaPerIter * issue_latency
    total_quad_cycles = cycles_per_iter * loopIters
    total_shader_cycles = total_quad_cycles * 4  # GFX9

    return total_shader_cycles
```

### 7.2 Memory Bound Estimate

```python
def memory_cycles(MT0, MT1, DepthU, bpe, GRVW, NumThreads, bandwidth_per_CU):
    bytes_A = MT0 * DepthU * bpe
    bytes_B = MT1 * DepthU * bpe
    total_bytes = bytes_A + bytes_B

    memory_cycles = total_bytes / bandwidth_per_CU
    return memory_cycles
```

### 7.3 Overall Inner Loop

```python
def inner_loop_cycles(compute_cycles, memory_cycles, PGR):
    return max(compute_cycles, memory_cycles)
```

### 7.4 Total Execution Time

```python
def total_time(M, N, K, MT0, MT1, DepthU, GSU, NumCUs, loop_cycles, freq):
    K_eff = ceil(K / GSU)
    loop_count = K_eff // DepthU
    M_tiles = ceil(M / MT0)
    N_tiles = ceil(N / MT1)
    num_WGs = M_tiles * N_tiles * GSU

    waves_per_CU = ceil(num_WGs / NumCUs)
    total_loops = loop_count * waves_per_CU
    total_cycles = total_loops * loop_cycles

    return total_cycles / freq
```

---

## 8. Hardware Constants by Architecture

### gfx942 (MI300X)
| Constant | Value |
|----------|-------|
| NumCUs | 304 |
| NumXCDs | 8 |
| L1 Cache | 32 KB |
| L2 Cache | 256 MB (32 MB/XCD) |
| L3 Cache (Infinity Cache) | 256 MB |
| HBM Bandwidth | 5.3 TB/s |
| Math Frequency | ~1.5 GHz |
| Wavefront Size | 64 |

### gfx950 (MI350)
| Constant | Value |
|----------|-------|
| NumCUs | 304 |
| NumXCDs | 8 |
| L1 Cache | 32 KB |
| L2 Cache | 256 MB |
| L3 Cache | 256 MB |
| HBM Bandwidth | ~6 TB/s |
| Math Frequency | ~1.5 GHz |
| Wavefront Size | 64 |

---

## 9. Source Code References

| Component | File Path |
|-----------|-----------|
| Parameter definitions | `tensilelite/Tensile/Common/ValidParameters.py` |
| Solution struct | `tensilelite/Tensile/SolutionStructs/Solution.py` |
| Assembly code generation | `tensilelite/Tensile/KernelWriterAssembly.py` |
| Scheduling algorithms | `tensilelite/Tensile/Components/SIA.py` |
| Cycle counting | `tensilelite/rocisa/rocisa/src/pass/cycle.cpp` |
| MFMA latencies | `tensilelite/rocisa/rocisa/include/instruction/mfma.hpp` |
| LDS instruction latencies | `tensilelite/rocisa/rocisa/include/instruction/mem.hpp` |
| Performance simulator | `shared/origami/.../formocast_simulator.hpp` |
| Static perf model | `tensilelite/src/ContractionSolution.cpp` |
