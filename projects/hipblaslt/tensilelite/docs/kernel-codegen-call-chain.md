# TensileLite: YAML Solution Parameters to Assembly Code Generation

This document describes the exact call chain, imports, and setup required to
generate assembly source code from a kernel recipe (YAML solution parameters)
using TensileLite's Python code generation infrastructure.

---

## 1. High-Level Call Chain

```
YAML logic file
  |
  v
LibraryIO.parseLibraryLogicFile()          # reads YAML, builds Solution objects
  |
  v
Solution(config_dict, ...)                 # constructs Solution from solution state dict
  |
  v
solution.getKernels()                      # returns [kernel] (kernel IS the Solution object)
  |
  v
KernelWriterAssembly.getSourceFileString(kernel)   # main entry point
  |
  v
KernelWriter._getKernelSource(kernel)      # calls _initKernel + kernelBody
  |
  v
KernelWriter._initKernel(kernel, tPA, tPB) # sets ISA via rocIsa, initializes state
  |
  v
KernelWriter.kernelBody(kernel, tPA, tPB)  # builds the Module tree (assembly IR)
  |
  v
str(KernelBody module)                     # serializes Module tree to assembly text
```

---

## 2. Key Classes and Files

| File | Class/Function | Role |
|------|---------------|------|
| `Tensile/KernelWriterAssembly.py` | `KernelWriterAssembly` | Subclass of `KernelWriter`; entry point is `getSourceFileString(kernel)` |
| `Tensile/KernelWriter.py` | `KernelWriter` (abstract) | Base class; contains `_getKernelSource()`, `_initKernel()`, `kernelBody()` |
| `Tensile/SolutionStructs/Solution.py` | `Solution` | `collections.abc.Mapping` subclass; constructed from a config dict; IS the kernel |
| `Tensile/SolutionStructs/Problem.py` | `ProblemType` | Parsed from YAML `ProblemType` section |
| `Tensile/Toolchain/Component.py` | `Assembler` | Wraps the ROCm assembler (amdclang); needed to construct `KernelWriterAssembly` |
| `Tensile/Toolchain/Assembly.py` | `makeAssemblyToolchain()` | Factory to create `AssemblyToolchain` from compiler paths |
| `Tensile/Common/Capabilities.py` | `makeIsaInfoMap()` | Calls `rocisa.rocIsa.getInstance().init(isa, cxxCompiler)` for each target ISA |
| `Tensile/Common/GlobalParameters.py` | `assignGlobalParameters()` | Must be called before kernel generation |
| `Tensile/LibraryIO.py` | `parseLibraryLogicFile()` | Reads YAML logic files, returns `Solution` objects |
| `Tensile/TensileCreateLibrary/Run.py` | `processKernelSource()` | Wrapper that calls `setRocIsa()` then `getSourceFileString()` |

---

## 3. Exact Function Signatures

### 3.1 `KernelWriterAssembly.__init__`
```python
def __init__(self, assembler: Assembler, debugConfig: DebugConfig):
```
- `assembler`: `Tensile.Toolchain.Component.Assembler` instance (wraps `amdclang++` path; constructed with `Assembler(Path(cxx_compiler), co_version)`)
- `debugConfig`: `Tensile.Common.DebugConfig` named tuple (all fields have defaults)

### 3.2 `KernelWriterAssembly.getSourceFileString` (main entry point)
```python
def getSourceFileString(self, kernel: Solution) -> Tuple[int, str]:
```
- `kernel`: a `Solution` object (which is also a `Mapping`)
- Returns `(error_code, assembly_source_text)`. Error code 0 means success.

### 3.3 `KernelWriter._getKernelSource` (internal, called by `getSourceFileString`)
```python
def _getKernelSource(self, kernel: Solution) -> str:
```
Calls `_initKernel()` then `kernelBody()`, converts the Module tree to a string.

### 3.4 `KernelWriter._initKernel`
```python
def _initKernel(self, kernel, tensorParametersA, tensorParametersB):
```
Sets the ISA on the `rocIsa` singleton via `rocIsa.getInstance().setKernel(version, wavefrontSize)`.

### 3.5 `KernelWriter.setRocIsa`
```python
def setRocIsa(self, data, outOptions):
```
Must be called before `getSourceFileString` in the production pipeline. Sets
`rocIsa` data and output options.

### 3.6 `Solution.__init__`
```python
def __init__(
    self,
    config,                            # dict of solution parameters from YAML
    splitGSU: bool,                    # typically False
    printSolutionRejectionReason: bool, # typically True
    printIndexAssignmentInfo: bool,     # typically False
    assembler: Assembler,              # Assembler instance
    isaInfoMap: Dict[IsaVersion, IsaInfo],  # from makeIsaInfoMap()
    srcName: str = ""                  # source file name (for diagnostics)
):
```

### 3.7 `processKernelSource` (production wrapper in Run.py)
```python
def processKernelSource(
    kernelWriterAssembly,  # KernelWriterAssembly instance
    data,                  # rocisa.rocIsa.getInstance().getData()
    outOptions,            # rocisa.rocIsa.getInstance().getOutputOptions()
    splitGSU,              # bool, typically False
    kernel,                # Solution (kernel) object
    compress=False         # whether to zlib-compress the result
) -> KernelCodeGenResult:
```

---

## 4. ISA/Architecture Setup Requirements

### 4.0 ROCm Toolchain Prerequisite

The `Assembler` class inherits from `Tensile.Toolchain.Component.Component`, which
calls `hipconfig --version` at **module import time** to determine `_rocm_version`.
This means ROCm must be installed and `hipconfig` must be in `$PATH` before any
import of `Tensile.Toolchain.Component`.  Similarly, the `Assembler` constructor
calls `amdclang++ --version` to extract the compiler version.

### 4.1 `rocisa` Singleton Initialization

The `rocisa` C++ extension module provides GPU ISA information. It must be
initialized before kernel generation:

```python
import rocisa

ti = rocisa.rocIsa.getInstance()

# For each target ISA, initialize capabilities:
isa = IsaVersion(9, 4, 2)  # e.g. gfx942
ti.init(isa, cxxCompiler_path, False)

# Query capabilities (used by makeIsaInfoMap):
isaInfo = ti.getIsaInfo(isa)
# isaInfo.asmCaps, isaInfo.archCaps, isaInfo.regCaps, isaInfo.asmBugs
```

### 4.2 Before Generating Assembly

Before calling `getSourceFileString()`, the production code calls:

```python
outOptions = rocisa.rocIsa.getInstance().getOutputOptions()
outOptions.outputNoComment = True  # or False for comments

kernelWriter.setRocIsa(
    rocisa.rocIsa.getInstance().getData(),
    outOptions
)
```

### 4.3 ISA on the Solution

The solution dict must have `"ISA"` set as an `IsaVersion` tuple, e.g.
`IsaVersion(9, 4, 2)` for gfx942. This is typically done by
`LibraryIO.parseLibraryLogicData()` which calls `gfxToIsa(architectureName)`.

---

## 5. Complete Steps: YAML Solution Parameters to Assembly Text

### Step 1: Set up the toolchain and ISA map

```python
from Tensile.Common.Architectures import gfxToIsa
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.GlobalParameters import assignGlobalParameters
from Tensile.Common import IsaVersion, DebugConfig
from Tensile.Toolchain.Component import Assembler
from Tensile.KernelWriterAssembly import KernelWriterAssembly
import rocisa

# Path to the ROCm C++ compiler (e.g., amdclang++)
cxx_compiler = "/opt/rocm/bin/amdclang++"
co_version = "4"  # code object version (default; see globalParameters["CodeObjectVersion"])

# Target ISA
isa = IsaVersion(9, 4, 2)  # gfx942

# Build ISA info map (initializes rocisa internally)
isaInfoMap = makeIsaInfoMap([isa], cxx_compiler)

# Assign global parameters (can pass empty dict if no overrides)
assignGlobalParameters({}, isaInfoMap)
```

### Step 2: Create the Assembler and KernelWriterAssembly

```python
from pathlib import Path

assembler = Assembler(Path(cxx_compiler), co_version)
debugConfig = DebugConfig()  # all defaults
kernelWriter = KernelWriterAssembly(assembler, debugConfig)
```

### Step 3: Create a Solution object from a config dict

```python
from Tensile.SolutionStructs import Solution

# This is the YAML "solution state" dict (kernel recipe)
config = {
    "ProblemType": {
        "OperationType": "GEMM",
        "DataType": "H",         # half precision
        "DestDataType": "H",
        "ComputeDataType": "S",  # float accumulate
        "HighPrecisionAccumulate": True,
        "TransposeA": False,
        "TransposeB": True,
        "UseBeta": True,
        "Batched": True,
    },
    "KernelLanguage": "Assembly",
    "ISA": [9, 4, 2],
    "MatrixInstruction": [16, 16, 16, 1],
    "DepthU": 32,
    "PrefetchGlobalRead": 2,
    "PrefetchLocalRead": 1,
    "ThreadTile": [4, 4],
    "WorkGroup": [16, 16, 1],
    # ... other solution parameters ...
}

solution = Solution(
    config,
    splitGSU=False,
    printSolutionRejectionReason=True,
    printIndexAssignmentInfo=False,
    assembler=assembler,
    isaInfoMap=isaInfoMap,
)
```

### Step 4: Get kernels from the solution

```python
kernels = solution.getKernels()
# kernels is [solution] - the Solution IS the kernel (it sets _state["Kernel"]=True)
kernel = kernels[0]

# The 'duplicate' attribute is normally set by writeSolutionsAndKernelsTCL() when
# iterating over multiple kernels.  getSourceFileString() checks kernel.duplicate
# and returns early with (0, "") if True.  For standalone codegen, always set False.
kernel.duplicate = False
```

### Step 5: Set rocIsa state and generate assembly

```python
ti = rocisa.rocIsa.getInstance()
outOptions = ti.getOutputOptions()
outOptions.outputNoComment = False  # include comments in assembly

kernelWriter.setRocIsa(ti.getData(), outOptions)

# Generate assembly source
error_code, assembly_text = kernelWriter.getSourceFileString(kernel)

if error_code == 0:
    print("Assembly generated successfully!")
    print(assembly_text[:500])  # first 500 chars
else:
    print(f"Error generating assembly: {error_code}")
```

---

## 6. Production Pipeline (TensileCreateLibrary)

In the production build (`TensileCreateLibrary/Run.py`), the flow is:

```
TensileCreateLibrary(arguments)
  |
  +-- validateToolchain()              # find amdclang++, offload-bundler
  +-- makeIsaInfoMap(targetIsas, cxx)  # init rocisa for each ISA
  +-- assignGlobalParameters({}, map)  # set global params
  +-- makeAssemblyToolchain(...)       # create Assembler, Linker, Bundler
  |
  +-- generateLogicDataAndSolutions()
  |     +-- LibraryIO.parseLibraryLogicFile()  # for each YAML
  |           +-- parseLibraryLogicData()
  |                 +-- Solution(solutionState, ...) for each solution in YAML
  |
  +-- generateKernelObjectsFromSolutions(solutions)
  |     +-- solution.getKernels() for each solution
  |
  +-- KernelWriterAssembly(assembler, DebugConfig())
  |
  +-- writeSolutionsAndKernelsTCL(...)
        +-- processKernelSource(writer, data, opts, splitGSU, kernel)
        |     +-- writer.setRocIsa(data, outOptions)
        |     +-- writer.getSourceFileString(kernel)
        |           +-- _getKernelSource(kernel)
        |                 +-- _initKernel(kernel, tPA, tPB)
        |                 +-- kernelBody(kernel, tPA, tPB) -> (error, Module)
        |                 +-- str(Module) -> assembly text
        |
        +-- writeAssembly(asmPath, result)    # writes .s file
        +-- assembler(gfx, wfsize, .s, .o)   # amdclang++ -x assembler
        +-- buildAssemblyCodeObjectFiles()    # link .o -> .co
```

---

## 7. Is There a Simple Single-Kernel Generation Test?

**No standalone test exists** that directly generates assembly from a solution
dict. The existing tests fall into two categories:

1. **Unit tests** (`Tests/unit/test_CustomSchedule.py`, etc.) - these test
   scheduling/validation logic using plain dicts (not full `Solution` objects)
   and never invoke `KernelWriterAssembly.getSourceFileString()`.

2. **Integration tests** (`Tests/common/*.yaml`) - these are YAML configs
   consumed by the full `Tensile` benchmarking pipeline which internally
   calls `TensileCreateLibrary`.

The closest to a "single kernel generation" script is the
`processKernelSource()` function in `TensileCreateLibrary/Run.py`, which is
the minimal wrapper around the codegen pipeline.

---

## 8. Library Logic YAML Format (Input to TensileCreateLibrary)

The YAML files consumed by `TensileCreateLibrary` have this structure:

```yaml
- {MinimumRequiredVersion: "5.0.0"}
- ScheduleName                      # e.g. "aldebaran"
- {Architecture: "gfx942", CUCount: 110}
- [Device 0049, Device 0050]        # device names
- {OperationType: GEMM, DataType: H, ...}  # ProblemType
- [                                  # Solutions list
    {SolutionIndex: 0, KernelLanguage: Assembly, ...},
    {SolutionIndex: 1, KernelLanguage: Assembly, ...},
  ]
- null                               # index 6 (unused)
- [[size_mapping_entries]]           # ExactLogic
- [[range_entries]]                  # RangeLogic
```

Or the newer dict-based format with keys:
`MinimumRequiredVersion`, `ScheduleName`, `ArchitectureName`, `DeviceNames`,
`ProblemType`, `Solutions`, `ExactLogic`, `RangeLogic`, `LibraryType`, etc.

---

## 9. Key Imports Summary

```python
# Core codegen
from Tensile.KernelWriterAssembly import KernelWriterAssembly
from Tensile.KernelWriter import KernelWriter
from Tensile.SolutionStructs import Solution
from Tensile.SolutionStructs.Problem import ProblemType

# Toolchain
from Tensile.Toolchain.Component import Assembler
from Tensile.Toolchain.Assembly import makeAssemblyToolchain

# ISA and capabilities
from Tensile.Common import IsaVersion, DebugConfig
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.Architectures import gfxToIsa
from Tensile.Common.GlobalParameters import assignGlobalParameters

# rocisa (C++ extension)
import rocisa
from rocisa import rocIsa

# Library I/O (for reading YAML logic files)
from Tensile import LibraryIO
```
