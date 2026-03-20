// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <miopen/config.hpp>
#include <miopen/miopen.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if MIOPEN_BACKEND_HIP
#include <hip/hip_runtime_api.h>
#endif

struct CKKernelListHandle;

namespace miopen {
struct ExecutionContext;
namespace conv {
struct ProblemDescription;
} // namespace conv
namespace solver {
struct ConvSolution;

enum class CKConvDirection { Fwd = 0, Bwd = 1, Wrw = 2 };

/// Query the HIP runtime for the current device's architecture name.
/// Returns an empty string on failure or when not using the HIP backend.
inline std::string GetCurrentDeviceName()
{
#if MIOPEN_BACKEND_HIP
    int device = 0;
    if(hipGetDevice(&device) != hipSuccess)
        return {};
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, device) != hipSuccess)
        return {};
    return std::string(props.gcnArchName);
#else
    return {};
#endif
}

class CKGroupedConvLibLoader
{
public:
    /// Thread-safe accessor returning a cached per-device singleton.
    MIOPEN_INTERNALS_EXPORT static const CKGroupedConvLibLoader&
    Get(const std::string& device_name);

    MIOPEN_INTERNALS_EXPORT bool IsLoaded() const { return loaded_; }

    // -- Direction-parameterized wrappers -------------------------------------
    MIOPEN_INTERNALS_EXPORT std::vector<std::string>
    fill_valid_kernels(CKConvDirection dir,
                       const miopen::conv::ProblemDescription& problem,
                       miopenDataType_t dtype,
                       bool use_tf32) const;

    MIOPEN_INTERNALS_EXPORT bool
    is_applicable(CKConvDirection dir,
                  const miopen::conv::ProblemDescription& problem,
                  miopenDataType_t dtype,
                  bool use_tf32) const;

    MIOPEN_INTERNALS_EXPORT bool
    is_args_supported(CKConvDirection dir,
                      const miopen::conv::ProblemDescription& problem,
                      const std::string& kernel_id,
                      miopenDataType_t dtype,
                      bool use_tf32) const;

    MIOPEN_INTERNALS_EXPORT size_t
    get_workspace_size(CKConvDirection dir,
                       const miopen::conv::ProblemDescription& problem,
                       miopenDataType_t dtype) const;

    MIOPEN_INTERNALS_EXPORT ConvSolution
    get_solution(CKConvDirection dir,
                 const ExecutionContext& ctx,
                 const miopen::conv::ProblemDescription& problem,
                 const std::string& kernel_id,
                 bool use_tf32) const;

    ~CKGroupedConvLibLoader();

    CKGroupedConvLibLoader(const CKGroupedConvLibLoader&)            = delete;
    CKGroupedConvLibLoader& operator=(const CKGroupedConvLibLoader&) = delete;

private:
    explicit CKGroupedConvLibLoader(const std::string& device_name);

    void LoadLibrary(const std::string& device_name);
    bool LoadSymbols();

    // Singleton cache
    static std::mutex& CacheMutex();
    static std::unordered_map<std::string, std::unique_ptr<CKGroupedConvLibLoader>>& Cache();

    void* lib_handle_ = nullptr;
    bool loaded_      = false;

    // -- Function pointer types -----------------------------------------------
    // CKKernelListHandle is declared at global scope in the interface header.

    using GetApiVersionFn = int (*)();

    using KernelListSizeFn = size_t (*)(const ::CKKernelListHandle*);
    using KernelListGetFn  = const char* (*)(const ::CKKernelListHandle*, size_t);
    using KernelListFreeFn = void (*)(::CKKernelListHandle*);

    using SolutionFreeFn = void (*)(ConvSolution*);

    using FillValidKernelsFn = ::CKKernelListHandle* (*)(const miopen::conv::ProblemDescription*,
                                                         miopenDataType_t,
                                                         bool);
    using IsApplicableFn     = bool (*)(const miopen::conv::ProblemDescription*,
                                    miopenDataType_t,
                                    bool);
    using IsArgsSupportedFn  = bool (*)(const miopen::conv::ProblemDescription*,
                                       const char*,
                                       miopenDataType_t,
                                       bool);
    using GetWorkspaceSizeFn = size_t (*)(const miopen::conv::ProblemDescription*,
                                          miopenDataType_t);
    using GetSolutionFn      = ConvSolution* (*)(const ExecutionContext*,
                                            const miopen::conv::ProblemDescription*,
                                            const char*,
                                            bool);

    // -- Function pointers ----------------------------------------------------
    GetApiVersionFn get_api_version_fn_ = nullptr;

    KernelListSizeFn kernel_list_size_fn_ = nullptr;
    KernelListGetFn kernel_list_get_fn_   = nullptr;
    KernelListFreeFn kernel_list_free_fn_ = nullptr;

    SolutionFreeFn solution_free_fn_ = nullptr;

    struct DirectionFns
    {
        FillValidKernelsFn fill_valid_kernels = nullptr;
        IsApplicableFn is_applicable          = nullptr;
        IsArgsSupportedFn is_args_supported   = nullptr;
        GetWorkspaceSizeFn get_workspace_size = nullptr;
        GetSolutionFn get_solution            = nullptr;
    };

    DirectionFns dir_fns_[3]; // indexed by static_cast<int>(CKConvDirection)

    // Helper: extract kernel list from handle
    std::vector<std::string> ExtractKernelList(::CKKernelListHandle* handle) const;

    // Helper: extract ConvSolution from pointer
    ConvSolution ExtractSolution(ConvSolution* ptr) const;
};

} // namespace solver
} // namespace miopen
