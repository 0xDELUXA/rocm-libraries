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

struct CKKernelListHandle;

namespace miopen {
struct ExecutionContext;
namespace conv {
struct ProblemDescription;
} // namespace conv
namespace solver {
struct ConvSolution;

class CKGroupedConvLibLoader
{
public:
    /// Thread-safe accessor returning a cached per-device singleton.
    MIOPEN_INTERNALS_EXPORT static const CKGroupedConvLibLoader& Get(const std::string& device_name);

    MIOPEN_INTERNALS_EXPORT bool IsLoaded() const { return loaded_; }

    // -- FWD wrappers ---------------------------------------------------------
    MIOPEN_INTERNALS_EXPORT std::vector<std::string>
    fwd_fill_valid_kernels(const miopen::conv::ProblemDescription& problem,
                           miopenDataType_t dtype,
                           bool use_tf32) const;

    MIOPEN_INTERNALS_EXPORT bool fwd_is_applicable(const miopen::conv::ProblemDescription& problem,
                                                   miopenDataType_t dtype,
                                                   bool use_tf32) const;

    MIOPEN_INTERNALS_EXPORT bool fwd_is_args_supported(const miopen::conv::ProblemDescription& problem,
                                                       const std::string& kernel_id,
                                                       miopenDataType_t dtype,
                                                       bool use_tf32) const;

    MIOPEN_INTERNALS_EXPORT size_t
    fwd_get_workspace_size(const miopen::conv::ProblemDescription& problem,
                           miopenDataType_t dtype) const;

    MIOPEN_INTERNALS_EXPORT ConvSolution fwd_get_solution(const ExecutionContext& ctx,
                                                          const miopen::conv::ProblemDescription& problem,
                                                          const std::string& kernel_id,
                                                          bool use_tf32) const;

    // -- BWD wrappers ---------------------------------------------------------
    MIOPEN_INTERNALS_EXPORT std::vector<std::string>
    bwd_fill_valid_kernels(const miopen::conv::ProblemDescription& problem,
                           miopenDataType_t dtype,
                           bool use_tf32) const;

    MIOPEN_INTERNALS_EXPORT bool bwd_is_applicable(const miopen::conv::ProblemDescription& problem,
                                                   miopenDataType_t dtype,
                                                   bool use_tf32) const;

    MIOPEN_INTERNALS_EXPORT bool bwd_is_args_supported(const miopen::conv::ProblemDescription& problem,
                                                       const std::string& kernel_id,
                                                       miopenDataType_t dtype,
                                                       bool use_tf32) const;

    MIOPEN_INTERNALS_EXPORT size_t
    bwd_get_workspace_size(const miopen::conv::ProblemDescription& problem,
                           miopenDataType_t dtype) const;

    MIOPEN_INTERNALS_EXPORT ConvSolution bwd_get_solution(const ExecutionContext& ctx,
                                                          const miopen::conv::ProblemDescription& problem,
                                                          const std::string& kernel_id,
                                                          bool use_tf32) const;

    // -- WRW wrappers ---------------------------------------------------------
    MIOPEN_INTERNALS_EXPORT std::vector<std::string>
    wrw_fill_valid_kernels(const miopen::conv::ProblemDescription& problem,
                           miopenDataType_t dtype,
                           bool use_tf32) const;

    MIOPEN_INTERNALS_EXPORT bool wrw_is_applicable(const miopen::conv::ProblemDescription& problem,
                                                   miopenDataType_t dtype,
                                                   bool use_tf32) const;

    MIOPEN_INTERNALS_EXPORT bool wrw_is_args_supported(const miopen::conv::ProblemDescription& problem,
                                                       const std::string& kernel_id,
                                                       miopenDataType_t dtype,
                                                       bool use_tf32) const;

    MIOPEN_INTERNALS_EXPORT size_t
    wrw_get_workspace_size(const miopen::conv::ProblemDescription& problem,
                           miopenDataType_t dtype) const;

    MIOPEN_INTERNALS_EXPORT ConvSolution wrw_get_solution(const ExecutionContext& ctx,
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
    bool loaded_       = false;

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
    using IsApplicableFn     = bool (*)(const miopen::conv::ProblemDescription*, miopenDataType_t, bool);
    using IsArgsSupportedFn  = bool (*)(const miopen::conv::ProblemDescription*,
                                       const char*,
                                       miopenDataType_t,
                                       bool);
    using GetWorkspaceSizeFn = size_t (*)(const miopen::conv::ProblemDescription*, miopenDataType_t);
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

    // FWD
    FillValidKernelsFn fwd_fill_valid_kernels_fn_ = nullptr;
    IsApplicableFn fwd_is_applicable_fn_          = nullptr;
    IsArgsSupportedFn fwd_is_args_supported_fn_   = nullptr;
    GetWorkspaceSizeFn fwd_get_workspace_size_fn_ = nullptr;
    GetSolutionFn fwd_get_solution_fn_            = nullptr;

    // BWD
    FillValidKernelsFn bwd_fill_valid_kernels_fn_ = nullptr;
    IsApplicableFn bwd_is_applicable_fn_          = nullptr;
    IsArgsSupportedFn bwd_is_args_supported_fn_   = nullptr;
    GetWorkspaceSizeFn bwd_get_workspace_size_fn_ = nullptr;
    GetSolutionFn bwd_get_solution_fn_            = nullptr;

    // WRW
    FillValidKernelsFn wrw_fill_valid_kernels_fn_ = nullptr;
    IsApplicableFn wrw_is_applicable_fn_          = nullptr;
    IsArgsSupportedFn wrw_is_args_supported_fn_   = nullptr;
    GetWorkspaceSizeFn wrw_get_workspace_size_fn_ = nullptr;
    GetSolutionFn wrw_get_solution_fn_            = nullptr;

    // Helper: extract kernel list from handle
    std::vector<std::string> ExtractKernelList(::CKKernelListHandle* handle) const;

    // Helper: extract ConvSolution from pointer
    ConvSolution ExtractSolution(ConvSolution* ptr) const;
};

} // namespace solver
} // namespace miopen
