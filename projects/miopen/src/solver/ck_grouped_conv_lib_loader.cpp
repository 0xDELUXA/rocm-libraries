// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/solver/ck_grouped_conv_lib_loader.hpp>
#include <miopen/solver/ck_grouped_conv_interface.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/env.hpp>
#include <miopen/logger.hpp>

#include <cstdlib>
#include <dlfcn.h>

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_CK_LIB_PATH)

namespace miopen {
namespace solver {

namespace {

/// Strip architecture-specific suffixes like ":sramecc+:xnack-" from a device
/// name, returning only the base GPU identifier (e.g. "gfx90a").
std::string StripDeviceSuffix(const std::string& device_name)
{
    auto pos = device_name.find(':');
    if(pos != std::string::npos)
        return device_name.substr(0, pos);
    return device_name;
}

/// Build the expected shared library filename for a given device.
std::string MakeLibraryFilename(const std::string& device_name)
{
    return "libMIOpenCKGroupedConv_" + StripDeviceSuffix(device_name) + ".so";
}

/// Resolve the directory containing libMIOpen.so via dladdr.
/// Uses realpath to canonicalize symlinks so per-arch CK libraries
/// are found even when libMIOpen.so is accessed through a symlink.
std::string GetMIOpenLibDir()
{
    Dl_info info;
    if(dladdr(reinterpret_cast<void*>(miopenCreate), &info) != 0)
    {
        // Canonicalize to resolve symlinks (e.g. /opt/rocm/lib -> /opt/rocm-X.Y.Z/lib)
        char* real = realpath(info.dli_fname, nullptr);
        std::string path(real ? real : info.dli_fname);
        free(real);
        auto slash = path.rfind('/');
        if(slash != std::string::npos)
            return path.substr(0, slash);
    }
    return {};
}

} // namespace

// -- Singleton infrastructure -------------------------------------------------

std::mutex& CKGroupedConvLibLoader::CacheMutex()
{
    static std::mutex mtx;
    return mtx;
}

std::unordered_map<std::string, std::unique_ptr<CKGroupedConvLibLoader>>&
CKGroupedConvLibLoader::Cache()
{
    static std::unordered_map<std::string, std::unique_ptr<CKGroupedConvLibLoader>> cache;
    return cache;
}

const CKGroupedConvLibLoader& CKGroupedConvLibLoader::Get(const std::string& device_name)
{
    const auto key = StripDeviceSuffix(device_name);
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto& cache = Cache();
    auto it     = cache.find(key);
    if(it == cache.end())
    {
        // Use new + reset instead of make_unique because the constructor is private.
        std::unique_ptr<CKGroupedConvLibLoader> ptr(new CKGroupedConvLibLoader(device_name));
        it = cache.emplace(key, std::move(ptr)).first;
    }
    return *it->second;
}

// -- Construction / Destruction -----------------------------------------------

CKGroupedConvLibLoader::CKGroupedConvLibLoader(const std::string& device_name)
{
    LoadLibrary(device_name);
}

CKGroupedConvLibLoader::~CKGroupedConvLibLoader()
{
    // RTLD_NODELETE keeps the library mapped, so dlclose only decrements the
    // reference count without unmapping.  We still call it for correctness.
    if(lib_handle_ != nullptr)
        dlclose(lib_handle_);
}

// -- Library loading ----------------------------------------------------------

void CKGroupedConvLibLoader::LoadLibrary(const std::string& device_name)
{
    const auto filename = MakeLibraryFilename(device_name);
    constexpr int flags = RTLD_NOW | RTLD_NODELETE;

    // 1. Try MIOPEN_CK_LIB_PATH environment variable
    const auto env_path = env::value(MIOPEN_CK_LIB_PATH);
    if(!env_path.empty())
    {
        auto full_path = env_path + "/" + filename;
        lib_handle_    = dlopen(full_path.c_str(), flags);
        if(lib_handle_ != nullptr)
        {
            MIOPEN_LOG_I2("Loaded CK grouped conv library from env path: " << full_path);
        }
    }

    // 2. Try the directory containing libMIOpen.so
    if(lib_handle_ == nullptr)
    {
        auto lib_dir = GetMIOpenLibDir();
        if(!lib_dir.empty())
        {
            auto full_path = lib_dir + "/" + filename;
            lib_handle_    = dlopen(full_path.c_str(), flags);
            if(lib_handle_ != nullptr)
            {
                MIOPEN_LOG_I2("Loaded CK grouped conv library from lib dir: " << full_path);
            }
        }
    }

    // 3. Fall back to default search path
    if(lib_handle_ == nullptr)
    {
        lib_handle_ = dlopen(filename.c_str(), flags);
        if(lib_handle_ != nullptr)
        {
            MIOPEN_LOG_I2("Loaded CK grouped conv library from default path: " << filename);
        }
    }

    if(lib_handle_ == nullptr)
    {
        MIOPEN_LOG_W("CK grouped conv library not found for device "
                     << StripDeviceSuffix(device_name) << ": " << dlerror());
        return;
    }

    if(!LoadSymbols())
    {
        MIOPEN_LOG_W("Failed to resolve symbols in CK grouped conv library for device "
                     << StripDeviceSuffix(device_name));
        loaded_ = false;
        return;
    }

    // API version check
    const int lib_version = get_api_version_fn_();
    if(lib_version != CK_GROUPED_CONV_API_VERSION)
    {
        MIOPEN_LOG_W("CK grouped conv API version mismatch for device "
                     << StripDeviceSuffix(device_name) << ": expected "
                     << CK_GROUPED_CONV_API_VERSION << ", got " << lib_version);
        loaded_ = false;
        return;
    }

    loaded_ = true;
}

// -- Symbol resolution --------------------------------------------------------

bool CKGroupedConvLibLoader::LoadSymbols()
{
    // Helper macro: resolve a symbol or return false on failure.
#define LOAD_SYM(member, name)                                                  \
    do                                                                          \
    {                                                                           \
        member = reinterpret_cast<decltype(member)>(dlsym(lib_handle_, #name)); \
        if(member == nullptr)                                                   \
        {                                                                       \
            MIOPEN_LOG_W("dlsym failed for " #name ": " << dlerror());          \
            return false;                                                       \
        }                                                                       \
    } while(false)

    // Common
    LOAD_SYM(get_api_version_fn_, ckgrpconv_get_api_version);
    LOAD_SYM(kernel_list_size_fn_, ckgrpconv_kernel_list_size);
    LOAD_SYM(kernel_list_get_fn_, ckgrpconv_kernel_list_get);
    LOAD_SYM(kernel_list_free_fn_, ckgrpconv_kernel_list_free);
    LOAD_SYM(solution_free_fn_, ckgrpconv_solution_free);

    // FWD
    LOAD_SYM(fwd_fill_valid_kernels_fn_, ckgrpconv_fwd_fill_valid_kernels);
    LOAD_SYM(fwd_is_applicable_fn_, ckgrpconv_fwd_is_applicable);
    LOAD_SYM(fwd_is_args_supported_fn_, ckgrpconv_fwd_is_args_supported);
    LOAD_SYM(fwd_get_workspace_size_fn_, ckgrpconv_fwd_get_workspace_size);
    LOAD_SYM(fwd_get_solution_fn_, ckgrpconv_fwd_get_solution);

    // BWD
    LOAD_SYM(bwd_fill_valid_kernels_fn_, ckgrpconv_bwd_fill_valid_kernels);
    LOAD_SYM(bwd_is_applicable_fn_, ckgrpconv_bwd_is_applicable);
    LOAD_SYM(bwd_is_args_supported_fn_, ckgrpconv_bwd_is_args_supported);
    LOAD_SYM(bwd_get_workspace_size_fn_, ckgrpconv_bwd_get_workspace_size);
    LOAD_SYM(bwd_get_solution_fn_, ckgrpconv_bwd_get_solution);

    // WRW
    LOAD_SYM(wrw_fill_valid_kernels_fn_, ckgrpconv_wrw_fill_valid_kernels);
    LOAD_SYM(wrw_is_applicable_fn_, ckgrpconv_wrw_is_applicable);
    LOAD_SYM(wrw_is_args_supported_fn_, ckgrpconv_wrw_is_args_supported);
    LOAD_SYM(wrw_get_workspace_size_fn_, ckgrpconv_wrw_get_workspace_size);
    LOAD_SYM(wrw_get_solution_fn_, ckgrpconv_wrw_get_solution);

#undef LOAD_SYM
    return true;
}

// -- Helpers ------------------------------------------------------------------

std::vector<std::string>
CKGroupedConvLibLoader::ExtractKernelList(CKKernelListHandle* handle) const
{
    if(handle == nullptr)
        return {};
    std::vector<std::string> result;
    const size_t n = kernel_list_size_fn_(handle);
    result.reserve(n);
    for(size_t i = 0; i < n; ++i)
    {
        const char* s = kernel_list_get_fn_(handle, i);
        if(s != nullptr)
            result.emplace_back(s);
    }
    kernel_list_free_fn_(handle);
    return result;
}

ConvSolution CKGroupedConvLibLoader::ExtractSolution(ConvSolution* ptr) const
{
    if(ptr == nullptr)
        return ConvSolution{miopenStatusInternalError};
    ConvSolution result = std::move(*ptr);
    solution_free_fn_(ptr);
    return result;
}

// -- FWD wrappers -------------------------------------------------------------

std::vector<std::string>
CKGroupedConvLibLoader::fwd_fill_valid_kernels(const conv::ProblemDescription& problem,
                                               miopenDataType_t dtype,
                                               bool use_tf32) const
{
    if(!IsLoaded())
        return {};
    return ExtractKernelList(fwd_fill_valid_kernels_fn_(&problem, dtype, use_tf32));
}

bool CKGroupedConvLibLoader::fwd_is_applicable(const conv::ProblemDescription& problem,
                                               miopenDataType_t dtype,
                                               bool use_tf32) const
{
    if(!IsLoaded())
        return false;
    return fwd_is_applicable_fn_(&problem, dtype, use_tf32);
}

bool CKGroupedConvLibLoader::fwd_is_args_supported(const conv::ProblemDescription& problem,
                                                   const std::string& kernel_id,
                                                   miopenDataType_t dtype,
                                                   bool use_tf32) const
{
    if(!IsLoaded())
        return false;
    return fwd_is_args_supported_fn_(&problem, kernel_id.c_str(), dtype, use_tf32);
}

size_t CKGroupedConvLibLoader::fwd_get_workspace_size(const conv::ProblemDescription& problem,
                                                      miopenDataType_t dtype) const
{
    if(!IsLoaded())
        return 0;
    return fwd_get_workspace_size_fn_(&problem, dtype);
}

ConvSolution CKGroupedConvLibLoader::fwd_get_solution(const ExecutionContext& ctx,
                                                      const conv::ProblemDescription& problem,
                                                      const std::string& kernel_id,
                                                      bool use_tf32) const
{
    if(!IsLoaded())
        return ConvSolution{miopenStatusInternalError};
    return ExtractSolution(fwd_get_solution_fn_(&ctx, &problem, kernel_id.c_str(), use_tf32));
}

// -- BWD wrappers -------------------------------------------------------------

std::vector<std::string>
CKGroupedConvLibLoader::bwd_fill_valid_kernels(const conv::ProblemDescription& problem,
                                               miopenDataType_t dtype,
                                               bool use_tf32) const
{
    if(!IsLoaded())
        return {};
    return ExtractKernelList(bwd_fill_valid_kernels_fn_(&problem, dtype, use_tf32));
}

bool CKGroupedConvLibLoader::bwd_is_applicable(const conv::ProblemDescription& problem,
                                               miopenDataType_t dtype,
                                               bool use_tf32) const
{
    if(!IsLoaded())
        return false;
    return bwd_is_applicable_fn_(&problem, dtype, use_tf32);
}

bool CKGroupedConvLibLoader::bwd_is_args_supported(const conv::ProblemDescription& problem,
                                                   const std::string& kernel_id,
                                                   miopenDataType_t dtype,
                                                   bool use_tf32) const
{
    if(!IsLoaded())
        return false;
    return bwd_is_args_supported_fn_(&problem, kernel_id.c_str(), dtype, use_tf32);
}

size_t CKGroupedConvLibLoader::bwd_get_workspace_size(const conv::ProblemDescription& problem,
                                                      miopenDataType_t dtype) const
{
    if(!IsLoaded())
        return 0;
    return bwd_get_workspace_size_fn_(&problem, dtype);
}

ConvSolution CKGroupedConvLibLoader::bwd_get_solution(const ExecutionContext& ctx,
                                                      const conv::ProblemDescription& problem,
                                                      const std::string& kernel_id,
                                                      bool use_tf32) const
{
    if(!IsLoaded())
        return ConvSolution{miopenStatusInternalError};
    return ExtractSolution(bwd_get_solution_fn_(&ctx, &problem, kernel_id.c_str(), use_tf32));
}

// -- WRW wrappers -------------------------------------------------------------

std::vector<std::string>
CKGroupedConvLibLoader::wrw_fill_valid_kernels(const conv::ProblemDescription& problem,
                                               miopenDataType_t dtype,
                                               bool use_tf32) const
{
    if(!IsLoaded())
        return {};
    return ExtractKernelList(wrw_fill_valid_kernels_fn_(&problem, dtype, use_tf32));
}

bool CKGroupedConvLibLoader::wrw_is_applicable(const conv::ProblemDescription& problem,
                                               miopenDataType_t dtype,
                                               bool use_tf32) const
{
    if(!IsLoaded())
        return false;
    return wrw_is_applicable_fn_(&problem, dtype, use_tf32);
}

bool CKGroupedConvLibLoader::wrw_is_args_supported(const conv::ProblemDescription& problem,
                                                   const std::string& kernel_id,
                                                   miopenDataType_t dtype,
                                                   bool use_tf32) const
{
    if(!IsLoaded())
        return false;
    return wrw_is_args_supported_fn_(&problem, kernel_id.c_str(), dtype, use_tf32);
}

size_t CKGroupedConvLibLoader::wrw_get_workspace_size(const conv::ProblemDescription& problem,
                                                      miopenDataType_t dtype) const
{
    if(!IsLoaded())
        return 0;
    return wrw_get_workspace_size_fn_(&problem, dtype);
}

ConvSolution CKGroupedConvLibLoader::wrw_get_solution(const ExecutionContext& ctx,
                                                      const conv::ProblemDescription& problem,
                                                      const std::string& kernel_id,
                                                      bool use_tf32) const
{
    if(!IsLoaded())
        return ConvSolution{miopenStatusInternalError};
    return ExtractSolution(wrw_get_solution_fn_(&ctx, &problem, kernel_id.c_str(), use_tf32));
}

} // namespace solver
} // namespace miopen
