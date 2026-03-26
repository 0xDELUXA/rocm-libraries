// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/numeric/e8m0.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
// #include "ck_tile/core/numeric/pk_fp6.hpp"

#include <cstdint>
#include <type_traits>
#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER
#include <concepts>
#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

namespace ck_tile::core::arch::mma {

template <typename T>
struct TypeToFlagValue;

template <>
struct TypeToFlagValue<fp8_t> // e4m3
{
    static constexpr std::uint8_t value = 0;
};

template <>
struct TypeToFlagValue<bf8_t> // e5m2
{
    static constexpr std::uint8_t value = 1;
};

// template <>
// struct TypeToFlagValue<pk_fp6_t<1>> // e2m3
// {
//     static constexpr std::uint8_t value = 2;
// };

// template <>
// struct TypeToFlagValue<bf6_t> // e3m2
// {
//     static constexpr std::uint8_t value = 3;
// };

template <>
struct TypeToFlagValue<pk_fp4_t> // e2m1
{
    static constexpr std::uint8_t value = 4;
};

template <typename T>
inline constexpr std::uint8_t TypeToFlagValue_v = TypeToFlagValue<T>::value;

template <typename ADataType, typename BDataType, std::uint8_t OPSELA = 0, std::uint8_t OPSELB = 0>
struct DefaultScaleMfmaCtrlFlags
{
    static constexpr std::uint8_t type_A  = TypeToFlagValue_v<ADataType>;
    static constexpr std::uint8_t type_B  = TypeToFlagValue_v<BDataType>;
    static constexpr std::uint8_t OPSEL_A = OPSELA;
    static constexpr std::uint8_t OPSEL_B = OPSELB;
};

#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

/**
 * @concept ScaleMfmaCtrlFlags
 * @brief  Expresses the interface of required members for each CtrlFlags type on Gfx9
 */
template <typename CtrlFlags>
concept ScaleMfmaCtrlFlags = requires(CtrlFlags ctrlFlags) {
    // Flag members for scale MFMA instructions
    { CtrlFlags::type_A } -> std::convertible_to<int>;
    { CtrlFlags::type_B } -> std::convertible_to<int>;
    { CtrlFlags::OPSEL_A } -> std::convertible_to<int>;
    { CtrlFlags::OPSEL_B } -> std::convertible_to<int>;
};

#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

template <typename T>
struct ScaleTypeToFlagValue;

template <>
struct ScaleTypeToFlagValue<e8m0_t> // e8m0
{
    static constexpr std::uint8_t value = 0;
};

// template <>
// struct ScaleTypeToFlagValue<e5m3_t> // e5m3
// {
//     static constexpr std::uint8_t value = 1;
// };

template <>
struct ScaleTypeToFlagValue<fp8_t> // e4m3
{
    static constexpr std::uint8_t value = 2;
};

template <typename T>
inline constexpr std::uint8_t ScaleTypeToFlagValue_v = ScaleTypeToFlagValue<T>::value;

template <typename ADataType,
          typename BDataType,
          typename ScaleADataType,
          typename ScaleBDataType,
          std::uint8_t OPSELA = 0,
          std::uint8_t OPSELB = 0>
struct DefaultScaleWmmaCtrlFlags
{
    static_assert(
        (std::is_same_v<ScaleADataType, e8m0_t> && std::is_same_v<ScaleBDataType, e8m0_t>) ||
            (std::is_same_v<ScaleADataType, e8m0_t> && !std::is_same_v<ADataType, pk_fp4_t> &&
             std::is_same_v<BDataType, pk_fp4_t>) ||
            (std::is_same_v<ScaleBDataType, e8m0_t> && std::is_same_v<ADataType, pk_fp4_t> &&
             !std::is_same_v<BDataType, pk_fp4_t>) ||
            (std::is_same_v<ScaleADataType, ScaleBDataType> &&
             std::is_same_v<ADataType, pk_fp4_t> && std::is_same_v<BDataType, pk_fp4_t>),
        "The combination of ADataType, BDataType, ScaleADataType and ScaleBDataType is invalid.");

    static constexpr std::uint8_t type_A       = TypeToFlagValue_v<ADataType>;
    static constexpr std::uint8_t type_B       = TypeToFlagValue_v<BDataType>;
    static constexpr std::uint8_t type_scale_A = ScaleTypeToFlagValue_v<ScaleADataType>;
    static constexpr std::uint8_t type_scale_B = ScaleTypeToFlagValue_v<ScaleBDataType>;
    static constexpr std::uint8_t OPSEL_A      = OPSELA;
    static constexpr std::uint8_t OPSEL_B      = OPSELB;
};

#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

/**
 * @concept ScaleMfmaCtrlFlags
 * @brief  Expresses the interface of required members for each CtrlFlags type on Gfx9
 */
template <typename CtrlFlags>
concept ScaleWmmaCtrlFlags = requires(CtrlFlags ctrlFlags) {
    // Flag members for scale MFMA instructions
    { CtrlFlags::type_A } -> std::convertible_to<int>;
    { CtrlFlags::type_B } -> std::convertible_to<int>;
    { CtrlFlags::type_scale_A } -> std::convertible_to<int>;
    { CtrlFlags::type_scale_B } -> std::convertible_to<int>;
    { CtrlFlags::OPSEL_A } -> std::convertible_to<int>;
    { CtrlFlags::OPSEL_B } -> std::convertible_to<int>;
};

#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

} // namespace ck_tile::core::arch::mma
