/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FRAMEWORK_DATA_TYPES_HPP
#define FRAMEWORK_DATA_TYPES_HPP

#include <cstdint>

#include <library_types.h> // Still needed for CUDA_C_XYZ enums
#include <vector_types.h>  // Still needed for int2, char2, etc.

#include <cuda/std/complex> // Modern replacement for cuComplex.h  // NOLINT(misc-unused-includes)
#include <cuda_fp16.h>

namespace framework::tensor {

/**
 * @brief Data type enumeration for NV operations
 *
 * This enumeration defines the supported data types for NV operations,
 * including various integer, floating-point, and complex number formats.
 * The values are compatible with CUDA library types where applicable.
 *
 * @see CUDA library types for compatibility information
 */
// clang-format off
 enum NvDataType : std::int8_t
 {
     TensorVoid  = -1,         /*!< uninitialized type                       */
     TensorBit   = 20,         /*!< 1-bit value                              */
     TensorR8I   = CUDA_R_8I,  /*!< 8-bit signed integer real values         */
     TensorC8I   = CUDA_C_8I,  /*!< 8-bit signed integer complex values      */
     TensorR8U   = CUDA_R_8U,  /*!< 8-bit unsigned integer real values       */
     TensorC8U   = CUDA_C_8U,  /*!< 8-bit unsigned integer complex values    */
     TensorR16I  = 21,         /*!< 16-bit signed integer real values        */
     TensorC16I  = 22,         /*!< 16-bit signed integer complex values     */
     TensorR16U  = 23,         /*!< 16-bit unsigned integer real values      */
     TensorC16U  = 24,         /*!< 16-bit unsigned integer complex values   */
     TensorR32I  = CUDA_R_32I, /*!< 32-bit signed integer real values        */
     TensorC32I  = CUDA_C_32I, /*!< 32-bit signed integer complex values     */
     TensorR32U  = CUDA_R_32U, /*!< 32-bit unsigned integer real values      */
     TensorC32U  = CUDA_C_32U, /*!< 32-bit unsigned integer complex values   */
     TensorR16F  = CUDA_R_16F, /*!< half precision (16-bit) real values      */
     TensorC16F  = CUDA_C_16F, /*!< half precision (16-bit) complex values   */
     TensorR32F  = CUDA_R_32F, /*!< single precision (32-bit) real values    */
     TensorC32F  = CUDA_C_32F, /*!< single precision (32-bit) complex values */
     TensorR64F  = CUDA_R_64F, /*!< double precision (64-bit) real values    */
     TensorC64F  = CUDA_C_64F  /*!< double precision (64-bit) complex values */
 };
 
 static_assert(TensorVoid  >= INT8_MIN && TensorVoid  <= INT8_MAX,  "TensorVoid out of range");
 static_assert(TensorBit   >= INT8_MIN && TensorBit   <= INT8_MAX,  "TensorBit out of range");
 static_assert(TensorR8I   >= INT8_MIN && TensorR8I   <= INT8_MAX,  "TensorR8I out of range");
 static_assert(TensorC8I   >= INT8_MIN && TensorC8I   <= INT8_MAX,  "TensorC8I out of range");
 static_assert(TensorR8U   >= INT8_MIN && TensorR8U   <= INT8_MAX,  "TensorR8U out of range");
 static_assert(TensorC8U   >= INT8_MIN && TensorC8U   <= INT8_MAX,  "TensorC8U out of range");
 static_assert(TensorR16I  >= INT8_MIN && TensorR16I  <= INT8_MAX,  "TensorR16I out of range");
 static_assert(TensorC16I  >= INT8_MIN && TensorC16I  <= INT8_MAX,  "TensorC16I out of range");
 static_assert(TensorR16U  >= INT8_MIN && TensorR16U  <= INT8_MAX,  "TensorR16U out of range");
 static_assert(TensorC16U  >= INT8_MIN && TensorC16U  <= INT8_MAX,  "TensorC16U out of range");
 static_assert(TensorR32I  >= INT8_MIN && TensorR32I  <= INT8_MAX,  "TensorR32I out of range");
 static_assert(TensorC32I  >= INT8_MIN && TensorC32I  <= INT8_MAX,  "TensorC32I out of range");
 static_assert(TensorR32U  >= INT8_MIN && TensorR32U  <= INT8_MAX,  "TensorR32U out of range");
 static_assert(TensorC32U  >= INT8_MIN && TensorC32U  <= INT8_MAX,  "TensorC32U out of range");
 static_assert(TensorR16F  >= INT8_MIN && TensorR16F  <= INT8_MAX,  "TensorR16F out of range");
 static_assert(TensorC16F  >= INT8_MIN && TensorC16F  <= INT8_MAX,  "TensorC16F out of range");
 static_assert(TensorR32F  >= INT8_MIN && TensorR32F  <= INT8_MAX,  "TensorR32F out of range");
 static_assert(TensorC32F  >= INT8_MIN && TensorC32F  <= INT8_MAX,  "TensorC32F out of range");
 static_assert(TensorR64F  >= INT8_MIN && TensorR64F  <= INT8_MAX,  "TensorR64F out of range");
 static_assert(TensorC64F  >= INT8_MIN && TensorC64F  <= INT8_MAX,  "TensorC64F out of range");

// clang-format on

/**
 * Get string representation of NV data type
 *
 * This function returns a human-readable string representation of the given
 * NV data type enumeration value. Useful for debugging, logging, and
 * error reporting.
 *
 * @param[in] type The NV data type to convert to string
 * @return const char* String representation of the data type
 * @retval "TensorVoid" for uninitialized type
 * @retval "TensorBit" for 1-bit values
 * @retval "TensorR8I", "TensorR8U", "TensorR16I", "TensorR16U", "TensorR32I",
 * "TensorR32U" for integer real number types
 * @retval "TensorR16F", "TensorR32F", "TensorR64F" for floating-point real
 * number types
 * @retval "TensorC8I", "TensorC8U", "TensorC16I", "TensorC16U", "TensorC32I",
 * "TensorC32U" for integer complex number types
 * @retval "TensorC16F", "TensorC32F", "TensorC64F" for floating-point complex
 * number types
 * @retval "UNKNOWN_TYPE" for invalid or unrecognized data types
 *
 * @note This function is marked [[nodiscard]] to encourage checking the return
 * value
 * @note The returned pointer points to static string literals and does not need
 * to be freed
 *
 * @see NvDataType for available data type enumeration values
 *
 * @par Example:
 * @code
 * NvDataType type = TensorR32F;
 * const char* typeStr = nv_get_data_type_string(type);
 * printf("Data type: %s\n", typeStr); // Output: "Data type: TensorR32F"
 * @endcode
 */
// clang-format off
 [[nodiscard]] constexpr const char* nv_get_data_type_string(const NvDataType type) noexcept
 {
     switch(type)
     {
     case TensorVoid:  return "TensorVoid";
     case TensorBit:   return "TensorBit";
     case TensorR16F:  return "TensorR16F";
     case TensorC16F:  return "TensorC16F";
     case TensorR32F:  return "TensorR32F";
     case TensorC32F:  return "TensorC32F";
     case TensorR8I:   return "TensorR8I";
     case TensorC8I:   return "TensorC8I";
     case TensorR8U:   return "TensorR8U";
     case TensorC8U:   return "TensorC8U";
     case TensorR16I:  return "TensorR16I";
     case TensorC16I:  return "TensorC16I";
     case TensorR16U:  return "TensorR16U";
     case TensorC16U:  return "TensorC16U";
     case TensorR32I:  return "TensorR32I";
     case TensorC32I:  return "TensorC32I";
     case TensorR32U:  return "TensorR32U";
     case TensorC32U:  return "TensorC32U";
     case TensorR64F:  return "TensorR64F";
     case TensorC64F:  return "TensorC64F";
     default:         return "UNKNOWN_TYPE";
     }
 }
// clang-format on

/**
 * Type traits for NV data types
 *
 * Template structure providing type mapping from NvDataType enumeration
 * values to their corresponding C++ types. Each specialization defines a
 * Type alias that represents the actual C++ type used for storage and
 * computation.
 *
 * @tparam T The NvDataType enumeration value
 *
 * @note TensorBit uses std::uint32_t for storage as multiple bits are packed
 * into a single word
 * @note Complex types use CUDA vector types (e.g., char2, float2) or cuComplex
 * types
 *
 * @see NvDataType for available enumeration values
 * @see type_to_tensor_type for reverse mapping from C++ types to NvDataType
 *
 * @par Example:
 * @code
 * using FloatType = data_type_traits<TensorR32F>::Type;  // Resolves to float
 * using ComplexType = data_type_traits<TensorC32F>::Type; // Resolves to
 * cuComplex
 * @endcode
 */
// clang-format off
 template <NvDataType T> struct data_type_traits;
 /// Type traits specialization for void/uninitialized tensor data
 template <>
 struct data_type_traits<TensorVoid> {
     using Type = void; //!< Void type for uninitialized
 };
 /// Type traits specialization for bit-packed tensor data
 template <>
 struct data_type_traits<TensorBit> {
     using Type = std::uint32_t; //!< 32-bit storage for packed bits
 };
 /// Type traits specialization for 8-bit signed integer real tensor data
 template <>
 struct data_type_traits<TensorR8I> {
     using Type = signed char; //!< 8-bit signed integer
 };
 /// Type traits specialization for 8-bit signed integer complex tensor data
 template <>
 struct data_type_traits<TensorC8I> {
     using Type = char2; //!< 8-bit signed integer complex
 };
 /// Type traits specialization for 8-bit unsigned integer real tensor data
 template <>
 struct data_type_traits<TensorR8U> {
     using Type = unsigned char; //!< 8-bit unsigned integer
 };
 /// Type traits specialization for 8-bit unsigned integer complex tensor data
 template <>
 struct data_type_traits<TensorC8U> {
     using Type = uchar2; //!< 8-bit unsigned integer complex
 };
 /// Type traits specialization for 16-bit signed integer real tensor data
 template <>
 struct data_type_traits<TensorR16I> {
     using Type = short; //!< 16-bit signed integer
 };
 /// Type traits specialization for 16-bit signed integer complex tensor data
 template <>
 struct data_type_traits<TensorC16I> {
     using Type = short2; //!< 16-bit signed integer complex
 };
 /// Type traits specialization for 16-bit unsigned integer real tensor data
 template <>
 struct data_type_traits<TensorR16U> {
     using Type = unsigned short; //!< 16-bit unsigned integer
 };
 /// Type traits specialization for 16-bit unsigned integer complex tensor data
 template <>
 struct data_type_traits<TensorC16U> {
     using Type = ushort2; //!< 16-bit unsigned integer complex
 };
 /// Type traits specialization for 32-bit signed integer real tensor data
 template <>
 struct data_type_traits<TensorR32I> {
     using Type = int; //!< 32-bit signed integer
 };
 /// Type traits specialization for 32-bit signed integer complex tensor data
 template <>
 struct data_type_traits<TensorC32I> {
     using Type = int2; //!< 32-bit signed integer complex
 };
 /// Type traits specialization for 32-bit unsigned integer real tensor data
 template <>
 struct data_type_traits<TensorR32U> {
     using Type = unsigned int; //!< 32-bit unsigned integer
 };
 /// Type traits specialization for 32-bit unsigned integer complex tensor data
 template <>
 struct data_type_traits<TensorC32U> {
     using Type = uint2; //!< 32-bit unsigned integer complex
 };
 /// Type traits specialization for 16-bit half-precision real tensor data
 template <>
 struct data_type_traits<TensorR16F> {
     using Type = __half; //!< 16-bit half-precision float
 };
 /// Type traits specialization for 16-bit half-precision complex tensor data
 template <>
 struct data_type_traits<TensorC16F> {
     using Type = __half2; //!< 16-bit half-precision complex
 };
 /// Type traits specialization for 32-bit single-precision real tensor data
 template <>
 struct data_type_traits<TensorR32F> {
     using Type = float; //!< 32-bit single-precision float
 };
 /// Type traits specialization for 32-bit single-precision complex tensor data
 template <>
 struct data_type_traits<TensorC32F> {
     using Type = cuda::std::complex<float>; //!< 32-bit single-precision complex
 };
 /// Type traits specialization for 64-bit double-precision real tensor data
 template <>
 struct data_type_traits<TensorR64F> {
     using Type = double; //!< 64-bit double-precision float
 };
 /// Type traits specialization for 64-bit double-precision complex tensor data
 template <>
 struct data_type_traits<TensorC64F> {
     using Type = cuda::std::complex<double>; //!< 64-bit double-precision complex
 };
// clang-format on

/**
 * Reverse type mapping from C++ types to NV data types
 *
 * Template structure providing reverse type mapping from C++ types to their
 * corresponding NvDataType enumeration values. Each specialization defines
 * a VALUE constant that represents the NvDataType for the given C++ type.
 *
 * @tparam T The C++ type to map to NvDataType
 *
 * @note This template provides compile-time type-to-enum mapping
 * @note Not all C++ types have corresponding NvDataType values
 *
 * @see data_type_traits for forward mapping from NvDataType to C++ types
 * @see NvDataType for available enumeration values
 *
 * @par Example:
 * @code
 * constexpr auto floatType = type_to_tensor_type<float>::VALUE;        //
 * TensorR32F constexpr auto complexType =
 * type_to_tensor_type<cuComplex>::VALUE;
 * // TensorC32F
 * @endcode
 */
// clang-format off
 template <typename T> struct type_to_tensor_type;
 /// Reverse type mapping for void type
 template <>
 struct type_to_tensor_type<void> {
     static constexpr NvDataType VALUE = TensorVoid; //!< Void type mapping
 };
 /// Reverse type mapping for 8-bit signed integer
 template <>
 struct type_to_tensor_type<signed char> {
     static constexpr NvDataType VALUE = TensorR8I; //!< 8-bit signed integer
 };
 /// Reverse type mapping for 8-bit signed integer complex
 template <>
 struct type_to_tensor_type<char2> {
     static constexpr NvDataType VALUE = TensorC8I; //!< 8-bit signed integer complex
 };
 /// Reverse type mapping for 8-bit unsigned integer
 template <>
 struct type_to_tensor_type<unsigned char> {
     static constexpr NvDataType VALUE = TensorR8U; //!< 8-bit unsigned integer
 };
 /// Reverse type mapping for 8-bit unsigned integer complex
 template <>
 struct type_to_tensor_type<uchar2> {
     static constexpr NvDataType VALUE = TensorC8U; //!< 8-bit unsigned integer complex
 };
 /// Reverse type mapping for 16-bit signed integer
 template <>
 struct type_to_tensor_type<short> {
     static constexpr NvDataType VALUE = TensorR16I; //!< 16-bit signed integer
 };
 /// Reverse type mapping for 16-bit signed integer complex
 template <>
 struct type_to_tensor_type<short2> {
     static constexpr NvDataType VALUE = TensorC16I; //!< 16-bit signed integer complex
 };
 /// Reverse type mapping for 16-bit unsigned integer
 template <>
 struct type_to_tensor_type<unsigned short> {
     static constexpr NvDataType VALUE = TensorR16U; //!< 16-bit unsigned integer
 };
 /// Reverse type mapping for 16-bit unsigned integer complex
 template <>
 struct type_to_tensor_type<ushort2> {
     static constexpr NvDataType VALUE = TensorC16U; //!< 16-bit unsigned integer complex
 };
 /// Reverse type mapping for 32-bit signed integer
 template <>
 struct type_to_tensor_type<int> {
     static constexpr NvDataType VALUE = TensorR32I; //!< 32-bit signed integer
 };
 /// Reverse type mapping for 32-bit signed integer complex
 template <>
 struct type_to_tensor_type<int2> {
     static constexpr NvDataType VALUE = TensorC32I; //!< 32-bit signed integer complex
 };
 /// Reverse type mapping for 32-bit unsigned integer
 template <>
 struct type_to_tensor_type<unsigned int> {
     static constexpr NvDataType VALUE = TensorR32U; //!< 32-bit unsigned integer
 };
 /// Reverse type mapping for 32-bit unsigned integer complex
 template <>
 struct type_to_tensor_type<uint2> {
     static constexpr NvDataType VALUE = TensorC32U; //!< 32-bit unsigned integer complex
 };
 /// Reverse type mapping for 16-bit half-precision float
 template <>
 struct type_to_tensor_type<__half> {
     static constexpr NvDataType VALUE = TensorR16F; //!< 16-bit half-precision float
 };
 /// Reverse type mapping for 16-bit half-precision complex
 template <>
 struct type_to_tensor_type<__half2> {
     static constexpr NvDataType VALUE = TensorC16F; //!< 16-bit half-precision complex
 };
 /// Reverse type mapping for 32-bit single-precision float
 template <>
 struct type_to_tensor_type<float> {
     static constexpr NvDataType VALUE = TensorR32F; //!< 32-bit single-precision float
 };
 /// Reverse type mapping for 32-bit single-precision complex
 template <>
 struct type_to_tensor_type<cuda::std::complex<float>> {
     static constexpr NvDataType VALUE = TensorC32F; //!< 32-bit single-precision complex
 };
 /// Reverse type mapping for 64-bit double-precision float
 template <>
 struct type_to_tensor_type<double> {
     static constexpr NvDataType VALUE = TensorR64F; //!< 64-bit double-precision float
 };
 /// Reverse type mapping for 64-bit double-precision complex
 template <>
 struct type_to_tensor_type<cuda::std::complex<double>> {
     static constexpr NvDataType VALUE = TensorC64F; //!< 64-bit double-precision complex
 };
// clang-format on

/**
 * Check if data type is sub-byte precision
 *
 * Determines whether the given NV data type represents a sub-byte data type,
 * meaning multiple values can be packed into a single byte. Currently only
 * the TensorBit type is considered sub-byte.
 *
 * @param[in] type The NV data type to check
 * @return true if the type is sub-byte precision, false otherwise
 * @retval true for TensorBit type (1-bit values)
 * @retval false for all other data types
 *
 * @see NvDataType for available data type enumeration values
 * @see get_nv_type_storage_element_size for related storage size information
 */
[[nodiscard]]
constexpr bool type_is_sub_byte(const NvDataType type) noexcept {
    return TensorBit == type;
}

/**
 * Get storage element size for NV data type
 *
 * Returns the size in bytes of the storage element for a given NV data type.
 * In general, this is the size of the type used to store the given
 * NvDataType. However, for sub-byte types, multiple elements are stored in a
 * machine word. For these types, the returned size is the size of a machine
 * word which stores multiple elements.
 *
 * @param[in] type The NV data type to get storage size for
 * @return Size in bytes of the storage element
 * @retval 0 for TensorVoid (uninitialized type)
 * @retval 1 for 8-bit types (TensorR8I, TensorR8U)
 * @retval 2 for 8-bit complex types (TensorC8I, TensorC8U) and 16-bit types
 * (TensorR16I, TensorR16U, TensorR16F)
 * @retval 4 for TensorBit (sub-byte type stored in uint32_t), 16-bit complex
 * types (TensorC16I, TensorC16U, TensorC16F), and 32-bit types (TensorR32I,
 * TensorR32U, TensorR32F)
 * @retval 8 for 32-bit complex types (TensorC32I, TensorC32U, TensorC32F) and
 * 64-bit real types (TensorR64F)
 * @retval 16 for 64-bit complex types (TensorC64F)
 *
 * @note For sub-byte types, the storage size represents the container size, not
 * the logical element size
 *
 * @see NvDataType for available data type enumeration values
 * @see type_is_sub_byte to check if a type is sub-byte precision
 * @see data_type_traits for type mapping information
 */
[[nodiscard]] std::size_t get_nv_type_storage_element_size(const NvDataType type) noexcept;

} // namespace framework::tensor

#endif // FRAMEWORK_DATA_TYPES_HPP
