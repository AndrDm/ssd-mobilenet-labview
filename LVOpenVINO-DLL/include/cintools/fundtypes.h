#ifndef _fundtypes_H
#define _fundtypes_H
/*
    (c) Copyright 1990-2025 by National Instruments Corp.
    All rights reserved.

    Owners: greg.richardson, craig.smith

    fundtypes.h - Fundamental data types.

    Header intended for use by C and C++.
*/

#include "platdefines.h"

#if Compiler == kGCC && __GNUC__ <= 4 && !defined(__STDC_LIMIT_MACROS)
#define __STDC_LIMIT_MACROS // only required for older compilers
#endif

#include <stdint.h>

// Core integer types

#ifndef _NI_int8_DEFINED_
#define _NI_int8_DEFINED_
#ifdef USE_C99_TYPES
typedef int8_t int8;
#elif Unix
typedef signed char int8;
#else
typedef char int8;
#endif
#endif

#ifndef _NI_uInt8_DEFINED_
#define _NI_uInt8_DEFINED_
#ifdef USE_C99_TYPES
typedef uint8_t uInt8;
#else
typedef unsigned char uInt8;
#endif
#endif

#ifndef _NI_int16_DEFINED_
#define _NI_int16_DEFINED_
#ifdef USE_C99_TYPES
typedef int16_t int16;
#elif Mac
typedef short int16;
#else
typedef short int int16;
#endif
#endif

#ifndef _NI_uInt16_DEFINED_
#define _NI_uInt16_DEFINED_
#ifdef USE_C99_TYPES
typedef uint16_t uInt16;
#elif Mac
typedef unsigned short uInt16;
#else
typedef unsigned short int uInt16;
#endif
#endif

#ifndef _NI_int32_DEFINED_
#define _NI_int32_DEFINED_
#ifdef USE_C99_TYPES
typedef int32_t int32;
#elif MSWin
typedef long int32;
#else
typedef int int32;
#endif
#endif

#ifndef _NI_uInt32_DEFINED_
#define _NI_uInt32_DEFINED_
#ifdef USE_C99_TYPES
typedef uint32_t uInt32;
#elif MSWin
typedef unsigned long uInt32;
#else
typedef unsigned int uInt32;
#endif
#endif

#ifndef _NI_int64_DEFINED_
#define _NI_int64_DEFINED_
#ifdef USE_C99_TYPES
typedef int64_t int64;
#elif MSWin
typedef __int64 int64;
#else
typedef long long int64;
#endif
#endif

#ifndef _NI_uInt64_DEFINED_
#define _NI_uInt64_DEFINED_
#ifdef USE_C99_TYPES
typedef uint64_t uInt64;
#elif MSWin
typedef unsigned __int64 uInt64;
#else
typedef unsigned long long uInt64;
#endif
#endif

// Types based on core integer types

#ifndef _NI_uChar_DEFINED_
#define _NI_uChar_DEFINED_
typedef uInt8 uChar;
#endif

#ifndef _NI_Bool32_DEFINED_
#define _NI_Bool32_DEFINED_
typedef int32 Bool32;
#endif
#undef TRUE
#undef FALSE
#define TRUE 1
#define FALSE 0

#ifndef _NI_HRESULT_DEFINED_
#define _NI_HRESULT_DEFINED_
typedef int32 HRESULT;
typedef int32 SCODE;
#endif // _NI_HRESULT_DEFINED_

// Core floating point types

#ifndef _NI_float32_DEFINED_
#define _NI_float32_DEFINED_
typedef float float32;
#endif
#ifndef _NI_float64_DEFINED_
#define _NI_float64_DEFINED_
typedef double float64;
#endif

// Limits for floating point types.
#ifndef FLOAT32_MAX
#define FLOAT32_MAX FLT_MAX
#endif
#ifndef FLOAT64_MAX
#define FLOAT64_MAX DBL_MAX
#endif

#include "lv_prolog.h"

#if Mac // ##############################################################

#if MacX64
#ifndef _NI_floatExt_DEFINED_
#define _NI_floatExt_DEFINED_
typedef struct
{ // force GCC to make this struct 10 bytes
    int32 mlo __attribute__((packed, aligned(2)));
    int32 mhi __attribute__((packed, aligned(2)));
    int16 e __attribute__((packed, aligned(2)));
} floatExt;
#endif
#define CompilerSupportsExt 0
#elif MacARM64
#ifndef _NI_floatExt_DEFINED_
#define _NI_floatExt_DEFINED_
typedef double floatExt;
#endif
#define CompilerSupportsExt 1
#else
#ifndef _NI_floatExt_DEFINED_
#define _NI_floatExt_DEFINED_
typedef struct
{
    double hi;
    double lo;
} floatExt;
#endif
#define ExtHiLo 1
#define CompilerSupportsExt 1
#endif

#elif Unix // ###########################################################

#define ExtHiLo 0
#if ProcessorType == kX86 || ProcessorType == kX64
#if (Compiler == kGCC)
#ifndef _NI_floatExt_DEFINED_
#define _NI_floatExt_DEFINED_
typedef struct
{ // force GCC to make this struct 10 bytes
    int32 mlo __attribute__((packed, aligned(2)));
    int32 mhi __attribute__((packed, aligned(2)));
    int16 e __attribute__((packed, aligned(2)));
} floatExt;
#endif
#else
#ifndef _NI_floatExt_DEFINED_
#define _NI_floatExt_DEFINED_
typedef struct
{
    int32 mlo;
    int32 mhi;
    int16 e;
} floatExt;
#endif
#endif
#define CompilerSupportsExt 0
#elif ProcessorType == kARM
#ifndef _NI_floatExt_DEFINED_
#define _NI_floatExt_DEFINED_
typedef double floatExt;
#endif
#define CompilerSupportsExt 1
#else
#ifndef _NI_floatExt_DEFINED_
#define _NI_floatExt_DEFINED_
typedef long double floatExt;
#endif
#define CompilerSupportsExt 1
#endif

#elif MSWin // ##########################################################

#define ExtHiLo 0
#if ProcessorType == kX64
#pragma pack(push)
#pragma pack(2)
#endif
#ifndef _NI_floatExt_DEFINED_
#define _NI_floatExt_DEFINED_
typedef struct
{
    int32 mlo;
    int32 mhi;
    int16 e;
} floatExt;
#endif
#define CompilerSupportsExt 0
#if ProcessorType == kX64
#pragma pack(pop)
#endif

#endif

#ifndef EXTCODE_CMPLX_TYPES
#define EXTCODE_CMPLX_TYPES

#ifndef _NI_cmplx64_DEFINED_
#define _NI_cmplx64_DEFINED_
/// @brief Complex single-precision floating point number.
typedef struct
{
    float32 re;
    float32 im;
} cmplx64;
#endif

#ifndef _NI_cmplx128_DEFINED_
#define _NI_cmplx128_DEFINED_
/// @brief Complex double-precision floating point number.
typedef struct
{
    float64 re;
    float64 im;
} cmplx128;
#endif

#ifndef _NI_cmplxExt_DEFINED_
#define _NI_cmplxExt_DEFINED_
/// @brief Complex extended-precision floating point number.
typedef struct
{
    floatExt re;
    floatExt im;
} cmplxExt;
#endif

#endif // EXTCODE_CMPLX_TYPES

#include "lv_epilog.h"

// For making void* pts that are distinguishable.
#define NIPrivatePtr(x)  \
    typedef struct x##_t \
    {                    \
        void* p;         \
    } * x

#ifndef Unused
/* The macro Unused can be used to avoid compiler warnings for
unused parameters or locals. */
#ifdef __cplusplus
// This implementation of Unused is safe for const parameters.
#define Unused(var_or_param) _Unused(reinterpret_cast<const void*>(&var_or_param))
inline void _Unused(const void*)
{}
#elif MSWin
// This implementation of Unused is not safe for const parameters.
#define Unused(var_or_param) var_or_param = var_or_param
#else
#define Unused(var_or_param)
#endif
#endif // Unused

#ifndef ConstNotEq
/* The ConstNotEq macro can be used to avoid
"expression is constant" warnings in "if"
statements -- only avoids the warning if you
compile with a C++ compiler. */
#ifdef __cplusplus
inline bool _ConstNotEq(const int32 a, const int32 b)
{
    return a != b;
}
#define ConstNotEq(a, b) _ConstNotEq(a, b)
#else
#define ConstNotEq(a, b) (a != b)
#endif // __cplusplus
#endif // ConstNotEq

#ifndef ConstEvalBool
/* The ConstEvalBool macro can be used to avoid
"expression is constant" warnings in "if"
statements -- only avoids the warning if you
compile with a C++ compiler. */
#ifdef __cplusplus
inline bool _ConstEvalBool(const bool a)
{
    return a;
}
#define ConstEvalBool(a) _ConstEvalBool(a)
#else
#define ConstEvalBool(a) (a)
#endif // __cplusplus
#endif // ConstEvalBool

#ifdef _CVI_
#include <cvidef.h>
#endif // _CVI_

#include <stddef.h>

#if (!defined(__cplusplus) || (NI_CPLUSPLUS_VERSION < 201103L)) && !defined(nullptr)
#define nullptr NULL
#endif

/* Macros to convert from pointers to non-pointer types.
On 64-bit systems, these will truncate the values to 32 bits. */
#define PtrToUInt32(p) ((uInt32)(uintptr_t)p)

/* Localize the warnings for float comparison to one place
see http://www.codeguru.com/forum/showthread.php?t=323835 */
#ifdef __cplusplus
#include <float.h> // DBL_EPSILON
#include <math.h> // fabs
enum FloatComparisonMethod
{
    FloatComparisonExact,
    FloatComparisonWithinEpsilon
};
namespace LVFloats
{
inline bool Equal64(const float64& f1, const float64& f2, FloatComparisonMethod method = FloatComparisonExact)
{
    if (FloatComparisonWithinEpsilon == method) {
        return fabs(f1 - f2) < DBL_EPSILON;
    }
    return (f1 <= f2) && (f1 >= f2); // f1 == f2;
}
inline bool Equal32(const float32& f1, const float32& f2, FloatComparisonMethod method = FloatComparisonExact)
{
    if (FloatComparisonWithinEpsilon == method) {
        return fabsf(f1 - f2) < FLT_EPSILON;
    }
    return (f1 <= f2) && (f1 >= f2); // f1 == f2;
}
template <typename T>
inline bool Equal(const T& v1, const T& v2, FloatComparisonMethod /* method */ = FloatComparisonExact)
{
    return (v1 <= v2) && (v1 >= v2); // v1 == v2;
}
template <>
inline bool Equal<float64>(const float64& v1, const float64& v2, FloatComparisonMethod method)
{
    return Equal64(v1, v2, method);
}
template <>
inline bool Equal<float32>(const float32& v1, const float32& v2, FloatComparisonMethod method)
{
    return Equal32(v1, v2, method);
}
} // namespace LVFloats
#endif // __cplusplus

#define CompilerSupportsInt64 1 // Needed until newer iak exports are used. AB#2146526

#endif // _fundtypes_H
