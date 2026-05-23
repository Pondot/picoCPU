// Emulator: fundamental types.
//
// All host/guest addresses are 64-bit. `GuestAddr` is a value type, not a
// pointer -- never dereference one with native code.

#pragma once

#include <cstddef>
#include <cstdint>

namespace emu {

using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8  = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using usize = size_t;
using isize = ptrdiff_t;

// Strongly-typed guest address. Implicit conversion to/from u64 for now;
// will tighten once we have real call sites.
using GuestAddr = u64;

// Sizes commonly recurring in x86 decoding.
constexpr u32 PAGE_SIZE     = 4096;
constexpr u32 PAGE_SHIFT    = 12;
constexpr u64 PAGE_MASK     = ~u64{0xFFF};

constexpr u32 CACHE_LINE    = 64;
constexpr u32 CACHE_LINE_SHIFT = 6;

constexpr u32 L3_REGION     = 2u * 1024u * 1024u; // 2 MiB
constexpr u32 L3_REGION_SHIFT = 21;

} // namespace emu
