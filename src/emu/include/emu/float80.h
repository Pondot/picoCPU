// 80-bit extended-precision float type.
//
// Storage layout (Intel x87 / IEEE 754 extended):
//   bytes [0..7]   -- 64-bit significand (explicit integer bit + 63 mantissa)
//   bytes [8..9]   -- 15-bit biased exponent + 1-bit sign
//
// Phase 12 scope: round-trip preserving storage. `to_double()` /
// `from_double()` convert via host double -- full Newton-Raphson soft-float
// is Phase 13. Workloads that only pass long-doubles through registers
// without computing on them (typical for ABI-edge cases) get bit-exact
// behavior already.

#pragma once

#include "emu/types.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace emu {

struct Float80 {
    u64 sig    = 0;       // 64-bit significand (explicit integer bit at MSB)
    u16 signexp = 0;      // bit 15 = sign; bits 0..14 = biased exponent (bias 0x3FFF)

    // Convert from a host double. The 52-bit double mantissa is left-shifted
    // by 11 to land in the upper bits of the 64-bit significand, and the
    // explicit integer bit (bit 63) is set for normals.
    static Float80 from_double(double d) noexcept {
        u64 db = 0;
        std::memcpy(&db, &d, 8);
        const bool sign = (db >> 63) != 0;
        const i32 dexp  = static_cast<i32>((db >> 52) & 0x7FF) - 1023;
        const u64 dmant =  db & ((u64{1} << 52) - 1);
        Float80 f{};
        if ((db & ~(u64{1} << 63)) == 0) {
            // ±zero
            f.sig    = 0;
            f.signexp = sign ? 0x8000 : 0;
            return f;
        }
        if (dexp == 1024) {
            // ±inf / NaN  ->  exp = all-1s (0x7FFF biased), sig = upper bits of mant.
            f.sig     = (u64{1} << 63) | (dmant << 11);
            f.signexp = static_cast<u16>((sign ? 0x8000 : 0) | 0x7FFF);
            return f;
        }
        // Normalized.
        f.sig     = (u64{1} << 63) | (dmant << 11);
        f.signexp = static_cast<u16>((sign ? 0x8000 : 0) | static_cast<u16>(dexp + 0x3FFF));
        return f;
    }

    [[nodiscard]] double to_double() const noexcept {
        const bool sign = (signexp & 0x8000) != 0;
        const i32 exp = static_cast<i32>(signexp & 0x7FFF);
        // ±zero
        if (exp == 0 && (sig & ~u64{0}) == 0) return sign ? -0.0 : 0.0;
        if (exp == 0x7FFF) {
            // ±inf or NaN
            const bool is_inf = ((sig & (u64{1} << 63)) != 0) && ((sig << 1) == 0);
            if (is_inf) return sign ? -std::numeric_limits<double>::infinity()
                                    :  std::numeric_limits<double>::infinity();
            return std::numeric_limits<double>::quiet_NaN();
        }
        // Normal -- extract the implicit-1 + 52 bits.
        const u64 mant52 = (sig >> 11) & ((u64{1} << 52) - 1);
        const i32 dexp = exp - 0x3FFF + 1023;
        u64 dbits;
        if (dexp >= 2047)       dbits = (sign ? (u64{1} << 63) : 0) | (u64{0x7FF} << 52);   // overflow -> inf
        else if (dexp <= 0)     dbits = sign ? (u64{1} << 63) : 0;                          // underflow -> 0
        else                    dbits = (sign ? (u64{1} << 63) : 0)
                                       | (static_cast<u64>(dexp) << 52) | mant52;
        double d;
        std::memcpy(&d, &dbits, 8);
        return d;
    }

    // Bit-exact load from 10-byte memory image.
    static Float80 from_bytes(const u8 b[10]) noexcept {
        Float80 f;
        std::memcpy(&f.sig,    b,     8);
        std::memcpy(&f.signexp,b + 8, 2);
        return f;
    }
    void to_bytes(u8 b[10]) const noexcept {
        std::memcpy(b,     &sig,    8);
        std::memcpy(b + 8, &signexp, 2);
    }

    // Accessors.
    [[nodiscard]] bool is_sign_negative() const noexcept { return (signexp & 0x8000) != 0; }
    [[nodiscard]] u16  raw_exp()          const noexcept { return signexp & 0x7FFF; }
    [[nodiscard]] bool is_zero()          const noexcept { return raw_exp() == 0 && sig == 0; }
    [[nodiscard]] bool is_inf()           const noexcept { return raw_exp() == 0x7FFF && (sig << 1) == 0 && (sig & (u64{1} << 63)); }
    [[nodiscard]] bool is_nan()           const noexcept {
        if (raw_exp() != 0x7FFF) return false;
        return ((sig << 1) != 0) || (sig & (u64{1} << 63)) == 0;
    }

    // Constants.
    static Float80 zero(bool negative) noexcept {
        Float80 f{};
        f.sig    = 0;
        f.signexp = negative ? 0x8000 : 0;
        return f;
    }
    static Float80 inf(bool negative) noexcept {
        Float80 f{};
        f.sig     = u64{1} << 63;
        f.signexp = static_cast<u16>((negative ? 0x8000 : 0) | 0x7FFF);
        return f;
    }
    static Float80 qnan() noexcept {
        Float80 f{};
        f.sig     = (u64{1} << 63) | (u64{1} << 62);
        f.signexp = 0x7FFF;
        return f;
    }
};

// Full-precision soft-float arithmetic at the 80-bit extended format.
// Round-to-nearest-even (the default x87 rounding mode); does not consult
// FCW for other rounding modes. Denormals are flushed to zero on result
// underflow; subnormal inputs are normalized into the operation.
Float80 f80_add(Float80 a, Float80 b) noexcept;
Float80 f80_sub(Float80 a, Float80 b) noexcept;
Float80 f80_mul(Float80 a, Float80 b) noexcept;
Float80 f80_div(Float80 a, Float80 b) noexcept;

// Transcendentals -- computed at extended-precision intermediate but with
// inputs/outputs round-tripping through the 80-bit format.
//
// FSIN / FCOS use Taylor series after range-reducing into [-π/4, π/4].
// F2XM1 (2^x - 1) is the x87 primitive for exp; uses a degree-6 minimax.
// FYL2X computes y * log2(x); reduces x to [1, 2) then uses degree-6 minimax.
// FSCALE multiplies the source by 2^trunc(scale) -- direct exponent bump.
//
// Precision: ~52-bit equivalent (one round-trip through double inside the
// approximation). Bit-exact match to a real x87 is not guaranteed (the real
// FPU uses table-based approximations that we don't emulate); identical
// results to ucrtbase's sin/cos to ~14 decimal digits.
Float80 f80_sin   (Float80 a) noexcept;
Float80 f80_cos   (Float80 a) noexcept;
Float80 f80_2xm1  (Float80 a) noexcept;     // 2^x - 1, valid for |x| ≤ 1
Float80 f80_yl2x  (Float80 y, Float80 x) noexcept;
Float80 f80_scale (Float80 a, Float80 scale) noexcept;

} // namespace emu
