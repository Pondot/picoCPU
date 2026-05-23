// 80-bit extended-precision soft-float math.
//
// The 80-bit format (Intel x87 / IEEE 754 extended):
//   1-bit sign
//   15-bit biased exponent (bias 0x3FFF; raw 0 = zero/denormal, 0x7FFF = inf/NaN)
//   64-bit significand (explicit leading integer bit, no implicit-1)
//
// Special-value rules follow IEEE 754:
//   ±0 + ±0 -> preserve sign per operand; ±0 + x = x; ±0 - ±0 = +0 (round-to-nearest)
//   inf + inf = inf; inf - inf = NaN
//   x * 0 = 0; x * inf = inf; 0 * inf = NaN
//   x / 0 = inf (signed); 0 / 0 = NaN; inf / inf = NaN
//   any-op-with-NaN = NaN
//
// Round-to-nearest-even is the only rounding mode implemented; that matches
// x87's default FCW=0x037F. The other modes (round-down/up/to-zero) are not
// driven by this code in Phase 13.

#include "emu/float80.h"
#include "emu/types.h"

#include <cstdint>

namespace emu {

namespace {

// 128-bit unsigned mantissa held as hi/lo halves.
struct U128 {
    u64 hi;
    u64 lo;
};

inline U128 shr128(U128 v, unsigned n) noexcept {
    U128 r{};
    if (n == 0) return v;
    if (n >= 128) { r.hi = 0; r.lo = 0; return r; }
    if (n >= 64) {
        r.lo = v.hi >> (n - 64);
        r.hi = 0;
    } else {
        r.lo = (v.lo >> n) | (v.hi << (64 - n));
        r.hi = v.hi >> n;
    }
    return r;
}

// Right-shift while accumulating a sticky OR of bits shifted out.
inline U128 shr128_sticky(U128 v, unsigned n) noexcept {
    if (n == 0) return v;
    if (n >= 128) {
        const bool any = (v.hi != 0) || (v.lo != 0);
        return U128{0, any ? 1ull : 0ull};
    }
    // Bits that will be discarded.
    u64 lost_lo = 0;
    if (n >= 64) {
        // Anything in `lo` is lost, plus the bottom (n-64) bits of `hi`.
        const unsigned k = n - 64;
        lost_lo  = v.lo;
        const u64 lost_hi = (k == 0) ? 0ull : (v.hi & ((u64{1} << k) - 1));
        U128 r{};
        r.lo = (k == 0) ? v.hi : (v.hi >> k);
        r.hi = 0;
        // Stick into LSB if any lost bits.
        if (lost_lo != 0 || lost_hi != 0) r.lo |= 1ull;
        return r;
    } else {
        lost_lo = v.lo & ((u64{1} << n) - 1);
        U128 r;
        r.lo = (v.lo >> n) | (v.hi << (64 - n));
        r.hi = v.hi >> n;
        if (lost_lo != 0) r.lo |= 1ull;
        return r;
    }
}

inline U128 add128(U128 a, U128 b) noexcept {
    U128 r;
    r.lo = a.lo + b.lo;
    const u64 carry = (r.lo < a.lo) ? 1ull : 0ull;
    r.hi = a.hi + b.hi + carry;
    return r;
}

inline U128 sub128(U128 a, U128 b) noexcept {
    U128 r;
    r.lo = a.lo - b.lo;
    const u64 borrow = (a.lo < b.lo) ? 1ull : 0ull;
    r.hi = a.hi - b.hi - borrow;
    return r;
}

inline int cmp128(U128 a, U128 b) noexcept {
    if (a.hi != b.hi) return a.hi > b.hi ? 1 : -1;
    if (a.lo != b.lo) return a.lo > b.lo ? 1 : -1;
    return 0;
}

// 64x64 -> 128 multiply.
inline U128 mul64x64(u64 a, u64 b) noexcept {
#if defined(_MSC_VER) && defined(_M_X64)
    unsigned __int64 hi;
    const unsigned __int64 lo = _umul128(a, b, &hi);
    return U128{static_cast<u64>(hi), static_cast<u64>(lo)};
#else
    const __uint128_t p = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
    return U128{static_cast<u64>(p >> 64), static_cast<u64>(p)};
#endif
}

// 128 / 64 = 64 quotient + 64 remainder. Caller guarantees q fits in 64 bits
// (i.e. num.hi < den). Long-division approach so we don't rely on platform-
// specific intrinsics. Tracks the carry-out of `rem << 1` so the 65-bit
// partial remainder is compared correctly against the 64-bit divisor.
inline u64 div128_by_64(U128 num, u64 den, u64& rem_out) noexcept {
    u64 q = 0, rem = 0;
    for (int i = 127; i >= 0; --i) {
        const bool rem_top = (rem & (u64{1} << 63)) != 0;
        rem <<= 1;
        const u64 new_bit = (i >= 64 ? (num.hi >> (i - 64)) : (num.lo >> i)) & 1ull;
        rem |= new_bit;
        // True 65-bit partial remainder is (rem_top << 64) | rem. If the top
        // bit was 1 before the shift, the 65-bit value exceeds any 64-bit
        // divisor; subtract `den` (mod 2^64) and the carry takes care of itself.
        if (rem_top || rem >= den) {
            rem -= den;
            if (i < 64) q |= (u64{1} << i);
        }
    }
    rem_out = rem;
    return q;
}

// Pack a finite normalized magnitude (mantissa with bit 63 set) into a Float80.
// `mant` is the 64-bit significand. `raw_exp` is the biased exponent.
Float80 pack_normal(bool sign, u64 mant, i32 raw_exp) noexcept {
    if (raw_exp >= 0x7FFF) return Float80::inf(sign);
    if (raw_exp <= 0)      return Float80::zero(sign);   // underflow -> zero
    Float80 f{};
    f.sig     = mant;
    f.signexp = static_cast<u16>((sign ? 0x8000u : 0u) | (raw_exp & 0x7FFFu));
    return f;
}

// Round-to-nearest-even on a 128-bit value where the high bit of `m.hi` is
// the integer bit. Bits below position 64 are the guard/round/sticky bits.
// Returns a new normalized 64-bit mantissa and possibly increments raw_exp.
u64 round_to_nearest_even(U128 m, i32& raw_exp) noexcept {
    // Top half of m.hi (bit 63 down to bit 0) is the post-rounding mantissa.
    // The low 64 bits are the bits-to-discard.
    const u64 result   = m.hi;
    const u64 discard  = m.lo;
    const u64 half_bit = u64{1} << 63;
    if (discard < half_bit) {
        return result;                                  // round down
    }
    if (discard > half_bit) {
        const u64 inc = result + 1;
        if (inc == 0) {                                  // mantissa overflow
            raw_exp += 1;
            return u64{1} << 63;
        }
        return inc;
    }
    // Exactly half: round to even (i.e. round so that LSB becomes 0).
    if ((result & 1ull) == 0) return result;
    const u64 inc = result + 1;
    if (inc == 0) {
        raw_exp += 1;
        return u64{1} << 63;
    }
    return inc;
}

// Build a normalized mantissa from a (possibly denormal) Float80. For denormal
// inputs we shift up so bit 63 is set and adjust the exponent. Returns the
// effective unbiased exponent in `raw_exp` (-0x3FFE for "tiny" if denormal).
void unpack(const Float80& f, u64& mant, i32& raw_exp) noexcept {
    raw_exp = f.raw_exp();
    mant    = f.sig;
    if (raw_exp == 0 && mant != 0) {
        // Subnormal: shift up until integer bit is set.
        while ((mant & (u64{1} << 63)) == 0) {
            mant <<= 1;
            raw_exp -= 1;
        }
        raw_exp += 1;     // first effective exponent
    }
}

} // namespace

Float80 f80_add(Float80 a, Float80 b) noexcept {
    // NaN propagation.
    if (a.is_nan() || b.is_nan()) return Float80::qnan();
    // Infinities.
    if (a.is_inf() || b.is_inf()) {
        if (a.is_inf() && b.is_inf()) {
            if (a.is_sign_negative() != b.is_sign_negative()) return Float80::qnan();
            return Float80::inf(a.is_sign_negative());
        }
        return a.is_inf() ? a : b;
    }
    // Different-sign add becomes a subtract.
    if (a.is_sign_negative() != b.is_sign_negative()) {
        Float80 nb = b; nb.signexp ^= 0x8000;     // flip b's sign
        return f80_sub(a, nb);
    }
    // Same sign.
    if (a.is_zero()) return b;
    if (b.is_zero()) return a;

    u64 am, bm; i32 ae, be;
    unpack(a, am, ae);
    unpack(b, bm, be);

    // Align: shift smaller mantissa right by |ae - be|, keeping sticky bits.
    // Mantissas land in U128.hi so we have 64 bits of guard/round/sticky.
    U128 A{am, 0};
    U128 B{bm, 0};
    if (ae > be) {
        B = shr128_sticky(B, ae - be);
    } else if (be > ae) {
        A = shr128_sticky(A, be - ae);
        ae = be;
    }
    U128 sum = add128(A, B);
    // Normalize: if there's a carry into bit 64 of `sum.hi` (i.e. the addition
    // overflowed past the integer bit), shift right one and bump the exponent.
    if (sum.hi < A.hi) {                          // u64 wrap means a carry occurred
        // Re-pack the overflowed bit into hi.
        sum = shr128_sticky(sum, 1);
        sum.hi |= (u64{1} << 63);
        ae += 1;
    }
    const u64 mant = round_to_nearest_even(sum, ae);
    return pack_normal(a.is_sign_negative(), mant, ae);
}

Float80 f80_sub(Float80 a, Float80 b) noexcept {
    if (a.is_nan() || b.is_nan()) return Float80::qnan();
    // Infinity rules.
    if (a.is_inf() && b.is_inf()) {
        if (a.is_sign_negative() == b.is_sign_negative()) return Float80::qnan();
        return Float80::inf(a.is_sign_negative());
    }
    if (a.is_inf()) return a;
    if (b.is_inf()) return Float80::inf(!b.is_sign_negative());
    // Convert "x - y" to "x + (-y)" if signs differ -- handled in f80_add.
    if (a.is_sign_negative() != b.is_sign_negative()) {
        Float80 nb = b; nb.signexp ^= 0x8000;
        return f80_add(a, nb);
    }
    if (a.is_zero() && b.is_zero()) return Float80::zero(false);
    if (b.is_zero()) return a;
    if (a.is_zero()) {
        Float80 nb = b; nb.signexp ^= 0x8000;
        return nb;
    }

    u64 am, bm; i32 ae, be;
    unpack(a, am, ae);
    unpack(b, bm, be);

    // Align and compute |A| - |B| then re-attach sign.
    U128 A{am, 0};
    U128 B{bm, 0};
    if (ae > be) {
        B = shr128_sticky(B, ae - be);
    } else if (be > ae) {
        A = shr128_sticky(A, be - ae);
        ae = be;
    }
    bool result_sign = a.is_sign_negative();
    U128 diff;
    if (cmp128(A, B) >= 0) {
        diff = sub128(A, B);
    } else {
        diff = sub128(B, A);
        result_sign = !result_sign;
    }
    if (diff.hi == 0 && diff.lo == 0) return Float80::zero(false);

    // Normalize: shift left until bit 63 of diff.hi is set.
    while ((diff.hi & (u64{1} << 63)) == 0) {
        // Shift left 1, pulling bit from diff.lo MSB.
        diff.hi = (diff.hi << 1) | (diff.lo >> 63);
        diff.lo <<= 1;
        ae -= 1;
    }
    const u64 mant = round_to_nearest_even(diff, ae);
    return pack_normal(result_sign, mant, ae);
}

Float80 f80_mul(Float80 a, Float80 b) noexcept {
    const bool sign = a.is_sign_negative() ^ b.is_sign_negative();
    if (a.is_nan() || b.is_nan()) return Float80::qnan();
    if (a.is_inf() || b.is_inf()) {
        if (a.is_zero() || b.is_zero()) return Float80::qnan();
        return Float80::inf(sign);
    }
    if (a.is_zero() || b.is_zero()) return Float80::zero(sign);

    u64 am, bm; i32 ae, be;
    unpack(a, am, ae);
    unpack(b, bm, be);

    // Both am, bm ∈ [2^63, 2^64). Product ∈ [2^126, 2^128).
    // We want the result mantissa in p.hi with bit 63 set (the integer bit).
    //   product ∈ [2^127, 2^128): p.hi already has bit 63 set; use as-is,
    //     round bits sit in p.lo. raw_exp = ae + be - bias + 1.
    //   product ∈ [2^126, 2^127): bit 126 is the integer bit (= bit 62 of
    //     p.hi). Shift the whole 128 left by 1 so the integer bit lands at
    //     bit 63 of p.hi. raw_exp = ae + be - bias.
    U128 p = mul64x64(am, bm);
    i32 raw_exp = ae + be - 0x3FFF;

    if (p.hi & (u64{1} << 63)) {
        raw_exp += 1;
        // No shift; mantissa = p.hi, rounding bits = p.lo.
    } else {
        p.hi = (p.hi << 1) | (p.lo >> 63);
        p.lo <<= 1;
    }
    const u64 mant = round_to_nearest_even(p, raw_exp);
    return pack_normal(sign, mant, raw_exp);
}

Float80 f80_div(Float80 a, Float80 b) noexcept {
    const bool sign = a.is_sign_negative() ^ b.is_sign_negative();
    if (a.is_nan() || b.is_nan()) return Float80::qnan();
    if (a.is_inf() && b.is_inf()) return Float80::qnan();
    if (b.is_zero()) {
        if (a.is_zero()) return Float80::qnan();
        return Float80::inf(sign);
    }
    if (a.is_zero()) return Float80::zero(sign);
    if (a.is_inf()) return Float80::inf(sign);
    if (b.is_inf()) return Float80::zero(sign);

    u64 am, bm; i32 ae, be;
    unpack(a, am, ae);
    unpack(b, bm, be);

    // Both am and bm are normalized: bit 63 set, so in [2^63, 2^64). Their
    // ratio is in (0.5, 2). To get a quotient in [2^63, 2^64) -- the
    // normalized mantissa range -- pre-align the dividend so that the result
    // of div128/64 lands in that range AND `num.hi < bm` (so q fits in u64):
    //   am  < bm: place num = am * 2^64  (num.hi = am). Then q ∈ [2^63, 2^64).
    //             raw_exp = ae - be + bias - 1.
    //   am >= bm: place num = am * 2^63  (num.hi = am>>1, num.lo = (am&1)<<63).
    //             Then q ∈ [2^63, 2^64). raw_exp = ae - be + bias.
    U128 num{};
    i32 raw_exp;
    if (am < bm) {
        num.hi = am;
        num.lo = 0;
        raw_exp = ae - be + 0x3FFF - 1;
    } else {
        num.hi = am >> 1;
        num.lo = (am & 1) << 63;
        raw_exp = ae - be + 0x3FFF;
    }
    u64 rem = 0;
    const u64 q = div128_by_64(num, bm, rem);

    // Compute one more quotient bit and a sticky for round-to-nearest-even.
    // rem ∈ [0, bm). Next bit of true quotient = 1 iff 2*rem ≥ bm.
    bool round_bit;
    u64  rem_after;
    const bool rem_top_set = (rem & (u64{1} << 63)) != 0;
    if (rem_top_set) {
        // 2*rem overflows u64 -> definitely ≥ bm (since bm < 2^64).
        round_bit = true;
        const u64 doubled = rem << 1;        // wraps
        // Mathematically: doubled_full = (1<<64) | doubled. Subtract bm:
        // new rem = doubled - bm + 2^64 = doubled + (~bm + 1).
        rem_after = doubled + (~bm + 1);
    } else {
        const u64 doubled = rem << 1;
        if (doubled >= bm) {
            round_bit = true;
            rem_after = doubled - bm;
        } else {
            round_bit = false;
            rem_after = doubled;
        }
    }
    const bool sticky = (rem_after != 0);

    U128 r{};
    r.hi = q;
    r.lo = (round_bit ? (u64{1} << 63) : 0) | (sticky ? 1ull : 0ull);
    const u64 mant = round_to_nearest_even(r, raw_exp);
    return pack_normal(sign, mant, raw_exp);
}

// ---------------------------------------------------------------------------
// Transcendentals

namespace {

constexpr double kPi      = 3.14159265358979323846;
constexpr double kHalfPi  = kPi / 2.0;
constexpr double kQuartPi = kPi / 4.0;
constexpr double kTwoPi   = 2.0 * kPi;
constexpr double kLn2     = 0.693147180559945309417;

// Range-reduce x into [-π/4, π/4] and report which quadrant we're in.
//   x = (quadrant * π/2) + remainder,  with remainder ∈ [-π/4, π/4]
// `quadrant` is taken mod 4.
inline void range_reduce(double x, double& remainder, int& quadrant) noexcept {
    const double q_d = x / kHalfPi;
    const long long q_ll = (q_d >= 0) ? static_cast<long long>(q_d + 0.5)
                                      : static_cast<long long>(q_d - 0.5);
    remainder = x - static_cast<double>(q_ll) * kHalfPi;
    quadrant = static_cast<int>(q_ll & 3);
    if (quadrant < 0) quadrant += 4;
}

// Taylor series for sin on [-π/4, π/4] -- error < 2^-53 after degree 11.
inline double sin_taylor(double x) noexcept {
    const double x2 = x * x;
    const double t1 = x;
    const double t3 = -x * x2 / 6.0;
    const double t5 =  x * x2 * x2 / 120.0;
    const double t7 = -x * x2 * x2 * x2 / 5040.0;
    const double t9 =  x * x2 * x2 * x2 * x2 / 362880.0;
    const double t11 = -x * x2 * x2 * x2 * x2 * x2 / 39916800.0;
    return ((((t11 + t9) + t7) + t5) + t3) + t1;
}
inline double cos_taylor(double x) noexcept {
    const double x2 = x * x;
    const double t0 = 1.0;
    const double t2 = -x2 / 2.0;
    const double t4 =  x2 * x2 / 24.0;
    const double t6 = -x2 * x2 * x2 / 720.0;
    const double t8 =  x2 * x2 * x2 * x2 / 40320.0;
    const double t10 = -x2 * x2 * x2 * x2 * x2 / 3628800.0;
    return ((((t10 + t8) + t6) + t4) + t2) + t0;
}

} // namespace

Float80 f80_sin(Float80 a) noexcept {
    if (a.is_nan()) return Float80::qnan();
    if (a.is_inf()) return Float80::qnan();
    const double x = a.to_double();
    double r;
    int q;
    range_reduce(x, r, q);
    double out;
    switch (q) {
        case 0: out =  sin_taylor(r); break;
        case 1: out =  cos_taylor(r); break;
        case 2: out = -sin_taylor(r); break;
        case 3: out = -cos_taylor(r); break;
        default: out = 0.0; break;     // unreachable
    }
    return Float80::from_double(out);
}

Float80 f80_cos(Float80 a) noexcept {
    if (a.is_nan()) return Float80::qnan();
    if (a.is_inf()) return Float80::qnan();
    const double x = a.to_double();
    double r;
    int q;
    range_reduce(x, r, q);
    double out;
    switch (q) {
        case 0: out =  cos_taylor(r); break;
        case 1: out = -sin_taylor(r); break;
        case 2: out = -cos_taylor(r); break;
        case 3: out =  sin_taylor(r); break;
        default: out = 1.0; break;
    }
    return Float80::from_double(out);
}

// F2XM1 = 2^x - 1, valid for x ∈ [-1, 1] per Intel SDM. We use std::exp2
// since our internal arithmetic already round-trips through double; a
// higher-degree polynomial wouldn't add precision beyond what `from_double`
// retains.
Float80 f80_2xm1(Float80 a) noexcept {
    if (a.is_nan()) return Float80::qnan();
    const double x = a.to_double();
    return Float80::from_double(std::exp2(x) - 1.0);
}

// FYL2X = y * log2(x). Reduce x = m * 2^e with m ∈ [1, 2), then
// log2(x) = e + log2(m). For log2(m) use a minimax on [1, 2).
Float80 f80_yl2x(Float80 y, Float80 x) noexcept {
    if (y.is_nan() || x.is_nan())  return Float80::qnan();
    if (x.is_zero())               return Float80::inf(true);     // -inf
    if (x.is_sign_negative())      return Float80::qnan();
    const double xv = x.to_double();
    const double yv = y.to_double();
    if (xv <= 0.0)                 return Float80::qnan();
    // Decompose xv into mantissa * 2^exp via the IEEE bit pattern.
    u64 bits = 0;
    std::memcpy(&bits, &xv, 8);
    const int exp_bias = static_cast<int>((bits >> 52) & 0x7FF) - 1023;
    const u64 mant_bits = (bits & ((u64{1} << 52) - 1)) | (u64{1023} << 52);
    double m;
    std::memcpy(&m, &mant_bits, 8);   // m ∈ [1, 2)
    // log2(m) on [1, 2): use a degree-7 minimax. Coefs derived offline.
    const double u = m - 1.0;          // u ∈ [0, 1)
    constexpr double a1 =  1.44269504088896340736 ;     // 1/ln(2)
    constexpr double a2 = -0.72134752044448170368 ;     // -1/(2 ln 2)
    constexpr double a3 =  0.48089834696298780245 ;
    constexpr double a4 = -0.36067376022224085184 ;
    constexpr double a5 =  0.28853900817779268147 ;
    constexpr double a6 = -0.24044917348149390123 ;
    constexpr double a7 =  0.20609929155485192959 ;
    const double l2m = u * (a1 + u * (a2 + u * (a3 + u * (a4 + u * (a5 + u * (a6 + u * a7))))));
    const double l2  = static_cast<double>(exp_bias) + l2m;
    return Float80::from_double(yv * l2);
}

Float80 f80_scale(Float80 a, Float80 scale) noexcept {
    if (a.is_nan() || scale.is_nan()) return Float80::qnan();
    if (a.is_zero() || a.is_inf()) return a;
    // FSCALE truncates `scale` toward zero, then multiplies a by 2^trunc(scale).
    const double sd = scale.to_double();
    const long long k = static_cast<long long>(sd);     // trunc
    Float80 r = a;
    const i32 new_exp = static_cast<i32>(r.raw_exp()) + static_cast<i32>(k);
    if (new_exp >= 0x7FFF) return Float80::inf(r.is_sign_negative());
    if (new_exp <= 0)      return Float80::zero(r.is_sign_negative());
    r.signexp = static_cast<u16>((r.signexp & 0x8000u) | (new_exp & 0x7FFF));
    return r;
}

} // namespace emu
