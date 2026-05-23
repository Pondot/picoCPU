// Flag computation: lazy stash + lazy compute + eager reference implementation.
//
// Lazy model (the path handlers use): each `stash_*` writes the cc_op state
// without touching `rflags_`. The next flag read calls `sync_flags_()`, which
// computes all six arithmetic flags from cc_op state, writes them into
// `rflags_`, and transitions cc_op to `Eflags` so subsequent reads short-
// circuit.
//
// Eager reference (the path the cross-check tests compare against): each
// `set_flags_*_eager` computes flags directly. Kept around so tests can
// validate the lazy implementation independently of the handlers.

#include "emu/cpu.h"

namespace emu {

namespace {

constexpr u8 PARITY_TABLE[256] = {
    1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,
    0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,
    0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,
    1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,
    0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,
    1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,
    1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,
    0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,
    0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,
    1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,
    1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,
    0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,
    1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,
    0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,
    0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,
    1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,
};

inline bool parity8(u8 v) noexcept { return PARITY_TABLE[v] != 0; }

// Apply ZF/SF/PF to a `bits` accumulator. Returns the accumulator with those
// three bits set/cleared and others untouched.
inline u64 apply_zsp(u64 bits, u8 size, u64 result) noexcept {
    const u64 mask = mask_for(size);
    const u64 mr   = result & mask;
    bits &= ~(rflags::ZF | rflags::SF | rflags::PF);
    if (mr == 0)                          bits |= rflags::ZF;
    if (mr & sign_mask(size))             bits |= rflags::SF;
    if (parity8(static_cast<u8>(mr)))     bits |= rflags::PF;
    return bits;
}

inline u64 with_bit(u64 bits, u64 mask, bool on) noexcept {
    bits &= ~mask;
    if (on) bits |= mask;
    return bits;
}

// ---- Per-op compute helpers (lazy path) ------------------------------------

u64 compute_logic(u8 size, u64 cc_dst) noexcept {
    u64 bits = rflags::ALWAYS_ONE;     // start with no arith bits
    bits = apply_zsp(bits, size, cc_dst);
    // CF=OF=AF=0 for logical ops.
    return bits;
}

u64 compute_add(u8 size, u64 cc_src, u64 cc_dst) noexcept {
    const u64 m  = mask_for(size);
    const u64 sb = sign_mask(size);
    const u64 b  = cc_src & m;
    const u64 r  = cc_dst & m;
    const u64 a  = (r - b) & m;     // recovered first operand

    u64 bits = rflags::ALWAYS_ONE;
    bits = apply_zsp(bits, size, r);
    bits = with_bit(bits, rflags::CF, r < b);
    bits = with_bit(bits, rflags::OF, (((a ^ r) & (b ^ r)) & sb) != 0);
    bits = with_bit(bits, rflags::AF, ((a ^ b ^ r) & 0x10ull) != 0);
    return bits;
}

u64 compute_adc(u8 size, u64 cc_src, u64 cc_dst, bool cin) noexcept {
    const u64 m  = mask_for(size);
    const u64 sb = sign_mask(size);
    const u64 b  = cc_src & m;
    const u64 r  = cc_dst & m;
    const u64 a  = (r - b - (cin ? 1u : 0u)) & m;

    u64 bits = rflags::ALWAYS_ONE;
    bits = apply_zsp(bits, size, r);
    bits = with_bit(bits, rflags::CF, cin ? (r <= b) : (r < b));
    bits = with_bit(bits, rflags::OF, (((a ^ r) & (b ^ r)) & sb) != 0);
    bits = with_bit(bits, rflags::AF, ((a ^ b ^ r) & 0x10ull) != 0);
    return bits;
}

u64 compute_sub(u8 size, u64 cc_src, u64 cc_dst) noexcept {
    const u64 m  = mask_for(size);
    const u64 sb = sign_mask(size);
    const u64 b  = cc_src & m;
    const u64 r  = cc_dst & m;
    const u64 a  = (r + b) & m;

    u64 bits = rflags::ALWAYS_ONE;
    bits = apply_zsp(bits, size, r);
    bits = with_bit(bits, rflags::CF, a < b);
    bits = with_bit(bits, rflags::OF, (((a ^ b) & (a ^ r)) & sb) != 0);
    bits = with_bit(bits, rflags::AF, ((a ^ b ^ r) & 0x10ull) != 0);
    return bits;
}

u64 compute_sbb(u8 size, u64 cc_src, u64 cc_dst, bool bin) noexcept {
    const u64 m  = mask_for(size);
    const u64 sb = sign_mask(size);
    const u64 b  = cc_src & m;
    const u64 r  = cc_dst & m;
    const u64 a  = (r + b + (bin ? 1u : 0u)) & m;

    u64 bits = rflags::ALWAYS_ONE;
    bits = apply_zsp(bits, size, r);
    bits = with_bit(bits, rflags::CF, bin ? (a <= b) : (a < b));
    bits = with_bit(bits, rflags::OF, (((a ^ b) & (a ^ r)) & sb) != 0);
    bits = with_bit(bits, rflags::AF, ((a ^ b ^ r) & 0x10ull) != 0);
    return bits;
}

u64 compute_inc(u8 size, bool saved_cf, u64 cc_dst) noexcept {
    const u64 m  = mask_for(size);
    const u64 sb = sign_mask(size);
    const u64 r  = cc_dst & m;
    const u64 prev = (r - 1) & m;

    u64 bits = rflags::ALWAYS_ONE;
    bits = apply_zsp(bits, size, r);
    bits = with_bit(bits, rflags::CF, saved_cf);
    bits = with_bit(bits, rflags::OF, (prev == (sb - 1ull)) && ((r & sb) != 0));
    bits = with_bit(bits, rflags::AF, ((prev ^ r) & 0x10ull) != 0);
    return bits;
}

u64 compute_dec(u8 size, bool saved_cf, u64 cc_dst) noexcept {
    const u64 m  = mask_for(size);
    const u64 sb = sign_mask(size);
    const u64 r  = cc_dst & m;
    const u64 prev = (r + 1) & m;

    u64 bits = rflags::ALWAYS_ONE;
    bits = apply_zsp(bits, size, r);
    bits = with_bit(bits, rflags::CF, saved_cf);
    bits = with_bit(bits, rflags::OF, (prev == sb) && ((r & sb) == 0));
    bits = with_bit(bits, rflags::AF, ((prev ^ r) & 0x10ull) != 0);
    return bits;
}

u64 compute_neg(u8 size, u64 cc_src, u64 cc_dst) noexcept {
    const u64 m  = mask_for(size);
    const u64 sb = sign_mask(size);
    const u64 a  = cc_src & m;
    const u64 r  = cc_dst & m;

    u64 bits = rflags::ALWAYS_ONE;
    bits = apply_zsp(bits, size, r);
    bits = with_bit(bits, rflags::CF, a != 0);
    bits = with_bit(bits, rflags::OF, a == sb);
    bits = with_bit(bits, rflags::AF, ((a | r) & 0x10ull) != 0);
    return bits;
}

u64 compute_shl(u8 size, u64 original, u64 cc_dst, u8 count) noexcept {
    if (count == 0) {
        // Shifts by 0 don't modify flags. Caller should preserve.
        // We can't preserve from here; return ALWAYS_ONE | ZSP from result.
        // Mitigated at stash sites: don't stash if count == 0.
        u64 bits = rflags::ALWAYS_ONE;
        bits = apply_zsp(bits, size, cc_dst);
        return bits;
    }
    const u8  width = static_cast<u8>(size * 8);
    const u64 sb    = sign_mask(size);

    u64 bits = rflags::ALWAYS_ONE;
    bits = apply_zsp(bits, size, cc_dst);
    bool cf;
    if (count <= width) cf = ((original >> (width - count)) & 1ull) != 0;
    else                cf = false;
    bits = with_bit(bits, rflags::CF, cf);
    if (count == 1) {
        const bool new_sign = (cc_dst & sb) != 0;
        bits = with_bit(bits, rflags::OF, new_sign != cf);
    }
    return bits;
}

u64 compute_shr(u8 size, u64 original, u64 cc_dst, u8 count) noexcept {
    if (count == 0) {
        u64 bits = rflags::ALWAYS_ONE;
        bits = apply_zsp(bits, size, cc_dst);
        return bits;
    }
    const u8 width = static_cast<u8>(size * 8);

    u64 bits = rflags::ALWAYS_ONE;
    bits = apply_zsp(bits, size, cc_dst);
    bool cf;
    if (count <= width) cf = ((original >> (count - 1u)) & 1ull) != 0;
    else                cf = false;
    bits = with_bit(bits, rflags::CF, cf);
    if (count == 1) {
        const u64 sb = sign_mask(size);
        bits = with_bit(bits, rflags::OF, (original & sb) != 0);
    }
    return bits;
}

u64 compute_sar(u8 size, u64 original, u64 cc_dst, u8 count) noexcept {
    if (count == 0) {
        u64 bits = rflags::ALWAYS_ONE;
        bits = apply_zsp(bits, size, cc_dst);
        return bits;
    }
    const u8 width = static_cast<u8>(size * 8);

    u64 bits = rflags::ALWAYS_ONE;
    bits = apply_zsp(bits, size, cc_dst);
    bool cf;
    if (count <= width) {
        cf = ((original >> (count - 1u)) & 1ull) != 0;
    } else {
        const u64 sb = sign_mask(size);
        cf = (original & sb) != 0;
    }
    bits = with_bit(bits, rflags::CF, cf);
    if (count == 1) {
        bits = with_bit(bits, rflags::OF, false);
    }
    return bits;
}

u64 compute_rol(u8 size, u64 cc_dst, u8 effective_count, u64 preserved) noexcept {
    if (effective_count == 0) {
        // ROL by 0 leaves all flags unchanged. Re-use the preserved bits.
        return preserved | rflags::ALWAYS_ONE;
    }
    const u64 sb = sign_mask(size);
    u64 bits = preserved;     // preserve everything but CF/OF
    const bool cf = (cc_dst & 1ull) != 0;
    bits = with_bit(bits, rflags::CF, cf);
    if (effective_count == 1) {
        bits = with_bit(bits, rflags::OF, ((cc_dst & sb) != 0) != cf);
    }
    return bits | rflags::ALWAYS_ONE;
}

u64 compute_ror(u8 size, u64 cc_dst, u8 effective_count, u64 preserved) noexcept {
    if (effective_count == 0) {
        return preserved | rflags::ALWAYS_ONE;
    }
    const u64 sb  = sign_mask(size);
    const u64 sb2 = sb >> 1;
    u64 bits = preserved;
    const bool cf = (cc_dst & sb) != 0;
    bits = with_bit(bits, rflags::CF, cf);
    if (effective_count == 1) {
        const bool b_top    = (cc_dst & sb)  != 0;
        const bool b_sec    = (cc_dst & sb2) != 0;
        bits = with_bit(bits, rflags::OF, b_top != b_sec);
    }
    return bits | rflags::ALWAYS_ONE;
}

u64 compute_mul_ov(u64 preserved, bool overflow) noexcept {
    // ZSP undefined per Intel SDM. Preserve whatever was there.
    u64 bits = preserved;
    bits = with_bit(bits, rflags::CF, overflow);
    bits = with_bit(bits, rflags::OF, overflow);
    return bits | rflags::ALWAYS_ONE;
}

} // namespace

// ---- sync_flags_ -----------------------------------------------------------

void Cpu::sync_flags_() const noexcept {
    if (cc_op_ == CcOp::Eflags) return;

    const u8  size = cc_size_;
    const u64 dst  = cc_dst_;
    const u64 src  = cc_src_;
    const u64 src2 = cc_src2_;
    const u64 preserved = rflags_;

    u64 bits = preserved;

    switch (cc_op_) {
        case CcOp::Eflags:                                                 break;
        case CcOp::Logic: bits = compute_logic(size, dst);                 break;
        case CcOp::Add:   bits = compute_add  (size, src, dst);            break;
        case CcOp::Adc:   bits = compute_adc  (size, src, dst, src2 != 0); break;
        case CcOp::Sub:   bits = compute_sub  (size, src, dst);            break;
        case CcOp::Sbb:   bits = compute_sbb  (size, src, dst, src2 != 0); break;
        case CcOp::Inc:   bits = compute_inc  (size, src != 0, dst);       break;
        case CcOp::Dec:   bits = compute_dec  (size, src != 0, dst);       break;
        case CcOp::Neg:   bits = compute_neg  (size, src, dst);            break;
        case CcOp::Shl:   bits = compute_shl  (size, src, dst, static_cast<u8>(src2)); break;
        case CcOp::Shr:   bits = compute_shr  (size, src, dst, static_cast<u8>(src2)); break;
        case CcOp::Sar:   bits = compute_sar  (size, src, dst, static_cast<u8>(src2)); break;
        case CcOp::Rol:   bits = compute_rol  (size, dst, static_cast<u8>(src2), preserved); break;
        case CcOp::Ror:   bits = compute_ror  (size, dst, static_cast<u8>(src2), preserved); break;
        case CcOp::MulOv: bits = compute_mul_ov(preserved, src != 0);      break;
    }

    // Preserve non-arith bits (DF, IF, etc.) -- clobber only ALWAYS_ONE bit
    // and the six arith bits.
    constexpr u64 arith_mask = rflags::CF | rflags::PF | rflags::AF
                             | rflags::ZF | rflags::SF | rflags::OF;
    rflags_ = (preserved & ~(arith_mask | rflags::ALWAYS_ONE))
            | (bits & (arith_mask | rflags::ALWAYS_ONE));
    cc_op_ = CcOp::Eflags;
}

// ---- Lazy stash entry points -----------------------------------------------

void Cpu::stash_logic(u8 size, u64 r) noexcept {
    cc_op_   = CcOp::Logic;
    cc_size_ = size;
    cc_dst_  = r;
    cc_src_  = 0;
    cc_src2_ = 0;
}

void Cpu::stash_add(u8 size, u64 b, u64 r) noexcept {
    cc_op_   = CcOp::Add;
    cc_size_ = size;
    cc_src_  = b;
    cc_dst_  = r;
    cc_src2_ = 0;
}

void Cpu::stash_adc(u8 size, u64 b, bool cin, u64 r) noexcept {
    cc_op_   = CcOp::Adc;
    cc_size_ = size;
    cc_src_  = b;
    cc_dst_  = r;
    cc_src2_ = cin ? 1ull : 0ull;
}

void Cpu::stash_sub(u8 size, u64 b, u64 r) noexcept {
    cc_op_   = CcOp::Sub;
    cc_size_ = size;
    cc_src_  = b;
    cc_dst_  = r;
    cc_src2_ = 0;
}

void Cpu::stash_sbb(u8 size, u64 b, bool bin, u64 r) noexcept {
    cc_op_   = CcOp::Sbb;
    cc_size_ = size;
    cc_src_  = b;
    cc_dst_  = r;
    cc_src2_ = bin ? 1ull : 0ull;
}

void Cpu::stash_inc(u8 size, u64 r) noexcept {
    // INC must preserve CF -- sync now to capture it, stash saved_cf in cc_src.
    sync_flags_();
    const bool saved = (rflags_ & rflags::CF) != 0;
    cc_op_   = CcOp::Inc;
    cc_size_ = size;
    cc_src_  = saved ? 1ull : 0ull;
    cc_dst_  = r;
    cc_src2_ = 0;
}

void Cpu::stash_dec(u8 size, u64 r) noexcept {
    sync_flags_();
    const bool saved = (rflags_ & rflags::CF) != 0;
    cc_op_   = CcOp::Dec;
    cc_size_ = size;
    cc_src_  = saved ? 1ull : 0ull;
    cc_dst_  = r;
    cc_src2_ = 0;
}

void Cpu::stash_neg(u8 size, u64 a, u64 r) noexcept {
    cc_op_   = CcOp::Neg;
    cc_size_ = size;
    cc_src_  = a;
    cc_dst_  = r;
    cc_src2_ = 0;
}

void Cpu::stash_shl(u8 size, u64 original, u8 count, u64 r) noexcept {
    if (count == 0) return;          // shifts by 0 leave flags untouched
    cc_op_   = CcOp::Shl;
    cc_size_ = size;
    cc_src_  = original;
    cc_dst_  = r;
    cc_src2_ = count;
}

void Cpu::stash_shr(u8 size, u64 original, u8 count, u64 r) noexcept {
    if (count == 0) return;
    cc_op_   = CcOp::Shr;
    cc_size_ = size;
    cc_src_  = original;
    cc_dst_  = r;
    cc_src2_ = count;
}

void Cpu::stash_sar(u8 size, u64 original, u8 count, u64 r) noexcept {
    if (count == 0) return;
    cc_op_   = CcOp::Sar;
    cc_size_ = size;
    cc_src_  = original;
    cc_dst_  = r;
    cc_src2_ = count;
}

void Cpu::stash_rol(u8 size, u64 original, u8 count, u64 r) noexcept {
    const u8 width = static_cast<u8>(size * 8);
    const u8 mod   = static_cast<u8>(count & (width - 1u));
    if (mod == 0 && count == 0) return;   // ROL by 0 leaves flags untouched
    // We must sync first since we want to *preserve* ZSP across this op.
    sync_flags_();
    cc_op_   = CcOp::Rol;
    cc_size_ = size;
    cc_src_  = original;
    cc_dst_  = r;
    cc_src2_ = count;
}

void Cpu::stash_ror(u8 size, u64 original, u8 count, u64 r) noexcept {
    const u8 width = static_cast<u8>(size * 8);
    const u8 mod   = static_cast<u8>(count & (width - 1u));
    if (mod == 0 && count == 0) return;
    sync_flags_();
    cc_op_   = CcOp::Ror;
    cc_size_ = size;
    cc_src_  = original;
    cc_dst_  = r;
    cc_src2_ = count;
}

void Cpu::stash_mul_ov(bool overflow) noexcept {
    sync_flags_();
    cc_op_   = CcOp::MulOv;
    cc_size_ = 8;
    cc_src_  = overflow ? 1ull : 0ull;
    cc_dst_  = 0;
    cc_src2_ = 0;
}

// ---- Eager reference helpers (cross-check only) ----------------------------
//
// These write `rflags_` directly. They are NOT called by handlers. They exist
// only so cross-check unit tests can compare against an independent
// implementation of the same flag semantics.

void Cpu::set_flags_logical_eager(u8 size, u64 r) noexcept {
    cc_op_ = CcOp::Eflags;     // bypass the lazy machine
    rflags_ = (rflags_ & ~(rflags::CF | rflags::PF | rflags::AF
                          | rflags::ZF | rflags::SF | rflags::OF));
    rflags_ = apply_zsp(rflags_, size, r) | rflags::ALWAYS_ONE;
}

void Cpu::set_flags_add_eager(u8 size, u64 a, u64 b, u64 r) noexcept {
    cc_op_ = CcOp::Eflags;
    const u64 sb = sign_mask(size);
    const u64 m  = mask_for(size);
    const u64 ma = a & m, mb = b & m, mr = r & m;
    rflags_ = apply_zsp(rflags_, size, r);
    rflags_ = with_bit(rflags_, rflags::CF, mr < ma);
    rflags_ = with_bit(rflags_, rflags::OF, (((ma ^ mr) & (mb ^ mr)) & sb) != 0);
    rflags_ = with_bit(rflags_, rflags::AF, ((ma ^ mb ^ mr) & 0x10ull) != 0);
    rflags_ |= rflags::ALWAYS_ONE;
}

void Cpu::set_flags_adc_eager(u8 size, u64 a, u64 b, bool cin, u64 r) noexcept {
    cc_op_ = CcOp::Eflags;
    const u64 sb = sign_mask(size);
    const u64 m  = mask_for(size);
    const u64 ma = a & m, mb = b & m, mr = r & m;
    rflags_ = apply_zsp(rflags_, size, r);
    rflags_ = with_bit(rflags_, rflags::CF, cin ? (mr <= ma) : (mr < ma));
    rflags_ = with_bit(rflags_, rflags::OF, (((ma ^ mr) & (mb ^ mr)) & sb) != 0);
    rflags_ = with_bit(rflags_, rflags::AF, ((ma ^ mb ^ mr) & 0x10ull) != 0);
    rflags_ |= rflags::ALWAYS_ONE;
}

void Cpu::set_flags_sub_eager(u8 size, u64 a, u64 b, u64 r) noexcept {
    cc_op_ = CcOp::Eflags;
    const u64 sb = sign_mask(size);
    const u64 m  = mask_for(size);
    const u64 ma = a & m, mb = b & m, mr = r & m;
    rflags_ = apply_zsp(rflags_, size, r);
    rflags_ = with_bit(rflags_, rflags::CF, ma < mb);
    rflags_ = with_bit(rflags_, rflags::OF, (((ma ^ mb) & (ma ^ mr)) & sb) != 0);
    rflags_ = with_bit(rflags_, rflags::AF, ((ma ^ mb ^ mr) & 0x10ull) != 0);
    rflags_ |= rflags::ALWAYS_ONE;
}

void Cpu::set_flags_sbb_eager(u8 size, u64 a, u64 b, bool bin, u64 r) noexcept {
    cc_op_ = CcOp::Eflags;
    const u64 sb = sign_mask(size);
    const u64 m  = mask_for(size);
    const u64 ma = a & m, mb = b & m, mr = r & m;
    rflags_ = apply_zsp(rflags_, size, r);
    rflags_ = with_bit(rflags_, rflags::CF, bin ? (ma <= mb) : (ma < mb));
    rflags_ = with_bit(rflags_, rflags::OF, (((ma ^ mb) & (ma ^ mr)) & sb) != 0);
    rflags_ = with_bit(rflags_, rflags::AF, ((ma ^ mb ^ mr) & 0x10ull) != 0);
    rflags_ |= rflags::ALWAYS_ONE;
}

void Cpu::set_flags_inc_eager(u8 size, u64 prev, u64 r) noexcept {
    // CF preserved.
    cc_op_ = CcOp::Eflags;
    const u64 sb = sign_mask(size);
    const u64 m  = mask_for(size);
    const u64 mp = prev & m, mr = r & m;
    rflags_ = apply_zsp(rflags_, size, r);
    rflags_ = with_bit(rflags_, rflags::OF, (mp == (sb - 1ull)) && ((mr & sb) != 0));
    rflags_ = with_bit(rflags_, rflags::AF, ((mp ^ mr) & 0x10ull) != 0);
    rflags_ |= rflags::ALWAYS_ONE;
}

void Cpu::set_flags_dec_eager(u8 size, u64 prev, u64 r) noexcept {
    cc_op_ = CcOp::Eflags;
    const u64 sb = sign_mask(size);
    const u64 m  = mask_for(size);
    const u64 mp = prev & m, mr = r & m;
    rflags_ = apply_zsp(rflags_, size, r);
    rflags_ = with_bit(rflags_, rflags::OF, (mp == sb) && ((mr & sb) == 0));
    rflags_ = with_bit(rflags_, rflags::AF, ((mp ^ mr) & 0x10ull) != 0);
    rflags_ |= rflags::ALWAYS_ONE;
}

void Cpu::set_flags_neg_eager(u8 size, u64 src, u64 r) noexcept {
    cc_op_ = CcOp::Eflags;
    const u64 sb = sign_mask(size);
    const u64 m  = mask_for(size);
    const u64 ms = src & m, mr = r & m;
    rflags_ = apply_zsp(rflags_, size, r);
    rflags_ = with_bit(rflags_, rflags::CF, ms != 0);
    rflags_ = with_bit(rflags_, rflags::OF, ms == sb);
    rflags_ = with_bit(rflags_, rflags::AF, ((ms | mr) & 0x10ull) != 0);
    rflags_ |= rflags::ALWAYS_ONE;
}

void Cpu::set_flags_shl_eager(u8 size, u64 src, u8 count, u64 r) noexcept {
    if (count == 0) return;
    cc_op_ = CcOp::Eflags;
    const u8 width = static_cast<u8>(size * 8);
    const u64 sb = sign_mask(size);
    rflags_ = apply_zsp(rflags_, size, r);
    const bool cf = (count <= width) ? (((src >> (width - count)) & 1ull) != 0) : false;
    rflags_ = with_bit(rflags_, rflags::CF, cf);
    if (count == 1) {
        const bool new_sign = (r & sb) != 0;
        rflags_ = with_bit(rflags_, rflags::OF, new_sign != cf);
    }
    rflags_ = with_bit(rflags_, rflags::AF, false);
    rflags_ |= rflags::ALWAYS_ONE;
}

void Cpu::set_flags_shr_eager(u8 size, u64 src, u8 count, u64 r) noexcept {
    if (count == 0) return;
    cc_op_ = CcOp::Eflags;
    const u8 width = static_cast<u8>(size * 8);
    rflags_ = apply_zsp(rflags_, size, r);
    const bool cf = (count <= width) ? (((src >> (count - 1u)) & 1ull) != 0) : false;
    rflags_ = with_bit(rflags_, rflags::CF, cf);
    if (count == 1) {
        const u64 sb = sign_mask(size);
        rflags_ = with_bit(rflags_, rflags::OF, (src & sb) != 0);
    }
    rflags_ = with_bit(rflags_, rflags::AF, false);
    rflags_ |= rflags::ALWAYS_ONE;
}

void Cpu::set_flags_sar_eager(u8 size, u64 src, u8 count, u64 r) noexcept {
    if (count == 0) return;
    cc_op_ = CcOp::Eflags;
    const u8 width = static_cast<u8>(size * 8);
    rflags_ = apply_zsp(rflags_, size, r);
    bool cf;
    if (count <= width) cf = ((src >> (count - 1u)) & 1ull) != 0;
    else                cf = (src & sign_mask(size)) != 0;
    rflags_ = with_bit(rflags_, rflags::CF, cf);
    if (count == 1) rflags_ = with_bit(rflags_, rflags::OF, false);
    rflags_ = with_bit(rflags_, rflags::AF, false);
    rflags_ |= rflags::ALWAYS_ONE;
}

void Cpu::set_flags_rol_eager(u8 size, u64 /*src*/, u8 count, u64 r) noexcept {
    const u8 width = static_cast<u8>(size * 8);
    const u8 mod   = static_cast<u8>(count & (width - 1u));
    if (mod == 0) return;
    cc_op_ = CcOp::Eflags;
    const bool cf = (r & 1ull) != 0;
    rflags_ = with_bit(rflags_, rflags::CF, cf);
    if (count == 1) {
        const u64 sb = sign_mask(size);
        rflags_ = with_bit(rflags_, rflags::OF, ((r & sb) != 0) != cf);
    }
    rflags_ |= rflags::ALWAYS_ONE;
}

void Cpu::set_flags_ror_eager(u8 size, u64 /*src*/, u8 count, u64 r) noexcept {
    const u8 width = static_cast<u8>(size * 8);
    const u8 mod   = static_cast<u8>(count & (width - 1u));
    if (mod == 0) return;
    cc_op_ = CcOp::Eflags;
    const u64 sb  = sign_mask(size);
    const u64 sb2 = sb >> 1;
    const bool cf = (r & sb) != 0;
    rflags_ = with_bit(rflags_, rflags::CF, cf);
    if (count == 1) {
        const bool b_top = (r & sb)  != 0;
        const bool b_sec = (r & sb2) != 0;
        rflags_ = with_bit(rflags_, rflags::OF, b_top != b_sec);
    }
    rflags_ |= rflags::ALWAYS_ONE;
}

} // namespace emu
