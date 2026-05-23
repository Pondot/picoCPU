// Arithmetic handlers: ADD/ADC/SUB/SBB/INC/DEC/NEG/IMUL.
//
// MUL/DIV/IDIV/1-op IMUL come in Phase 4.

#include "../handlers/handlers.h"
#include "../handlers/handler_util.h"

#include "emu/cpu.h"

#include <climits>
#include <intrin.h>

namespace emu::handlers {

void op_add(Cpu& cpu, const Insn& insn) {
    u64 a = 0, b = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, a)) return;
    if (!read_operand(cpu, insn, insn.src, insn.op_size, b)) return;
    const u64 m = mask_for(insn.op_size);
    const u64 r = (a + b) & m;
    if (!write_operand(cpu, insn, insn.dst, insn.op_size, r)) return;
    cpu.stash_add(insn.op_size, b, r);
}

void op_adc(Cpu& cpu, const Insn& insn) {
    u64 a = 0, b = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, a)) return;
    if (!read_operand(cpu, insn, insn.src, insn.op_size, b)) return;
    const bool cin = cpu.cf();
    const u64 m = mask_for(insn.op_size);
    const u64 r = (a + b + (cin ? 1u : 0u)) & m;
    if (!write_operand(cpu, insn, insn.dst, insn.op_size, r)) return;
    cpu.stash_adc(insn.op_size, b, cin, r);
}

void op_sub(Cpu& cpu, const Insn& insn) {
    u64 a = 0, b = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, a)) return;
    if (!read_operand(cpu, insn, insn.src, insn.op_size, b)) return;
    const u64 m = mask_for(insn.op_size);
    const u64 r = (a - b) & m;
    if (!write_operand(cpu, insn, insn.dst, insn.op_size, r)) return;
    cpu.stash_sub(insn.op_size, b, r);
}

void op_sbb(Cpu& cpu, const Insn& insn) {
    u64 a = 0, b = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, a)) return;
    if (!read_operand(cpu, insn, insn.src, insn.op_size, b)) return;
    const bool bin = cpu.cf();
    const u64 m = mask_for(insn.op_size);
    const u64 r = (a - b - (bin ? 1u : 0u)) & m;
    if (!write_operand(cpu, insn, insn.dst, insn.op_size, r)) return;
    cpu.stash_sbb(insn.op_size, b, bin, r);
}

void op_inc(Cpu& cpu, const Insn& insn) {
    u64 a = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, a)) return;
    const u64 m = mask_for(insn.op_size);
    const u64 r = (a + 1) & m;
    if (!write_operand(cpu, insn, insn.dst, insn.op_size, r)) return;
    cpu.stash_inc(insn.op_size, r);
}

void op_dec(Cpu& cpu, const Insn& insn) {
    u64 a = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, a)) return;
    const u64 m = mask_for(insn.op_size);
    const u64 r = (a - 1) & m;
    if (!write_operand(cpu, insn, insn.dst, insn.op_size, r)) return;
    cpu.stash_dec(insn.op_size, r);
}

void op_neg(Cpu& cpu, const Insn& insn) {
    u64 a = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, a)) return;
    const u64 m = mask_for(insn.op_size);
    const u64 r = (~a + 1u) & m;
    if (!write_operand(cpu, insn, insn.dst, insn.op_size, r)) return;
    cpu.stash_neg(insn.op_size, a, r);
}

void op_imul(Cpu& cpu, const Insn& insn) {
    // OpKind::Imul3 -> dst = src * imm_extra (3-operand: 0x69 / 0x6B)
    // OpKind::Imul  -> dst = dst * src      (2-operand: 0x0F 0xAF)
    u64 a = 0, b = 0;
    if (insn.kind == OpKind::Imul3) {
        if (!read_operand(cpu, insn, insn.src, insn.op_size, a)) return;
        b = static_cast<u64>(insn.imm_extra);
    } else {
        if (!read_operand(cpu, insn, insn.dst, insn.op_size, a)) return;
        if (!read_operand(cpu, insn, insn.src, insn.op_size, b)) return;
    }

    const u8  size = insn.op_size;
    const u64 m    = mask_for(size);
    u64 r = 0;
    bool overflow = false;

    if (size == 8) {
        // 128-bit signed multiply, take low 64.
#if defined(_MSC_VER) && defined(_M_X64)
        i64 high = 0;
        const i64 lo = _mul128(static_cast<i64>(a), static_cast<i64>(b), &high);
        r = static_cast<u64>(lo);
        overflow = high != (static_cast<i64>(lo) >> 63);
#else
        r = a * b;
        // Without 128-bit math we can't compute overflow precisely; conservatively false.
        overflow = false;
#endif
    } else {
        const i64 sa  = sign_extend(a, size);
        const i64 sb_ = sign_extend(b, size);
        const i64 full = sa * sb_;
        r = static_cast<u64>(full) & m;
        const i64 truncated_se = sign_extend(r, size);
        overflow = (full != truncated_se);
    }

    (void)write_operand(cpu, insn, insn.dst, size, r);
    cpu.stash_mul_ov(overflow);
}

// ---- 1-operand MUL/IMUL/DIV/IDIV ---------------------------------------
//
// All four operate on an implicit accumulator (AL / AX / EAX / RAX) and an
// explicit r/m source.
//
//   MUL  r/m{8,16,32,64}    -> AX = AL*src;  DX:AX = AX*src;  EDX:EAX = EAX*src;  RDX:RAX = RAX*src
//   IMUL r/m{8,16,32,64}    -> same split, signed
//   DIV  r/m{8,16,32,64}    -> quotient in AL/AX/EAX/RAX, remainder in AH/DX/EDX/RDX
//   IDIV r/m{8,16,32,64}    -> same, signed

void op_mul(Cpu& cpu, const Insn& insn) {
    u64 b = 0;
    if (!read_operand(cpu, insn, insn.src, insn.op_size, b)) return;
    const u64 a    = cpu.r(reg::RAX) & mask_for(insn.op_size);
    const u64 mask = mask_for(insn.op_size);
    bool overflow = false;
    u64 lo = 0, hi = 0;
    switch (insn.op_size) {
        case 1: {
            const u16 full = static_cast<u16>(a) * static_cast<u16>(b);
            lo = full & 0xFFu;
            hi = (full >> 8) & 0xFFu;
            overflow = hi != 0;
            // AX = full
            cpu.set_r16(reg::RAX, full);
            break;
        }
        case 2: {
            const u32 full = static_cast<u32>(a) * static_cast<u32>(b);
            lo = full & 0xFFFFu;
            hi = (full >> 16) & 0xFFFFu;
            overflow = hi != 0;
            cpu.set_r16(reg::RAX, static_cast<u16>(lo));
            cpu.set_r16(reg::RDX, static_cast<u16>(hi));
            break;
        }
        case 4: {
            const u64 full = static_cast<u64>(a) * static_cast<u64>(b);
            lo = full & 0xFFFFFFFFu;
            hi = (full >> 32) & 0xFFFFFFFFu;
            overflow = hi != 0;
            cpu.set_r32(reg::RAX, static_cast<u32>(lo));
            cpu.set_r32(reg::RDX, static_cast<u32>(hi));
            break;
        }
        case 8: {
#if defined(_MSC_VER) && defined(_M_X64)
            unsigned __int64 high = 0;
            lo = _umul128(a, b, &high);
            hi = high;
#else
            lo = a * b;
            hi = 0;
#endif
            overflow = hi != 0;
            cpu.set_r64(reg::RAX, lo);
            cpu.set_r64(reg::RDX, hi);
            break;
        }
        default: break;
    }
    (void)mask;
    cpu.stash_mul_ov(overflow);
}

void op_imul1(Cpu& cpu, const Insn& insn) {
    u64 b = 0;
    if (!read_operand(cpu, insn, insn.src, insn.op_size, b)) return;
    const u64 a = cpu.r(reg::RAX) & mask_for(insn.op_size);
    bool overflow = false;
    u64 lo = 0, hi = 0;
    switch (insn.op_size) {
        case 1: {
            const i16 sa = static_cast<i16>(static_cast<i8>(a));
            const i16 sb = static_cast<i16>(static_cast<i8>(b));
            const i16 full = static_cast<i16>(sa * sb);
            const u16 u = static_cast<u16>(full);
            lo = u & 0xFFu;
            hi = (u >> 8) & 0xFFu;
            cpu.set_r16(reg::RAX, u);
            overflow = (static_cast<i8>(lo) != full);     // top byte != sign-extension of low
            break;
        }
        case 2: {
            const i32 sa = static_cast<i32>(static_cast<i16>(a));
            const i32 sb = static_cast<i32>(static_cast<i16>(b));
            const i32 full = sa * sb;
            const u32 u = static_cast<u32>(full);
            lo = u & 0xFFFFu;
            hi = (u >> 16) & 0xFFFFu;
            cpu.set_r16(reg::RAX, static_cast<u16>(lo));
            cpu.set_r16(reg::RDX, static_cast<u16>(hi));
            overflow = (static_cast<i32>(static_cast<i16>(lo)) != full);
            break;
        }
        case 4: {
            const i64 sa = static_cast<i64>(static_cast<i32>(a));
            const i64 sb = static_cast<i64>(static_cast<i32>(b));
            const i64 full = sa * sb;
            const u64 u = static_cast<u64>(full);
            lo = u & 0xFFFFFFFFull;
            hi = (u >> 32) & 0xFFFFFFFFull;
            cpu.set_r32(reg::RAX, static_cast<u32>(lo));
            cpu.set_r32(reg::RDX, static_cast<u32>(hi));
            overflow = (static_cast<i64>(static_cast<i32>(lo)) != full);
            break;
        }
        case 8: {
#if defined(_MSC_VER) && defined(_M_X64)
            __int64 high = 0;
            const __int64 lo_s = _mul128(static_cast<__int64>(a), static_cast<__int64>(b), &high);
            lo = static_cast<u64>(lo_s);
            hi = static_cast<u64>(high);
            overflow = (high != (static_cast<__int64>(lo_s) >> 63));
#else
            lo = a * b;
            hi = 0;
            overflow = false;
#endif
            cpu.set_r64(reg::RAX, lo);
            cpu.set_r64(reg::RDX, hi);
            break;
        }
        default: break;
    }
    cpu.stash_mul_ov(overflow);
}

void op_div(Cpu& cpu, const Insn& insn) {
    u64 divisor = 0;
    if (!read_operand(cpu, insn, insn.src, insn.op_size, divisor)) return;
    if (divisor == 0) {
        cpu.set_fault(FaultKind::DivideError, cpu.rip(), Status::Ok, "DIV by zero");
        return;
    }
    switch (insn.op_size) {
        case 1: {
            const u16 num = static_cast<u16>(cpu.r16(reg::RAX));
            const u16 q = num / divisor;
            const u16 r = num % divisor;
            if (q > 0xFFu) {
                cpu.set_fault(FaultKind::DivideError, cpu.rip(), Status::Ok, "DIV overflow");
                return;
            }
            cpu.set_r8(reg::RAX, static_cast<u8>(q));
            // AH = remainder
            const u64 ax = cpu.r(reg::RAX);
            cpu.set_r64(reg::RAX, (ax & ~0xFF00ull) | (static_cast<u64>(r & 0xFF) << 8));
            return;
        }
        case 2: {
            const u32 num = (static_cast<u32>(cpu.r16(reg::RDX)) << 16) | cpu.r16(reg::RAX);
            const u32 q = num / divisor;
            const u32 r = num % divisor;
            if (q > 0xFFFFu) {
                cpu.set_fault(FaultKind::DivideError, cpu.rip(), Status::Ok, "DIV overflow");
                return;
            }
            cpu.set_r16(reg::RAX, static_cast<u16>(q));
            cpu.set_r16(reg::RDX, static_cast<u16>(r));
            return;
        }
        case 4: {
            const u64 num = (static_cast<u64>(cpu.r32(reg::RDX)) << 32) | cpu.r32(reg::RAX);
            const u64 q = num / divisor;
            const u64 r = num % divisor;
            if (q > 0xFFFFFFFFull) {
                cpu.set_fault(FaultKind::DivideError, cpu.rip(), Status::Ok, "DIV overflow");
                return;
            }
            cpu.set_r32(reg::RAX, static_cast<u32>(q));
            cpu.set_r32(reg::RDX, static_cast<u32>(r));
            return;
        }
        case 8: {
#if defined(_MSC_VER) && defined(_M_X64)
            const u64 hi = cpu.r(reg::RDX);
            const u64 lo = cpu.r(reg::RAX);
            unsigned __int64 q = 0, r = 0;
            // _udiv128 requires hi < divisor; we'll fall back on a soft check.
            if (hi >= divisor) {
                cpu.set_fault(FaultKind::DivideError, cpu.rip(), Status::Ok, "DIV overflow");
                return;
            }
            q = _udiv128(hi, lo, divisor, &r);
            cpu.set_r64(reg::RAX, q);
            cpu.set_r64(reg::RDX, r);
            return;
#else
            const u64 num = cpu.r(reg::RAX);   // 128-bit divide unavailable; degrade to 64-bit
            cpu.set_r64(reg::RAX, num / divisor);
            cpu.set_r64(reg::RDX, num % divisor);
            return;
#endif
        }
    }
}

void op_idiv(Cpu& cpu, const Insn& insn) {
    u64 divisor_u = 0;
    if (!read_operand(cpu, insn, insn.src, insn.op_size, divisor_u)) return;
    if (divisor_u == 0) {
        cpu.set_fault(FaultKind::DivideError, cpu.rip(), Status::Ok, "IDIV by zero");
        return;
    }
    switch (insn.op_size) {
        case 1: {
            const i16 num = static_cast<i16>(cpu.r16(reg::RAX));
            const i8 div8 = static_cast<i8>(divisor_u);
            const i16 q = static_cast<i16>(num / div8);
            const i16 r = static_cast<i16>(num % div8);
            if (q < -128 || q > 127) {
                cpu.set_fault(FaultKind::DivideError, cpu.rip(), Status::Ok, "IDIV overflow");
                return;
            }
            cpu.set_r8(reg::RAX, static_cast<u8>(q));
            const u64 ax = cpu.r(reg::RAX);
            cpu.set_r64(reg::RAX, (ax & ~0xFF00ull) | (static_cast<u64>(static_cast<u8>(r)) << 8));
            return;
        }
        case 2: {
            const i32 num = (static_cast<i32>(static_cast<i16>(cpu.r16(reg::RDX))) << 16)
                          |  static_cast<i32>(static_cast<u16>(cpu.r16(reg::RAX)));
            const i16 div16 = static_cast<i16>(divisor_u);
            const i32 q = num / div16;
            const i32 r = num % div16;
            if (q < INT16_MIN || q > INT16_MAX) {
                cpu.set_fault(FaultKind::DivideError, cpu.rip(), Status::Ok, "IDIV overflow");
                return;
            }
            cpu.set_r16(reg::RAX, static_cast<u16>(q));
            cpu.set_r16(reg::RDX, static_cast<u16>(r));
            return;
        }
        case 4: {
            const i64 num = (static_cast<i64>(static_cast<i32>(cpu.r32(reg::RDX))) << 32)
                          |  static_cast<i64>(static_cast<u32>(cpu.r32(reg::RAX)));
            const i32 div32 = static_cast<i32>(divisor_u);
            const i64 q = num / div32;
            const i64 r = num % div32;
            if (q < INT32_MIN || q > INT32_MAX) {
                cpu.set_fault(FaultKind::DivideError, cpu.rip(), Status::Ok, "IDIV overflow");
                return;
            }
            cpu.set_r32(reg::RAX, static_cast<u32>(q));
            cpu.set_r32(reg::RDX, static_cast<u32>(r));
            return;
        }
        case 8: {
#if defined(_MSC_VER) && defined(_M_X64)
            const i64 hi = static_cast<i64>(cpu.r(reg::RDX));
            const u64 lo = cpu.r(reg::RAX);
            __int64 q = 0, r = 0;
            // _div128 not always available; do simple 64-bit fallback when hi is the sign-extension.
            const i64 div = static_cast<i64>(divisor_u);
            if (hi == (static_cast<i64>(lo) >> 63)) {
                q = static_cast<i64>(lo) / div;
                r = static_cast<i64>(lo) % div;
                cpu.set_r64(reg::RAX, static_cast<u64>(q));
                cpu.set_r64(reg::RDX, static_cast<u64>(r));
                return;
            }
            cpu.set_fault(FaultKind::DivideError, cpu.rip(), Status::Ok, "IDIV128 unsupported");
            return;
#else
            const i64 lo = static_cast<i64>(cpu.r(reg::RAX));
            const i64 div = static_cast<i64>(divisor_u);
            cpu.set_r64(reg::RAX, static_cast<u64>(lo / div));
            cpu.set_r64(reg::RDX, static_cast<u64>(lo % div));
            return;
#endif
        }
    }
}

} // namespace emu::handlers
