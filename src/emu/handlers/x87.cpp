// x87 FPU -- minimal subset.
//
// We approximate 80-bit extended precision with host doubles. That's close
// enough for compiler-emitted x87 (which mostly does double math and converts
// to long-double only at function boundaries) and dramatically simpler.
//
// Convention:
//   - `insn.dst` either an Xmm-ish (we don't use that) or empty; `op.reg`
//      holds the ST(i) index for register-form ops via the secondary stash:
//      decoder packs the i index in `insn.dst.index`.
//   - `insn.src` is either a Mem operand (for memory-form ops with operand
//      size 4/8/2/etc) or empty.
//   - `insn.op_size` carries memory-operand width (4 = float, 8 = double,
//      10 = extended -- we treat 10 the same as 8 with precision loss,
//      2 = i16, 4 = i32, 8 = i64 for FILD/FISTP).

#include "../handlers/handlers.h"
#include "../handlers/handler_util.h"

#include "emu/cpu.h"

#include <cmath>
#include <cstring>

namespace emu::handlers {

namespace {

inline u8 st_index_from(const Insn& insn) noexcept {
    return insn.dst.index;     // decoder stashes the ST(i) reg index here
}

bool read_mem_f(Cpu& cpu, const Insn& insn, double& out) noexcept {
    auto* mp = cpu.mem_read();
    if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "x87 read"); return false; }
    const GuestAddr addr = ea(cpu, insn, insn.src);
    if (insn.op_size == 4) {
        u8 b[4]={};
        if (Status s = mp->read(addr, 4, b); fail(s)) { cpu.set_fault(FaultKind::PageFault, addr, s, "x87 read"); return false; }
        float f; std::memcpy(&f, b, 4); out = static_cast<double>(f);
    } else if (insn.op_size == 8) {
        u8 b[8]={};
        if (Status s = mp->read(addr, 8, b); fail(s)) { cpu.set_fault(FaultKind::PageFault, addr, s, "x87 read"); return false; }
        std::memcpy(&out, b, 8);
    } else if (insn.op_size == 10) {
        // 80-bit: read low 64 bits of mantissa as double approximation.
        u8 b[10]={};
        if (Status s = mp->read(addr, 10, b); fail(s)) { cpu.set_fault(FaultKind::PageFault, addr, s, "x87 read"); return false; }
        // Crude decode: combine sign+exp+mantissa to double.
        u64 mant = 0; std::memcpy(&mant, b, 8);
        u16 sign_exp = 0; std::memcpy(&sign_exp, b + 8, 2);
        const bool sign = (sign_exp & 0x8000) != 0;
        const int exp_unbiased = static_cast<int>(sign_exp & 0x7FFF) - 16383;
        if (exp_unbiased == -16383) { out = sign ? -0.0 : 0.0; return true; }
        const double m = static_cast<double>(mant) / static_cast<double>(1ull << 63);
        out = std::ldexp(m, exp_unbiased);
        if (sign) out = -out;
    } else {
        cpu.set_fault(FaultKind::InvalidOpcode, cpu.rip(), Status::InvalidInstruction, "x87 size");
        return false;
    }
    return true;
}

bool write_mem_f(Cpu& cpu, const Insn& insn, double v) noexcept {
    auto* mp = cpu.mem_write();
    if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "x87 write"); return false; }
    const GuestAddr addr = ea(cpu, insn, insn.src);
    if (insn.op_size == 4) {
        const float f = static_cast<float>(v);
        return ok(mp->write(addr, 4, &f));
    }
    if (insn.op_size == 8) {
        return ok(mp->write(addr, 8, &v));
    }
    if (insn.op_size == 10) {
        u8 b[10] = {};
        // Approximate 80-bit encoding from a double. Pack sign+exp+mantissa.
        u64 dv = 0; std::memcpy(&dv, &v, 8);
        const bool sign = (dv >> 63) != 0;
        const int exp_d = static_cast<int>((dv >> 52) & 0x7FF) - 1023;
        const u64 mant_d = (dv & ((1ull << 52) - 1));
        if (exp_d == -1023 && mant_d == 0) {
            // ±0 -- leave bytes zero except sign in high bit.
            b[9] = sign ? 0x80 : 0x00;
        } else {
            const u64 mant80 = (mant_d << 11) | (1ull << 63);   // explicit leading 1
            const u16 sign_exp = static_cast<u16>((sign ? 0x8000 : 0) | (static_cast<u16>(exp_d + 16383) & 0x7FFF));
            std::memcpy(b, &mant80, 8);
            std::memcpy(b + 8, &sign_exp, 2);
        }
        return ok(mp->write(addr, 10, b));
    }
    cpu.set_fault(FaultKind::InvalidOpcode, cpu.rip(), Status::InvalidInstruction, "x87 size");
    return false;
}

bool read_mem_i(Cpu& cpu, const Insn& insn, i64& out) noexcept {
    auto* mp = cpu.mem_read();
    if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "fild"); return false; }
    const GuestAddr addr = ea(cpu, insn, insn.src);
    u8 b[8]={};
    if (Status s = mp->read(addr, insn.op_size, b); fail(s)) { cpu.set_fault(FaultKind::PageFault, addr, s, "fild"); return false; }
    if (insn.op_size == 2) { i16 v; std::memcpy(&v, b, 2); out = v; }
    else if (insn.op_size == 4) { i32 v; std::memcpy(&v, b, 4); out = v; }
    else if (insn.op_size == 8) { std::memcpy(&out, b, 8); }
    else { cpu.set_fault(FaultKind::InvalidOpcode, cpu.rip(), Status::InvalidInstruction, "fild size"); return false; }
    return true;
}

bool write_mem_i(Cpu& cpu, const Insn& insn, i64 v) noexcept {
    auto* mp = cpu.mem_write();
    if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "fistp"); return false; }
    const GuestAddr addr = ea(cpu, insn, insn.src);
    if (insn.op_size == 2) { i16 x = static_cast<i16>(v); return ok(mp->write(addr, 2, &x)); }
    if (insn.op_size == 4) { i32 x = static_cast<i32>(v); return ok(mp->write(addr, 4, &x)); }
    if (insn.op_size == 8) { return ok(mp->write(addr, 8, &v)); }
    cpu.set_fault(FaultKind::InvalidOpcode, cpu.rip(), Status::InvalidInstruction, "fistp size");
    return false;
}

// Set the C3/C2/C1/C0 condition codes in the status word after FCOM family.
void set_x87_cond(Cpu& cpu, bool gt, bool eq, bool unordered) noexcept {
    u16 sw = cpu.fpu_sw();
    sw &= ~u16(0x4700);   // clear C3 (bit 14), C2 (bit 10), C1 (bit 9), C0 (bit 8)
    if (unordered) sw |= (1u << 14) | (1u << 10) | (1u << 8);   // C3=C2=C0=1
    else if (eq)   sw |=  1u << 14;                              // C3
    else if (gt)   {}                                            // all zero
    else           sw |=  1u << 8;                               // C0 (less)
    cpu.set_fpu_sw(sw);
}

} // namespace

void op_fld(Cpu& cpu, const Insn& insn) {
    if (insn.src.kind == OperandKind::Mem || insn.src.kind == OperandKind::RipRelMem) {
        double v;
        if (!read_mem_f(cpu, insn, v)) return;
        cpu.fpu_push(v);
    } else {
        // Reg form: FLD ST(i)  -- push a copy of ST(i).
        const u8 i = st_index_from(insn);
        cpu.fpu_push(cpu.st(i));
    }
}

void op_fild(Cpu& cpu, const Insn& insn) {
    i64 v;
    if (!read_mem_i(cpu, insn, v)) return;
    cpu.fpu_push(static_cast<double>(v));
}

void op_fst (Cpu& cpu, const Insn& insn) {
    if (insn.src.kind == OperandKind::Mem || insn.src.kind == OperandKind::RipRelMem) {
        (void)write_mem_f(cpu, insn, cpu.st(0));
    } else {
        cpu.set_st(st_index_from(insn), cpu.st(0));
    }
}
void op_fstp(Cpu& cpu, const Insn& insn) {
    if (insn.src.kind == OperandKind::Mem || insn.src.kind == OperandKind::RipRelMem) {
        if (!write_mem_f(cpu, insn, cpu.st(0))) return;
    } else {
        cpu.set_st(st_index_from(insn), cpu.st(0));
    }
    (void)cpu.fpu_pop();
}

void op_fistp(Cpu& cpu, const Insn& insn) {
    const i64 v = static_cast<i64>(cpu.st(0));
    if (!write_mem_i(cpu, insn, v)) return;
    (void)cpu.fpu_pop();
}

namespace {
// Inner: ST(0) op= other, where `other` is either a memory float or ST(i).
template <typename F>
void fp_op_with(Cpu& cpu, const Insn& insn, F f) noexcept {
    double rhs;
    if (insn.src.kind == OperandKind::Mem || insn.src.kind == OperandKind::RipRelMem) {
        if (!read_mem_f(cpu, insn, rhs)) return;
    } else {
        rhs = cpu.st(st_index_from(insn));
    }
    cpu.set_st(0, f(cpu.st(0), rhs));
}
template <typename F>
void fp_op_pop(Cpu& cpu, const Insn& insn, F f) noexcept {
    const u8 i = st_index_from(insn);
    cpu.set_st(i, f(cpu.st(i), cpu.st(0)));
    (void)cpu.fpu_pop();
}
} // namespace

void op_fadd(Cpu& cpu, const Insn& insn) { fp_op_with(cpu, insn, [](double a, double b){ return a + b; }); }
void op_fsub(Cpu& cpu, const Insn& insn) { fp_op_with(cpu, insn, [](double a, double b){ return a - b; }); }
void op_fmul(Cpu& cpu, const Insn& insn) { fp_op_with(cpu, insn, [](double a, double b){ return a * b; }); }
void op_fdiv(Cpu& cpu, const Insn& insn) { fp_op_with(cpu, insn, [](double a, double b){ return a / b; }); }

void op_faddp(Cpu& cpu, const Insn& insn) { fp_op_pop(cpu, insn, [](double a, double b){ return a + b; }); }
void op_fsubp(Cpu& cpu, const Insn& insn) { fp_op_pop(cpu, insn, [](double a, double b){ return a - b; }); }
void op_fmulp(Cpu& cpu, const Insn& insn) { fp_op_pop(cpu, insn, [](double a, double b){ return a * b; }); }
void op_fdivp(Cpu& cpu, const Insn& insn) { fp_op_pop(cpu, insn, [](double a, double b){ return a / b; }); }

void op_fchs(Cpu& cpu, const Insn&)  { cpu.set_st(0, -cpu.st(0)); }
void op_fabs(Cpu& cpu, const Insn&)  { cpu.set_st(0, std::fabs(cpu.st(0))); }
void op_fsqrt(Cpu& cpu, const Insn&) { cpu.set_st(0, std::sqrt(cpu.st(0))); }

void op_fxch(Cpu& cpu, const Insn& insn) {
    const u8 i = st_index_from(insn);
    const double tmp = cpu.st(0);
    cpu.set_st(0, cpu.st(i));
    cpu.set_st(i, tmp);
}

void op_fcom(Cpu& cpu, const Insn& insn) {
    double rhs;
    if (insn.src.kind == OperandKind::Mem || insn.src.kind == OperandKind::RipRelMem) {
        if (!read_mem_f(cpu, insn, rhs)) return;
    } else {
        rhs = cpu.st(st_index_from(insn));
    }
    const double a = cpu.st(0);
    const bool u = std::isnan(a) || std::isnan(rhs);
    set_x87_cond(cpu, !u && a > rhs, !u && a == rhs, u);
}
void op_fcomp (Cpu& cpu, const Insn& insn) { op_fcom(cpu, insn); (void)cpu.fpu_pop(); }
void op_fcompp(Cpu& cpu, const Insn& insn) { op_fcom(cpu, insn); (void)cpu.fpu_pop(); (void)cpu.fpu_pop(); }

void op_fninit(Cpu& cpu, const Insn&) {
    for (u8 i = 0; i < 8; ++i) cpu.set_st(i, 0.0);
    cpu.set_fpu_cw(0x037F);
    cpu.set_fpu_sw(0);
}
void op_fnstsw(Cpu& cpu, const Insn& insn) {
    const u16 sw = cpu.fpu_sw();
    // Two forms: FNSTSW AX (writes to AX register) or FNSTSW m16 (memory).
    if (insn.dst.kind == OperandKind::Reg) {
        cpu.set_r16(reg::RAX, sw);
    } else {
        auto* mp = cpu.mem_write();
        if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "fnstsw"); return; }
        (void)mp->write(ea(cpu, insn, insn.dst), 2, &sw);
    }
}
void op_fnstcw(Cpu& cpu, const Insn& insn) {
    const u16 cw = cpu.fpu_cw();
    auto* mp = cpu.mem_write();
    if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "fnstcw"); return; }
    (void)mp->write(ea(cpu, insn, insn.src), 2, &cw);
}
void op_fldcw(Cpu& cpu, const Insn& insn) {
    auto* mp = cpu.mem_read();
    if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "fldcw"); return; }
    u16 cw = 0;
    if (Status s = mp->read(ea(cpu, insn, insn.src), 2, &cw); fail(s)) {
        cpu.set_fault(FaultKind::PageFault, cpu.rip(), s, "fldcw");
        return;
    }
    cpu.set_fpu_cw(cw);
}

} // namespace emu::handlers
