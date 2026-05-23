// SSE / SSE2 floating-point handlers.
//
// Encoding map (handled by the decoder, not these handlers):
//   F3 0F xx  ->  scalar single  (SS) -- only lane 0 active; lanes 1-3 untouched
//   F2 0F xx  ->  scalar double  (SD) -- only lane 0 active
//   0F xx     ->  packed single  (PS) -- 4 × f32
//   66 0F xx  ->  packed double  (PD) -- 2 × f64
//
// The CPU's XMM registers are raw byte arrays; we reinterpret as float / double
// via memcpy (the standard C++17 way that avoids UB).

#include "../handlers/handlers.h"
#include "../handlers/handler_util.h"

#include "emu/cpu.h"

#include <cmath>
#include <cstring>

namespace emu::handlers {

namespace {

inline float  f32_from_bits(u32 b) noexcept { float  f; std::memcpy(&f, &b, 4); return f; }
inline u32    bits_from_f32(float  f) noexcept { u32 b; std::memcpy(&b, &f, 4); return b; }
inline double f64_from_bits(u64 b) noexcept { double d; std::memcpy(&d, &b, 8); return d; }
inline u64    bits_from_f64(double d) noexcept { u64 b; std::memcpy(&b, &d, 8); return b; }

// Read 16 bytes of an XMM-or-memory operand.
bool read_x16(Cpu& cpu, const Insn& insn, const Operand& op, u8 out[16]) noexcept {
    if (op.kind == OperandKind::Xmm) {
        std::memcpy(out, cpu.xmm(op.reg), 16);
        return true;
    }
    if (op.kind == OperandKind::Mem || op.kind == OperandKind::RipRelMem) {
        auto* mp = cpu.mem_read();
        if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "fp read"); return false; }
        const GuestAddr addr = ea(cpu, insn, op);
        if (Status s = mp->read(addr, 16, out); fail(s)) {
            cpu.set_fault(FaultKind::PageFault, addr, s, "fp read");
            return false;
        }
        return true;
    }
    cpu.set_fault(FaultKind::InvalidOpcode, cpu.rip(), Status::InvalidInstruction, "bad fp src");
    return false;
}

// Read only `size` bytes (4 or 8) for scalar ops -- but the source can still
// be a 16-byte XMM register or memory. We read the low `size` bytes.
bool read_scalar(Cpu& cpu, const Insn& insn, const Operand& op, u8 size, u8* out) noexcept {
    if (op.kind == OperandKind::Xmm) {
        std::memcpy(out, cpu.xmm(op.reg), size);
        return true;
    }
    if (op.kind == OperandKind::Mem || op.kind == OperandKind::RipRelMem) {
        auto* mp = cpu.mem_read();
        if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "fp scalar read"); return false; }
        const GuestAddr addr = ea(cpu, insn, op);
        if (Status s = mp->read(addr, size, out); fail(s)) {
            cpu.set_fault(FaultKind::PageFault, addr, s, "fp scalar read");
            return false;
        }
        return true;
    }
    if (op.kind == OperandKind::Reg) {
        // Integer-source operand (used by CVTSI2SS/CVTSI2SD).
        const u64 v = cpu.r(op.reg);
        for (int i = 0; i < size; ++i) out[i] = static_cast<u8>((v >> (8 * i)) & 0xFFu);
        return true;
    }
    cpu.set_fault(FaultKind::InvalidOpcode, cpu.rip(), Status::InvalidInstruction, "bad scalar src");
    return false;
}

bool write_x16(Cpu& cpu, const Insn& insn, const Operand& op, const u8 v[16]) noexcept {
    if (op.kind == OperandKind::Xmm) {
        std::memcpy(cpu.xmm(op.reg), v, 16);
        return true;
    }
    if (op.kind == OperandKind::Mem || op.kind == OperandKind::RipRelMem) {
        auto* mp = cpu.mem_write();
        if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "fp write"); return false; }
        const GuestAddr addr = ea(cpu, insn, op);
        if (Status s = mp->write(addr, 16, v); fail(s)) {
            cpu.set_fault(kind_from_status(s), addr, s, "fp write");
            return false;
        }
        return true;
    }
    cpu.set_fault(FaultKind::InvalidOpcode, cpu.rip(), Status::InvalidInstruction, "bad fp dst");
    return false;
}

// Scalar-op driver: read src's low `size` bytes, read dst's full xmm, perform
// op on lane 0, write back full xmm (lanes 1..n preserved).
template <typename Op32, typename Op64>
void scalar_op(Cpu& cpu, const Insn& insn, u8 size, Op32 op32, Op64 op64) noexcept {
    u8 dst_bytes[16]; if (!read_x16(cpu, insn, insn.dst, dst_bytes)) return;
    u8 src_bytes[8]  = {}; if (!read_scalar(cpu, insn, insn.src, size, src_bytes)) return;
    if (size == 4) {
        u32 a_bits = 0, b_bits = 0;
        std::memcpy(&a_bits, dst_bytes, 4);
        std::memcpy(&b_bits, src_bytes, 4);
        const float r = op32(f32_from_bits(a_bits), f32_from_bits(b_bits));
        const u32 rb  = bits_from_f32(r);
        std::memcpy(dst_bytes, &rb, 4);
    } else {
        u64 a_bits = 0, b_bits = 0;
        std::memcpy(&a_bits, dst_bytes, 8);
        std::memcpy(&b_bits, src_bytes, 8);
        const double r = op64(f64_from_bits(a_bits), f64_from_bits(b_bits));
        const u64 rb   = bits_from_f64(r);
        std::memcpy(dst_bytes, &rb, 8);
    }
    (void)write_x16(cpu, insn, insn.dst, dst_bytes);
}

// Packed-op driver.
template <typename Op32, typename Op64>
void packed_op(Cpu& cpu, const Insn& insn, bool dbl, Op32 op32, Op64 op64) noexcept {
    u8 a[16], b[16];
    if (!read_x16(cpu, insn, insn.dst, a)) return;
    if (!read_x16(cpu, insn, insn.src, b)) return;
    if (!dbl) {
        for (int i = 0; i < 4; ++i) {
            u32 ab = 0, bb = 0;
            std::memcpy(&ab, a + 4*i, 4);
            std::memcpy(&bb, b + 4*i, 4);
            const float r  = op32(f32_from_bits(ab), f32_from_bits(bb));
            const u32 rb   = bits_from_f32(r);
            std::memcpy(a + 4*i, &rb, 4);
        }
    } else {
        for (int i = 0; i < 2; ++i) {
            u64 ab = 0, bb = 0;
            std::memcpy(&ab, a + 8*i, 8);
            std::memcpy(&bb, b + 8*i, 8);
            const double r = op64(f64_from_bits(ab), f64_from_bits(bb));
            const u64 rb   = bits_from_f64(r);
            std::memcpy(a + 8*i, &rb, 8);
        }
    }
    (void)write_x16(cpu, insn, insn.dst, a);
}

} // namespace

// ---- Scalar single (SS) -----------------------------------------------------

void op_movss(Cpu& cpu, const Insn& insn) {
    // Reg-to-reg MOVSS preserves upper 96 bits of dst.
    // Mem-to-reg MOVSS zeroes upper 96 bits.
    // Reg-to-mem MOVSS writes only the low 32 bits.
    if (insn.dst.kind == OperandKind::Xmm && insn.src.kind == OperandKind::Xmm) {
        std::memcpy(cpu.xmm(insn.dst.reg), cpu.xmm(insn.src.reg), 4);
        return;
    }
    if (insn.dst.kind == OperandKind::Xmm) {
        u8 src[4] = {};
        if (!read_scalar(cpu, insn, insn.src, 4, src)) return;
        u8 dst[16] = {};
        std::memcpy(dst, src, 4);   // upper 12 bytes zero
        (void)write_x16(cpu, insn, insn.dst, dst);
        return;
    }
    // store: only low 4 bytes of src to memory
    u8 src_full[16]; if (!read_x16(cpu, insn, insn.src, src_full)) return;
    auto* mp = cpu.mem_write();
    if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "MOVSS store"); return; }
    const GuestAddr addr = ea(cpu, insn, insn.dst);
    if (Status s = mp->write(addr, 4, src_full); fail(s)) cpu.set_fault(kind_from_status(s), addr, s, "MOVSS store");
}
void op_movsd_fp(Cpu& cpu, const Insn& insn) {
    if (insn.dst.kind == OperandKind::Xmm && insn.src.kind == OperandKind::Xmm) {
        std::memcpy(cpu.xmm(insn.dst.reg), cpu.xmm(insn.src.reg), 8);
        return;
    }
    if (insn.dst.kind == OperandKind::Xmm) {
        u8 src[8] = {};
        if (!read_scalar(cpu, insn, insn.src, 8, src)) return;
        u8 dst[16] = {};
        std::memcpy(dst, src, 8);
        (void)write_x16(cpu, insn, insn.dst, dst);
        return;
    }
    u8 src_full[16]; if (!read_x16(cpu, insn, insn.src, src_full)) return;
    auto* mp = cpu.mem_write();
    if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "MOVSD store"); return; }
    const GuestAddr addr = ea(cpu, insn, insn.dst);
    if (Status s = mp->write(addr, 8, src_full); fail(s)) cpu.set_fault(kind_from_status(s), addr, s, "MOVSD store");
}

void op_addss(Cpu& cpu, const Insn& insn) { scalar_op(cpu, insn, 4, [](float a, float b){return a+b;}, [](double,double){return 0.0;}); }
void op_subss(Cpu& cpu, const Insn& insn) { scalar_op(cpu, insn, 4, [](float a, float b){return a-b;}, [](double,double){return 0.0;}); }
void op_mulss(Cpu& cpu, const Insn& insn) { scalar_op(cpu, insn, 4, [](float a, float b){return a*b;}, [](double,double){return 0.0;}); }
void op_divss(Cpu& cpu, const Insn& insn) { scalar_op(cpu, insn, 4, [](float a, float b){return a/b;}, [](double,double){return 0.0;}); }
void op_sqrtss(Cpu& cpu, const Insn& insn){ scalar_op(cpu, insn, 4, [](float, float b){return std::sqrt(b);}, [](double,double){return 0.0;}); }
void op_minss(Cpu& cpu, const Insn& insn) { scalar_op(cpu, insn, 4, [](float a, float b){return a<b?a:b;}, [](double,double){return 0.0;}); }
void op_maxss(Cpu& cpu, const Insn& insn) { scalar_op(cpu, insn, 4, [](float a, float b){return a>b?a:b;}, [](double,double){return 0.0;}); }

void op_addsd(Cpu& cpu, const Insn& insn) { scalar_op(cpu, insn, 8, [](float,float){return 0.0f;}, [](double a, double b){return a+b;}); }
void op_subsd(Cpu& cpu, const Insn& insn) { scalar_op(cpu, insn, 8, [](float,float){return 0.0f;}, [](double a, double b){return a-b;}); }
void op_mulsd(Cpu& cpu, const Insn& insn) { scalar_op(cpu, insn, 8, [](float,float){return 0.0f;}, [](double a, double b){return a*b;}); }
void op_divsd(Cpu& cpu, const Insn& insn) { scalar_op(cpu, insn, 8, [](float,float){return 0.0f;}, [](double a, double b){return a/b;}); }
void op_sqrtsd(Cpu& cpu, const Insn& insn){ scalar_op(cpu, insn, 8, [](float,float){return 0.0f;}, [](double, double b){return std::sqrt(b);}); }
void op_minsd(Cpu& cpu, const Insn& insn) { scalar_op(cpu, insn, 8, [](float,float){return 0.0f;}, [](double a, double b){return a<b?a:b;}); }
void op_maxsd(Cpu& cpu, const Insn& insn) { scalar_op(cpu, insn, 8, [](float,float){return 0.0f;}, [](double a, double b){return a>b?a:b;}); }

// UCOMISS / UCOMISD: set EFLAGS based on the comparison.
//   ZF=1, PF=1, CF=1 -> unordered (NaN)
//   ZF=0, PF=0, CF=0 -> greater
//   ZF=0, PF=0, CF=1 -> less
//   ZF=1, PF=0, CF=0 -> equal
// AF, SF, OF are cleared.
namespace {
void set_compare_flags(Cpu& cpu, bool unordered, bool less, bool equal) noexcept {
    cpu.force_sync_flags();
    cpu.set_zf(unordered || equal);
    cpu.set_pf(unordered);
    cpu.set_cf(unordered || less);
    cpu.set_af(false);
    cpu.set_sf(false);
    cpu.set_of(false);
}
} // namespace

void op_ucomiss(Cpu& cpu, const Insn& insn) {
    u8 a_bytes[4]={}, b_bytes[4]={};
    if (!read_scalar(cpu, insn, insn.dst, 4, a_bytes)) return;
    if (!read_scalar(cpu, insn, insn.src, 4, b_bytes)) return;
    u32 ab, bb; std::memcpy(&ab, a_bytes, 4); std::memcpy(&bb, b_bytes, 4);
    const float a = f32_from_bits(ab), b = f32_from_bits(bb);
    const bool unordered = std::isnan(a) || std::isnan(b);
    set_compare_flags(cpu, unordered, !unordered && a < b, !unordered && a == b);
}

void op_ucomisd(Cpu& cpu, const Insn& insn) {
    u8 a_bytes[8]={}, b_bytes[8]={};
    if (!read_scalar(cpu, insn, insn.dst, 8, a_bytes)) return;
    if (!read_scalar(cpu, insn, insn.src, 8, b_bytes)) return;
    u64 ab, bb; std::memcpy(&ab, a_bytes, 8); std::memcpy(&bb, b_bytes, 8);
    const double a = f64_from_bits(ab), b = f64_from_bits(bb);
    const bool unordered = std::isnan(a) || std::isnan(b);
    set_compare_flags(cpu, unordered, !unordered && a < b, !unordered && a == b);
}

// CVTSI2SS / CVTSI2SD -- integer source (32 or 64-bit) -> float / double in dst[0]
void op_cvtsi2ss(Cpu& cpu, const Insn& insn) {
    u8 src[8]={};
    if (!read_scalar(cpu, insn, insn.src, insn.op_size, src)) return;
    i64 v = 0;
    if (insn.op_size == 4) { i32 t; std::memcpy(&t, src, 4); v = t; }
    else                   { std::memcpy(&v, src, 8); }
    const float f = static_cast<float>(v);
    u8 dst[16]; if (!read_x16(cpu, insn, insn.dst, dst)) return;
    const u32 bits = bits_from_f32(f);
    std::memcpy(dst, &bits, 4);
    (void)write_x16(cpu, insn, insn.dst, dst);
}
void op_cvtsi2sd(Cpu& cpu, const Insn& insn) {
    u8 src[8]={};
    if (!read_scalar(cpu, insn, insn.src, insn.op_size, src)) return;
    i64 v = 0;
    if (insn.op_size == 4) { i32 t; std::memcpy(&t, src, 4); v = t; }
    else                   { std::memcpy(&v, src, 8); }
    const double d = static_cast<double>(v);
    u8 dst[16]; if (!read_x16(cpu, insn, insn.dst, dst)) return;
    const u64 bits = bits_from_f64(d);
    std::memcpy(dst, &bits, 8);
    (void)write_x16(cpu, insn, insn.dst, dst);
}

// CVTTSS2SI / CVTTSD2SI -- truncated float -> signed integer.
void op_cvttss2si(Cpu& cpu, const Insn& insn) {
    u8 src[4]={};
    if (!read_scalar(cpu, insn, insn.src, 4, src)) return;
    u32 b; std::memcpy(&b, src, 4);
    const float f = f32_from_bits(b);
    const i64 v = static_cast<i64>(f);    // truncate toward zero
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, static_cast<u64>(v));
}
void op_cvttsd2si(Cpu& cpu, const Insn& insn) {
    u8 src[8]={};
    if (!read_scalar(cpu, insn, insn.src, 8, src)) return;
    u64 b; std::memcpy(&b, src, 8);
    const double d = f64_from_bits(b);
    const i64 v = static_cast<i64>(d);
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, static_cast<u64>(v));
}

// CVTSS2SD / CVTSD2SS -- single ↔ double conversion within an XMM.
void op_cvtss2sd(Cpu& cpu, const Insn& insn) {
    u8 src[4]={};
    if (!read_scalar(cpu, insn, insn.src, 4, src)) return;
    u32 b; std::memcpy(&b, src, 4);
    const double d = static_cast<double>(f32_from_bits(b));
    u8 dst[16]; if (!read_x16(cpu, insn, insn.dst, dst)) return;
    const u64 bits = bits_from_f64(d);
    std::memcpy(dst, &bits, 8);
    (void)write_x16(cpu, insn, insn.dst, dst);
}
void op_cvtsd2ss(Cpu& cpu, const Insn& insn) {
    u8 src[8]={};
    if (!read_scalar(cpu, insn, insn.src, 8, src)) return;
    u64 b; std::memcpy(&b, src, 8);
    const float f = static_cast<float>(f64_from_bits(b));
    u8 dst[16]; if (!read_x16(cpu, insn, insn.dst, dst)) return;
    const u32 bits = bits_from_f32(f);
    std::memcpy(dst, &bits, 4);
    (void)write_x16(cpu, insn, insn.dst, dst);
}

// ---- Packed -----------------------------------------------------------------
void op_movups(Cpu& cpu, const Insn& insn) {
    u8 v[16]; if (!read_x16(cpu, insn, insn.src, v)) return;
    (void)write_x16(cpu, insn, insn.dst, v);
}
void op_movupd(Cpu& cpu, const Insn& insn) { op_movups(cpu, insn); }

void op_addps(Cpu& cpu, const Insn& insn) { packed_op(cpu, insn, false, [](float a, float b){return a+b;}, [](double,double){return 0.0;}); }
void op_subps(Cpu& cpu, const Insn& insn) { packed_op(cpu, insn, false, [](float a, float b){return a-b;}, [](double,double){return 0.0;}); }
void op_mulps(Cpu& cpu, const Insn& insn) { packed_op(cpu, insn, false, [](float a, float b){return a*b;}, [](double,double){return 0.0;}); }
void op_divps(Cpu& cpu, const Insn& insn) { packed_op(cpu, insn, false, [](float a, float b){return a/b;}, [](double,double){return 0.0;}); }
void op_sqrtps(Cpu& cpu, const Insn& insn){ packed_op(cpu, insn, false, [](float, float b){return std::sqrt(b);}, [](double,double){return 0.0;}); }

void op_addpd(Cpu& cpu, const Insn& insn) { packed_op(cpu, insn, true,  [](float,float){return 0.0f;}, [](double a, double b){return a+b;}); }
void op_subpd(Cpu& cpu, const Insn& insn) { packed_op(cpu, insn, true,  [](float,float){return 0.0f;}, [](double a, double b){return a-b;}); }
void op_mulpd(Cpu& cpu, const Insn& insn) { packed_op(cpu, insn, true,  [](float,float){return 0.0f;}, [](double a, double b){return a*b;}); }
void op_divpd(Cpu& cpu, const Insn& insn) { packed_op(cpu, insn, true,  [](float,float){return 0.0f;}, [](double a, double b){return a/b;}); }
void op_sqrtpd(Cpu& cpu, const Insn& insn){ packed_op(cpu, insn, true,  [](float,float){return 0.0f;}, [](double, double b){return std::sqrt(b);}); }

// XORPS/PD/ANDPS/PD/ORPS/PD operate on raw bits, identical to PXOR/PAND/POR.
void op_xorps(Cpu& cpu, const Insn& insn) { return op_pxor(cpu, insn); }
void op_xorpd(Cpu& cpu, const Insn& insn) { return op_pxor(cpu, insn); }
void op_andps(Cpu& cpu, const Insn& insn) { return op_pand(cpu, insn); }
void op_orps (Cpu& cpu, const Insn& insn) { return op_por (cpu, insn); }

} // namespace emu::handlers
