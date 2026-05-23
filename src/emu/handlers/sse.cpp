// SSE / SSE2 handlers.
//
// All operate on 16-byte XMM operands. The source operand is either another
// XMM register or 16 bytes of guest memory.
//
// Flags are not touched by any of these ops (Intel SDM: SSE integer ops
// leave RFLAGS untouched).

#include "../handlers/handlers.h"
#include "../handlers/handler_util.h"

#include "emu/cpu.h"

#include <cstring>

namespace emu::handlers {

namespace {

bool read_xmm16(Cpu& cpu, const Insn& insn, const Operand& op, u8 out[16]) noexcept {
    if (op.kind == OperandKind::Xmm) {
        std::memcpy(out, cpu.xmm(op.reg), 16);
        return true;
    }
    if (op.kind == OperandKind::Mem || op.kind == OperandKind::RipRelMem) {
        auto* mp = cpu.mem_read();
        if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "xmm read"); return false; }
        const GuestAddr addr = ea(cpu, insn, op);
        if (Status s = mp->read(addr, 16, out); fail(s)) {
            cpu.set_fault(FaultKind::PageFault, addr, s, "xmm read");
            return false;
        }
        return true;
    }
    cpu.set_fault(FaultKind::InvalidOpcode, cpu.rip(), Status::InvalidInstruction, "bad xmm src");
    return false;
}

bool write_xmm16(Cpu& cpu, const Insn& insn, const Operand& op, const u8 v[16]) noexcept {
    if (op.kind == OperandKind::Xmm) {
        std::memcpy(cpu.xmm(op.reg), v, 16);
        return true;
    }
    if (op.kind == OperandKind::Mem || op.kind == OperandKind::RipRelMem) {
        auto* mp = cpu.mem_write();
        if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "xmm write"); return false; }
        const GuestAddr addr = ea(cpu, insn, op);
        if (Status s = mp->write(addr, 16, v); fail(s)) {
            cpu.set_fault(kind_from_status(s), addr, s, "xmm write");
            return false;
        }
        return true;
    }
    cpu.set_fault(FaultKind::InvalidOpcode, cpu.rip(), Status::InvalidInstruction, "bad xmm dst");
    return false;
}

template <typename Op>
void binary_xmm(Cpu& cpu, const Insn& insn, Op op) noexcept {
    u8 a[16], b[16], r[16];
    if (!read_xmm16(cpu, insn, insn.dst, a)) return;
    if (!read_xmm16(cpu, insn, insn.src, b)) return;
    op(a, b, r);
    (void)write_xmm16(cpu, insn, insn.dst, r);
}

} // namespace

void op_movdqa(Cpu& cpu, const Insn& insn) {
    u8 v[16];
    if (!read_xmm16(cpu, insn, insn.src, v)) return;
    (void)write_xmm16(cpu, insn, insn.dst, v);
}
void op_movdqu(Cpu& cpu, const Insn& insn) { op_movdqa(cpu, insn); }   // identical without alignment check

void op_pxor(Cpu& cpu, const Insn& insn) {
    binary_xmm(cpu, insn, [](const u8* a, const u8* b, u8* r) {
        for (int i = 0; i < 16; ++i) r[i] = static_cast<u8>(a[i] ^ b[i]);
    });
}
void op_pand(Cpu& cpu, const Insn& insn) {
    binary_xmm(cpu, insn, [](const u8* a, const u8* b, u8* r) {
        for (int i = 0; i < 16; ++i) r[i] = static_cast<u8>(a[i] & b[i]);
    });
}
void op_por(Cpu& cpu, const Insn& insn) {
    binary_xmm(cpu, insn, [](const u8* a, const u8* b, u8* r) {
        for (int i = 0; i < 16; ++i) r[i] = static_cast<u8>(a[i] | b[i]);
    });
}

void op_paddb(Cpu& cpu, const Insn& insn) {
    binary_xmm(cpu, insn, [](const u8* a, const u8* b, u8* r) {
        for (int i = 0; i < 16; ++i) r[i] = static_cast<u8>(a[i] + b[i]);
    });
}
void op_paddw(Cpu& cpu, const Insn& insn) {
    binary_xmm(cpu, insn, [](const u8* a, const u8* b, u8* r) {
        for (int i = 0; i < 8; ++i) {
            const u16 av = u16(a[2*i]) | (u16(a[2*i+1]) << 8);
            const u16 bv = u16(b[2*i]) | (u16(b[2*i+1]) << 8);
            const u16 sum = static_cast<u16>(av + bv);
            r[2*i]   = static_cast<u8>(sum & 0xFFu);
            r[2*i+1] = static_cast<u8>((sum >> 8) & 0xFFu);
        }
    });
}
void op_paddd(Cpu& cpu, const Insn& insn) {
    binary_xmm(cpu, insn, [](const u8* a, const u8* b, u8* r) {
        for (int i = 0; i < 4; ++i) {
            u32 av = 0, bv = 0;
            for (int j = 0; j < 4; ++j) {
                av |= (u32(a[4*i+j]) << (8*j));
                bv |= (u32(b[4*i+j]) << (8*j));
            }
            const u32 sum = av + bv;
            for (int j = 0; j < 4; ++j) r[4*i+j] = static_cast<u8>((sum >> (8*j)) & 0xFFu);
        }
    });
}
void op_paddq(Cpu& cpu, const Insn& insn) {
    binary_xmm(cpu, insn, [](const u8* a, const u8* b, u8* r) {
        for (int i = 0; i < 2; ++i) {
            u64 av = 0, bv = 0;
            for (int j = 0; j < 8; ++j) {
                av |= (u64(a[8*i+j]) << (8*j));
                bv |= (u64(b[8*i+j]) << (8*j));
            }
            const u64 sum = av + bv;
            for (int j = 0; j < 8; ++j) r[8*i+j] = static_cast<u8>((sum >> (8*j)) & 0xFFu);
        }
    });
}

void op_psubb(Cpu& cpu, const Insn& insn) {
    binary_xmm(cpu, insn, [](const u8* a, const u8* b, u8* r) {
        for (int i = 0; i < 16; ++i) r[i] = static_cast<u8>(a[i] - b[i]);
    });
}
void op_psubw(Cpu& cpu, const Insn& insn) {
    binary_xmm(cpu, insn, [](const u8* a, const u8* b, u8* r) {
        for (int i = 0; i < 8; ++i) {
            const u16 av = u16(a[2*i]) | (u16(a[2*i+1]) << 8);
            const u16 bv = u16(b[2*i]) | (u16(b[2*i+1]) << 8);
            const u16 d = static_cast<u16>(av - bv);
            r[2*i]   = static_cast<u8>(d & 0xFFu);
            r[2*i+1] = static_cast<u8>((d >> 8) & 0xFFu);
        }
    });
}
void op_psubd(Cpu& cpu, const Insn& insn) {
    binary_xmm(cpu, insn, [](const u8* a, const u8* b, u8* r) {
        for (int i = 0; i < 4; ++i) {
            u32 av = 0, bv = 0;
            for (int j = 0; j < 4; ++j) {
                av |= (u32(a[4*i+j]) << (8*j));
                bv |= (u32(b[4*i+j]) << (8*j));
            }
            const u32 d = av - bv;
            for (int j = 0; j < 4; ++j) r[4*i+j] = static_cast<u8>((d >> (8*j)) & 0xFFu);
        }
    });
}
void op_psubq(Cpu& cpu, const Insn& insn) {
    binary_xmm(cpu, insn, [](const u8* a, const u8* b, u8* r) {
        for (int i = 0; i < 2; ++i) {
            u64 av = 0, bv = 0;
            for (int j = 0; j < 8; ++j) {
                av |= (u64(a[8*i+j]) << (8*j));
                bv |= (u64(b[8*i+j]) << (8*j));
            }
            const u64 d = av - bv;
            for (int j = 0; j < 8; ++j) r[8*i+j] = static_cast<u8>((d >> (8*j)) & 0xFFu);
        }
    });
}

// PSHUFD: 4-way 32-bit shuffle controlled by imm8 (two bits per output lane).
// Encoded as `66 0F 70 /r ib`. Source = src (xmm/m128), dst = dst (xmm),
// imm8 in insn.imm_extra.
void op_pshufd(Cpu& cpu, const Insn& insn) {
    u8 s[16], r[16];
    if (!read_xmm16(cpu, insn, insn.src, s)) return;
    const u8 ctrl = static_cast<u8>(insn.imm_extra & 0xFFu);
    for (int lane = 0; lane < 4; ++lane) {
        const u8 sel = (ctrl >> (lane * 2)) & 0x3u;
        for (int j = 0; j < 4; ++j) r[lane * 4 + j] = s[sel * 4 + j];
    }
    (void)write_xmm16(cpu, insn, insn.dst, r);
}

void op_pcmpeqb(Cpu& cpu, const Insn& insn) {
    binary_xmm(cpu, insn, [](const u8* a, const u8* b, u8* r) {
        for (int i = 0; i < 16; ++i) r[i] = (a[i] == b[i]) ? 0xFFu : 0u;
    });
}
void op_pcmpeqw(Cpu& cpu, const Insn& insn) {
    binary_xmm(cpu, insn, [](const u8* a, const u8* b, u8* r) {
        for (int i = 0; i < 8; ++i) {
            const bool eq =  a[2*i] == b[2*i] && a[2*i+1] == b[2*i+1];
            r[2*i]   = eq ? 0xFFu : 0u;
            r[2*i+1] = eq ? 0xFFu : 0u;
        }
    });
}
void op_pcmpeqd(Cpu& cpu, const Insn& insn) {
    binary_xmm(cpu, insn, [](const u8* a, const u8* b, u8* r) {
        for (int i = 0; i < 4; ++i) {
            bool eq = true;
            for (int j = 0; j < 4; ++j) if (a[4*i+j] != b[4*i+j]) { eq = false; break; }
            for (int j = 0; j < 4; ++j) r[4*i+j] = eq ? 0xFFu : 0u;
        }
    });
}

// MOVD/MOVQ xmm, r/m{32,64}  -- 66 0F 6E /r (W=0: 32-bit; W=1: 64-bit).
// Low op_size bytes of dst XMM = source value; upper bytes zeroed.
void op_movd_to_xmm(Cpu& cpu, const Insn& insn) {
    u64 v = 0;
    if (!read_operand(cpu, insn, insn.src, insn.op_size, v)) return;
    u8 out[16] = {};
    for (u8 i = 0; i < insn.op_size; ++i) out[i] = static_cast<u8>((v >> (8 * i)) & 0xFFu);
    std::memcpy(cpu.xmm(insn.dst.reg), out, 16);
}

// MOVD/MOVQ r/m{32,64}, xmm  -- 66 0F 7E /r (W=0: low 32 of xmm; W=1: low 64).
// Also F3 0F 7E /r: MOVQ xmm, xmm/m64 -- handled by decoder routing as
// op_size=8, src=Xmm, dst=Xmm with this same handler reading from src.
void op_movd_from_xmm(Cpu& cpu, const Insn& insn) {
    // The decoder routes both r/m<-xmm and xmm<-xmm/m64 to this handler. Pick
    // the source xmm based on which operand is Xmm: in r/m<-xmm form,
    // src.kind==Xmm; in xmm<-xmm/m64 form (F3 prefix), dst.kind==Xmm and
    // src.kind∈{Xmm,Mem}.
    if (insn.dst.kind == OperandKind::Xmm && insn.src.kind == OperandKind::Xmm
        && insn.op_size == 8) {
        // F3 0F 7E: MOVQ xmm, xmm/m64 -- copy low 8 bytes of src.xmm into
        // dst.xmm, zero upper 8.
        u8 buf[16] = {};
        std::memcpy(buf, cpu.xmm(insn.src.reg), 8);
        std::memcpy(cpu.xmm(insn.dst.reg), buf, 16);
        return;
    }
    if (insn.dst.kind == OperandKind::Xmm
        && (insn.src.kind == OperandKind::Mem || insn.src.kind == OperandKind::RipRelMem)
        && insn.op_size == 8) {
        auto* mp = cpu.mem_read();
        u8 buf[16] = {};
        const GuestAddr addr = ea(cpu, insn, insn.src);
        if (!mp || fail(mp->read(addr, 8, buf))) {
            cpu.set_fault(FaultKind::PageFault, addr, Status::ProviderFailure, "movq xmm m64");
            return;
        }
        std::memcpy(cpu.xmm(insn.dst.reg), buf, 16);
        return;
    }
    // 66 0F 7E form: r/m{32,64} <- xmm (low op_size bytes).
    const u8* xb = cpu.xmm(insn.src.reg);
    u64 v = 0;
    for (u8 i = 0; i < insn.op_size; ++i) v |= (u64{xb[i]} << (8 * i));
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, v);
}

} // namespace emu::handlers
