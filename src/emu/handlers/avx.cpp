// AVX (VEX-encoded) handlers.
//
// VEX-128 (`op_size = 16`):
//   dst    = YMM[reg].low128 = src1 OP src2,
//   YMM[reg].hi128 = 0       (per VEX semantics)
//   src1 reg id is stashed in `insn.dst.index` (== VEX.vvvv)
//   src2 is `insn.src` (XMM register or memory)
//
// VEX-256 (`op_size = 32`):  the same shape, but over the full 32-byte YMM.
//
// VMOVDQA/U: 2-operand (no vvvv); dst = src (16 or 32 bytes), upper YMM
// zeroed only for the load-into-reg form when L=0.

#include "../handlers/handlers.h"
#include "../handlers/handler_util.h"

#include "emu/cpu.h"

#include <cstring>

namespace emu::handlers {

namespace {

bool read_vec(Cpu& cpu, const Insn& insn, const Operand& op, u8 size, u8* out) noexcept {
    if (op.kind == OperandKind::Xmm) {
        std::memcpy(out, cpu.ymm(op.reg), size);
        return true;
    }
    if (op.kind == OperandKind::Mem || op.kind == OperandKind::RipRelMem) {
        auto* mp = cpu.mem_read();
        if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "vex read"); return false; }
        const GuestAddr addr = ea(cpu, insn, op);
        if (Status s = mp->read(addr, size, out); fail(s)) {
            cpu.set_fault(FaultKind::PageFault, addr, s, "vex read");
            return false;
        }
        return true;
    }
    cpu.set_fault(FaultKind::InvalidOpcode, cpu.rip(), Status::InvalidInstruction, "bad vex src");
    return false;
}

bool write_vec(Cpu& cpu, const Insn& insn, const Operand& op, u8 size, const u8* v) noexcept {
    if (op.kind == OperandKind::Xmm) {
        std::memcpy(cpu.ymm(op.reg), v, size);
        // VEX-128: zero upper 128 bits of the destination YMM.
        if (size == 16) cpu.zero_ymm_upper(op.reg);
        return true;
    }
    if (op.kind == OperandKind::Mem || op.kind == OperandKind::RipRelMem) {
        auto* mp = cpu.mem_write();
        if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, cpu.rip(), Status::ProviderFailure, "vex write"); return false; }
        const GuestAddr addr = ea(cpu, insn, op);
        if (Status s = mp->write(addr, size, v); fail(s)) {
            cpu.set_fault(kind_from_status(s), addr, s, "vex write");
            return false;
        }
        return true;
    }
    cpu.set_fault(FaultKind::InvalidOpcode, cpu.rip(), Status::InvalidInstruction, "bad vex dst");
    return false;
}

template <typename F>
void vex_binop(Cpu& cpu, const Insn& insn, F f) noexcept {
    const u8 size = insn.op_size;          // 16 or 32
    u8 src1[32]={}, src2[32]={}, out[32]={};
    std::memcpy(src1, cpu.ymm(insn.dst.index), size);
    if (!read_vec(cpu, insn, insn.src, size, src2)) return;
    f(src1, src2, out, size);
    (void)write_vec(cpu, insn, insn.dst, size, out);
}

} // namespace

void op_vmovdqa(Cpu& cpu, const Insn& insn) {
    const u8 size = insn.op_size;
    u8 buf[32]={};
    if (!read_vec(cpu, insn, insn.src, size, buf)) return;
    (void)write_vec(cpu, insn, insn.dst, size, buf);
}
void op_vmovdqu(Cpu& cpu, const Insn& insn) { op_vmovdqa(cpu, insn); }

void op_vpxor(Cpu& cpu, const Insn& insn) {
    vex_binop(cpu, insn, [](const u8* a, const u8* b, u8* r, u8 n) {
        for (u8 i = 0; i < n; ++i) r[i] = static_cast<u8>(a[i] ^ b[i]);
    });
}
void op_vpand(Cpu& cpu, const Insn& insn) {
    vex_binop(cpu, insn, [](const u8* a, const u8* b, u8* r, u8 n) {
        for (u8 i = 0; i < n; ++i) r[i] = static_cast<u8>(a[i] & b[i]);
    });
}
void op_vpor(Cpu& cpu, const Insn& insn) {
    vex_binop(cpu, insn, [](const u8* a, const u8* b, u8* r, u8 n) {
        for (u8 i = 0; i < n; ++i) r[i] = static_cast<u8>(a[i] | b[i]);
    });
}

void op_vpaddb(Cpu& cpu, const Insn& insn) {
    vex_binop(cpu, insn, [](const u8* a, const u8* b, u8* r, u8 n) {
        for (u8 i = 0; i < n; ++i) r[i] = static_cast<u8>(a[i] + b[i]);
    });
}
void op_vpaddw(Cpu& cpu, const Insn& insn) {
    vex_binop(cpu, insn, [](const u8* a, const u8* b, u8* r, u8 n) {
        for (u8 i = 0; i < n/2; ++i) {
            const u16 av = u16(a[2*i]) | (u16(a[2*i+1]) << 8);
            const u16 bv = u16(b[2*i]) | (u16(b[2*i+1]) << 8);
            const u16 sum = static_cast<u16>(av + bv);
            r[2*i]   = static_cast<u8>(sum & 0xFFu);
            r[2*i+1] = static_cast<u8>((sum >> 8) & 0xFFu);
        }
    });
}
void op_vpaddd(Cpu& cpu, const Insn& insn) {
    vex_binop(cpu, insn, [](const u8* a, const u8* b, u8* r, u8 n) {
        for (u8 i = 0; i < n/4; ++i) {
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
void op_vpaddq(Cpu& cpu, const Insn& insn) {
    vex_binop(cpu, insn, [](const u8* a, const u8* b, u8* r, u8 n) {
        for (u8 i = 0; i < n/8; ++i) {
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

void op_vpsubb(Cpu& cpu, const Insn& insn) {
    vex_binop(cpu, insn, [](const u8* a, const u8* b, u8* r, u8 n) {
        for (u8 i = 0; i < n; ++i) r[i] = static_cast<u8>(a[i] - b[i]);
    });
}
void op_vpsubw(Cpu& cpu, const Insn& insn) {
    vex_binop(cpu, insn, [](const u8* a, const u8* b, u8* r, u8 n) {
        for (u8 i = 0; i < n/2; ++i) {
            const u16 av = u16(a[2*i]) | (u16(a[2*i+1]) << 8);
            const u16 bv = u16(b[2*i]) | (u16(b[2*i+1]) << 8);
            const u16 d = static_cast<u16>(av - bv);
            r[2*i]   = static_cast<u8>(d & 0xFFu);
            r[2*i+1] = static_cast<u8>((d >> 8) & 0xFFu);
        }
    });
}
void op_vpsubd(Cpu& cpu, const Insn& insn) {
    vex_binop(cpu, insn, [](const u8* a, const u8* b, u8* r, u8 n) {
        for (u8 i = 0; i < n/4; ++i) {
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
void op_vpsubq(Cpu& cpu, const Insn& insn) {
    vex_binop(cpu, insn, [](const u8* a, const u8* b, u8* r, u8 n) {
        for (u8 i = 0; i < n/8; ++i) {
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

// VZEROUPPER (`C5 F8 77`): zero upper 128 bits of all YMM regs.
void op_vzeroupper(Cpu& cpu, const Insn&) {
    for (u8 i = 0; i < 16; ++i) cpu.zero_ymm_upper(i);
}
// VZEROALL  (`C5 FC 77`): zero entire YMM regs.
void op_vzeroall(Cpu& cpu, const Insn&) {
    for (u8 i = 0; i < 16; ++i) {
        std::memset(cpu.ymm(i), 0, 32);
    }
}

} // namespace emu::handlers
