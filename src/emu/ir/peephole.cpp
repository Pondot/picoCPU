// Junk-fold peephole pass.

#include "emu/peephole.h"

#include "../handlers/handlers.h"

#include <atomic>

namespace emu {

namespace {

std::atomic<u64> g_folded{0};

bool is_self_mov(const Insn& i) noexcept {
    return i.kind == OpKind::Mov
        && i.dst.kind == OperandKind::Reg
        && i.src.kind == OperandKind::Reg
        && i.dst.reg  == i.src.reg;
}

bool is_self_xchg(const Insn& i) noexcept {
    return i.kind == OpKind::Xchg
        && i.dst.kind == OperandKind::Reg
        && i.src.kind == OperandKind::Reg
        && i.dst.reg  == i.src.reg;
}

bool is_self_lea(const Insn& i) noexcept {
    return i.kind == OpKind::Lea
        && i.dst.kind == OperandKind::Reg
        && i.src.kind == OperandKind::Mem
        && i.src.reg == i.dst.reg
        && i.src.index == reg::NONE
        && i.src.imm == 0;
}

void make_nop(Insn& i) noexcept {
    i.kind    = OpKind::Nop;
    i.handler = &handlers::op_nop;
    // Operand fields left untouched; op_nop ignores them.
    g_folded.fetch_add(1, std::memory_order_relaxed);
}

} // namespace

// Opaque-predicate fold: after `xor reg, reg` we know
//   ZF=1, CF=0, SF=0, OF=0
// so any Jcc whose condition is implied by those flags can be folded.
// We only convert "always-taken" Jcc -> unconditional Jmp here (the always-
// not-taken case would require dropping the block terminator, which is more
// invasive -- handled in Phase 7).
void fold_opaque_jcc(DecodedBlock& block) noexcept {
    if (block.insns.size() < 2) return;
    for (size_t i = 0; i + 1 < block.insns.size(); ++i) {
        Insn& a = block.insns[i];
        Insn& b = block.insns[i + 1];
        const bool xor_self = a.kind == OpKind::Xor
            && a.dst.kind == OperandKind::Reg
            && a.src.kind == OperandKind::Reg
            && a.dst.reg  == a.src.reg;
        if (!xor_self) continue;
        if (b.kind != OpKind::Jcc) continue;

        // Known flags after xor self: ZF=1, CF=0, SF=0, OF=0, AF=0.
        bool always_taken = false;
        switch (b.cond) {
            case Cc::Z:   case Cc::BE: case Cc::LE:
            case Cc::NB:  case Cc::NS: case Cc::NO: case Cc::NL:
                always_taken = true; break;
            default: break;
        }
        if (always_taken) {
            b.kind    = OpKind::Jmp;
            b.handler = &handlers::op_jmp;
            b.cond    = Cc::None;
            g_folded.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void peephole_block(DecodedBlock& block) noexcept {
    for (Insn& i : block.insns) {
        if (is_self_mov(i) || is_self_xchg(i) || is_self_lea(i)) {
            make_nop(i);
        }
    }
    fold_opaque_jcc(block);
}

u64 peephole_folded_count() noexcept {
    return g_folded.load(std::memory_order_relaxed);
}

} // namespace emu
