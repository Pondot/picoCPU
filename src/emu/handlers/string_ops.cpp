// REP-prefixed string operations.
//
// Each handler runs the inner step in a tight loop when a REP/REPE/REPNE
// prefix is present (insn.flags & INSN_FLAG_REP/REPNE). The loop:
//   while (rcx != 0) {
//     do_one_step();
//     --rcx;
//     if (cc_terminates) break;   // REPE/REPNE only
//   }
// Direction (RDI/RSI increment/decrement) follows DF.
//
// Inner loops live entirely inside the handler -- this is the one place the
// dispatcher-cost-per-instruction argument doesn't apply, because a single
// REP can drive thousands of iterations from one decoded instruction.

#include "../handlers/handlers.h"
#include "../handlers/handler_util.h"

#include "emu/cpu.h"

namespace emu::handlers {

namespace {

// Compute step direction (signed step size in bytes) given DF and operand size.
i64 dir_step(const Cpu& cpu, u8 size) noexcept {
    return cpu.df() ? -static_cast<i64>(size) : static_cast<i64>(size);
}

// Read `size` bytes from guest memory through `cpu.mem_read`.
bool read_bytes(Cpu& cpu, GuestAddr addr, u8 size, u64& out) noexcept {
    auto* mp = cpu.mem_read();
    if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, addr, Status::ProviderFailure, "string-op read"); return false; }
    u8 buf[8] = {};
    if (Status s = mp->read(addr, size, buf); fail(s)) {
        cpu.set_fault(FaultKind::PageFault, addr, s, "string-op read");
        return false;
    }
    u64 v = 0; for (int i = 0; i < size; ++i) v |= (u64{buf[i]} << (8 * i));
    out = v;
    return true;
}

bool write_bytes(Cpu& cpu, GuestAddr addr, u8 size, u64 value) noexcept {
    auto* mp = cpu.mem_write();
    if (!mp) { cpu.set_fault(FaultKind::ProviderFailure, addr, Status::ProviderFailure, "string-op write"); return false; }
    u8 buf[8] = {};
    for (int i = 0; i < size; ++i) buf[i] = static_cast<u8>((value >> (8 * i)) & 0xFFu);
    if (Status s = mp->write(addr, size, buf); fail(s)) {
        cpu.set_fault(kind_from_status(s), addr, s, "string-op write");
        return false;
    }
    return true;
}

bool has_rep(const Insn& i) noexcept { return (i.flags & (INSN_FLAG_REP | INSN_FLAG_REPNE)) != 0; }

// Inner-loop driver. `step()` performs one iteration of the string op and
// (for SCAS/CMPS) returns whether ZF was set so REPE/REPNE can decide
// termination.
template <typename Step>
void run_string(Cpu& cpu, const Insn& insn, Step step) noexcept {
    if (has_rep(insn)) {
        const bool repe  = (insn.flags & INSN_FLAG_REP)   != 0;   // F3 prefix
        const bool repne = (insn.flags & INSN_FLAG_REPNE) != 0;   // F2 prefix
        while (cpu.r(reg::RCX) != 0 && !cpu.halted()) {
            bool zf_after = false;
            if (!step(zf_after)) return;
            cpu.set_r64(reg::RCX, cpu.r(reg::RCX) - 1);
            if (repe  && !zf_after) break;     // REPE/REPZ: stop when ZF=0
            if (repne &&  zf_after) break;     // REPNE/REPNZ: stop when ZF=1
        }
    } else {
        bool ignored = false;
        (void)step(ignored);
    }
}

} // namespace

// MOVS -- copy [RSI] -> [RDI] of size op_size.
void op_movs(Cpu& cpu, const Insn& insn) {
    run_string(cpu, insn, [&](bool& /*zf*/) -> bool {
        const GuestAddr rsi = cpu.r(reg::RSI);
        const GuestAddr rdi = cpu.r(reg::RDI);
        u64 v = 0;
        if (!read_bytes(cpu, rsi, insn.op_size, v)) return false;
        if (!write_bytes(cpu, rdi, insn.op_size, v)) return false;
        const i64 step = dir_step(cpu, insn.op_size);
        cpu.set_r64(reg::RSI, rsi + static_cast<u64>(step));
        cpu.set_r64(reg::RDI, rdi + static_cast<u64>(step));
        return true;
    });
}

// STOS -- store [r]AX -> [RDI].
void op_stos(Cpu& cpu, const Insn& insn) {
    run_string(cpu, insn, [&](bool& /*zf*/) -> bool {
        const GuestAddr rdi = cpu.r(reg::RDI);
        const u64 v = cpu.r(reg::RAX) & mask_for(insn.op_size);
        if (!write_bytes(cpu, rdi, insn.op_size, v)) return false;
        const i64 step = dir_step(cpu, insn.op_size);
        cpu.set_r64(reg::RDI, rdi + static_cast<u64>(step));
        return true;
    });
}

// LODS -- load [RSI] -> [r]AX.
void op_lods(Cpu& cpu, const Insn& insn) {
    run_string(cpu, insn, [&](bool& /*zf*/) -> bool {
        const GuestAddr rsi = cpu.r(reg::RSI);
        u64 v = 0;
        if (!read_bytes(cpu, rsi, insn.op_size, v)) return false;
        switch (insn.op_size) {
            case 1: cpu.set_r8 (reg::RAX, static_cast<u8 >(v)); break;
            case 2: cpu.set_r16(reg::RAX, static_cast<u16>(v)); break;
            case 4: cpu.set_r32(reg::RAX, static_cast<u32>(v)); break;
            case 8: cpu.set_r64(reg::RAX, v); break;
        }
        const i64 step = dir_step(cpu, insn.op_size);
        cpu.set_r64(reg::RSI, rsi + static_cast<u64>(step));
        return true;
    });
}

// SCAS -- compare [r]AX vs [RDI], set flags (SUB semantics).
void op_scas(Cpu& cpu, const Insn& insn) {
    run_string(cpu, insn, [&](bool& zf_after) -> bool {
        const GuestAddr rdi = cpu.r(reg::RDI);
        u64 v = 0;
        if (!read_bytes(cpu, rdi, insn.op_size, v)) return false;
        const u64 a = cpu.r(reg::RAX) & mask_for(insn.op_size);
        const u64 r = (a - v) & mask_for(insn.op_size);
        cpu.stash_sub(insn.op_size, v, r);
        zf_after = cpu.zf();
        const i64 step = dir_step(cpu, insn.op_size);
        cpu.set_r64(reg::RDI, rdi + static_cast<u64>(step));
        return true;
    });
}

// CMPS -- compare [RSI] vs [RDI].
void op_cmps(Cpu& cpu, const Insn& insn) {
    run_string(cpu, insn, [&](bool& zf_after) -> bool {
        const GuestAddr rsi = cpu.r(reg::RSI);
        const GuestAddr rdi = cpu.r(reg::RDI);
        u64 vs = 0, vd = 0;
        if (!read_bytes(cpu, rsi, insn.op_size, vs)) return false;
        if (!read_bytes(cpu, rdi, insn.op_size, vd)) return false;
        const u64 r = (vs - vd) & mask_for(insn.op_size);
        cpu.stash_sub(insn.op_size, vd, r);
        zf_after = cpu.zf();
        const i64 step = dir_step(cpu, insn.op_size);
        cpu.set_r64(reg::RSI, rsi + static_cast<u64>(step));
        cpu.set_r64(reg::RDI, rdi + static_cast<u64>(step));
        return true;
    });
}

} // namespace emu::handlers
