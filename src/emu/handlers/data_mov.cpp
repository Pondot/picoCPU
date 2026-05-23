// Data-movement handlers: MOV / MOVZX / MOVSX / LEA / PUSH / POP / XCHG / NOP.
//
// PUSH/POP write to memory through the *write* provider. For Phase 1 the
// tester provides a writable shadow region for the stack (a host-backed
// scratch buffer); see tester.cpp.

#include "../handlers/handlers.h"
#include "../handlers/handler_util.h"

#include "emu/cpu.h"
#include "emu/memory.h"

namespace emu::handlers {

namespace {

// Convenience writer for the stack -- uses cpu.mem_write() at the given addr.
// Returns true on success; faults the CPU on failure.
bool write_mem_bytes(Cpu& cpu, GuestAddr addr, u8 size, u64 value) noexcept {
    auto* mp = cpu.mem_write();
    if (mp == nullptr) {
        cpu.set_fault(FaultKind::ProviderFailure, addr, Status::ProviderFailure,
                      "no write provider");
        return false;
    }
    u8 buf[8] = {};
    for (int i = 0; i < size; ++i) buf[i] = static_cast<u8>((value >> (8 * i)) & 0xFFu);
    Status s = mp->write(addr, size, buf);
    if (fail(s)) {
        cpu.set_fault(kind_from_status(s), addr, s, "stack/mem write");
        return false;
    }
    return true;
}

bool read_mem_bytes(Cpu& cpu, GuestAddr addr, u8 size, u64& value) noexcept {
    auto* mp = cpu.mem_read();
    if (mp == nullptr) {
        cpu.set_fault(FaultKind::ProviderFailure, addr, Status::ProviderFailure,
                      "no read provider");
        return false;
    }
    u8 buf[8] = {};
    Status s = mp->read(addr, size, buf);
    if (fail(s)) {
        cpu.set_fault(FaultKind::PageFault, addr, s, "mem read");
        return false;
    }
    u64 v = 0;
    for (int i = 0; i < size; ++i) v |= (u64{buf[i]} << (8 * i));
    value = v;
    return true;
}

} // namespace

void op_invalid(Cpu& cpu, const Insn&) {
    cpu.set_fault(FaultKind::InvalidOpcode, cpu.rip(), Status::InvalidInstruction,
                  "invalid handler bound by decoder");
}

void op_nop(Cpu&, const Insn&) { /* nothing */ }

void op_mov(Cpu& cpu, const Insn& insn) {
    u64 v = 0;
    if (!read_operand(cpu, insn, insn.src, insn.op_size, v)) return;
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, v);
}

void op_movzx(Cpu& cpu, const Insn& insn) {
    // src.scale was stuffed with the source size at decode time.
    const u8 src_size = insn.src.scale ? insn.src.scale : 1;
    u64 v = 0;
    if (!read_operand(cpu, insn, insn.src, src_size, v)) return;
    v &= mask_for(src_size);     // zero-extend
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, v);
}

void op_movsx(Cpu& cpu, const Insn& insn) {
    const u8 src_size = insn.src.scale ? insn.src.scale : 1;
    u64 v = 0;
    if (!read_operand(cpu, insn, insn.src, src_size, v)) return;
    const i64 ext = sign_extend(v, src_size);
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, static_cast<u64>(ext));
}

void op_movsxd(Cpu& cpu, const Insn& insn) {
    u64 v = 0;
    if (!read_operand(cpu, insn, insn.src, 4, v)) return;
    const i64 ext = sign_extend(v, 4);
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, static_cast<u64>(ext));
}

void op_lea(Cpu& cpu, const Insn& insn) {
    // LEA computes the effective address of a memory operand and stores it
    // in a register. It never touches memory.
    if (insn.src.kind != OperandKind::Mem && insn.src.kind != OperandKind::RipRelMem) {
        cpu.set_fault(FaultKind::InvalidOpcode, cpu.rip(), Status::InvalidInstruction,
                      "LEA requires memory operand");
        return;
    }
    const GuestAddr addr = ea(cpu, insn, insn.src);
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, addr);
}

void op_push(Cpu& cpu, const Insn& insn) {
    u64 v = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, v)) return;
    const u64 new_rsp = cpu.r(reg::RSP) - insn.op_size;
    if (!write_mem_bytes(cpu, new_rsp, insn.op_size, v)) return;
    cpu.set_r64(reg::RSP, new_rsp);
}

void op_pop(Cpu& cpu, const Insn& insn) {
    u64 v = 0;
    const u64 rsp = cpu.r(reg::RSP);
    if (!read_mem_bytes(cpu, rsp, insn.op_size, v)) return;
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, v);
    cpu.set_r64(reg::RSP, rsp + insn.op_size);
}

void op_cmov(Cpu& cpu, const Insn& insn) {
    if (cpu.cond_holds(insn.cond)) {
        u64 v = 0;
        if (!read_operand(cpu, insn, insn.src, insn.op_size, v)) return;
        (void)write_operand(cpu, insn, insn.dst, insn.op_size, v);
    }
    // Else: do nothing -- CMOVcc is a no-op when the condition is false.
}

void op_xchg(Cpu& cpu, const Insn& insn) {
    u64 a = 0, b = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, a)) return;
    if (!read_operand(cpu, insn, insn.src, insn.op_size, b)) return;
    if (!write_operand(cpu, insn, insn.dst, insn.op_size, b)) return;
    (void)write_operand(cpu, insn, insn.src, insn.op_size, a);
}

// BSWAP r32/r64 -- byte-swap. 32-bit form zero-extends to 64; 16-bit is
// architecturally undefined and not supported here.
void op_bswap(Cpu& cpu, const Insn& insn) {
    u64 v = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, v)) return;
    u64 swapped = 0;
    if (insn.op_size == 8) {
        swapped = ((v & 0x00000000000000FFull) << 56) |
                  ((v & 0x000000000000FF00ull) << 40) |
                  ((v & 0x0000000000FF0000ull) << 24) |
                  ((v & 0x00000000FF000000ull) <<  8) |
                  ((v & 0x000000FF00000000ull) >>  8) |
                  ((v & 0x0000FF0000000000ull) >> 24) |
                  ((v & 0x00FF000000000000ull) >> 40) |
                  ((v & 0xFF00000000000000ull) >> 56);
    } else {
        const u32 w = static_cast<u32>(v);
        const u32 s = ((w & 0x000000FFu) << 24) |
                      ((w & 0x0000FF00u) <<  8) |
                      ((w & 0x00FF0000u) >>  8) |
                      ((w & 0xFF000000u) >> 24);
        swapped = static_cast<u64>(s);
    }
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, swapped);
}

// CWDE / CDQE: sign-extend the rAX accumulator within itself.
//   op_size=2 -> AL->AX     (CBW)
//   op_size=4 -> AX->EAX    (CWDE)
//   op_size=8 -> EAX->RAX   (CDQE)
// The same opcode 0x98 covers all three; op_size carries the operand size.
void op_cwde(Cpu& cpu, const Insn& insn) {
    const u64 raw = cpu.r(reg::RAX);
    u64 out = raw;
    if (insn.op_size == 2) {
        const i8  s = static_cast<i8>(raw & 0xFF);
        out = (raw & ~u64{0xFFFF}) | (static_cast<u64>(static_cast<u16>(static_cast<i16>(s))) & 0xFFFF);
    } else if (insn.op_size == 4) {
        const i16 s = static_cast<i16>(raw & 0xFFFF);
        cpu.set_r32(reg::RAX, static_cast<u32>(static_cast<i32>(s)));
        return;
    } else if (insn.op_size == 8) {
        const i32 s = static_cast<i32>(raw & 0xFFFFFFFFu);
        cpu.set_r64(reg::RAX, static_cast<u64>(static_cast<i64>(s)));
        return;
    }
    cpu.set_r64(reg::RAX, out);
}

// CWD / CDQ / CQO: sign-extend rAX into rDX (high half of double-precision).
//   op_size=2 -> AX  -> DX:AX     (CWD)
//   op_size=4 -> EAX -> EDX:EAX   (CDQ)
//   op_size=8 -> RAX -> RDX:RAX   (CQO)
void op_cdq(Cpu& cpu, const Insn& insn) {
    const u64 a = cpu.r(reg::RAX);
    u64 d = 0;
    if (insn.op_size == 2) {
        d = (a & 0x8000) ? 0xFFFFull : 0ull;
        cpu.set_r16(reg::RDX, static_cast<u16>(d));
    } else if (insn.op_size == 4) {
        d = (a & 0x80000000ull) ? 0xFFFFFFFFull : 0ull;
        cpu.set_r32(reg::RDX, static_cast<u32>(d));
    } else if (insn.op_size == 8) {
        d = (a & (u64{1} << 63)) ? ~u64{0} : u64{0};
        cpu.set_r64(reg::RDX, d);
    }
}

} // namespace emu::handlers
