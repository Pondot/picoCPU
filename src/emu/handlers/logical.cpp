// Logical handlers: AND / OR / XOR / NOT / TEST / CMP.

#include "../handlers/handlers.h"
#include "../handlers/handler_util.h"

#include "emu/cpu.h"

namespace emu::handlers {

void op_and(Cpu& cpu, const Insn& insn) {
    u64 a = 0, b = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, a)) return;
    if (!read_operand(cpu, insn, insn.src, insn.op_size, b)) return;
    const u64 r = (a & b) & mask_for(insn.op_size);
    if (!write_operand(cpu, insn, insn.dst, insn.op_size, r)) return;
    cpu.stash_logic(insn.op_size, r);
}

void op_or(Cpu& cpu, const Insn& insn) {
    u64 a = 0, b = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, a)) return;
    if (!read_operand(cpu, insn, insn.src, insn.op_size, b)) return;
    const u64 r = (a | b) & mask_for(insn.op_size);
    if (!write_operand(cpu, insn, insn.dst, insn.op_size, r)) return;
    cpu.stash_logic(insn.op_size, r);
}

void op_xor(Cpu& cpu, const Insn& insn) {
    u64 a = 0, b = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, a)) return;
    if (!read_operand(cpu, insn, insn.src, insn.op_size, b)) return;
    const u64 r = (a ^ b) & mask_for(insn.op_size);
    if (!write_operand(cpu, insn, insn.dst, insn.op_size, r)) return;
    cpu.stash_logic(insn.op_size, r);
}

void op_not(Cpu& cpu, const Insn& insn) {
    u64 a = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, a)) return;
    const u64 r = (~a) & mask_for(insn.op_size);
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, r);
    // NOT does not affect flags (Intel SDM Vol 2A, NOT).
}

void op_test(Cpu& cpu, const Insn& insn) {
    // result = dst AND src; flags only -- no writeback.
    u64 a = 0, b = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, a)) return;
    if (!read_operand(cpu, insn, insn.src, insn.op_size, b)) return;
    const u64 r = (a & b) & mask_for(insn.op_size);
    cpu.stash_logic(insn.op_size, r);
}

void op_cmp(Cpu& cpu, const Insn& insn) {
    // result = dst - src; flags only.
    u64 a = 0, b = 0;
    if (!read_operand(cpu, insn, insn.dst, insn.op_size, a)) return;
    if (!read_operand(cpu, insn, insn.src, insn.op_size, b)) return;
    const u64 r = (a - b) & mask_for(insn.op_size);
    cpu.stash_sub(insn.op_size, b, r);
}

} // namespace emu::handlers
