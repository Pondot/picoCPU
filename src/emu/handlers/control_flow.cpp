// Control flow: JMP / Jcc / CALL / RET.
//
// PUSH/POP support for CALL/RET is gated on a writable provider; for the
// Phase-1 gate (RNG mixer), the function ends in a bare `ret` and our
// dispatcher treats that as "end of run" without actually popping.
//
// For mid-function CALL/RET we'll add a writable shadow region in Phase 3.

#include "../handlers/handlers.h"
#include "../handlers/handler_util.h"

#include "emu/cpu.h"

namespace emu::handlers {

namespace {

GuestAddr branch_target_from_operand(Cpu& cpu, const Insn& insn) noexcept {
    // dst is either Rel (rel8/rel32) or Reg/Mem (indirect).
    const Operand& op = insn.dst;
    if (op.kind == OperandKind::Rel) {
        // Target = (RIP at start) + len + rel.
        return insn.rip + insn.len + static_cast<u64>(op.imm);
    }
    if (op.kind == OperandKind::Reg) {
        return cpu.r(op.reg);
    }
    if (op.kind == OperandKind::Mem || op.kind == OperandKind::RipRelMem) {
        u64 v = 0;
        if (!read_operand(cpu, insn, op, 8, v)) return 0;
        return v;
    }
    cpu.set_fault(FaultKind::InvalidOpcode, cpu.rip(), Status::InvalidInstruction,
                  "bad branch operand kind");
    return 0;
}

} // namespace

void op_jmp(Cpu& cpu, const Insn& insn) {
    const GuestAddr tgt = branch_target_from_operand(cpu, insn);
    if (cpu.halted()) return;
    cpu.take_branch(tgt);
}

void op_jcc(Cpu& cpu, const Insn& insn) {
    if (cpu.cond_holds(insn.cond)) {
        const GuestAddr tgt = insn.rip + insn.len + static_cast<u64>(insn.dst.imm);
        cpu.take_branch(tgt);
    }
    // Not taken: dispatcher advances RIP normally.
}

void op_call(Cpu& cpu, const Insn& insn) {
    // Push (RIP-of-next-instruction) onto the stack, then jump.
    const u64 return_addr = insn.rip + insn.len;
    auto* mp = cpu.mem_write();
    if (mp == nullptr) {
        cpu.set_fault(FaultKind::ProviderFailure, cpu.r(reg::RSP),
                      Status::ProviderFailure, "no write provider for CALL push");
        return;
    }
    const u64 new_rsp = cpu.r(reg::RSP) - 8;
    u8 buf[8];
    for (int i = 0; i < 8; ++i) buf[i] = static_cast<u8>((return_addr >> (8 * i)) & 0xFFu);
    if (Status s = mp->write(new_rsp, 8, buf); fail(s)) {
        cpu.set_fault(kind_from_status(s), new_rsp, s, "CALL push");
        return;
    }
    cpu.set_r64(reg::RSP, new_rsp);
    cpu.take_branch(branch_target_from_operand(cpu, insn));
}

// LOOP / LOOPE / LOOPNE -- RCX-- ; jump rel8 if RCX != 0 (and cc)
// LOOP    insn.cond = Cc::None
// LOOPE   insn.cond = Cc::Z      (jump if RCX != 0 && ZF == 1)
// LOOPNE  insn.cond = Cc::NZ     (jump if RCX != 0 && ZF == 0)
void op_loop(Cpu& cpu, const Insn& insn) {
    const u64 new_rcx = cpu.r(reg::RCX) - 1;
    cpu.set_r64(reg::RCX, new_rcx);
    bool take = (new_rcx != 0);
    if (take && insn.cond != Cc::None) {
        take = cpu.cond_holds(insn.cond);
    }
    if (take) {
        const GuestAddr tgt = insn.rip + insn.len + static_cast<u64>(insn.dst.imm);
        cpu.take_branch(tgt);
    }
}

void op_jcxz(Cpu& cpu, const Insn& insn) {
    if (cpu.r(reg::RCX) == 0) {
        const GuestAddr tgt = insn.rip + insn.len + static_cast<u64>(insn.dst.imm);
        cpu.take_branch(tgt);
    }
}

// CPUID -- returns a fixed Intel-style feature vector unless hooked.
// Goal: code that probes CPUID for SSE/SSE2/AVX/etc. finds those features
// (since we emulate them). Hookable via HookKind::Insn / filter_op = Cpuid.
void op_cpuid(Cpu& cpu, const Insn& insn) {
    (void)insn;
    const u32 leaf    = cpu.r32(reg::RAX);
    const u32 subleaf = cpu.r32(reg::RCX);
    u32 eax = 0, ebx = 0, ecx = 0, edx = 0;

    // User-set override takes precedence over the canned defaults below.
    if (cpu.cpuid_lookup(leaf, subleaf, eax, ebx, ecx, edx)) {
        cpu.set_r32(reg::RAX, eax);
        cpu.set_r32(reg::RBX, ebx);
        cpu.set_r32(reg::RCX, ecx);
        cpu.set_r32(reg::RDX, edx);
        return;
    }

    switch (leaf) {
        case 0x00000000:
            // Max standard leaf + vendor "GenuineIntel"
            eax = 0x0000000D;
            ebx = 0x756E6547;   // "Genu"
            edx = 0x49656E69;   // "ineI"
            ecx = 0x6C65746E;   // "ntel"
            break;
        case 0x00000001: {
            // Family/Model/Stepping -- pick "Skylake-ish": family=6, model=85.
            eax = (6u << 8) | (0x05 << 4) | 0x03 | (0u << 12) | (5u << 16);
            // EBX: brand index + CLFLUSH line + logical CPUs + initial APIC.
            ebx = (1u << 0) | (8u << 8) | (1u << 16) | (0u << 24);
            // ECX feature flags: SSE3, PCLMULQDQ, SSSE3, FMA, CMPXCHG16B,
            // SSE4.1, SSE4.2, MOVBE, POPCNT, AESNI, XSAVE, OSXSAVE, AVX,
            // F16C, RDRAND.
            ecx = (1u << 0)  | (1u << 1)  | (1u << 9)  | (1u << 12) |
                  (1u << 13) | (1u << 19) | (1u << 20) | (1u << 22) |
                  (1u << 23) | (1u << 25) | (1u << 26) | (1u << 27) |
                  (1u << 28) | (1u << 29) | (1u << 30);
            // EDX feature flags: FPU, VME, DE, PSE, TSC, MSR, PAE, MCE,
            // CX8, APIC, SEP, MTRR, PGE, MCA, CMOV, PAT, PSE36, CLFSH,
            // DS, ACPI, MMX, FXSR, SSE, SSE2, SS, HTT.
            edx = (1u << 0)  | (1u << 1)  | (1u << 2)  | (1u << 3)  |
                  (1u << 4)  | (1u << 5)  | (1u << 6)  | (1u << 7)  |
                  (1u << 8)  | (1u << 9)  | (1u << 11) | (1u << 12) |
                  (1u << 13) | (1u << 14) | (1u << 15) | (1u << 16) |
                  (1u << 17) | (1u << 19) | (1u << 21) | (1u << 22) |
                  (1u << 23) | (1u << 24) | (1u << 25) | (1u << 26) |
                  (1u << 27) | (1u << 28);
            break;
        }
        case 0x00000007: {
            // Extended features (subleaf 0): FSGSBASE, BMI1, AVX2, BMI2,
            // ERMS (REP MOVSB), AVX512F (we don't emulate but the bit costs
            // nothing to advertise zero -- leave AVX-512 off).
            if (subleaf == 0) {
                eax = 0;
                ebx = (1u << 0)  | (1u << 3)  | (1u << 5)  | (1u << 8)  |
                      (1u << 9);
                ecx = 0;
                edx = 0;
            }
            break;
        }
        case 0x80000000:
            eax = 0x80000004;   // max extended leaf
            ebx = ecx = edx = 0;
            break;
        case 0x80000001:
            // EDX bit 29 = LM (long mode). RDTSCP (bit 27) too.
            edx = (1u << 29) | (1u << 27);
            break;
        case 0x80000002:
            // Brand string part 1: "Emulator x86-64 CPU"
            eax = 0x6C756D45;   // "Emul"
            ebx = 0x726F7461;   // "ator"
            ecx = 0x38782820;   // " (x8"
            edx = 0x34362D36;   // "6-64"
            break;
        case 0x80000003:
            eax = 0x55504320;   //  " CPU"
            ebx = 0x00000029;   //  ")\0"
            ecx = 0; edx = 0;
            break;
        case 0x80000004:
            eax = ebx = ecx = edx = 0;
            break;
        default: break;
    }
    cpu.set_r32(reg::RAX, eax);
    cpu.set_r32(reg::RBX, ebx);
    cpu.set_r32(reg::RCX, ecx);
    cpu.set_r32(reg::RDX, edx);
}

// RDTSC -- returns a deterministic counter unless hooked. Counter increments
// once per call so tight RDTSC-RDTSC loops still see progress.
void op_rdtsc(Cpu& cpu, const Insn& insn) {
    (void)insn;
    static u64 fake_tsc = 0;
    ++fake_tsc;
    cpu.set_r32(reg::RAX, static_cast<u32>(fake_tsc));
    cpu.set_r32(reg::RDX, static_cast<u32>(fake_tsc >> 32));
}

// RDRAND -- fills dst with a deterministic xorshift64* sequence unless hooked.
// Sets CF = 1 to signal "value valid".
void op_rdrand(Cpu& cpu, const Insn& insn) {
    static u64 state = 0x12345678ABCDEF01ull;
    state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
    const u64 v = state * 0x2545F4914F6CDD1Dull;
    (void)write_operand(cpu, insn, insn.dst, insn.op_size, v);
    cpu.force_sync_flags();
    cpu.set_cf(true);
    cpu.set_of(false);
    cpu.set_zf(false);
    cpu.set_sf(false);
    cpu.set_af(false);
    cpu.set_pf(false);
}

void op_hlt(Cpu& cpu, const Insn& insn) {
    (void)insn;
    cpu.set_fault(FaultKind::GeneralProtection, cpu.rip(), Status::Ok, "HLT in user mode");
}

// SYSCALL -- default fault if no INSN hook claims it. Real workloads should
// install a hook with `filter_op = OpKind::Syscall` to dispatch the syscall
// (read RAX for the number, set up return values, etc.).
void op_syscall(Cpu& cpu, const Insn& insn) {
    (void)insn;
    // The dispatcher fires INSN hooks *before* the handler. If a hook
    // already serviced this syscall and set its own state, it should have
    // halted the CPU or modified RAX/RCX/R11 directly.
    cpu.set_fault(FaultKind::InvalidOpcode, cpu.rip(), Status::Ok,
                  "SYSCALL -- install an INSN hook with filter_op=Syscall to service");
}

void op_ret(Cpu& cpu, const Insn& insn) {
    // Pop return address from the stack and branch to it. The optional
    // imm16 (RET n) bumps RSP further to release callee-popped stack args.
    auto* mp = cpu.mem_read();
    if (mp == nullptr) {
        cpu.set_fault(FaultKind::ProviderFailure, cpu.r(reg::RSP),
                      Status::ProviderFailure, "no read provider for RET pop");
        return;
    }
    const u64 rsp = cpu.r(reg::RSP);
    u8 buf[8] = {};
    if (Status s = mp->read(rsp, 8, buf); fail(s)) {
        cpu.set_fault(kind_from_status(s), rsp, s, "RET pop");
        return;
    }
    u64 ret_addr = 0;
    for (int i = 0; i < 8; ++i) ret_addr |= (u64{buf[i]} << (8 * i));

    u64 imm16_pop = 0;
    if (insn.dst.kind == OperandKind::Imm) {
        imm16_pop = static_cast<u64>(insn.dst.imm) & 0xFFFFull;
    }
    cpu.set_r64(reg::RSP, rsp + 8 + imm16_pop);
    cpu.take_branch(ret_addr);
}

} // namespace emu::handlers
