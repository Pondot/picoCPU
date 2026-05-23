// Emulator IR.
//
// The decoder turns guest bytes into `Insn`s. Each `Insn` carries a
// pre-resolved handler pointer (so the dispatcher does one indirect call per
// instruction, no per-call branching on opcode/operand kind). `OpKind` is
// kept around for debugging, peephole passes and the tracer -- never read on
// the hot path.
//
// Insn is 64 bytes (one cache line). Operand is 24 bytes; three of them
// fit alongside the 16-byte header.

#pragma once

#include "emu/types.h"

#include <cstddef>
#include <vector>

namespace emu {

class Cpu;
struct Insn;

// Function-pointer signature for handlers. `cpu` carries all guest state; the
// handler reads/writes through it. `insn` is the decoded instruction.
using Handler = void (*)(Cpu& cpu, const Insn& insn);

// ---- Register identifiers ---------------------------------------------------
//
// 64-bit GPR indices match the x86-64 encoding: 0=RAX, 1=RCX, 2=RDX, 3=RBX,
// 4=RSP, 5=RBP, 6=RSI, 7=RDI, 8..15=R8..R15. RIP is *not* a GPR -- it lives
// directly on the Cpu and is referenced via OperandKind::RipRelMem when it
// appears in addressing.
namespace reg {
    constexpr u8 RAX = 0,  RCX = 1,  RDX = 2,  RBX = 3;
    constexpr u8 RSP = 4,  RBP = 5,  RSI = 6,  RDI = 7;
    constexpr u8 R8  = 8,  R9  = 9,  R10 = 10, R11 = 11;
    constexpr u8 R12 = 12, R13 = 13, R14 = 14, R15 = 15;
    constexpr u8 NONE = 0xFF;
}

enum class Seg : u8 {
    DS = 0,
    CS = 1,
    SS = 2,
    ES = 3,
    FS = 4,
    GS = 5,
};

// ---- Condition codes (Jcc / CMOVcc / SETcc) ---------------------------------
// Values match the low nibble of the x86 conditional opcode (O=0, NO=1, ...).
enum class Cc : u8 {
    O   = 0x0, NO  = 0x1,
    B   = 0x2, NB  = 0x3,  // CF=1 / CF=0
    Z   = 0x4, NZ  = 0x5,  // ZF=1 / ZF=0
    BE  = 0x6, NBE = 0x7,  // CF=1 || ZF=1 / both clear
    S   = 0x8, NS  = 0x9,
    P   = 0xA, NP  = 0xB,
    L   = 0xC, NL  = 0xD,  // SF != OF
    LE  = 0xE, NLE = 0xF,  // ZF=1 || SF!=OF
    None = 0xFF,
};

// ---- Operand kinds ----------------------------------------------------------
enum class OperandKind : u8 {
    None = 0,
    Reg,       // 64/32/16/8-bit GPR; size carried by `Insn::op_size`
    Imm,       // signed/unsigned immediate, sign-extended to 64 in `Operand::imm`
    Mem,       // [base + index*scale + disp], with optional segment override
    RipRelMem, // RIP-relative: effective addr = (RIP after this insn) + disp
    Rel,       // PC-relative branch target offset (Jcc/JMP rel/CALL rel)
    Xmm,       // XMM0..XMM15 (SSE); `op.reg` holds 0..15
};

struct Operand {
    OperandKind kind;       // 1
    u8          reg;        // 1: kind=Reg -> reg id; kind=Mem -> base reg id (or NONE)
    u8          index;      // 1: kind=Mem -> index reg id (or NONE)
    u8          scale;      // 1: 1/2/4/8 (only meaningful when index != NONE)
    Seg         seg;        // 1: kind=Mem only
    u8          pad[3];     // 3: pad to 8
    i64         imm;        // 8: kind=Imm -> value, kind=Mem -> disp32 sign-extended,
                            //    kind=RipRelMem -> disp, kind=Rel -> relative offset
};
static_assert(sizeof(Operand) == 16, "Operand must be 16 bytes");

// ---- Op kind (semantic) -----------------------------------------------------
//
// Kept compact (u16). New ops append at the end -- never re-number.
enum class OpKind : u16 {
    Invalid = 0,
    Nop,

    // Data movement.
    Mov,
    MovZx,
    MovSx,
    MovSxd,
    Lea,
    Push,
    Pop,
    Xchg,
    Bswap,
    Cmov,

    // Integer arithmetic.
    Add,
    Adc,
    Sub,
    Sbb,
    Inc,
    Dec,
    Neg,
    Mul,
    Imul,        // 1-operand (full-width) and 2-operand IMUL
    Imul3,       // 3-operand IMUL r, r/m, imm -- value lives in insn.imm_extra
    Div,
    Idiv,

    // Logical.
    And,
    Or,
    Xor,
    Not,
    Test,
    Cmp,

    // Shifts / rotates.
    Shl,
    Shr,
    Sar,
    Rol,
    Ror,
    Rcl,
    Rcr,
    Shld,
    Shrd,

    // Control flow.
    Jmp,         // direct (rel) and indirect -- operand distinguishes
    Jcc,         // Insn::cond holds the condition
    Call,
    Ret,
    Loop,
    LoopE,
    LoopNE,
    Jcxz,

    // System / misc.
    // String ops (REP/REPE/REPNE prefixes carried in Insn::flags)
    Movs,
    Stos,
    Lods,
    Scas,
    Cmps,

    // Bit manipulation
    Bsf,
    Bsr,
    Bt,
    Bts,
    Btr,
    Btc,
    Popcnt,
    Lzcnt,
    Tzcnt,

    Cpuid,
    Rdtsc,
    Rdrand,
    Rdseed,
    Hlt,
    Syscall,

    // SSE/SSE2 (operand size = 16; XMM register file)
    Movdqa,
    Movdqu,
    Pxor,
    Pand,
    Por,
    Paddb, Paddw, Paddd, Paddq,
    Psubb, Psubw, Psubd, Psubq,
    Pshufd,
    Pcmpeqb, Pcmpeqw, Pcmpeqd,

    // SSE floating-point (scalar + packed). The decoder picks which based on
    // the mandatory prefix (F3 = SS, F2 = SD, 66 = PD, none = PS).
    Movss, Movsd, Movups, Movupd, Movaps, Movapd,
    Addss, Subss, Mulss, Divss, Sqrtss, Minss, Maxss,
    Addsd, Subsd, Mulsd, Divsd, Sqrtsd, Minsd, Maxsd,
    Addps, Subps, Mulps, Divps, Sqrtps, Minps, Maxps,
    Addpd, Subpd, Mulpd, Divpd, Sqrtpd, Minpd, Maxpd,
    Ucomiss, Ucomisd, Comiss, Comisd,
    Cvtsi2ss, Cvtsi2sd, Cvttss2si, Cvttsd2si, Cvtss2sd, Cvtsd2ss,
    Xorps, Xorpd, Andps, Andpd, Orps, Orpd,

    // x87 FPU (minimal subset; ST(0)..ST(7) stack)
    Fld, Fst, Fstp, Fild, Fistp,
    Fadd, Fsub, Fmul, Fdiv, Fchs, Fabs, Fsqrt,
    Faddp, Fsubp, Fmulp, Fdivp, Fxch,
    Fcom, Fcomp, Fcompp,
    Fninit, Fnstsw, Fnstcw, Fldcw,

    // AVX (VEX-encoded). When INSN_FLAG_VEX is set, `insn.dst.index` carries
    // the vvvv (second source) register id. `op_size` is 16 (L=0, xmm) or
    // 32 (L=1, ymm).
    Vmovdqa, Vmovdqu,
    Vpxor, Vpand, Vpor,
    Vpaddb, Vpaddw, Vpaddd, Vpaddq,
    Vpsubb, Vpsubw, Vpsubd, Vpsubq,
    Vzeroupper, Vzeroall,

    // Misc legacy + memory-ordering
    Lahf, Sahf, Pushf, Popf,
    Int3, Int_n,
    Lfence, Mfence, Sfence, Pause, Clflush,
    Prefetcht0, Prefetcht1, Prefetcht2, Prefetchnta,
    Movbe,

    // Extra SSE
    Rcpss, Rsqrtss, Rcpps, Rsqrtps,
    Pmovmskb, Movmskps, Movmskpd,
    Shufps, Shufpd, Pshufb, Pshufhw, Pshuflw,
    Pmullw, Pmulhw, Pmulhuw, Pmaddwd,
    Psllw, Pslld, Psllq,
    Psrlw, Psrld, Psrlq,
    Psraw, Psrad,
    Punpcklbw, Punpcklwd, Punpckldq, Punpcklqdq,
    Punpckhbw, Punpckhwd, Punpckhdq, Punpckhqdq,

    // 0F 38 / 0F 3A -- SSSE3, SSE4.1, SSE4.2, AES-NI, CLMUL
    Aesenc, Aesenclast, Aesdec, Aesdeclast, Aesimc, Aeskeygenassist,
    Pclmulqdq,
    Crc32,
    Roundss, Roundsd, Roundps, Roundpd,
    Insertps,
    Pextrb, Pextrw, Pextrd, Pextrq,
    Pinsrb, Pinsrw, Pinsrd, Pinsrq,
    Pcmpgtb, Pcmpgtw, Pcmpgtd, Pcmpgtq,
    Pmaxub, Pmaxuw, Pmaxud, Pmaxsb, Pmaxsw, Pmaxsd,
    Pminub, Pminuw, Pminud, Pminsb, Pminsw, Pminsd,
    Pabsb, Pabsw, Pabsd,
    Psadbw,
    Pmovsxbw, Pmovsxbd, Pmovsxbq, Pmovsxwd, Pmovsxwq, Pmovsxdq,
    Pmovzxbw, Pmovzxbd, Pmovzxbq, Pmovzxwd, Pmovzxwq, Pmovzxdq,

    // More x87 (transcendentals)
    Fsin, Fcos, Fsincos, Fpatan, Fptan, Fyl2x, F2xm1, Fscale, Frndint, Ftst,

    // BMI1 / BMI2 (VEX-encoded GPR ops)
    Andn, Bextr, Blsi, Blsr, Blsmsk, Bzhi,
    Mulx, Pdep, Pext, Rorx, Sarx, Shlx, Shrx,

    // VEX-encoded 0F 38 / 0F 3A SSE 3-operand forms
    Vpshufb, Vaesenc, Vaesenclast, Vaesdec, Vaesdeclast, Vpcmpgtq,
    Vpclmulqdq,

    // SHA-NI
    Sha1rnds4, Sha1nexte, Sha1msg1, Sha1msg2,
    Sha256rnds2, Sha256msg1, Sha256msg2,

    // SSE4.2 string compare
    Pcmpestri, Pcmpestrm, Pcmpistri, Pcmpistrm,

    // AVX-512 (EVEX). `insn.pad` carries: bits 0-2 = aaa (opmask reg),
    // bit 3 = z (zeroing), bit 4 = b (broadcast -- single-lane mem broadcast
    // to all lanes when src.kind == Mem). `insn.op_size` is the vector width
    // in bytes (16/32/64).
    Vpxord, Vpxorq,
    Vpaddb_evex, Vpaddw_evex, Vpaddd_evex, Vpaddq_evex,
    Vpsubb_evex, Vpsubw_evex, Vpsubd_evex, Vpsubq_evex,
    Vpandd, Vpandq, Vpord, Vporq,
    Vmovdqu32, Vmovdqu64, Vmovdqa32, Vmovdqa64,

    // AVX-512 floating-point (packed single = 4-byte lanes, packed double = 8-byte)
    Vaddps_evex, Vaddpd_evex,
    Vsubps_evex, Vsubpd_evex,
    Vmulps_evex, Vmulpd_evex,
    Vdivps_evex, Vdivpd_evex,
    Vsqrtps_evex, Vsqrtpd_evex,
    Vminps_evex, Vminpd_evex,
    Vmaxps_evex, Vmaxpd_evex,

    // AVX2 GATHER
    Vpgatherdd, Vpgatherdq, Vpgatherqd, Vpgatherqq,
    Vgatherdps, Vgatherdpd, Vgatherqps, Vgatherqpd,

    // AVX-512 SCATTER (EVEX 0F 38 A0..A3) -- store lanes to per-lane-indexed
    // memory. Mask is an opmask register (aaa bits) not a vector mask.
    Vpscatterdd, Vpscatterdq, Vpscatterqd, Vpscatterqq,
    Vscatterdps, Vscatterdpd, Vscatterqps, Vscatterqpd,

    // Sign/zero-extend rAX accumulator family
    Cwde,   // 0x98 (op_size=2): AL->AX,  =4: AX->EAX, =8: EAX->RAX
    Cdq,    // 0x99 (op_size=2): AX->DX:AX, =4: EAX->EDX:EAX, =8: RAX->RDX:RAX

    // MOVD/MOVQ between XMM and GPR/memory.
    // 66 0F 6E /r  -- xmm <- r/m{32,64}  (zero upper XMM bits)
    // 66 0F 7E /r  -- r/m{32,64} <- xmm  (zero-extend GPR to 64 if 32-bit form)
    // F3 0F 7E /r  -- xmm <- xmm/m64     (move qword, zero upper)
    Movd,
    Movq,

    // ----
    Count,
};

// Returns a short uppercase mnemonic ("MOV", "XOR", "RET", ...). For tracing.
const char* op_kind_name(OpKind k) noexcept;

// ---- Insn -------------------------------------------------------------------
//
// `flags` field bitmap:
constexpr u8 INSN_FLAG_CONTROL_FLOW = 1u << 0;  // CALL/JMP/Jcc/RET/LOOP -- dispatcher exits after
constexpr u8 INSN_FLAG_LOCK         = 1u << 1;  // LOCK prefix present
constexpr u8 INSN_FLAG_REP          = 1u << 2;  // F3 REP/REPE
constexpr u8 INSN_FLAG_REPNE        = 1u << 3;  // F2 REPNE
constexpr u8 INSN_FLAG_HAS_SEG      = 1u << 4;  // segment override present
constexpr u8 INSN_FLAG_OPSIZE_PFX   = 1u << 5;  // 0x66 was present
constexpr u8 INSN_FLAG_ADDRSIZE_PFX = 1u << 6;  // 0x67 was present
constexpr u8 INSN_FLAG_VEX          = 1u << 7;  // VEX-encoded (3-operand AVX). vvvv in insn.dst.index.

struct Insn {
    Handler handler;        // 8: bound by the decoder; one indirect call per insn
    OpKind  kind;           // 2: semantic op (debug / peephole)
    u8      op_size;        // 1: operand size in bytes (1/2/4/8)
    u8      len;            // 1: total instruction length in bytes
    Cc      cond;           // 1: condition for Jcc/CMOVcc/SETcc; otherwise Cc::None
    u8      flags;          // 1: INSN_FLAG_* bitmap
    u16     pad;            // 2: explicit pad
    GuestAddr rip;          // 8: RIP of *this* instruction at decode time (for RIP-rel)
    Operand dst;            // 16
    Operand src;            // 16
    i64     imm_extra;      // 8: 3-operand IMUL's immediate; SHLD/SHRD count;
                            //    unused by all other ops.
};
static_assert(sizeof(Insn) <= 64, "Insn must fit in one cache line");

// ---- DecodedBlock -----------------------------------------------------------
//
// A basic block: a straight-line run of instructions starting at `pc` and
// ending at the first control-flow instruction (JMP/Jcc/CALL/RET/LOOP).
// Owned by `BlockCache`. Handlers iterate `insns` directly; the dispatcher
// reads `insn.rip` and `insn.len` to advance the program counter.
struct DecodedBlock {
    GuestAddr        pc        = 0;
    u32              byte_size = 0;
    std::vector<Insn> insns;
};

} // namespace emu
