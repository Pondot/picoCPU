// Per-opcode dispatch.
//
// Reads opcode bytes (1-byte primary, optional 0x0F + 2-byte), parses
// ModRM/SIB/disp/immediate as needed, fills the Insn's operand fields, and
// binds the handler pointer.
//
// Coverage (Phase 1):
//   Primary 0x00..0x3D -- ADD/OR/ADC/SBB/AND/SUB/XOR/CMP groups (full 8 forms each)
//   Primary 0x50..0x5F -- PUSH/POP r64
//   Primary 0x68/0x6A -- PUSH imm
//   Primary 0x70..0x7F -- Jcc short
//   Primary 0x80/0x81/0x83 -- group 1 r/m, imm
//   Primary 0x84/0x85 -- TEST r/m, r
//   Primary 0x86/0x87 -- XCHG r/m, r
//   Primary 0x88..0x8B -- MOV r/m,r and r,r/m
//   Primary 0x8D -- LEA r, m
//   Primary 0x90 -- NOP
//   Primary 0xB0..0xBF -- MOV r, imm
//   Primary 0xC0/0xC1 -- group 2 r/m, imm8 (SHL/SHR/SAR/ROL/ROR/RCL/RCR)
//   Primary 0xC2/0xC3 -- RET
//   Primary 0xC6/0xC7 -- MOV r/m, imm
//   Primary 0xD0/0xD1 -- group 2 r/m, 1
//   Primary 0xD2/0xD3 -- group 2 r/m, CL
//   Primary 0xE8 -- CALL rel32
//   Primary 0xE9 -- JMP rel32
//   Primary 0xEB -- JMP rel8
//   Primary 0xF6/0xF7 -- group 3 (TEST/NOT/NEG)  (MUL/DIV land in Phase 4)
//   Primary 0xFE/0xFF -- group 4/5 (INC/DEC/CALL/JMP/PUSH)
//   Secondary 0x0F 0x80..0x8F -- Jcc near
//   Secondary 0x0F 0xAF -- IMUL r, r/m
//   Secondary 0x0F 0xB6/0xB7 -- MOVZX
//   Secondary 0x0F 0xBE/0xBF -- MOVSX

#include "decoder_internal.h"
#include "../handlers/handlers.h"

#include "emu/cpu.h"

namespace emu::decoder {

namespace {

inline u8 alu_size(const Prefixes& px) noexcept {
    return default_gpr_op_size(px);
}

// 8-bit ALU op size override.
inline u8 alu_size_8() noexcept { return 1; }

// Build a register operand directly from a raw reg id (already combined with REX).
Operand make_reg(u8 reg_id) noexcept {
    Operand o{};
    o.kind = OperandKind::Reg;
    o.reg  = reg_id;
    return o;
}

Operand make_imm(i64 v) noexcept {
    Operand o{};
    o.kind = OperandKind::Imm;
    o.imm  = v;
    return o;
}

Operand make_rel(i64 disp) noexcept {
    Operand o{};
    o.kind = OperandKind::Rel;
    o.imm  = disp;
    return o;
}

// Decode an ALU op family (ADD/OR/ADC/SBB/AND/SUB/XOR/CMP).
// `family_index` ∈ 0..7 corresponds to ADD..CMP.
// `variant` ∈ 0..5 is the offset within the family (00, 01, 02, 03, 04, 05).
Status decode_alu_primary(ByteSource& bs, const Prefixes& px, u8 family_index,
                          u8 variant, Insn& out) noexcept {
    static constexpr OpKind kinds[8] = {
        OpKind::Add, OpKind::Or,  OpKind::Adc, OpKind::Sbb,
        OpKind::And, OpKind::Sub, OpKind::Xor, OpKind::Cmp,
    };
    static constexpr Handler handlers_[8] = {
        &handlers::op_add, &handlers::op_or,  &handlers::op_adc, &handlers::op_sbb,
        &handlers::op_and, &handlers::op_sub, &handlers::op_xor, &handlers::op_cmp,
    };

    out.kind    = kinds[family_index];
    out.handler = handlers_[family_index];
    if (out.kind == OpKind::Cmp) {
        // CMP is "SUB without writeback" -- no special flag here yet.
    }

    switch (variant) {
        case 0: {  // r/m8, r8  (MR)
            out.op_size = alu_size_8();
            u8 mb = 0;
            if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
            const ModRm m = split_modrm(mb);
            Operand dst{}, src{};
            if (Status s = decode_rm_operand(bs, px, m, out.op_size, dst); fail(s)) return s;
            src = reg_operand(px, m, out.op_size);
            out.dst = dst;
            out.src = src;
            return Status::Ok;
        }
        case 1: {  // r/m{16,32,64}, r{16,32,64}  (MR)
            out.op_size = alu_size(px);
            u8 mb = 0;
            if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
            const ModRm m = split_modrm(mb);
            Operand dst{}, src{};
            if (Status s = decode_rm_operand(bs, px, m, out.op_size, dst); fail(s)) return s;
            src = reg_operand(px, m, out.op_size);
            out.dst = dst;
            out.src = src;
            return Status::Ok;
        }
        case 2: {  // r8, r/m8  (RM)
            out.op_size = alu_size_8();
            u8 mb = 0;
            if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
            const ModRm m = split_modrm(mb);
            Operand src{}, dst = reg_operand(px, m, out.op_size);
            if (Status s = decode_rm_operand(bs, px, m, out.op_size, src); fail(s)) return s;
            out.dst = dst;
            out.src = src;
            return Status::Ok;
        }
        case 3: {  // r{16,32,64}, r/m{16,32,64}  (RM)
            out.op_size = alu_size(px);
            u8 mb = 0;
            if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
            const ModRm m = split_modrm(mb);
            Operand src{}, dst = reg_operand(px, m, out.op_size);
            if (Status s = decode_rm_operand(bs, px, m, out.op_size, src); fail(s)) return s;
            out.dst = dst;
            out.src = src;
            return Status::Ok;
        }
        case 4: {  // AL, imm8
            out.op_size = 1;
            i64 imm = 0;
            if (Status s = read_imm_sx(bs, 1, imm); fail(s)) return s;
            out.dst = make_reg(reg::RAX);
            out.src = make_imm(imm);
            return Status::Ok;
        }
        case 5: {  // rAX, imm (16/32/32sx)
            out.op_size = alu_size(px);
            const u8 imm_size = out.op_size == 2 ? 2 : 4;
            i64 imm = 0;
            if (Status s = read_imm_sx(bs, imm_size, imm); fail(s)) return s;
            out.dst = make_reg(reg::RAX);
            out.src = make_imm(imm);
            return Status::Ok;
        }
        default:
            return Status::UnsupportedOpcode;
    }
}

Status decode_group1(ByteSource& bs, const Prefixes& px, u8 opc, Insn& out) noexcept {
    // ModRM /reg-field selects op: 0=ADD, 1=OR, 2=ADC, 3=SBB, 4=AND, 5=SUB,
    // 6=XOR, 7=CMP.
    u8 mb = 0;
    if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
    const ModRm m = split_modrm(mb);

    const u8 op_size = (opc == 0x80) ? 1 : alu_size(px);
    u8 imm_size;
    switch (opc) {
        case 0x80: imm_size = 1; break;
        case 0x81: imm_size = (op_size == 2) ? 2 : 4; break;
        case 0x83: imm_size = 1; break;  // sign-extended
        default:   return Status::UnsupportedOpcode;
    }

    Operand dst{};
    if (Status s = decode_rm_operand(bs, px, m, op_size, dst); fail(s)) return s;

    i64 imm = 0;
    if (Status s = read_imm_sx(bs, imm_size, imm); fail(s)) return s;

    static constexpr OpKind kinds[8] = {
        OpKind::Add, OpKind::Or,  OpKind::Adc, OpKind::Sbb,
        OpKind::And, OpKind::Sub, OpKind::Xor, OpKind::Cmp,
    };
    static constexpr Handler handlers_[8] = {
        &handlers::op_add, &handlers::op_or,  &handlers::op_adc, &handlers::op_sbb,
        &handlers::op_and, &handlers::op_sub, &handlers::op_xor, &handlers::op_cmp,
    };

    out.kind    = kinds[m.reg];
    out.handler = handlers_[m.reg];
    out.op_size = op_size;
    out.dst     = dst;
    out.src     = make_imm(imm);
    return Status::Ok;
}

Status decode_group2(ByteSource& bs, const Prefixes& px, u8 opc, Insn& out) noexcept {
    // ModRM /reg field selects:
    //   0=ROL 1=ROR 2=RCL 3=RCR 4=SHL/SAL 5=SHR 6=SHL/SAL (alias) 7=SAR
    u8 mb = 0;
    if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
    const ModRm m = split_modrm(mb);

    const u8 op_size = (opc == 0xC0 || opc == 0xD0 || opc == 0xD2) ? 1 : alu_size(px);

    Operand dst{};
    if (Status s = decode_rm_operand(bs, px, m, op_size, dst); fail(s)) return s;

    Operand src{};
    switch (opc) {
        case 0xC0: case 0xC1: {  // imm8 count
            i64 imm = 0;
            if (Status s = read_imm_sx(bs, 1, imm); fail(s)) return s;
            src = make_imm(imm);
            break;
        }
        case 0xD0: case 0xD1:    // count = 1 (implicit)
            src = make_imm(1);
            break;
        case 0xD2: case 0xD3:    // count = CL (operand kind = Reg)
            src = make_reg(reg::RCX);
            // Encoded as 8-bit register but operand semantics use only low 8 bits.
            break;
        default:
            return Status::UnsupportedOpcode;
    }

    static constexpr OpKind kinds[8] = {
        OpKind::Rol, OpKind::Ror, OpKind::Rcl, OpKind::Rcr,
        OpKind::Shl, OpKind::Shr, OpKind::Shl, OpKind::Sar,
    };
    static constexpr Handler handlers_[8] = {
        &handlers::op_rol, &handlers::op_ror, &handlers::op_rcl, &handlers::op_rcr,
        &handlers::op_shl, &handlers::op_shr, &handlers::op_shl, &handlers::op_sar,
    };

    out.kind    = kinds[m.reg];
    out.handler = handlers_[m.reg];
    out.op_size = op_size;
    out.dst     = dst;
    out.src     = src;
    return Status::Ok;
}

Status decode_group3(ByteSource& bs, const Prefixes& px, u8 opc, Insn& out) noexcept {
    // ModRM /reg: 0=TEST imm, 1=TEST imm (alias), 2=NOT, 3=NEG,
    //             4=MUL, 5=IMUL (1-op), 6=DIV, 7=IDIV
    u8 mb = 0;
    if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
    const ModRm m = split_modrm(mb);

    const u8 op_size = (opc == 0xF6) ? 1 : alu_size(px);

    Operand rm{};
    if (Status s = decode_rm_operand(bs, px, m, op_size, rm); fail(s)) return s;

    switch (m.reg) {
        case 0: case 1: {
            // TEST r/m, imm
            const u8 imm_size = (opc == 0xF6) ? 1 : (op_size == 2 ? 2 : 4);
            i64 imm = 0;
            if (Status s = read_imm_sx(bs, imm_size, imm); fail(s)) return s;
            out.kind    = OpKind::Test;
            out.handler = &handlers::op_test;
            out.op_size = op_size;
            out.dst     = rm;
            out.src     = make_imm(imm);
            return Status::Ok;
        }
        case 2:   // NOT r/m
            out.kind    = OpKind::Not;
            out.handler = &handlers::op_not;
            out.op_size = op_size;
            out.dst     = rm;
            return Status::Ok;
        case 3:   // NEG r/m
            out.kind    = OpKind::Neg;
            out.handler = &handlers::op_neg;
            out.op_size = op_size;
            out.dst     = rm;
            return Status::Ok;
        case 4:   // MUL r/m  -- unsigned full-width product into [r]DX:[r]AX
            out.kind    = OpKind::Mul;
            out.handler = &handlers::op_mul;
            out.op_size = op_size;
            out.src     = rm;     // multiplicand B; multiplier A is implicit [r]AX
            return Status::Ok;
        case 5:   // IMUL r/m  -- signed full-width product
            out.kind    = OpKind::Imul;
            out.handler = &handlers::op_imul1;
            out.op_size = op_size;
            out.src     = rm;
            return Status::Ok;
        case 6:   // DIV r/m  -- unsigned divide [r]DX:[r]AX / r/m
            out.kind    = OpKind::Div;
            out.handler = &handlers::op_div;
            out.op_size = op_size;
            out.src     = rm;
            return Status::Ok;
        case 7:   // IDIV r/m -- signed divide
            out.kind    = OpKind::Idiv;
            out.handler = &handlers::op_idiv;
            out.op_size = op_size;
            out.src     = rm;
            return Status::Ok;
        default:
            return Status::UnsupportedOpcode;
    }
}

Status decode_group4_5(ByteSource& bs, const Prefixes& px, u8 opc, Insn& out) noexcept {
    // 0xFE -- group 4 (8-bit): /0 INC, /1 DEC
    // 0xFF -- group 5 (16/32/64): /0 INC, /1 DEC, /2 CALL r/m, /4 JMP r/m, /6 PUSH r/m
    u8 mb = 0;
    if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
    const ModRm m = split_modrm(mb);

    const u8 op_size = (opc == 0xFE) ? 1 : alu_size(px);

    Operand rm{};
    if (Status s = decode_rm_operand(bs, px, m, op_size, rm); fail(s)) return s;

    switch (m.reg) {
        case 0:
            out.kind    = OpKind::Inc;
            out.handler = &handlers::op_inc;
            out.op_size = op_size;
            out.dst     = rm;
            return Status::Ok;
        case 1:
            out.kind    = OpKind::Dec;
            out.handler = &handlers::op_dec;
            out.op_size = op_size;
            out.dst     = rm;
            return Status::Ok;
        case 2:
            if (opc != 0xFF) return Status::UnsupportedOpcode;
            out.kind    = OpKind::Call;
            out.handler = &handlers::op_call;
            out.op_size = 8;     // x86-64: indirect calls always 64-bit
            out.dst     = rm;    // call target operand
            out.flags  |= INSN_FLAG_CONTROL_FLOW;
            return Status::Ok;
        case 4:
            if (opc != 0xFF) return Status::UnsupportedOpcode;
            out.kind    = OpKind::Jmp;
            out.handler = &handlers::op_jmp;
            out.op_size = 8;
            out.dst     = rm;
            out.flags  |= INSN_FLAG_CONTROL_FLOW;
            return Status::Ok;
        case 6:
            if (opc != 0xFF) return Status::UnsupportedOpcode;
            out.kind    = OpKind::Push;
            out.handler = &handlers::op_push;
            out.op_size = 8;
            out.dst     = rm;
            return Status::Ok;
        default:
            return Status::UnsupportedOpcode;
    }
}

// Helper for SSE: decode (xmm dst, xmm/m128 src) ModRM. The "RM" form;
// for SSE the operand size is fixed at 16 bytes regardless of REX.W.
Status decode_xmm_rm(ByteSource& bs, const Prefixes& px,
                     Insn& out, OpKind kind, Handler handler) noexcept {
    u8 mb = 0;
    if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
    const ModRm m = split_modrm(mb);
    out.op_size = 16;
    out.kind    = kind;
    out.handler = handler;

    Operand dst{};
    dst.kind = OperandKind::Xmm;
    dst.reg  = static_cast<u8>((px.rex_r() ? 8u : 0u) | (m.reg & 7));
    out.dst  = dst;

    Operand src{};
    if (m.mod == 3) {
        src.kind = OperandKind::Xmm;
        src.reg  = static_cast<u8>((px.rex_b() ? 8u : 0u) | (m.rm & 7));
    } else {
        if (Status s = decode_rm_operand(bs, px, m, 16, src); fail(s)) return s;
    }
    out.src = src;
    return Status::Ok;
}

// "MR" form: r/m128 dst, xmm src (e.g. MOVDQA m, xmm).
Status decode_xmm_mr(ByteSource& bs, const Prefixes& px,
                     Insn& out, OpKind kind, Handler handler) noexcept {
    u8 mb = 0;
    if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
    const ModRm m = split_modrm(mb);
    out.op_size = 16;
    out.kind    = kind;
    out.handler = handler;

    Operand dst{};
    if (m.mod == 3) {
        dst.kind = OperandKind::Xmm;
        dst.reg  = static_cast<u8>((px.rex_b() ? 8u : 0u) | (m.rm & 7));
    } else {
        if (Status s = decode_rm_operand(bs, px, m, 16, dst); fail(s)) return s;
    }
    out.dst = dst;

    Operand src{};
    src.kind = OperandKind::Xmm;
    src.reg  = static_cast<u8>((px.rex_r() ? 8u : 0u) | (m.reg & 7));
    out.src  = src;
    return Status::Ok;
}

// ---- 0F 38 (SSSE3 / SSE4.x / AES-NI / CRC32 / MOVBE / BMI) ---------------
Status decode_0F38(ByteSource& bs, const Prefixes& px, Insn& out) noexcept {
    u8 op = 0;
    if (Status s = read_byte_ext(bs, op); fail(s)) return s;

    // 0F 38 F0 MOVBE r, m  /  0F 38 F1 MOVBE m, r
    if (op == 0xF0 || op == 0xF1) {
        // F2 0F 38 F0/F1 is CRC32 instead.
        if (px.repne) {
            u8 mb = 0;
            if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
            const ModRm m = split_modrm(mb);
            const u8 size = (op == 0xF0) ? 1 : alu_size(px);
            Operand dst = reg_operand(px, m, 4);
            Operand src{};
            if (Status s = decode_rm_operand(bs, px, m, size, src); fail(s)) return s;
            out.op_size = size; out.dst = dst; out.src = src;
            out.kind = OpKind::Crc32; out.handler = &handlers::op_crc32;
            return Status::Ok;
        }
        u8 mb = 0;
        if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
        const ModRm m = split_modrm(mb);
        const u8 size = (op == 0xF0) ? alu_size(px) : alu_size(px);
        Operand reg_op = reg_operand(px, m, size);
        Operand rm_op{};
        if (Status s = decode_rm_operand(bs, px, m, size, rm_op); fail(s)) return s;
        out.op_size = size;
        if (op == 0xF0) { out.dst = reg_op; out.src = rm_op; }
        else            { out.dst = rm_op;  out.src = reg_op; }
        out.kind = OpKind::Movbe; out.handler = &handlers::op_movbe;
        return Status::Ok;
    }

    // SHA-NI lives in the no-mandatory-prefix space of 0F 38.
    if (!px.opsize_pfx && !px.rep && !px.repne) {
        switch (op) {
            case 0xC8: return decode_xmm_rm(bs, px, out, OpKind::Sha1nexte,    &handlers::op_sha1nexte);
            case 0xC9: return decode_xmm_rm(bs, px, out, OpKind::Sha1msg1,     &handlers::op_sha1msg1);
            case 0xCA: return decode_xmm_rm(bs, px, out, OpKind::Sha1msg2,     &handlers::op_sha1msg2);
            case 0xCB: return decode_xmm_rm(bs, px, out, OpKind::Sha256rnds2,  &handlers::op_sha256rnds2);
            case 0xCC: return decode_xmm_rm(bs, px, out, OpKind::Sha256msg1,   &handlers::op_sha256msg1);
            case 0xCD: return decode_xmm_rm(bs, px, out, OpKind::Sha256msg2,   &handlers::op_sha256msg2);
            default: break;
        }
    }

    // 66 0F 38 xx -- SSE/AES family
    if (px.opsize_pfx) {
        switch (op) {
            case 0x00: return decode_xmm_rm(bs, px, out, OpKind::Pshufb, &handlers::op_pshufb);
            case 0x1C: return decode_xmm_rm(bs, px, out, OpKind::Pabsb,  &handlers::op_pabsb);
            case 0x1D: return decode_xmm_rm(bs, px, out, OpKind::Pabsw,  &handlers::op_pabsw);
            case 0x1E: return decode_xmm_rm(bs, px, out, OpKind::Pabsd,  &handlers::op_pabsd);
            case 0x37: return decode_xmm_rm(bs, px, out, OpKind::Pcmpgtq,&handlers::op_pcmpgtq);
            case 0x38: return decode_xmm_rm(bs, px, out, OpKind::Pminsb, &handlers::op_pminsw);   // approx
            case 0x39: return decode_xmm_rm(bs, px, out, OpKind::Pminsd, &handlers::op_pminsw);
            case 0x3A: return decode_xmm_rm(bs, px, out, OpKind::Pminuw, &handlers::op_pminub);
            case 0x3B: return decode_xmm_rm(bs, px, out, OpKind::Pminud, &handlers::op_pminub);
            case 0x3C: return decode_xmm_rm(bs, px, out, OpKind::Pmaxsb, &handlers::op_pmaxsw);
            case 0x3D: return decode_xmm_rm(bs, px, out, OpKind::Pmaxsd, &handlers::op_pmaxsw);
            case 0x3E: return decode_xmm_rm(bs, px, out, OpKind::Pmaxuw, &handlers::op_pmaxub);
            case 0x3F: return decode_xmm_rm(bs, px, out, OpKind::Pmaxud, &handlers::op_pmaxub);
            case 0x20: return decode_xmm_rm(bs, px, out, OpKind::Pmovsxbw, &handlers::op_pmovmskb);  // stubbed
            case 0xDB: return decode_xmm_rm(bs, px, out, OpKind::Aesimc,         &handlers::op_aesimc);
            case 0xDC: return decode_xmm_rm(bs, px, out, OpKind::Aesenc,         &handlers::op_aesenc);
            case 0xDD: return decode_xmm_rm(bs, px, out, OpKind::Aesenclast,     &handlers::op_aesenclast);
            case 0xDE: return decode_xmm_rm(bs, px, out, OpKind::Aesdec,         &handlers::op_aesdec);
            case 0xDF: return decode_xmm_rm(bs, px, out, OpKind::Aesdeclast,     &handlers::op_aesdeclast);
            default: break;
        }
    }
    return Status::UnsupportedOpcode;
}

// ---- 0F 3A (SSE4.x / AES-NI imm-form / CLMUL / ROUND* / INSERTPS / PEXTR / PINSR) ----
Status decode_0F3A(ByteSource& bs, const Prefixes& px, Insn& out) noexcept {
    u8 op = 0;
    if (Status s = read_byte_ext(bs, op); fail(s)) return s;

    // SHA1RNDS4 (no mandatory prefix): 0F 3A CC /r ib
    if (!px.opsize_pfx && !px.rep && !px.repne && op == 0xCC) {
        if (Status s = decode_xmm_rm(bs, px, out, OpKind::Sha1rnds4, &handlers::op_sha1rnds4); fail(s)) return s;
        i64 imm = 0;
        if (Status s = read_imm_sx(bs, 1, imm); fail(s)) return s;
        out.imm_extra = imm & 0xFF;
        return Status::Ok;
    }

    if (!px.opsize_pfx) return Status::UnsupportedOpcode;     // remaining 0F 3A ops need 66 prefix

    auto consume_imm = [&](Insn& o)->Status {
        i64 imm = 0;
        if (Status s = read_imm_sx(bs, 1, imm); fail(s)) return s;
        o.imm_extra = imm & 0xFF;
        return Status::Ok;
    };

    switch (op) {
        case 0x08: { if (Status s = decode_xmm_rm(bs, px, out, OpKind::Roundps,&handlers::op_roundps); fail(s)) return s; return consume_imm(out); }
        case 0x09: { if (Status s = decode_xmm_rm(bs, px, out, OpKind::Roundpd,&handlers::op_roundpd); fail(s)) return s; return consume_imm(out); }
        case 0x0A: { if (Status s = decode_xmm_rm(bs, px, out, OpKind::Roundss,&handlers::op_roundss); fail(s)) return s; return consume_imm(out); }
        case 0x0B: { if (Status s = decode_xmm_rm(bs, px, out, OpKind::Roundsd,&handlers::op_roundsd); fail(s)) return s; return consume_imm(out); }
        case 0x14: {
            // PEXTRB r/m8, xmm, imm8
            u8 mb = 0;
            if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
            const ModRm m = split_modrm(mb);
            Operand dst{};
            if (Status s = decode_rm_operand(bs, px, m, 1, dst); fail(s)) return s;
            Operand src{}; src.kind = OperandKind::Xmm;
            src.reg = static_cast<u8>((px.rex_r() ? 8u : 0u) | (m.reg & 7));
            out.op_size = 1; out.dst = dst; out.src = src;
            out.kind = OpKind::Pextrb; out.handler = &handlers::op_pextrb;
            return consume_imm(out);
        }
        case 0x15: {
            // PEXTRW r/m16, xmm, imm8
            u8 mb = 0;
            if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
            const ModRm m = split_modrm(mb);
            Operand dst{};
            if (Status s = decode_rm_operand(bs, px, m, 2, dst); fail(s)) return s;
            Operand src{}; src.kind = OperandKind::Xmm;
            src.reg = static_cast<u8>((px.rex_r() ? 8u : 0u) | (m.reg & 7));
            out.op_size = 2; out.dst = dst; out.src = src;
            out.kind = OpKind::Pextrw; out.handler = &handlers::op_pextrw;
            return consume_imm(out);
        }
        case 0x16: {
            // PEXTRD or PEXTRQ (REX.W) r/m, xmm, imm8
            u8 mb = 0;
            if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
            const ModRm m = split_modrm(mb);
            const u8 size = px.rex_w() ? 8 : 4;
            Operand dst{};
            if (Status s = decode_rm_operand(bs, px, m, size, dst); fail(s)) return s;
            Operand src{}; src.kind = OperandKind::Xmm;
            src.reg = static_cast<u8>((px.rex_r() ? 8u : 0u) | (m.reg & 7));
            out.op_size = size; out.dst = dst; out.src = src;
            if (size == 8) { out.kind = OpKind::Pextrq; out.handler = &handlers::op_pextrq; }
            else           { out.kind = OpKind::Pextrd; out.handler = &handlers::op_pextrd; }
            return consume_imm(out);
        }
        case 0x20: {
            // PINSRB xmm, r/m8, imm8
            u8 mb = 0;
            if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
            const ModRm m = split_modrm(mb);
            Operand src{};
            if (Status s = decode_rm_operand(bs, px, m, 1, src); fail(s)) return s;
            Operand dst{}; dst.kind = OperandKind::Xmm;
            dst.reg = static_cast<u8>((px.rex_r() ? 8u : 0u) | (m.reg & 7));
            out.op_size = 1; out.dst = dst; out.src = src;
            out.kind = OpKind::Pinsrb; out.handler = &handlers::op_pinsrb;
            return consume_imm(out);
        }
        case 0x21: { if (Status s = decode_xmm_rm(bs, px, out, OpKind::Insertps, &handlers::op_insertps); fail(s)) return s; return consume_imm(out); }
        case 0x22: {
            // PINSRD / PINSRQ (REX.W) xmm, r/m, imm8
            u8 mb = 0;
            if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
            const ModRm m = split_modrm(mb);
            const u8 size = px.rex_w() ? 8 : 4;
            Operand src{};
            if (Status s = decode_rm_operand(bs, px, m, size, src); fail(s)) return s;
            Operand dst{}; dst.kind = OperandKind::Xmm;
            dst.reg = static_cast<u8>((px.rex_r() ? 8u : 0u) | (m.reg & 7));
            out.op_size = size; out.dst = dst; out.src = src;
            if (size == 8) { out.kind = OpKind::Pinsrq; out.handler = &handlers::op_pinsrq; }
            else           { out.kind = OpKind::Pinsrd; out.handler = &handlers::op_pinsrd; }
            return consume_imm(out);
        }
        case 0x44: { if (Status s = decode_xmm_rm(bs, px, out, OpKind::Pclmulqdq, &handlers::op_pclmulqdq); fail(s)) return s; return consume_imm(out); }
        case 0x60: { if (Status s = decode_xmm_rm(bs, px, out, OpKind::Pcmpestrm,&handlers::op_pcmpestrm);fail(s)) return s; return consume_imm(out); }
        case 0x61: { if (Status s = decode_xmm_rm(bs, px, out, OpKind::Pcmpestri,&handlers::op_pcmpestri);fail(s)) return s; return consume_imm(out); }
        case 0x62: { if (Status s = decode_xmm_rm(bs, px, out, OpKind::Pcmpistrm,&handlers::op_pcmpistrm);fail(s)) return s; return consume_imm(out); }
        case 0x63: { if (Status s = decode_xmm_rm(bs, px, out, OpKind::Pcmpistri,&handlers::op_pcmpistri);fail(s)) return s; return consume_imm(out); }
        case 0xDF: { if (Status s = decode_xmm_rm(bs, px, out, OpKind::Aeskeygenassist, &handlers::op_aeskeygen); fail(s)) return s; return consume_imm(out); }
        default: break;
    }
    return Status::UnsupportedOpcode;
}

Status decode_two_byte(ByteSource& bs, const Prefixes& px, Insn& out, GuestAddr rip) noexcept {
    u8 sec = 0;
    if (Status s = read_byte_ext(bs, sec); fail(s)) return s;

    // 3-byte opcode prefixes:  0F 38 xx  and  0F 3A xx.
    if (sec == 0x38) return decode_0F38(bs, px, out);
    if (sec == 0x3A) return decode_0F3A(bs, px, out);

    // ---- SSE FP scalar single (F3 0F xx) and double (F2 0F xx) ----------
    if (px.rep || px.repne) {
        const bool is_sd = px.repne;
        switch (sec) {
            case 0x10: return decode_xmm_rm(bs, px, out,
                                            is_sd ? OpKind::Movsd : OpKind::Movss,
                                            is_sd ? &handlers::op_movsd_fp : &handlers::op_movss);
            case 0x11: return decode_xmm_mr(bs, px, out,
                                            is_sd ? OpKind::Movsd : OpKind::Movss,
                                            is_sd ? &handlers::op_movsd_fp : &handlers::op_movss);
            case 0x2A: {
                // CVTSI2SS / CVTSI2SD r/m{32,64} -> xmm[0]
                u8 mb = 0;
                if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
                const ModRm m = split_modrm(mb);
                const u8 int_size = px.rex_w() ? 8 : 4;
                Operand src{};
                if (Status s = decode_rm_operand(bs, px, m, int_size, src); fail(s)) return s;
                Operand dst{};
                dst.kind = OperandKind::Xmm;
                dst.reg  = static_cast<u8>((px.rex_r() ? 8u : 0u) | (m.reg & 7));
                out.op_size = int_size;
                out.dst = dst; out.src = src;
                out.kind    = is_sd ? OpKind::Cvtsi2sd : OpKind::Cvtsi2ss;
                out.handler = is_sd ? &handlers::op_cvtsi2sd : &handlers::op_cvtsi2ss;
                return Status::Ok;
            }
            case 0x2C: {
                // CVTTSS2SI / CVTTSD2SI xmm/m32 -> r{32,64}
                u8 mb = 0;
                if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
                const ModRm m = split_modrm(mb);
                const u8 int_size = px.rex_w() ? 8 : 4;
                Operand src{};
                const u8 src_size = is_sd ? 8 : 4;
                if (m.mod == 3) {
                    src.kind = OperandKind::Xmm;
                    src.reg  = static_cast<u8>((px.rex_b() ? 8u : 0u) | (m.rm & 7));
                } else {
                    if (Status s = decode_rm_operand(bs, px, m, src_size, src); fail(s)) return s;
                }
                Operand dst{};
                dst.kind = OperandKind::Reg;
                dst.reg  = static_cast<u8>((px.rex_r() ? 8u : 0u) | (m.reg & 7));
                out.op_size = int_size;
                out.dst = dst; out.src = src;
                out.kind    = is_sd ? OpKind::Cvttsd2si : OpKind::Cvttss2si;
                out.handler = is_sd ? &handlers::op_cvttsd2si : &handlers::op_cvttss2si;
                return Status::Ok;
            }
            case 0x5A: {
                // CVTSS2SD (F3) or CVTSD2SS (F2)
                return decode_xmm_rm(bs, px, out,
                                     is_sd ? OpKind::Cvtsd2ss : OpKind::Cvtss2sd,
                                     is_sd ? &handlers::op_cvtsd2ss : &handlers::op_cvtss2sd);
            }
            case 0x58: return decode_xmm_rm(bs, px, out,
                                            is_sd ? OpKind::Addsd : OpKind::Addss,
                                            is_sd ? &handlers::op_addsd : &handlers::op_addss);
            case 0x5C: return decode_xmm_rm(bs, px, out,
                                            is_sd ? OpKind::Subsd : OpKind::Subss,
                                            is_sd ? &handlers::op_subsd : &handlers::op_subss);
            case 0x59: return decode_xmm_rm(bs, px, out,
                                            is_sd ? OpKind::Mulsd : OpKind::Mulss,
                                            is_sd ? &handlers::op_mulsd : &handlers::op_mulss);
            case 0x5E: return decode_xmm_rm(bs, px, out,
                                            is_sd ? OpKind::Divsd : OpKind::Divss,
                                            is_sd ? &handlers::op_divsd : &handlers::op_divss);
            case 0x51: return decode_xmm_rm(bs, px, out,
                                            is_sd ? OpKind::Sqrtsd : OpKind::Sqrtss,
                                            is_sd ? &handlers::op_sqrtsd : &handlers::op_sqrtss);
            case 0x5D: return decode_xmm_rm(bs, px, out,
                                            is_sd ? OpKind::Minsd : OpKind::Minss,
                                            is_sd ? &handlers::op_minsd : &handlers::op_minss);
            case 0x5F: return decode_xmm_rm(bs, px, out,
                                            is_sd ? OpKind::Maxsd : OpKind::Maxss,
                                            is_sd ? &handlers::op_maxsd : &handlers::op_maxss);
            // F3 0F 53 RCPSS / 52 RSQRTSS -- single only
            case 0x53: if (!is_sd) return decode_xmm_rm(bs, px, out, OpKind::Rcpss,   &handlers::op_rcpss); break;
            case 0x52: if (!is_sd) return decode_xmm_rm(bs, px, out, OpKind::Rsqrtss, &handlers::op_rsqrtss); break;
            // F3 0F 70 PSHUFHW (no_sd), F2 0F 70 PSHUFLW (sd)
            case 0x70: {
                const OpKind k = is_sd ? OpKind::Pshuflw : OpKind::Pshufhw;
                Handler h = is_sd ? &handlers::op_pshuflw : &handlers::op_pshufhw;
                if (Status s = decode_xmm_rm(bs, px, out, k, h); fail(s)) return s;
                i64 imm = 0;
                if (Status s = read_imm_sx(bs, 1, imm); fail(s)) return s;
                out.imm_extra = imm & 0xFF;
                return Status::Ok;
            }
            default: break;
        }
    }

    // ---- SSE FP packed double (66 0F xx) and UCOMISD (66 0F 2E) ----------
    if (px.opsize_pfx) {
        switch (sec) {
            case 0x10: return decode_xmm_rm(bs, px, out, OpKind::Movupd, &handlers::op_movupd);
            case 0x11: return decode_xmm_mr(bs, px, out, OpKind::Movupd, &handlers::op_movupd);
            case 0x28: return decode_xmm_rm(bs, px, out, OpKind::Movapd, &handlers::op_movupd);  // MOVAPD = same as MOVUPD here
            case 0x29: return decode_xmm_mr(bs, px, out, OpKind::Movapd, &handlers::op_movupd);
            case 0x2E: case 0x2F:
                return decode_xmm_rm(bs, px, out, OpKind::Ucomisd, &handlers::op_ucomisd);
            case 0x58: return decode_xmm_rm(bs, px, out, OpKind::Addpd, &handlers::op_addpd);
            case 0x5C: return decode_xmm_rm(bs, px, out, OpKind::Subpd, &handlers::op_subpd);
            case 0x59: return decode_xmm_rm(bs, px, out, OpKind::Mulpd, &handlers::op_mulpd);
            case 0x5E: return decode_xmm_rm(bs, px, out, OpKind::Divpd, &handlers::op_divpd);
            case 0x51: return decode_xmm_rm(bs, px, out, OpKind::Sqrtpd,&handlers::op_sqrtpd);
            case 0x57: return decode_xmm_rm(bs, px, out, OpKind::Xorpd, &handlers::op_xorpd);
            case 0x54: return decode_xmm_rm(bs, px, out, OpKind::Andpd, &handlers::op_andps);  // bitwise = identical
            case 0x56: return decode_xmm_rm(bs, px, out, OpKind::Orpd,  &handlers::op_orps);
            default: break;   // fall through to SSE2 integer dispatch below
        }
    }

    // ---- SSE FP packed single (0F xx, no mandatory prefix) ---------------
    if (!px.opsize_pfx && !px.rep && !px.repne) {
        switch (sec) {
            case 0x10: return decode_xmm_rm(bs, px, out, OpKind::Movups, &handlers::op_movups);
            case 0x11: return decode_xmm_mr(bs, px, out, OpKind::Movups, &handlers::op_movups);
            case 0x28: return decode_xmm_rm(bs, px, out, OpKind::Movaps, &handlers::op_movups);
            case 0x29: return decode_xmm_mr(bs, px, out, OpKind::Movaps, &handlers::op_movups);
            case 0x2E: case 0x2F:
                return decode_xmm_rm(bs, px, out, OpKind::Ucomiss, &handlers::op_ucomiss);
            case 0x58: return decode_xmm_rm(bs, px, out, OpKind::Addps, &handlers::op_addps);
            case 0x5C: return decode_xmm_rm(bs, px, out, OpKind::Subps, &handlers::op_subps);
            case 0x59: return decode_xmm_rm(bs, px, out, OpKind::Mulps, &handlers::op_mulps);
            case 0x5E: return decode_xmm_rm(bs, px, out, OpKind::Divps, &handlers::op_divps);
            case 0x51: return decode_xmm_rm(bs, px, out, OpKind::Sqrtps,&handlers::op_sqrtps);
            case 0x57: return decode_xmm_rm(bs, px, out, OpKind::Xorps, &handlers::op_xorps);
            case 0x54: return decode_xmm_rm(bs, px, out, OpKind::Andps, &handlers::op_andps);
            case 0x56: return decode_xmm_rm(bs, px, out, OpKind::Orps,  &handlers::op_orps);
            case 0x53: return decode_xmm_rm(bs, px, out, OpKind::Rcpps, &handlers::op_rcpps);
            case 0x52: return decode_xmm_rm(bs, px, out, OpKind::Rsqrtps, &handlers::op_rsqrtps);
            // 0F C6 SHUFPS xmm, xmm/m128, imm8
            case 0xC6: {
                if (Status s = decode_xmm_rm(bs, px, out, OpKind::Shufps, &handlers::op_shufps); fail(s)) return s;
                i64 imm = 0;
                if (Status s = read_imm_sx(bs, 1, imm); fail(s)) return s;
                out.imm_extra = imm & 0xFF;
                return Status::Ok;
            }
            case 0x50: {
                // 0F 50 MOVMSKPS r32, xmm
                u8 mb = 0;
                if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
                const ModRm m = split_modrm(mb);
                Operand dst{}; dst.kind = OperandKind::Reg;
                dst.reg  = static_cast<u8>((px.rex_r() ? 8u : 0u) | (m.reg & 7));
                Operand src{}; src.kind = OperandKind::Xmm;
                src.reg  = static_cast<u8>((px.rex_b() ? 8u : 0u) | (m.rm & 7));
                out.op_size = 4; out.dst = dst; out.src = src;
                out.kind = OpKind::Movmskps; out.handler = &handlers::op_movmskps;
                return Status::Ok;
            }
            default: break;
        }
    }

    // ---- SSE2 integer (66 0F xx) -- leftover from earlier --------------
    if (px.opsize_pfx) {
        switch (sec) {
            case 0x6F: return decode_xmm_rm(bs, px, out, OpKind::Movdqa, &handlers::op_movdqa);
            case 0x7F: return decode_xmm_mr(bs, px, out, OpKind::Movdqa, &handlers::op_movdqa);
            case 0x6E: {
                // MOVD/MOVQ xmm, r/m{32,64}.  W bit picks size.
                u8 mb = 0;
                if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
                const ModRm m = split_modrm(mb);
                const u8 size = px.rex_w() ? 8 : 4;
                Operand dst{};
                dst.kind = OperandKind::Xmm;
                dst.reg  = static_cast<u8>((px.rex_r() ? 8u : 0u) | (m.reg & 7));
                Operand src{};
                if (Status s = decode_rm_operand(bs, px, m, size, src); fail(s)) return s;
                out.dst = dst; out.src = src;
                out.op_size = size;
                out.kind    = (size == 8) ? OpKind::Movq : OpKind::Movd;
                out.handler = &handlers::op_movd_to_xmm;
                return Status::Ok;
            }
            case 0x7E: {
                // MOVD/MOVQ r/m{32,64}, xmm.  W bit picks size.
                u8 mb = 0;
                if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
                const ModRm m = split_modrm(mb);
                const u8 size = px.rex_w() ? 8 : 4;
                Operand src{};
                src.kind = OperandKind::Xmm;
                src.reg  = static_cast<u8>((px.rex_r() ? 8u : 0u) | (m.reg & 7));
                Operand dst{};
                if (Status s = decode_rm_operand(bs, px, m, size, dst); fail(s)) return s;
                out.dst = dst; out.src = src;
                out.op_size = size;
                out.kind    = (size == 8) ? OpKind::Movq : OpKind::Movd;
                out.handler = &handlers::op_movd_from_xmm;
                return Status::Ok;
            }
            case 0xEF: return decode_xmm_rm(bs, px, out, OpKind::Pxor,   &handlers::op_pxor);
            case 0xDB: return decode_xmm_rm(bs, px, out, OpKind::Pand,   &handlers::op_pand);
            case 0xEB: return decode_xmm_rm(bs, px, out, OpKind::Por,    &handlers::op_por);
            case 0xFC: return decode_xmm_rm(bs, px, out, OpKind::Paddb,  &handlers::op_paddb);
            case 0xFD: return decode_xmm_rm(bs, px, out, OpKind::Paddw,  &handlers::op_paddw);
            case 0xFE: return decode_xmm_rm(bs, px, out, OpKind::Paddd,  &handlers::op_paddd);
            case 0xD4: return decode_xmm_rm(bs, px, out, OpKind::Paddq,  &handlers::op_paddq);
            case 0xF8: return decode_xmm_rm(bs, px, out, OpKind::Psubb,  &handlers::op_psubb);
            case 0xF9: return decode_xmm_rm(bs, px, out, OpKind::Psubw,  &handlers::op_psubw);
            case 0xFA: return decode_xmm_rm(bs, px, out, OpKind::Psubd,  &handlers::op_psubd);
            case 0xFB: return decode_xmm_rm(bs, px, out, OpKind::Psubq,  &handlers::op_psubq);
            case 0x74: return decode_xmm_rm(bs, px, out, OpKind::Pcmpeqb,&handlers::op_pcmpeqb);
            case 0x75: return decode_xmm_rm(bs, px, out, OpKind::Pcmpeqw,&handlers::op_pcmpeqw);
            case 0x76: return decode_xmm_rm(bs, px, out, OpKind::Pcmpeqd,&handlers::op_pcmpeqd);
            case 0xD5: return decode_xmm_rm(bs, px, out, OpKind::Pmullw, &handlers::op_pmullw);
            case 0xE5: return decode_xmm_rm(bs, px, out, OpKind::Pmulhw, &handlers::op_pmulhw);
            case 0xE4: return decode_xmm_rm(bs, px, out, OpKind::Pmulhuw,&handlers::op_pmulhuw);
            case 0xF5: return decode_xmm_rm(bs, px, out, OpKind::Pmaddwd,&handlers::op_pmaddwd);
            case 0xF1: return decode_xmm_rm(bs, px, out, OpKind::Psllw,  &handlers::op_psllw);
            case 0xF2: return decode_xmm_rm(bs, px, out, OpKind::Pslld,  &handlers::op_pslld);
            case 0xF3: return decode_xmm_rm(bs, px, out, OpKind::Psllq,  &handlers::op_psllq);
            case 0xD1: return decode_xmm_rm(bs, px, out, OpKind::Psrlw,  &handlers::op_psrlw);
            case 0xD2: return decode_xmm_rm(bs, px, out, OpKind::Psrld,  &handlers::op_psrld);
            case 0xD3: return decode_xmm_rm(bs, px, out, OpKind::Psrlq,  &handlers::op_psrlq);
            case 0xE1: return decode_xmm_rm(bs, px, out, OpKind::Psraw,  &handlers::op_psraw);
            case 0xE2: return decode_xmm_rm(bs, px, out, OpKind::Psrad,  &handlers::op_psrad);
            case 0x60: return decode_xmm_rm(bs, px, out, OpKind::Punpcklbw, &handlers::op_punpcklbw);
            case 0x61: return decode_xmm_rm(bs, px, out, OpKind::Punpcklwd, &handlers::op_punpcklwd);
            case 0x62: return decode_xmm_rm(bs, px, out, OpKind::Punpckldq, &handlers::op_punpckldq);
            case 0x6C: return decode_xmm_rm(bs, px, out, OpKind::Punpcklqdq,&handlers::op_punpcklqdq);
            case 0x68: return decode_xmm_rm(bs, px, out, OpKind::Punpckhbw, &handlers::op_punpckhbw);
            case 0x69: return decode_xmm_rm(bs, px, out, OpKind::Punpckhwd, &handlers::op_punpckhwd);
            case 0x6A: return decode_xmm_rm(bs, px, out, OpKind::Punpckhdq, &handlers::op_punpckhdq);
            case 0x6D: return decode_xmm_rm(bs, px, out, OpKind::Punpckhqdq,&handlers::op_punpckhqdq);
            // 66 0F C6 SHUFPD r/m, xmm, imm8
            case 0xC6: {
                if (Status s = decode_xmm_rm(bs, px, out, OpKind::Shufpd, &handlers::op_shufpd); fail(s)) return s;
                i64 imm = 0;
                if (Status s = read_imm_sx(bs, 1, imm); fail(s)) return s;
                out.imm_extra = imm & 0xFF;
                return Status::Ok;
            }
            // PSLLDQ / PSRLDQ via group 14 (66 0F 73 /6 or /2) -- covered as
            // PSLLQ/PSRLQ paths above for now.
            // 66 0F D7 PMOVMSKB r32, xmm
            case 0xD7: {
                u8 mb = 0;
                if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
                const ModRm m = split_modrm(mb);
                Operand dst{}; dst.kind = OperandKind::Reg;
                dst.reg  = static_cast<u8>((px.rex_r() ? 8u : 0u) | (m.reg & 7));
                Operand src{}; src.kind = OperandKind::Xmm;
                src.reg  = static_cast<u8>((px.rex_b() ? 8u : 0u) | (m.rm & 7));
                out.op_size = 4; out.dst = dst; out.src = src;
                out.kind = OpKind::Pmovmskb; out.handler = &handlers::op_pmovmskb;
                return Status::Ok;
            }
            // 66 0F 50 MOVMSKPD r32, xmm
            case 0x50: {
                u8 mb = 0;
                if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
                const ModRm m = split_modrm(mb);
                Operand dst{}; dst.kind = OperandKind::Reg;
                dst.reg  = static_cast<u8>((px.rex_r() ? 8u : 0u) | (m.reg & 7));
                Operand src{}; src.kind = OperandKind::Xmm;
                src.reg  = static_cast<u8>((px.rex_b() ? 8u : 0u) | (m.rm & 7));
                out.op_size = 4; out.dst = dst; out.src = src;
                out.kind = OpKind::Movmskpd; out.handler = &handlers::op_movmskpd;
                return Status::Ok;
            }
            case 0x70: {
                // PSHUFD xmm, xmm/m128, imm8
                if (Status s = decode_xmm_rm(bs, px, out, OpKind::Pshufd, &handlers::op_pshufd); fail(s)) return s;
                i64 imm = 0;
                if (Status s = read_imm_sx(bs, 1, imm); fail(s)) return s;
                out.imm_extra = imm & 0xFF;
                return Status::Ok;
            }
            default: break;
        }
    }

    // ---- SSE (F3 0F xx) -- unaligned MOVDQU, MOVQ xmm,xmm/m64 -------------
    if (px.rep) {
        switch (sec) {
            case 0x6F: return decode_xmm_rm(bs, px, out, OpKind::Movdqu, &handlers::op_movdqu);
            case 0x7F: return decode_xmm_mr(bs, px, out, OpKind::Movdqu, &handlers::op_movdqu);
            case 0x7E: {
                // F3 0F 7E /r -- MOVQ xmm, xmm/m64. Always 8-byte source.
                u8 mb = 0;
                if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
                const ModRm m = split_modrm(mb);
                Operand dst{};
                dst.kind = OperandKind::Xmm;
                dst.reg  = static_cast<u8>((px.rex_r() ? 8u : 0u) | (m.reg & 7));
                Operand src{};
                if (m.mod == 3) {
                    src.kind = OperandKind::Xmm;
                    src.reg  = static_cast<u8>((px.rex_b() ? 8u : 0u) | (m.rm & 7));
                } else {
                    if (Status s = decode_rm_operand(bs, px, m, 8, src); fail(s)) return s;
                }
                out.dst = dst; out.src = src;
                out.op_size = 8;
                out.kind    = OpKind::Movq;
                out.handler = &handlers::op_movd_from_xmm;
                return Status::Ok;
            }
            default: break;
        }
    }

    // F3 0F BC TZCNT, F3 0F BD LZCNT, F3 0F B8 POPCNT (mandatory F3 prefix)
    if (px.rep && (sec == 0xBC || sec == 0xBD || sec == 0xB8)) {
        u8 mb = 0;
        if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
        const ModRm m = split_modrm(mb);
        const u8 size = alu_size(px);
        Operand src{};
        Operand dst = reg_operand(px, m, size);
        if (Status s = decode_rm_operand(bs, px, m, size, src); fail(s)) return s;
        out.op_size = size;
        out.dst = dst; out.src = src;
        switch (sec) {
            case 0xB8: out.kind = OpKind::Popcnt; out.handler = &handlers::op_popcnt; break;
            case 0xBC: out.kind = OpKind::Tzcnt;  out.handler = &handlers::op_tzcnt;  break;
            case 0xBD: out.kind = OpKind::Lzcnt;  out.handler = &handlers::op_lzcnt;  break;
        }
        return Status::Ok;
    }

    // BSF (0F BC) / BSR (0F BD) without F3 prefix
    if (!px.rep && (sec == 0xBC || sec == 0xBD)) {
        u8 mb = 0;
        if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
        const ModRm m = split_modrm(mb);
        const u8 size = alu_size(px);
        Operand src{};
        Operand dst = reg_operand(px, m, size);
        if (Status s = decode_rm_operand(bs, px, m, size, src); fail(s)) return s;
        out.op_size = size;
        out.dst = dst; out.src = src;
        out.kind    = (sec == 0xBC) ? OpKind::Bsf : OpKind::Bsr;
        out.handler = (sec == 0xBC) ? &handlers::op_bsf : &handlers::op_bsr;
        return Status::Ok;
    }

    // BT (0F A3), BTS (0F AB), BTR (0F B3), BTC (0F BB) -- all r/m, r form
    if (sec == 0xA3 || sec == 0xAB || sec == 0xB3 || sec == 0xBB) {
        u8 mb = 0;
        if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
        const ModRm m = split_modrm(mb);
        const u8 size = alu_size(px);
        Operand dst{};
        if (Status s = decode_rm_operand(bs, px, m, size, dst); fail(s)) return s;
        Operand src = reg_operand(px, m, size);
        out.op_size = size;
        out.dst = dst; out.src = src;
        switch (sec) {
            case 0xA3: out.kind = OpKind::Bt;  out.handler = &handlers::op_bt;  break;
            case 0xAB: out.kind = OpKind::Bts; out.handler = &handlers::op_bts; break;
            case 0xB3: out.kind = OpKind::Btr; out.handler = &handlers::op_btr; break;
            case 0xBB: out.kind = OpKind::Btc; out.handler = &handlers::op_btc; break;
        }
        return Status::Ok;
    }

    // Group 8: 0F BA /4 BT, /5 BTS, /6 BTR, /7 BTC -- all r/m, imm8
    if (sec == 0xBA) {
        u8 mb = 0;
        if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
        const ModRm m = split_modrm(mb);
        const u8 size = alu_size(px);
        Operand dst{};
        if (Status s = decode_rm_operand(bs, px, m, size, dst); fail(s)) return s;
        i64 imm = 0;
        if (Status s = read_imm_sx(bs, 1, imm); fail(s)) return s;
        out.op_size = size;
        out.dst = dst;
        out.src.kind = OperandKind::Imm;
        out.src.imm  = imm;
        switch (m.reg) {
            case 4: out.kind = OpKind::Bt;  out.handler = &handlers::op_bt;  return Status::Ok;
            case 5: out.kind = OpKind::Bts; out.handler = &handlers::op_bts; return Status::Ok;
            case 6: out.kind = OpKind::Btr; out.handler = &handlers::op_btr; return Status::Ok;
            case 7: out.kind = OpKind::Btc; out.handler = &handlers::op_btc; return Status::Ok;
            default: return Status::UnsupportedOpcode;
        }
    }

    // CMOVcc (0F 40..4F):  CMOVcc r, r/m.
    if (sec >= 0x40 && sec <= 0x4F) {
        u8 mb = 0;
        if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
        const ModRm m = split_modrm(mb);
        const u8 size = alu_size(px);
        Operand src{};
        Operand dst = reg_operand(px, m, size);
        if (Status s = decode_rm_operand(bs, px, m, size, src); fail(s)) return s;
        out.kind    = OpKind::Cmov;
        out.handler = &handlers::op_cmov;
        out.op_size = size;
        out.cond    = static_cast<Cc>(sec - 0x40);
        out.dst     = dst;
        out.src     = src;
        return Status::Ok;
    }

    // Jcc near (0F 80..8F).
    if (sec >= 0x80 && sec <= 0x8F) {
        u32 rel32 = 0;
        if (Status s = read_u32_ext(bs, rel32); fail(s)) return s;
        out.kind    = OpKind::Jcc;
        out.handler = &handlers::op_jcc;
        out.op_size = 4;
        out.cond    = static_cast<Cc>(sec - 0x80);
        out.dst     = make_rel(static_cast<i64>(static_cast<i32>(rel32)));
        out.flags  |= INSN_FLAG_CONTROL_FLOW;
        (void)rip;
        return Status::Ok;
    }

    // 0F 05 SYSCALL
    if (sec == 0x05) {
        out.kind    = OpKind::Syscall;
        out.handler = &handlers::op_syscall;
        out.flags  |= INSN_FLAG_CONTROL_FLOW;   // syscalls usually return via a different path
        return Status::Ok;
    }

    // 0F 0B UD2 -- undefined instruction (faults). Treated as InvalidOpcode.
    if (sec == 0x0B) return Status::InvalidInstruction;

    // 0F 18 -- group 16 (PREFETCHT0/T1/T2/NTA). ModRM /reg selects level.
    if (sec == 0x18) {
        u8 mb = 0;
        if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
        const ModRm m = split_modrm(mb);
        Operand src{};
        if (Status s = decode_rm_operand(bs, px, m, 1, src); fail(s)) return s;
        out.op_size = 1; out.src = src;
        out.handler = &handlers::op_prefetch;
        switch (m.reg) {
            case 0: out.kind = OpKind::Prefetchnta; break;
            case 1: out.kind = OpKind::Prefetcht0;  break;
            case 2: out.kind = OpKind::Prefetcht1;  break;
            case 3: out.kind = OpKind::Prefetcht2;  break;
            default: out.kind = OpKind::Prefetchnta; break;
        }
        return Status::Ok;
    }

    // 0F AE -- group 15 (fences + CLFLUSH). reg-form: E8/F0/F8 LFENCE/MFENCE/SFENCE.
    //                                       mem-form /7 = CLFLUSH.
    if (sec == 0xAE) {
        u8 mb = 0;
        if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
        const ModRm m = split_modrm(mb);
        if (m.mod == 3) {
            if (m.reg == 5) { out.kind=OpKind::Lfence; out.handler=&handlers::op_fence; return Status::Ok; }
            if (m.reg == 6) { out.kind=OpKind::Mfence; out.handler=&handlers::op_fence; return Status::Ok; }
            if (m.reg == 7) { out.kind=OpKind::Sfence; out.handler=&handlers::op_fence; return Status::Ok; }
            return Status::UnsupportedOpcode;
        }
        if (m.reg == 7) {
            Operand src{};
            if (Status s = decode_rm_operand(bs, px, m, 1, src); fail(s)) return s;
            out.op_size = 1; out.src = src;
            out.kind = OpKind::Clflush; out.handler = &handlers::op_clflush;
            return Status::Ok;
        }
        return Status::UnsupportedOpcode;
    }

    // F3 0F 90 / 0F 1F = PAUSE / NOP-with-hint. PAUSE = F3 90 actually; we
    // catch F3 0F 1F /0 (multi-byte NOP, common in code padding) as a no-op.
    if (sec == 0x1F) {
        u8 mb = 0;
        if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
        const ModRm m = split_modrm(mb);
        Operand src{};
        if (m.mod != 3) {
            if (Status s = decode_rm_operand(bs, px, m, 1, src); fail(s)) return s;
            out.src = src;
        }
        out.op_size = 1;
        out.kind = OpKind::Nop; out.handler = &handlers::op_nop;
        return Status::Ok;
    }

    // 0F 31 RDTSC, 0F A2 CPUID
    if (sec == 0x31) {
        out.kind    = OpKind::Rdtsc;
        out.handler = &handlers::op_rdtsc;
        return Status::Ok;
    }
    if (sec == 0xA2) {
        out.kind    = OpKind::Cpuid;
        out.handler = &handlers::op_cpuid;
        return Status::Ok;
    }

    // 0F C8..CF -- BSWAP r32/r64. Low 3 bits select the register; combine
    // with REX.B for r8..r15. With REX.W the operation is 64-bit, otherwise
    // 32-bit (which zero-extends RAX-RDI to 64).
    if (sec >= 0xC8 && sec <= 0xCF) {
        const u8 size = px.rex_w() ? 8 : 4;
        const u8 reg  = static_cast<u8>((px.rex_b() ? 8u : 0u) | (sec - 0xC8));
        Operand d{};
        d.kind = OperandKind::Reg;
        d.reg  = reg;
        out.op_size = size;
        out.kind    = OpKind::Bswap;
        out.handler = &handlers::op_bswap;
        out.dst     = d;
        return Status::Ok;
    }

    // 0F C7 /6 RDRAND, /7 RDSEED
    if (sec == 0xC7) {
        u8 mb = 0;
        if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
        const ModRm m = split_modrm(mb);
        if (m.reg == 6 || m.reg == 7) {
            const u8 size = alu_size(px);
            out.kind    = (m.reg == 6) ? OpKind::Rdrand : OpKind::Rdseed;
            out.handler = &handlers::op_rdrand;
            out.op_size = size;
            // dst is the r/m register (mod must be 11 here).
            Operand dst{};
            if (Status s = decode_rm_operand(bs, px, m, size, dst); fail(s)) return s;
            out.dst = dst;
            return Status::Ok;
        }
        return Status::UnsupportedOpcode;
    }

    // SHLD r/m, r, imm8  (0F A4) | SHLD r/m, r, CL  (0F A5)
    // SHRD r/m, r, imm8  (0F AC) | SHRD r/m, r, CL  (0F AD)
    if (sec == 0xA4 || sec == 0xA5 || sec == 0xAC || sec == 0xAD) {
        u8 mb = 0;
        if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
        const ModRm m = split_modrm(mb);
        const u8 size = alu_size(px);
        Operand dst{};
        if (Status s = decode_rm_operand(bs, px, m, size, dst); fail(s)) return s;
        // SHLD/SHRD's second register source: stash its id in dst.index so the
        // handler can find it without growing Insn.
        dst.index = static_cast<u8>((px.rex_r() ? 8u : 0u) | (m.reg & 7));
        Operand src{};
        if (sec == 0xA4 || sec == 0xAC) {
            // imm8 count
            i64 imm = 0;
            if (Status s = read_imm_sx(bs, 1, imm); fail(s)) return s;
            src.kind = OperandKind::Imm;
            src.imm  = imm;
            out.imm_extra = imm;       // handler reads from imm_extra for clarity
        } else {
            // CL count
            src.kind = OperandKind::Reg;
            src.reg  = reg::RCX;
        }
        const bool is_shld = (sec == 0xA4 || sec == 0xA5);
        out.kind    = is_shld ? OpKind::Shld : OpKind::Shrd;
        out.handler = is_shld ? &handlers::op_shld : &handlers::op_shrd;
        out.op_size = size;
        out.dst     = dst;
        out.src     = src;
        return Status::Ok;
    }

    // IMUL r, r/m  (0F AF).
    if (sec == 0xAF) {
        u8 mb = 0;
        if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
        const ModRm m = split_modrm(mb);
        const u8 size = alu_size(px);
        Operand src{}, dst = reg_operand(px, m, size);
        if (Status s = decode_rm_operand(bs, px, m, size, src); fail(s)) return s;
        out.kind    = OpKind::Imul;
        out.handler = &handlers::op_imul;
        out.op_size = size;
        out.dst     = dst;
        out.src     = src;
        return Status::Ok;
    }

    // MOVZX (0F B6 = 8->32/64; 0F B7 = 16->32/64).
    if (sec == 0xB6 || sec == 0xB7) {
        u8 mb = 0;
        if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
        const ModRm m = split_modrm(mb);
        const u8 dst_size = alu_size(px);
        const u8 src_size = (sec == 0xB6) ? 1 : 2;
        Operand src{}, dst = reg_operand(px, m, dst_size);
        if (Status s = decode_rm_operand(bs, px, m, src_size, src); fail(s)) return s;
        out.kind    = OpKind::MovZx;
        out.handler = &handlers::op_movzx;
        out.op_size = dst_size;
        // Stash src_size in src.scale (handler reads it from there).
        src.scale = src_size;
        out.dst   = dst;
        out.src   = src;
        return Status::Ok;
    }

    // MOVSX (0F BE = 8->; 0F BF = 16->).
    if (sec == 0xBE || sec == 0xBF) {
        u8 mb = 0;
        if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
        const ModRm m = split_modrm(mb);
        const u8 dst_size = alu_size(px);
        const u8 src_size = (sec == 0xBE) ? 1 : 2;
        Operand src{}, dst = reg_operand(px, m, dst_size);
        if (Status s = decode_rm_operand(bs, px, m, src_size, src); fail(s)) return s;
        out.kind    = OpKind::MovSx;
        out.handler = &handlers::op_movsx;
        out.op_size = dst_size;
        src.scale = src_size;
        out.dst   = dst;
        out.src   = src;
        return Status::Ok;
    }

    return Status::UnsupportedOpcode;
}

} // namespace

// ---- VEX-specific (AVX) ----------------------------------------------------
//
// VEX opcodes live in the implicit 0F / 0F 38 / 0F 3A map (selected by
// `px.vex_map`). We only support `vex_map == 1` (0F) at this scope; 0F 38
// and 0F 3A would add VPSHUFB, AES-NI, BMI, etc.

Status decode_vex_op_rmv(ByteSource& bs, const Prefixes& px,
                         Insn& out, OpKind kind, Handler handler) noexcept {
    // Format: VEX-encoded `op dst, vvvv (src1), r/m (src2)` -- all xmm/ymm.
    u8 mb = 0;
    if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
    const ModRm m = split_modrm(mb);
    const u8 size = px.vex_l ? 32 : 16;
    out.op_size = size;
    out.kind    = kind;
    out.handler = handler;
    out.flags  |= INSN_FLAG_VEX;

    Operand dst{};
    dst.kind  = OperandKind::Xmm;
    dst.reg   = static_cast<u8>((px.rex_r() ? 8u : 0u) | (m.reg & 7));
    dst.index = px.vex_vvvv;        // src1 reg id
    out.dst   = dst;

    Operand src{};
    if (m.mod == 3) {
        src.kind = OperandKind::Xmm;
        src.reg  = static_cast<u8>((px.rex_b() ? 8u : 0u) | (m.rm & 7));
    } else {
        if (Status s = decode_rm_operand(bs, px, m, size, src); fail(s)) return s;
    }
    out.src = src;
    return Status::Ok;
}

// 2-operand VEX (VMOVDQA / VMOVDQU): no vvvv.
Status decode_vex_op_rm(ByteSource& bs, const Prefixes& px,
                        Insn& out, OpKind kind, Handler handler, bool reg_to_rm) noexcept {
    u8 mb = 0;
    if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
    const ModRm m = split_modrm(mb);
    const u8 size = px.vex_l ? 32 : 16;
    out.op_size = size;
    out.kind    = kind;
    out.handler = handler;
    out.flags  |= INSN_FLAG_VEX;

    Operand reg_op{};
    reg_op.kind = OperandKind::Xmm;
    reg_op.reg  = static_cast<u8>((px.rex_r() ? 8u : 0u) | (m.reg & 7));

    Operand rm_op{};
    if (m.mod == 3) {
        rm_op.kind = OperandKind::Xmm;
        rm_op.reg  = static_cast<u8>((px.rex_b() ? 8u : 0u) | (m.rm & 7));
    } else {
        if (Status s = decode_rm_operand(bs, px, m, size, rm_op); fail(s)) return s;
    }
    if (reg_to_rm) { out.dst = rm_op;  out.src = reg_op; }
    else           { out.dst = reg_op; out.src = rm_op; }
    return Status::Ok;
}

// VEX-encoded GPR op with three operands: dst (ModRM.reg), vvvv (in dst.index),
// r/m (insn.src). Used by BMI1/BMI2.
Status decode_vex_gpr_rmv(ByteSource& bs, const Prefixes& px,
                          Insn& out, OpKind kind, Handler handler) noexcept {
    u8 mb = 0;
    if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
    const ModRm m = split_modrm(mb);
    const u8 size = px.vex_w ? 8 : 4;
    out.op_size = size;
    out.kind    = kind;
    out.handler = handler;
    out.flags  |= INSN_FLAG_VEX;

    Operand dst{};
    dst.kind  = OperandKind::Reg;
    dst.reg   = static_cast<u8>((px.rex_r() ? 8u : 0u) | (m.reg & 7));
    dst.index = px.vex_vvvv;        // second source (GPR)
    out.dst   = dst;

    Operand src{};
    if (Status s = decode_rm_operand(bs, px, m, size, src); fail(s)) return s;
    out.src = src;
    return Status::Ok;
}

// VGATHER decode. The SIB.index field is a vector register; we use the same
// decoder path as a normal memory operand (which leaves `index` as a raw reg
// id, which we then *re-interpret* in the handler as a vector reg). We mark
// the operand by setting `src.pad[0] = 1` and stuff the per-lane sizes into
// `imm_extra`:
//   low byte  = data-lane size (4 or 8)
//   next byte = index-lane size (4 or 8)
Status decode_vgather(ByteSource& bs, const Prefixes& px, Insn& out,
                      OpKind kind, Handler handler,
                      u8 data_lane, u8 index_lane) noexcept {
    u8 mb = 0;
    if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
    const ModRm m = split_modrm(mb);
    // mod==3 is invalid for VGATHER (must be memory).
    if (m.mod == 3) return Status::InvalidInstruction;
    const u8 size = px.vex_l ? 32 : 16;
    out.op_size  = size;
    out.kind     = kind;
    out.handler  = handler;
    out.flags   |= INSN_FLAG_VEX;
    out.imm_extra = static_cast<i64>(data_lane | (u32(index_lane) << 8));

    Operand dst{};
    dst.kind  = OperandKind::Xmm;
    dst.reg   = static_cast<u8>((px.rex_r() ? 8u : 0u) | (m.reg & 7));
    dst.index = px.vex_vvvv;                // mask reg (vvvv)
    out.dst   = dst;

    Operand src{};
    if (Status s = decode_rm_operand(bs, px, m, size, src); fail(s)) return s;
    src.pad[0] = 1;                          // mark index as vector reg
    out.src    = src;
    return Status::Ok;
}

// VEX 0F 38 SSE 3-operand routing (when 66 prefix present).
Status decode_vex_map2_sse66(ByteSource& bs, const Prefixes& px, u8 op, Insn& out) noexcept {
    switch (op) {
        case 0x00: return decode_vex_op_rmv(bs, px, out, OpKind::Vpshufb,     &handlers::op_vpshufb);
        case 0x37: return decode_vex_op_rmv(bs, px, out, OpKind::Vpcmpgtq,    &handlers::op_vpcmpgtq);
        case 0xDC: return decode_vex_op_rmv(bs, px, out, OpKind::Vaesenc,     &handlers::op_vaesenc);
        case 0xDD: return decode_vex_op_rmv(bs, px, out, OpKind::Vaesenclast, &handlers::op_vaesenclast);
        case 0xDE: return decode_vex_op_rmv(bs, px, out, OpKind::Vaesdec,     &handlers::op_vaesdec);
        case 0xDF: return decode_vex_op_rmv(bs, px, out, OpKind::Vaesdeclast, &handlers::op_vaesdeclast);
        // VGATHER family -- VEX.W selects qword/dword *data* element.
        case 0x90: return px.vex_w
            ? decode_vgather(bs, px, out, OpKind::Vpgatherdq, &handlers::op_vpgatherdq, 8, 4)
            : decode_vgather(bs, px, out, OpKind::Vpgatherdd, &handlers::op_vpgatherdd, 4, 4);
        case 0x91: return px.vex_w
            ? decode_vgather(bs, px, out, OpKind::Vpgatherqq, &handlers::op_vpgatherqq, 8, 8)
            : decode_vgather(bs, px, out, OpKind::Vpgatherqd, &handlers::op_vpgatherqd, 4, 8);
        case 0x92: return px.vex_w
            ? decode_vgather(bs, px, out, OpKind::Vgatherdpd, &handlers::op_vgatherdpd, 8, 4)
            : decode_vgather(bs, px, out, OpKind::Vgatherdps, &handlers::op_vgatherdps, 4, 4);
        case 0x93: return px.vex_w
            ? decode_vgather(bs, px, out, OpKind::Vgatherqpd, &handlers::op_vgatherqpd, 8, 8)
            : decode_vgather(bs, px, out, OpKind::Vgatherqps, &handlers::op_vgatherqps, 4, 8);
        default: break;
    }
    return Status::UnsupportedOpcode;
}

Status decode_vex_map2_bmi(ByteSource& bs, const Prefixes& px, Insn& out) noexcept {
    u8 op = 0;
    if (Status s = read_byte_ext(bs, op); fail(s)) return s;

    // 66-prefix => SSE-style 3-operand AVX op
    if (px.opsize_pfx) return decode_vex_map2_sse66(bs, px, op, out);

    // BMI ops: pp bits select variants.
    // pp=00 (no prefix) -- ANDN F2, BEXTR F7, BZHI F5, BLS-group F3
    // pp=01 (66)        -- SHLX F7
    // pp=10 (F3)        -- SARX F7, PEXT F5
    // pp=11 (F2)        -- SHRX F7, MULX F6, PDEP F5
    if (op == 0xF2 && !px.opsize_pfx && !px.rep && !px.repne) {
        return decode_vex_gpr_rmv(bs, px, out, OpKind::Andn, &handlers::op_andn);
    }
    if (op == 0xF7) {
        if (!px.opsize_pfx && !px.rep && !px.repne)
            return decode_vex_gpr_rmv(bs, px, out, OpKind::Bextr, &handlers::op_bextr);
        if (px.opsize_pfx) return decode_vex_gpr_rmv(bs, px, out, OpKind::Shlx, &handlers::op_shlx);
        if (px.rep)        return decode_vex_gpr_rmv(bs, px, out, OpKind::Sarx, &handlers::op_sarx);
        if (px.repne)      return decode_vex_gpr_rmv(bs, px, out, OpKind::Shrx, &handlers::op_shrx);
    }
    if (op == 0xF5) {
        if (!px.opsize_pfx && !px.rep && !px.repne)
            return decode_vex_gpr_rmv(bs, px, out, OpKind::Bzhi, &handlers::op_bzhi);
        if (px.rep)   return decode_vex_gpr_rmv(bs, px, out, OpKind::Pext, &handlers::op_pext);
        if (px.repne) return decode_vex_gpr_rmv(bs, px, out, OpKind::Pdep, &handlers::op_pdep);
    }
    if (op == 0xF6 && px.repne) {
        return decode_vex_gpr_rmv(bs, px, out, OpKind::Mulx, &handlers::op_mulx);
    }
    if (op == 0xF3) {
        // BLSI/BLSR/BLSMSK -- group with /reg as sub-op selector. vvvv is the
        // destination; ModRM.reg field selects the op, r/m is source.
        u8 mb = 0;
        if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
        const ModRm m = split_modrm(mb);
        const u8 size = px.vex_w ? 8 : 4;
        Operand dst{};
        dst.kind = OperandKind::Reg;
        dst.reg  = px.vex_vvvv;
        out.dst  = dst;
        Operand src{};
        if (Status s = decode_rm_operand(bs, px, m, size, src); fail(s)) return s;
        out.src = src;
        out.op_size = size;
        out.flags |= INSN_FLAG_VEX;
        switch (m.reg) {
            case 1: out.kind = OpKind::Blsr;    out.handler = &handlers::op_blsr;    return Status::Ok;
            case 2: out.kind = OpKind::Blsmsk;  out.handler = &handlers::op_blsmsk;  return Status::Ok;
            case 3: out.kind = OpKind::Blsi;    out.handler = &handlers::op_blsi;    return Status::Ok;
            default: return Status::UnsupportedOpcode;
        }
    }
    return Status::UnsupportedOpcode;
}

Status decode_vex_map3_rorx(ByteSource& bs, const Prefixes& px, Insn& out) noexcept {
    u8 op = 0;
    if (Status s = read_byte_ext(bs, op); fail(s)) return s;

    // 66 prefix -> SSE-style 3-operand (VPCLMULQDQ)
    if (op == 0x44 && px.opsize_pfx) {
        if (Status s = decode_vex_op_rmv(bs, px, out, OpKind::Vpclmulqdq, &handlers::op_vpclmulqdq); fail(s)) return s;
        i64 imm = 0;
        if (Status s = read_imm_sx(bs, 1, imm); fail(s)) return s;
        out.imm_extra = imm & 0xFF;
        return Status::Ok;
    }

    if (op == 0xF0 && px.repne) {       // F2 0F 3A F0 ib  RORX
        u8 mb = 0;
        if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
        const ModRm m = split_modrm(mb);
        const u8 size = px.vex_w ? 8 : 4;
        Operand dst{};
        dst.kind = OperandKind::Reg;
        dst.reg  = static_cast<u8>((px.rex_r() ? 8u : 0u) | (m.reg & 7));
        out.dst  = dst;
        Operand src{};
        if (Status s = decode_rm_operand(bs, px, m, size, src); fail(s)) return s;
        out.src = src;
        i64 imm = 0;
        if (Status s = read_imm_sx(bs, 1, imm); fail(s)) return s;
        out.op_size = size;
        out.imm_extra = imm & 0xFF;
        out.kind = OpKind::Rorx; out.handler = &handlers::op_rorx;
        out.flags |= INSN_FLAG_VEX;
        return Status::Ok;
    }
    return Status::UnsupportedOpcode;
}

// EVEX 3-operand decode: dst (ModRM.reg), vvvv (in dst.index), r/m (in src).
// Same shape as decode_vex_op_rmv but the operand size comes from EVEX L'L.
Status decode_evex_op_rmv(ByteSource& bs, const Prefixes& px,
                          Insn& out, OpKind kind, Handler handler) noexcept {
    u8 mb = 0;
    if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
    const ModRm m = split_modrm(mb);
    const u8 size = px.evex_vector_size();
    out.op_size = size;
    out.kind    = kind;
    out.handler = handler;
    out.flags  |= INSN_FLAG_VEX;
    // Pack masking info into pad: bits 0-2 = aaa, bit 3 = z, bit 4 = b.
    // When src is a register AND evex.b is set, L'L is the rounding-control
    // override (not the vector size); store in pad bits 5-6.
    u16 pad = static_cast<u16>((px.evex_aaa & 0x7u) | (px.evex_z ? 0x8u : 0u) | (px.evex_b ? 0x10u : 0u));
    if (m.mod == 3 && px.evex_b) {
        const u8 rc = static_cast<u8>((px.evex_l2 ? 2u : 0u) | (px.vex_l ? 1u : 0u));
        pad |= static_cast<u16>(rc << 5);
    }
    out.pad = pad;

    Operand dst{};
    dst.kind  = OperandKind::Xmm;
    dst.reg   = static_cast<u8>((px.evex_r_ ? 16u : 0u) | (px.rex_r() ? 8u : 0u) | (m.reg & 7));
    dst.index = px.vex_vvvv;
    out.dst   = dst;

    Operand src{};
    if (m.mod == 3) {
        src.kind = OperandKind::Xmm;
        src.reg  = static_cast<u8>((px.rex_b() ? 8u : 0u) | (m.rm & 7));
    } else {
        if (Status s = decode_rm_operand(bs, px, m, size, src); fail(s)) return s;
    }
    out.src = src;
    return Status::Ok;
}

// 2-operand EVEX (VMOVDQU* / VMOVDQA*): no vvvv.
Status decode_evex_op_rm(ByteSource& bs, const Prefixes& px,
                         Insn& out, OpKind kind, Handler handler, bool reg_to_rm) noexcept {
    u8 mb = 0;
    if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
    const ModRm m = split_modrm(mb);
    const u8 size = px.evex_vector_size();
    out.op_size = size;
    out.kind    = kind;
    out.handler = handler;
    out.flags  |= INSN_FLAG_VEX;
    out.pad     = static_cast<u16>((px.evex_aaa & 0x7u) | (px.evex_z ? 0x8u : 0u) | (px.evex_b ? 0x10u : 0u));

    Operand reg_op{};
    reg_op.kind = OperandKind::Xmm;
    reg_op.reg  = static_cast<u8>((px.evex_r_ ? 16u : 0u) | (px.rex_r() ? 8u : 0u) | (m.reg & 7));

    Operand rm_op{};
    if (m.mod == 3) {
        rm_op.kind = OperandKind::Xmm;
        rm_op.reg  = static_cast<u8>((px.rex_b() ? 8u : 0u) | (m.rm & 7));
    } else {
        if (Status s = decode_rm_operand(bs, px, m, size, rm_op); fail(s)) return s;
    }
    if (reg_to_rm) { out.dst = rm_op;  out.src = reg_op; }
    else           { out.dst = reg_op; out.src = rm_op; }
    return Status::Ok;
}

// EVEX SCATTER decode (vex_map==2, opcodes 0xA0..0xA3).
// dst = memory (vector-index addressing); src = reg (data source).
// Mask = EVEX.aaa as opmask; not the vvvv form.
Status decode_evex_scatter(ByteSource& bs, const Prefixes& px, Insn& out,
                           OpKind kind, Handler handler,
                           u8 data_lane, u8 index_lane) noexcept {
    u8 mb = 0;
    if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
    const ModRm m = split_modrm(mb);
    if (m.mod == 3) return Status::InvalidInstruction;
    const u8 size = px.evex_vector_size();
    out.op_size    = size;
    out.kind       = kind;
    out.handler    = handler;
    out.flags     |= INSN_FLAG_VEX;
    out.pad        = static_cast<u16>((px.evex_aaa & 0x7u) | (px.evex_z ? 0x8u : 0u));
    out.imm_extra  = static_cast<i64>(data_lane | (u32(index_lane) << 8));

    Operand dst{};                  // source DATA register (we store FROM this).
    dst.kind  = OperandKind::Xmm;
    dst.reg   = static_cast<u8>((px.evex_r_ ? 16u : 0u) | (px.rex_r() ? 8u : 0u) | (m.reg & 7));
    out.dst   = dst;

    Operand src{};                  // destination MEMORY operand (we store TO this).
    if (Status s = decode_rm_operand(bs, px, m, size, src); fail(s)) return s;
    src.pad[0] = 1;                 // mark index as vector reg
    out.src    = src;
    return Status::Ok;
}

Status decode_evex_map2(ByteSource& bs, const Prefixes& px, Insn& out) noexcept {
    u8 op = 0;
    if (Status s = read_byte_ext(bs, op); fail(s)) return s;
    if (px.opsize_pfx) {
        switch (op) {
            case 0xA0: return px.vex_w
                ? decode_evex_scatter(bs, px, out, OpKind::Vpscatterdq, &handlers::op_vpscatterdq, 8, 4)
                : decode_evex_scatter(bs, px, out, OpKind::Vpscatterdd, &handlers::op_vpscatterdd, 4, 4);
            case 0xA1: return px.vex_w
                ? decode_evex_scatter(bs, px, out, OpKind::Vpscatterqq, &handlers::op_vpscatterqq, 8, 8)
                : decode_evex_scatter(bs, px, out, OpKind::Vpscatterqd, &handlers::op_vpscatterqd, 4, 8);
            case 0xA2: return px.vex_w
                ? decode_evex_scatter(bs, px, out, OpKind::Vscatterdpd, &handlers::op_vscatterdpd, 8, 4)
                : decode_evex_scatter(bs, px, out, OpKind::Vscatterdps, &handlers::op_vscatterdps, 4, 4);
            case 0xA3: return px.vex_w
                ? decode_evex_scatter(bs, px, out, OpKind::Vscatterqpd, &handlers::op_vscatterqpd, 8, 8)
                : decode_evex_scatter(bs, px, out, OpKind::Vscatterqps, &handlers::op_vscatterqps, 4, 8);
            default: break;
        }
    }
    return Status::UnsupportedOpcode;
}

Status decode_evex(ByteSource& bs, const Prefixes& px, Insn& out) noexcept {
    if (px.vex_map == 2) return decode_evex_map2(bs, px, out);
    if (px.vex_map != 1) return Status::UnsupportedOpcode;
    u8 op = 0;
    if (Status s = read_byte_ext(bs, op); fail(s)) return s;

    // pp=01 (66 prefix) -- integer SSE-style ops + packed-double FP
    if (px.opsize_pfx) {
        switch (op) {
            case 0xEF:                      // VPXORD/Q
                return px.vex_w
                    ? decode_evex_op_rmv(bs, px, out, OpKind::Vpxorq, &handlers::op_vpxorq)
                    : decode_evex_op_rmv(bs, px, out, OpKind::Vpxord, &handlers::op_vpxord);
            case 0xDB:                      // VPANDD/Q
                return px.vex_w
                    ? decode_evex_op_rmv(bs, px, out, OpKind::Vpandq, &handlers::op_vpandq)
                    : decode_evex_op_rmv(bs, px, out, OpKind::Vpandd, &handlers::op_vpandd);
            case 0xEB:                      // VPORD/Q
                return px.vex_w
                    ? decode_evex_op_rmv(bs, px, out, OpKind::Vporq, &handlers::op_vporq)
                    : decode_evex_op_rmv(bs, px, out, OpKind::Vpord, &handlers::op_vpord);
            case 0xFC: return decode_evex_op_rmv(bs, px, out, OpKind::Vpaddb_evex, &handlers::op_vpaddb_evex);
            case 0xFD: return decode_evex_op_rmv(bs, px, out, OpKind::Vpaddw_evex, &handlers::op_vpaddw_evex);
            case 0xFE: return decode_evex_op_rmv(bs, px, out, OpKind::Vpaddd_evex, &handlers::op_vpaddd_evex);
            case 0xD4: return decode_evex_op_rmv(bs, px, out, OpKind::Vpaddq_evex, &handlers::op_vpaddq_evex);
            case 0xF8: return decode_evex_op_rmv(bs, px, out, OpKind::Vpsubb_evex, &handlers::op_vpsubb_evex);
            case 0xF9: return decode_evex_op_rmv(bs, px, out, OpKind::Vpsubw_evex, &handlers::op_vpsubw_evex);
            case 0xFA: return decode_evex_op_rmv(bs, px, out, OpKind::Vpsubd_evex, &handlers::op_vpsubd_evex);
            case 0xFB: return decode_evex_op_rmv(bs, px, out, OpKind::Vpsubq_evex, &handlers::op_vpsubq_evex);
            case 0x6F:                      // VMOVDQA32 / VMOVDQA64
                return px.vex_w
                    ? decode_evex_op_rm(bs, px, out, OpKind::Vmovdqa64, &handlers::op_vmovdqa64, false)
                    : decode_evex_op_rm(bs, px, out, OpKind::Vmovdqa32, &handlers::op_vmovdqa32, false);
            case 0x7F:
                return px.vex_w
                    ? decode_evex_op_rm(bs, px, out, OpKind::Vmovdqa64, &handlers::op_vmovdqa64, true)
                    : decode_evex_op_rm(bs, px, out, OpKind::Vmovdqa32, &handlers::op_vmovdqa32, true);
            // Packed-double FP (W=1 selects PD; W=0 means PS but EVEX FP uses pp to pick).
            // For 66 0F xx we treat all FP arithmetic opcodes as PD.
            case 0x58: return decode_evex_op_rmv(bs, px, out, OpKind::Vaddpd_evex,  &handlers::op_vaddpd_evex);
            case 0x59: return decode_evex_op_rmv(bs, px, out, OpKind::Vmulpd_evex,  &handlers::op_vmulpd_evex);
            case 0x5C: return decode_evex_op_rmv(bs, px, out, OpKind::Vsubpd_evex,  &handlers::op_vsubpd_evex);
            case 0x5D: return decode_evex_op_rmv(bs, px, out, OpKind::Vminpd_evex,  &handlers::op_vminpd_evex);
            case 0x5E: return decode_evex_op_rmv(bs, px, out, OpKind::Vdivpd_evex,  &handlers::op_vdivpd_evex);
            case 0x5F: return decode_evex_op_rmv(bs, px, out, OpKind::Vmaxpd_evex,  &handlers::op_vmaxpd_evex);
            case 0x51: return decode_evex_op_rm (bs, px, out, OpKind::Vsqrtpd_evex, &handlers::op_vsqrtpd_evex, false);
            default: break;
        }
    }
    // pp=00 (no mandatory prefix) -- packed-single FP
    if (!px.opsize_pfx && !px.rep && !px.repne) {
        switch (op) {
            case 0x58: return decode_evex_op_rmv(bs, px, out, OpKind::Vaddps_evex,  &handlers::op_vaddps_evex);
            case 0x59: return decode_evex_op_rmv(bs, px, out, OpKind::Vmulps_evex,  &handlers::op_vmulps_evex);
            case 0x5C: return decode_evex_op_rmv(bs, px, out, OpKind::Vsubps_evex,  &handlers::op_vsubps_evex);
            case 0x5D: return decode_evex_op_rmv(bs, px, out, OpKind::Vminps_evex,  &handlers::op_vminps_evex);
            case 0x5E: return decode_evex_op_rmv(bs, px, out, OpKind::Vdivps_evex,  &handlers::op_vdivps_evex);
            case 0x5F: return decode_evex_op_rmv(bs, px, out, OpKind::Vmaxps_evex,  &handlers::op_vmaxps_evex);
            case 0x51: return decode_evex_op_rm (bs, px, out, OpKind::Vsqrtps_evex, &handlers::op_vsqrtps_evex, false);
            default: break;
        }
    }
    // pp=10 (F3) -- VMOVDQU32/64
    if (px.rep) {
        switch (op) {
            case 0x6F:
                return px.vex_w
                    ? decode_evex_op_rm(bs, px, out, OpKind::Vmovdqu64, &handlers::op_vmovdqu64, false)
                    : decode_evex_op_rm(bs, px, out, OpKind::Vmovdqu32, &handlers::op_vmovdqu32, false);
            case 0x7F:
                return px.vex_w
                    ? decode_evex_op_rm(bs, px, out, OpKind::Vmovdqu64, &handlers::op_vmovdqu64, true)
                    : decode_evex_op_rm(bs, px, out, OpKind::Vmovdqu32, &handlers::op_vmovdqu32, true);
            default: break;
        }
    }
    return Status::UnsupportedOpcode;
}

Status decode_vex(ByteSource& bs, const Prefixes& px, Insn& out) noexcept {
    if (px.is_evex)      return decode_evex(bs, px, out);
    if (px.vex_map == 2) return decode_vex_map2_bmi(bs, px, out);
    if (px.vex_map == 3) return decode_vex_map3_rorx(bs, px, out);
    if (px.vex_map != 1) return Status::UnsupportedOpcode;
    u8 op = 0;
    if (Status s = read_byte_ext(bs, op); fail(s)) return s;

    // 66 0F xx -- VEX-encoded SSE2-integer / FP-double family
    if (px.opsize_pfx) {
        switch (op) {
            case 0x6F: return decode_vex_op_rm(bs, px, out, OpKind::Vmovdqa, &handlers::op_vmovdqa, false);
            case 0x7F: return decode_vex_op_rm(bs, px, out, OpKind::Vmovdqa, &handlers::op_vmovdqa, true);
            case 0xEF: return decode_vex_op_rmv(bs, px, out, OpKind::Vpxor,  &handlers::op_vpxor);
            case 0xDB: return decode_vex_op_rmv(bs, px, out, OpKind::Vpand,  &handlers::op_vpand);
            case 0xEB: return decode_vex_op_rmv(bs, px, out, OpKind::Vpor,   &handlers::op_vpor);
            case 0xFC: return decode_vex_op_rmv(bs, px, out, OpKind::Vpaddb, &handlers::op_vpaddb);
            case 0xFD: return decode_vex_op_rmv(bs, px, out, OpKind::Vpaddw, &handlers::op_vpaddw);
            case 0xFE: return decode_vex_op_rmv(bs, px, out, OpKind::Vpaddd, &handlers::op_vpaddd);
            case 0xD4: return decode_vex_op_rmv(bs, px, out, OpKind::Vpaddq, &handlers::op_vpaddq);
            case 0xF8: return decode_vex_op_rmv(bs, px, out, OpKind::Vpsubb, &handlers::op_vpsubb);
            case 0xF9: return decode_vex_op_rmv(bs, px, out, OpKind::Vpsubw, &handlers::op_vpsubw);
            case 0xFA: return decode_vex_op_rmv(bs, px, out, OpKind::Vpsubd, &handlers::op_vpsubd);
            case 0xFB: return decode_vex_op_rmv(bs, px, out, OpKind::Vpsubq, &handlers::op_vpsubq);
            default: break;
        }
    }
    // F3 0F xx -- VEX-encoded MOVDQU
    if (px.rep) {
        switch (op) {
            case 0x6F: return decode_vex_op_rm(bs, px, out, OpKind::Vmovdqu, &handlers::op_vmovdqu, false);
            case 0x7F: return decode_vex_op_rm(bs, px, out, OpKind::Vmovdqu, &handlers::op_vmovdqu, true);
            default: break;
        }
    }
    // No mandatory prefix
    if (!px.opsize_pfx && !px.rep && !px.repne) {
        if (op == 0x77) {
            out.kind    = px.vex_l ? OpKind::Vzeroall : OpKind::Vzeroupper;
            out.handler = px.vex_l ? &handlers::op_vzeroall : &handlers::op_vzeroupper;
            return Status::Ok;
        }
    }
    return Status::UnsupportedOpcode;
}

// ---- Entry point -----------------------------------------------------------
Status decode_after_prefixes(ByteSource& bs, Prefixes& px, GuestAddr rip, Insn& out) noexcept {
    if (px.is_vex) {
        return decode_vex(bs, px, out);
    }

    u8 opc = 0;
    if (Status s = read_byte_ext(bs, opc); fail(s)) return s;

    // ALU primary families: ADD 0x00, OR 0x08, ADC 0x10, SBB 0x18,
    //                       AND 0x20, SUB 0x28, XOR 0x30, CMP 0x38.
    if (opc <= 0x3D && (opc & 0x06) != 0x06) {
        const u8 family = static_cast<u8>(opc >> 3);
        const u8 variant = static_cast<u8>(opc & 0x07);
        // Skip non-ALU opcodes inside these rows (e.g. 0x06/0x07 PUSH/POP ES,
        // 0x0E PUSH CS, etc. -- all invalid in 64-bit mode anyway).
        if (variant < 6) {
            return decode_alu_primary(bs, px, family, variant, out);
        }
    }

    // 0x50..0x57 PUSH r64
    if (opc >= 0x50 && opc <= 0x57) {
        const u8 reg_id = static_cast<u8>((px.rex_b() ? 8u : 0u) | (opc - 0x50u));
        out.kind    = OpKind::Push;
        out.handler = &handlers::op_push;
        out.op_size = 8;
        out.dst     = make_reg(reg_id);
        return Status::Ok;
    }
    // 0x58..0x5F POP r64
    if (opc >= 0x58 && opc <= 0x5F) {
        const u8 reg_id = static_cast<u8>((px.rex_b() ? 8u : 0u) | (opc - 0x58u));
        out.kind    = OpKind::Pop;
        out.handler = &handlers::op_pop;
        out.op_size = 8;
        out.dst     = make_reg(reg_id);
        return Status::Ok;
    }

    // 0x68 PUSH imm32 (sign-extended to 64)
    if (opc == 0x68) {
        i64 imm = 0;
        if (Status s = read_imm_sx(bs, 4, imm); fail(s)) return s;
        out.kind    = OpKind::Push;
        out.handler = &handlers::op_push;
        out.op_size = 8;
        out.dst     = make_imm(imm);
        return Status::Ok;
    }
    // 0x6A PUSH imm8 (sign-extended)
    if (opc == 0x6A) {
        i64 imm = 0;
        if (Status s = read_imm_sx(bs, 1, imm); fail(s)) return s;
        out.kind    = OpKind::Push;
        out.handler = &handlers::op_push;
        out.op_size = 8;
        out.dst     = make_imm(imm);
        return Status::Ok;
    }

    // 0x63 MOVSXD r{16,32,64}, r/m32  (sign-extend 32->64 when REX.W)
    if (opc == 0x63) {
        u8 mb = 0;
        if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
        const ModRm m = split_modrm(mb);
        const u8 dst_size = alu_size(px);
        const u8 src_size = 4;
        Operand src{};
        Operand dst = reg_operand(px, m, dst_size);
        if (Status s = decode_rm_operand(bs, px, m, src_size, src); fail(s)) return s;
        src.scale = src_size;
        out.kind    = OpKind::MovSxd;
        out.handler = &handlers::op_movsxd;
        out.op_size = dst_size;
        out.dst     = dst;
        out.src     = src;
        return Status::Ok;
    }

    // 0x69 / 0x6B -- IMUL r{16,32,64}, r/m, imm{16|32|8sx}
    if (opc == 0x69 || opc == 0x6B) {
        const u8 size = alu_size(px);
        u8 mb = 0;
        if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
        const ModRm m = split_modrm(mb);
        Operand src{};
        Operand dst = reg_operand(px, m, size);
        if (Status s = decode_rm_operand(bs, px, m, size, src); fail(s)) return s;
        const u8 imm_size = (opc == 0x6B) ? 1 : (size == 2 ? 2 : 4);
        i64 imm = 0;
        if (Status s = read_imm_sx(bs, imm_size, imm); fail(s)) return s;
        out.kind      = OpKind::Imul3;
        out.handler   = &handlers::op_imul;
        out.op_size   = size;
        out.dst       = dst;
        out.src       = src;
        out.imm_extra = imm;
        return Status::Ok;
    }

    // 0x70..0x7F Jcc short
    if (opc >= 0x70 && opc <= 0x7F) {
        u8 rel8 = 0;
        if (Status s = read_byte_ext(bs, rel8); fail(s)) return s;
        out.kind    = OpKind::Jcc;
        out.handler = &handlers::op_jcc;
        out.op_size = 1;
        out.cond    = static_cast<Cc>(opc - 0x70);
        out.dst     = make_rel(static_cast<i64>(static_cast<i8>(rel8)));
        out.flags  |= INSN_FLAG_CONTROL_FLOW;
        return Status::Ok;
    }

    // 0x80/0x81/0x83 -- group 1
    if (opc == 0x80 || opc == 0x81 || opc == 0x83) {
        return decode_group1(bs, px, opc, out);
    }

    // 0x84 TEST r/m8, r8 ; 0x85 TEST r/m, r
    if (opc == 0x84 || opc == 0x85) {
        const u8 size = (opc == 0x84) ? 1 : alu_size(px);
        u8 mb = 0;
        if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
        const ModRm m = split_modrm(mb);
        Operand dst{};
        if (Status s = decode_rm_operand(bs, px, m, size, dst); fail(s)) return s;
        Operand src = reg_operand(px, m, size);
        out.kind    = OpKind::Test;
        out.handler = &handlers::op_test;
        out.op_size = size;
        out.dst     = dst;
        out.src     = src;
        return Status::Ok;
    }

    // 0x86/0x87 XCHG r/m, r
    if (opc == 0x86 || opc == 0x87) {
        const u8 size = (opc == 0x86) ? 1 : alu_size(px);
        u8 mb = 0;
        if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
        const ModRm m = split_modrm(mb);
        Operand a{};
        if (Status s = decode_rm_operand(bs, px, m, size, a); fail(s)) return s;
        Operand b = reg_operand(px, m, size);
        out.kind    = OpKind::Xchg;
        out.handler = &handlers::op_xchg;
        out.op_size = size;
        out.dst     = a;
        out.src     = b;
        return Status::Ok;
    }

    // 0x88..0x8B MOV
    if (opc >= 0x88 && opc <= 0x8B) {
        const u8 size = (opc & 1u) ? alu_size(px) : 1;
        u8 mb = 0;
        if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
        const ModRm m = split_modrm(mb);
        Operand dst{}, src{};
        const bool reg_to_rm = (opc == 0x88 || opc == 0x89);
        if (reg_to_rm) {
            if (Status s = decode_rm_operand(bs, px, m, size, dst); fail(s)) return s;
            src = reg_operand(px, m, size);
        } else {
            dst = reg_operand(px, m, size);
            if (Status s = decode_rm_operand(bs, px, m, size, src); fail(s)) return s;
        }
        out.kind    = OpKind::Mov;
        out.handler = &handlers::op_mov;
        out.op_size = size;
        out.dst     = dst;
        out.src     = src;
        return Status::Ok;
    }

    // 0x8D LEA r, m
    if (opc == 0x8D) {
        const u8 size = alu_size(px);
        u8 mb = 0;
        if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
        const ModRm m = split_modrm(mb);
        if (m.mod == 3) return Status::InvalidInstruction;  // LEA requires mem operand
        Operand src{};
        if (Status s = decode_rm_operand(bs, px, m, size, src); fail(s)) return s;
        out.kind    = OpKind::Lea;
        out.handler = &handlers::op_lea;
        out.op_size = size;
        out.dst     = reg_operand(px, m, size);
        out.src     = src;
        return Status::Ok;
    }

    // 0x90 NOP (canonical XCHG rAX,rAX)
    if (opc == 0x90) {
        out.kind    = OpKind::Nop;
        out.handler = &handlers::op_nop;
        out.op_size = 0;
        return Status::Ok;
    }

    // 0xB0..0xB7 MOV r8, imm8
    if (opc >= 0xB0 && opc <= 0xB7) {
        const u8 reg_id = static_cast<u8>((px.rex_b() ? 8u : 0u) | (opc - 0xB0u));
        i64 imm = 0;
        if (Status s = read_imm_sx(bs, 1, imm); fail(s)) return s;
        out.kind    = OpKind::Mov;
        out.handler = &handlers::op_mov;
        out.op_size = 1;
        out.dst     = make_reg(reg_id);
        out.src     = make_imm(imm);
        return Status::Ok;
    }
    // 0xB8..0xBF MOV r{16,32,64}, imm{16,32,64}
    if (opc >= 0xB8 && opc <= 0xBF) {
        const u8 reg_id = static_cast<u8>((px.rex_b() ? 8u : 0u) | (opc - 0xB8u));
        const u8 size = alu_size(px);
        i64 imm = 0;
        if (Status s = read_imm_sx(bs, size, imm); fail(s)) return s;
        out.kind    = OpKind::Mov;
        out.handler = &handlers::op_mov;
        out.op_size = size;
        out.dst     = make_reg(reg_id);
        out.src     = make_imm(imm);
        return Status::Ok;
    }

    // 0xC0/0xC1 group 2 with imm8
    if (opc == 0xC0 || opc == 0xC1) {
        return decode_group2(bs, px, opc, out);
    }

    // 0xC2 RET imm16
    if (opc == 0xC2) {
        u16 imm = 0;
        if (Status s = read_u16_ext(bs, imm); fail(s)) return s;
        out.kind    = OpKind::Ret;
        out.handler = &handlers::op_ret;
        out.op_size = 8;
        out.dst     = make_imm(static_cast<i64>(imm));
        out.flags  |= INSN_FLAG_CONTROL_FLOW;
        return Status::Ok;
    }
    // 0xC3 RET near
    if (opc == 0xC3) {
        out.kind    = OpKind::Ret;
        out.handler = &handlers::op_ret;
        out.op_size = 8;
        out.flags  |= INSN_FLAG_CONTROL_FLOW;
        return Status::Ok;
    }

    // 0xC6 MOV r/m8, imm8 ; 0xC7 MOV r/m{16,32,64}, imm{16,32,32sx}
    if (opc == 0xC6 || opc == 0xC7) {
        const u8 size = (opc == 0xC6) ? 1 : alu_size(px);
        u8 mb = 0;
        if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
        const ModRm m = split_modrm(mb);
        if (m.reg != 0) return Status::UnsupportedOpcode;
        Operand dst{};
        if (Status s = decode_rm_operand(bs, px, m, size, dst); fail(s)) return s;
        const u8 imm_size = (size == 8) ? 4 : size;  // 64-bit MOV imm uses 32sx
        i64 imm = 0;
        if (Status s = read_imm_sx(bs, imm_size, imm); fail(s)) return s;
        out.kind    = OpKind::Mov;
        out.handler = &handlers::op_mov;
        out.op_size = size;
        out.dst     = dst;
        out.src     = make_imm(imm);
        return Status::Ok;
    }

    // 0xD0..0xD3 group 2 with implicit count
    if (opc >= 0xD0 && opc <= 0xD3) {
        return decode_group2(bs, px, opc, out);
    }

    // ---- x87 (D8-DF) ---------------------------------------------------
    // The opcode and the ModRM byte together select the operation. We handle
    // the common subset; rare instructions fall through as UnsupportedOpcode.
    if (opc >= 0xD8 && opc <= 0xDF) {
        u8 mb = 0;
        if (Status s = read_byte_ext(bs, mb); fail(s)) return s;
        const ModRm m = split_modrm(mb);
        const bool reg_form = (m.mod == 3);

        // Memory-form size mapping:
        //   D8/DC: 32-bit float (op_size=4) / 64-bit float (op_size=8)
        //   DA/DE: 32-bit / 16-bit integer
        //   DB/DF: 32-bit integer (DB), 16/32/64-bit integer (DF)
        //   DD:    64-bit float
        u8 mem_size = 4;
        switch (opc) {
            case 0xD8: mem_size = 4; break;   // float32
            case 0xD9: mem_size = 4; break;   // float32 mostly
            case 0xDA: mem_size = 4; break;   // i32
            case 0xDB: mem_size = 4; break;   // i32 / float80 depending on /reg
            case 0xDC: mem_size = 8; break;   // float64
            case 0xDD: mem_size = 8; break;   // float64 mostly
            case 0xDE: mem_size = 2; break;   // i16
            case 0xDF: mem_size = 2; break;   // i16 / i64 depending on /reg
        }
        out.op_size = mem_size;

        // Stash ST(i) index in dst.index when register form.
        if (reg_form) {
            out.dst.index = m.rm;
        } else {
            Operand src{};
            if (Status s = decode_rm_operand(bs, px, m, mem_size, src); fail(s)) return s;
            out.src = src;
        }

        // Tables: opc + /reg -> (OpKind, handler). We only cover the routine cases.
        // Reg-form (mod=3):
        if (reg_form) {
            if (opc == 0xD8) {
                // D8 C0-C7 FADD ST,ST(i)   D0-D7 FCOM  E0-E7 FSUB  E8-EF FSUBR  F0-F7 FDIV  F8-FF FDIVR
                switch (m.reg) {
                    case 0: out.kind=OpKind::Fadd; out.handler=&handlers::op_fadd; return Status::Ok;
                    case 1: out.kind=OpKind::Fmul; out.handler=&handlers::op_fmul; return Status::Ok;
                    case 2: case 3: out.kind=OpKind::Fcom; out.handler=&handlers::op_fcom; return Status::Ok;
                    case 4: out.kind=OpKind::Fsub; out.handler=&handlers::op_fsub; return Status::Ok;
                    case 6: out.kind=OpKind::Fdiv; out.handler=&handlers::op_fdiv; return Status::Ok;
                    default: return Status::UnsupportedOpcode;
                }
            }
            if (opc == 0xD9) {
                if (m.reg == 0) { out.kind=OpKind::Fld;  out.handler=&handlers::op_fld;  return Status::Ok; }
                if (m.reg == 1) { out.kind=OpKind::Fxch; out.handler=&handlers::op_fxch; return Status::Ok; }
                // D9 E0=FCHS, E1=FABS, ...
                switch (mb) {
                    case 0xE0: out.kind=OpKind::Fchs;    out.handler=&handlers::op_fchs;   return Status::Ok;
                    case 0xE1: out.kind=OpKind::Fabs;    out.handler=&handlers::op_fabs;   return Status::Ok;
                    case 0xFA: out.kind=OpKind::Fsqrt;   out.handler=&handlers::op_fsqrt;  return Status::Ok;
                    case 0xF0: out.kind=OpKind::F2xm1;   out.handler=&handlers::op_f2xm1;  return Status::Ok;
                    case 0xF1: out.kind=OpKind::Fyl2x;   out.handler=&handlers::op_fyl2x;  return Status::Ok;
                    case 0xF3: out.kind=OpKind::Fpatan;  out.handler=&handlers::op_fpatan; return Status::Ok;
                    case 0xFD: out.kind=OpKind::Fscale;  out.handler=&handlers::op_fscale; return Status::Ok;
                    case 0xFC: out.kind=OpKind::Frndint; out.handler=&handlers::op_frndint;return Status::Ok;
                    case 0xFE: out.kind=OpKind::Fsin;    out.handler=&handlers::op_fsin;   return Status::Ok;
                    case 0xFF: out.kind=OpKind::Fcos;    out.handler=&handlers::op_fcos;   return Status::Ok;
                    default: return Status::UnsupportedOpcode;
                }
            }
            if (opc == 0xDB && mb == 0xE3) {
                out.kind=OpKind::Fninit; out.handler=&handlers::op_fninit; return Status::Ok;
            }
            if (opc == 0xDC) {
                switch (m.reg) {
                    case 0: out.kind=OpKind::Fadd; out.handler=&handlers::op_fadd; return Status::Ok;
                    case 1: out.kind=OpKind::Fmul; out.handler=&handlers::op_fmul; return Status::Ok;
                    case 4: out.kind=OpKind::Fsub; out.handler=&handlers::op_fsub; return Status::Ok;
                    case 6: out.kind=OpKind::Fdiv; out.handler=&handlers::op_fdiv; return Status::Ok;
                    default: return Status::UnsupportedOpcode;
                }
            }
            if (opc == 0xDD) {
                if (m.reg == 2) { out.kind=OpKind::Fst;  out.handler=&handlers::op_fst;  return Status::Ok; }
                if (m.reg == 3) { out.kind=OpKind::Fstp; out.handler=&handlers::op_fstp; return Status::Ok; }
                return Status::UnsupportedOpcode;
            }
            if (opc == 0xDE) {
                switch (m.reg) {
                    case 0: out.kind=OpKind::Faddp; out.handler=&handlers::op_faddp; return Status::Ok;
                    case 1: out.kind=OpKind::Fmulp; out.handler=&handlers::op_fmulp; return Status::Ok;
                    case 4: out.kind=OpKind::Fsubp; out.handler=&handlers::op_fsubp; return Status::Ok;
                    case 6: out.kind=OpKind::Fdivp; out.handler=&handlers::op_fdivp; return Status::Ok;
                    default: return Status::UnsupportedOpcode;
                }
            }
            if (opc == 0xDF && mb == 0xE0) {
                // FNSTSW AX
                Operand dst{}; dst.kind = OperandKind::Reg; dst.reg = reg::RAX;
                out.dst = dst;
                out.kind = OpKind::Fnstsw; out.handler = &handlers::op_fnstsw;
                return Status::Ok;
            }
            return Status::UnsupportedOpcode;
        }

        // Memory form (mod != 3):
        switch (opc) {
            case 0xD9:
                if (m.reg == 0) { out.kind=OpKind::Fld;    out.handler=&handlers::op_fld;    return Status::Ok; }
                if (m.reg == 2) { out.kind=OpKind::Fst;    out.handler=&handlers::op_fst;    return Status::Ok; }
                if (m.reg == 3) { out.kind=OpKind::Fstp;   out.handler=&handlers::op_fstp;   return Status::Ok; }
                if (m.reg == 5) { out.op_size = 2; out.kind=OpKind::Fldcw; out.handler=&handlers::op_fldcw; return Status::Ok; }
                if (m.reg == 7) { out.op_size = 2;
                                  Operand dst = out.src; out.src = {}; out.dst = dst;
                                  out.kind=OpKind::Fnstcw; out.handler=&handlers::op_fnstcw;
                                  return Status::Ok; }
                return Status::UnsupportedOpcode;
            case 0xDB:
                if (m.reg == 0) { out.op_size = 4; out.kind=OpKind::Fild;  out.handler=&handlers::op_fild;  return Status::Ok; }
                if (m.reg == 3) { out.op_size = 4; out.kind=OpKind::Fistp; out.handler=&handlers::op_fistp; return Status::Ok; }
                if (m.reg == 5) { out.op_size = 10; out.kind=OpKind::Fld;  out.handler=&handlers::op_fld;   return Status::Ok; }
                if (m.reg == 7) { out.op_size = 10; out.kind=OpKind::Fstp; out.handler=&handlers::op_fstp;  return Status::Ok; }
                return Status::UnsupportedOpcode;
            case 0xDD:
                if (m.reg == 0) { out.kind=OpKind::Fld;  out.handler=&handlers::op_fld;  return Status::Ok; }
                if (m.reg == 2) { out.kind=OpKind::Fst;  out.handler=&handlers::op_fst;  return Status::Ok; }
                if (m.reg == 3) { out.kind=OpKind::Fstp; out.handler=&handlers::op_fstp; return Status::Ok; }
                return Status::UnsupportedOpcode;
            case 0xDF:
                if (m.reg == 0) { out.op_size = 2; out.kind=OpKind::Fild;  out.handler=&handlers::op_fild;  return Status::Ok; }
                if (m.reg == 5) { out.op_size = 8; out.kind=OpKind::Fild;  out.handler=&handlers::op_fild;  return Status::Ok; }
                if (m.reg == 7) { out.op_size = 8; out.kind=OpKind::Fistp; out.handler=&handlers::op_fistp; return Status::Ok; }
                return Status::UnsupportedOpcode;
            default:
                return Status::UnsupportedOpcode;
        }
    }

    // 0x98 CBW / CWDE / CDQE   /   0x99 CWD / CDQ / CQO
    if (opc == 0x98 || opc == 0x99) {
        out.op_size = alu_size(px);
        out.kind    = (opc == 0x98) ? OpKind::Cwde : OpKind::Cdq;
        out.handler = (opc == 0x98) ? &handlers::op_cwde : &handlers::op_cdq;
        return Status::Ok;
    }

    // 0xA8 TEST AL, imm8   /   0xA9 TEST rAX, imm16/32 (sign-ext w/ REX.W)
    if (opc == 0xA8 || opc == 0xA9) {
        const u8 size = (opc == 0xA8) ? 1 : alu_size(px);
        const u8 imm_size = (size == 8) ? 4 : size;
        i64 imm = 0;
        if (Status s = read_imm_sx(bs, imm_size, imm); fail(s)) return s;
        out.op_size = size;
        out.kind    = OpKind::Test;
        out.handler = &handlers::op_test;
        out.dst     = make_reg(reg::RAX);
        out.src     = make_imm(imm);
        return Status::Ok;
    }

    // String ops: A4/A5 MOVS, A6/A7 CMPS, AA/AB STOS, AC/AD LODS, AE/AF SCAS
    if (opc == 0xA4 || opc == 0xA5 || opc == 0xA6 || opc == 0xA7 ||
        opc == 0xAA || opc == 0xAB || opc == 0xAC || opc == 0xAD ||
        opc == 0xAE || opc == 0xAF) {
        const bool word_size = (opc & 1u) != 0;
        const u8 size = word_size ? alu_size(px) : 1;
        out.op_size = size;
        switch (opc) {
            case 0xA4: case 0xA5: out.kind = OpKind::Movs; out.handler = &handlers::op_movs; break;
            case 0xA6: case 0xA7: out.kind = OpKind::Cmps; out.handler = &handlers::op_cmps; break;
            case 0xAA: case 0xAB: out.kind = OpKind::Stos; out.handler = &handlers::op_stos; break;
            case 0xAC: case 0xAD: out.kind = OpKind::Lods; out.handler = &handlers::op_lods; break;
            case 0xAE: case 0xAF: out.kind = OpKind::Scas; out.handler = &handlers::op_scas; break;
        }
        return Status::Ok;
    }

    // 0xE0 LOOPNE, 0xE1 LOOPE, 0xE2 LOOP, 0xE3 JCXZ -- all rel8
    if (opc >= 0xE0 && opc <= 0xE3) {
        u8 rel = 0;
        if (Status s = read_byte_ext(bs, rel); fail(s)) return s;
        out.op_size = 1;
        out.dst = make_rel(static_cast<i64>(static_cast<i8>(rel)));
        out.flags |= INSN_FLAG_CONTROL_FLOW;
        switch (opc) {
            case 0xE0: out.kind = OpKind::LoopNE; out.cond = Cc::NZ; out.handler = &handlers::op_loop; break;
            case 0xE1: out.kind = OpKind::LoopE;  out.cond = Cc::Z;  out.handler = &handlers::op_loop; break;
            case 0xE2: out.kind = OpKind::Loop;   out.cond = Cc::None; out.handler = &handlers::op_loop; break;
            case 0xE3: out.kind = OpKind::Jcxz;   out.handler = &handlers::op_jcxz; break;
        }
        return Status::Ok;
    }

    // 0xF4 HLT
    if (opc == 0xF4) {
        out.kind    = OpKind::Hlt;
        out.handler = &handlers::op_hlt;
        out.flags  |= INSN_FLAG_CONTROL_FLOW;
        return Status::Ok;
    }

    // 0x9C PUSHFQ, 0x9D POPFQ, 0x9E SAHF, 0x9F LAHF
    if (opc == 0x9C) { out.kind=OpKind::Pushf; out.handler=&handlers::op_pushf; out.op_size=8; return Status::Ok; }
    if (opc == 0x9D) { out.kind=OpKind::Popf;  out.handler=&handlers::op_popf;  out.op_size=8; return Status::Ok; }
    if (opc == 0x9E) { out.kind=OpKind::Sahf;  out.handler=&handlers::op_sahf;  return Status::Ok; }
    if (opc == 0x9F) { out.kind=OpKind::Lahf;  out.handler=&handlers::op_lahf;  return Status::Ok; }

    // 0xCC INT3, 0xCD INT n
    if (opc == 0xCC) { out.kind=OpKind::Int3; out.handler=&handlers::op_int3; out.flags |= INSN_FLAG_CONTROL_FLOW; return Status::Ok; }
    if (opc == 0xCD) {
        u8 n=0;
        if (Status s = read_byte_ext(bs, n); fail(s)) return s;
        out.kind = OpKind::Int_n; out.handler = &handlers::op_int_n;
        out.dst.kind = OperandKind::Imm; out.dst.imm = n;
        out.flags |= INSN_FLAG_CONTROL_FLOW;
        return Status::Ok;
    }

    // 0xE8 CALL rel32
    if (opc == 0xE8) {
        u32 rel = 0;
        if (Status s = read_u32_ext(bs, rel); fail(s)) return s;
        out.kind    = OpKind::Call;
        out.handler = &handlers::op_call;
        out.op_size = 4;
        out.dst     = make_rel(static_cast<i64>(static_cast<i32>(rel)));
        out.flags  |= INSN_FLAG_CONTROL_FLOW;
        return Status::Ok;
    }
    // 0xE9 JMP rel32
    if (opc == 0xE9) {
        u32 rel = 0;
        if (Status s = read_u32_ext(bs, rel); fail(s)) return s;
        out.kind    = OpKind::Jmp;
        out.handler = &handlers::op_jmp;
        out.op_size = 4;
        out.dst     = make_rel(static_cast<i64>(static_cast<i32>(rel)));
        out.flags  |= INSN_FLAG_CONTROL_FLOW;
        return Status::Ok;
    }
    // 0xEB JMP rel8
    if (opc == 0xEB) {
        u8 rel = 0;
        if (Status s = read_byte_ext(bs, rel); fail(s)) return s;
        out.kind    = OpKind::Jmp;
        out.handler = &handlers::op_jmp;
        out.op_size = 1;
        out.dst     = make_rel(static_cast<i64>(static_cast<i8>(rel)));
        out.flags  |= INSN_FLAG_CONTROL_FLOW;
        return Status::Ok;
    }

    // 0xF6/0xF7 group 3
    if (opc == 0xF6 || opc == 0xF7) {
        return decode_group3(bs, px, opc, out);
    }

    // 0xFE/0xFF group 4/5
    if (opc == 0xFE || opc == 0xFF) {
        return decode_group4_5(bs, px, opc, out);
    }

    // 0x0F secondary
    if (opc == 0x0F) {
        return decode_two_byte(bs, px, out, rip);
    }

    return Status::UnsupportedOpcode;
}

} // namespace emu::decoder
