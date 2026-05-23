// Handler declarations. Each handler is bound by the decoder into Insn::handler.
//
// Phase 1: one handler per OpKind with internal branching on operand kind.
// Phase 4 splits hot ops into specialized per-operand variants for the
// dispatch fast path.

#pragma once

#include "emu/ir.h"

namespace emu {

class Cpu;

namespace handlers {

void op_invalid(Cpu&, const Insn&);
void op_nop    (Cpu&, const Insn&);

// Data movement
void op_mov    (Cpu&, const Insn&);
void op_movzx  (Cpu&, const Insn&);
void op_movsx  (Cpu&, const Insn&);
void op_movsxd (Cpu&, const Insn&);
void op_lea    (Cpu&, const Insn&);
void op_push   (Cpu&, const Insn&);
void op_pop    (Cpu&, const Insn&);
void op_xchg   (Cpu&, const Insn&);
void op_cmov   (Cpu&, const Insn&);
void op_bswap  (Cpu&, const Insn&);
void op_cwde   (Cpu&, const Insn&);
void op_cdq    (Cpu&, const Insn&);
void op_movd_to_xmm   (Cpu&, const Insn&);  // 66 0F 6E
void op_movd_from_xmm (Cpu&, const Insn&);  // 66 0F 7E (and F3 0F 7E for xmm<-xmm/m64)

// Arithmetic
void op_add    (Cpu&, const Insn&);
void op_adc    (Cpu&, const Insn&);
void op_sub    (Cpu&, const Insn&);
void op_sbb    (Cpu&, const Insn&);
void op_inc    (Cpu&, const Insn&);
void op_dec    (Cpu&, const Insn&);
void op_neg    (Cpu&, const Insn&);
void op_imul   (Cpu&, const Insn&);
void op_imul1  (Cpu&, const Insn&);   // 1-operand IMUL (full-width signed product)
void op_mul    (Cpu&, const Insn&);   // 1-operand MUL (unsigned)
void op_div    (Cpu&, const Insn&);
void op_idiv   (Cpu&, const Insn&);

// Logical
void op_and    (Cpu&, const Insn&);
void op_or     (Cpu&, const Insn&);
void op_xor    (Cpu&, const Insn&);
void op_not    (Cpu&, const Insn&);
void op_test   (Cpu&, const Insn&);
void op_cmp    (Cpu&, const Insn&);

// Shifts / rotates
void op_shl    (Cpu&, const Insn&);
void op_shr    (Cpu&, const Insn&);
void op_sar    (Cpu&, const Insn&);
void op_rol    (Cpu&, const Insn&);
void op_ror    (Cpu&, const Insn&);
void op_rcl    (Cpu&, const Insn&);
void op_rcr    (Cpu&, const Insn&);
void op_shld   (Cpu&, const Insn&);
void op_shrd   (Cpu&, const Insn&);

// String ops
void op_movs   (Cpu&, const Insn&);
void op_stos   (Cpu&, const Insn&);
void op_lods   (Cpu&, const Insn&);
void op_scas   (Cpu&, const Insn&);
void op_cmps   (Cpu&, const Insn&);

// Bit manipulation
void op_bsf    (Cpu&, const Insn&);
void op_bsr    (Cpu&, const Insn&);
void op_bt     (Cpu&, const Insn&);
void op_bts    (Cpu&, const Insn&);
void op_btr    (Cpu&, const Insn&);
void op_btc    (Cpu&, const Insn&);
void op_popcnt (Cpu&, const Insn&);
void op_lzcnt  (Cpu&, const Insn&);
void op_tzcnt  (Cpu&, const Insn&);

// Control flow
void op_jmp    (Cpu&, const Insn&);
void op_jcc    (Cpu&, const Insn&);
void op_call   (Cpu&, const Insn&);
void op_ret    (Cpu&, const Insn&);
void op_loop   (Cpu&, const Insn&);   // cc selects LOOP / LOOPE / LOOPNE via insn.cond
void op_jcxz   (Cpu&, const Insn&);

// System
void op_cpuid  (Cpu&, const Insn&);
void op_rdtsc  (Cpu&, const Insn&);
void op_rdrand (Cpu&, const Insn&);
void op_hlt    (Cpu&, const Insn&);
void op_syscall(Cpu&, const Insn&);

// SSE / SSE2
void op_movdqa (Cpu&, const Insn&);
void op_movdqu (Cpu&, const Insn&);
void op_pxor   (Cpu&, const Insn&);
void op_pand   (Cpu&, const Insn&);
void op_por    (Cpu&, const Insn&);
void op_paddb  (Cpu&, const Insn&);
void op_paddw  (Cpu&, const Insn&);
void op_paddd  (Cpu&, const Insn&);
void op_paddq  (Cpu&, const Insn&);
void op_psubb  (Cpu&, const Insn&);
void op_psubw  (Cpu&, const Insn&);
void op_psubd  (Cpu&, const Insn&);
void op_psubq  (Cpu&, const Insn&);
void op_pshufd (Cpu&, const Insn&);
void op_pcmpeqb(Cpu&, const Insn&);
void op_pcmpeqw(Cpu&, const Insn&);
void op_pcmpeqd(Cpu&, const Insn&);

// SSE FP -- scalar
void op_movss   (Cpu&, const Insn&);
void op_movsd_fp(Cpu&, const Insn&);   // FP MOVSD (F2 0F 10/11) -- *not* the string-op MOVSD
void op_addss   (Cpu&, const Insn&);
void op_subss   (Cpu&, const Insn&);
void op_mulss   (Cpu&, const Insn&);
void op_divss   (Cpu&, const Insn&);
void op_sqrtss  (Cpu&, const Insn&);
void op_minss   (Cpu&, const Insn&);
void op_maxss   (Cpu&, const Insn&);
void op_addsd   (Cpu&, const Insn&);
void op_subsd   (Cpu&, const Insn&);
void op_mulsd   (Cpu&, const Insn&);
void op_divsd   (Cpu&, const Insn&);
void op_sqrtsd  (Cpu&, const Insn&);
void op_minsd   (Cpu&, const Insn&);
void op_maxsd   (Cpu&, const Insn&);
void op_ucomiss (Cpu&, const Insn&);
void op_ucomisd (Cpu&, const Insn&);
void op_cvtsi2ss(Cpu&, const Insn&);
void op_cvtsi2sd(Cpu&, const Insn&);
void op_cvttss2si(Cpu&, const Insn&);
void op_cvttsd2si(Cpu&, const Insn&);
void op_cvtss2sd(Cpu&, const Insn&);
void op_cvtsd2ss(Cpu&, const Insn&);

// SSE FP -- packed
void op_movups  (Cpu&, const Insn&);
void op_movupd  (Cpu&, const Insn&);
void op_addps   (Cpu&, const Insn&);
void op_subps   (Cpu&, const Insn&);
void op_mulps   (Cpu&, const Insn&);
void op_divps   (Cpu&, const Insn&);
void op_sqrtps  (Cpu&, const Insn&);
void op_addpd   (Cpu&, const Insn&);
void op_subpd   (Cpu&, const Insn&);
void op_mulpd   (Cpu&, const Insn&);
void op_divpd   (Cpu&, const Insn&);
void op_sqrtpd  (Cpu&, const Insn&);
void op_xorps   (Cpu&, const Insn&);
void op_xorpd   (Cpu&, const Insn&);
void op_andps   (Cpu&, const Insn&);
void op_orps    (Cpu&, const Insn&);

// x87 FPU
void op_fld     (Cpu&, const Insn&);
void op_fst     (Cpu&, const Insn&);
void op_fstp    (Cpu&, const Insn&);
void op_fild    (Cpu&, const Insn&);
void op_fistp   (Cpu&, const Insn&);
void op_fadd    (Cpu&, const Insn&);
void op_fsub    (Cpu&, const Insn&);
void op_fmul    (Cpu&, const Insn&);
void op_fdiv    (Cpu&, const Insn&);
void op_fchs    (Cpu&, const Insn&);
void op_fabs    (Cpu&, const Insn&);
void op_fsqrt   (Cpu&, const Insn&);
void op_faddp   (Cpu&, const Insn&);
void op_fsubp   (Cpu&, const Insn&);
void op_fmulp   (Cpu&, const Insn&);
void op_fdivp   (Cpu&, const Insn&);
void op_fxch    (Cpu&, const Insn&);
void op_fcom    (Cpu&, const Insn&);
void op_fcomp   (Cpu&, const Insn&);
void op_fcompp  (Cpu&, const Insn&);
void op_fninit  (Cpu&, const Insn&);
void op_fnstsw  (Cpu&, const Insn&);
void op_fnstcw  (Cpu&, const Insn&);
void op_fldcw   (Cpu&, const Insn&);

// AVX (VEX-encoded). `insn.dst.index` carries the vvvv source register id.
void op_vmovdqa  (Cpu&, const Insn&);
void op_vmovdqu  (Cpu&, const Insn&);
void op_vpxor    (Cpu&, const Insn&);
void op_vpand    (Cpu&, const Insn&);
void op_vpor     (Cpu&, const Insn&);
void op_vpaddb   (Cpu&, const Insn&);
void op_vpaddw   (Cpu&, const Insn&);
void op_vpaddd   (Cpu&, const Insn&);
void op_vpaddq   (Cpu&, const Insn&);
void op_vpsubb   (Cpu&, const Insn&);
void op_vpsubw   (Cpu&, const Insn&);
void op_vpsubd   (Cpu&, const Insn&);
void op_vpsubq   (Cpu&, const Insn&);
void op_vzeroupper(Cpu&, const Insn&);
void op_vzeroall (Cpu&, const Insn&);

// Misc legacy
void op_lahf     (Cpu&, const Insn&);
void op_sahf     (Cpu&, const Insn&);
void op_pushf    (Cpu&, const Insn&);
void op_popf     (Cpu&, const Insn&);
void op_int3     (Cpu&, const Insn&);
void op_int_n    (Cpu&, const Insn&);

// Memory-ordering / cache control (effectively no-op on the emulator)
void op_fence    (Cpu&, const Insn&);
void op_pause    (Cpu&, const Insn&);
void op_clflush  (Cpu&, const Insn&);
void op_prefetch (Cpu&, const Insn&);

// MOVBE -- byte-reverse on load/store
void op_movbe    (Cpu&, const Insn&);

// Extra SSE
void op_rcpss    (Cpu&, const Insn&);
void op_rsqrtss  (Cpu&, const Insn&);
void op_rcpps    (Cpu&, const Insn&);
void op_rsqrtps  (Cpu&, const Insn&);
void op_pmovmskb (Cpu&, const Insn&);
void op_movmskps (Cpu&, const Insn&);
void op_movmskpd (Cpu&, const Insn&);
void op_shufps   (Cpu&, const Insn&);
void op_shufpd   (Cpu&, const Insn&);
void op_pshufb   (Cpu&, const Insn&);
void op_pshufhw  (Cpu&, const Insn&);
void op_pshuflw  (Cpu&, const Insn&);
void op_pmullw   (Cpu&, const Insn&);
void op_pmulhw   (Cpu&, const Insn&);
void op_pmulhuw  (Cpu&, const Insn&);
void op_pmaddwd  (Cpu&, const Insn&);
void op_psllw    (Cpu&, const Insn&);
void op_pslld    (Cpu&, const Insn&);
void op_psllq    (Cpu&, const Insn&);
void op_psrlw    (Cpu&, const Insn&);
void op_psrld    (Cpu&, const Insn&);
void op_psrlq    (Cpu&, const Insn&);
void op_psraw    (Cpu&, const Insn&);
void op_psrad    (Cpu&, const Insn&);
void op_punpcklbw  (Cpu&, const Insn&);
void op_punpcklwd  (Cpu&, const Insn&);
void op_punpckldq  (Cpu&, const Insn&);
void op_punpcklqdq (Cpu&, const Insn&);
void op_punpckhbw  (Cpu&, const Insn&);
void op_punpckhwd  (Cpu&, const Insn&);
void op_punpckhdq  (Cpu&, const Insn&);
void op_punpckhqdq (Cpu&, const Insn&);

// SSSE3 / SSE4.x / AES-NI / CRC32 / CLMUL
void op_aesenc       (Cpu&, const Insn&);
void op_aesenclast   (Cpu&, const Insn&);
void op_aesdec       (Cpu&, const Insn&);
void op_aesdeclast   (Cpu&, const Insn&);
void op_aesimc       (Cpu&, const Insn&);
void op_aeskeygen    (Cpu&, const Insn&);
void op_pclmulqdq    (Cpu&, const Insn&);
void op_crc32        (Cpu&, const Insn&);
void op_roundss      (Cpu&, const Insn&);
void op_roundsd      (Cpu&, const Insn&);
void op_roundps      (Cpu&, const Insn&);
void op_roundpd      (Cpu&, const Insn&);
void op_insertps     (Cpu&, const Insn&);
void op_pextrb       (Cpu&, const Insn&);
void op_pextrw       (Cpu&, const Insn&);
void op_pextrd       (Cpu&, const Insn&);
void op_pextrq       (Cpu&, const Insn&);
void op_pinsrb       (Cpu&, const Insn&);
void op_pinsrw       (Cpu&, const Insn&);
void op_pinsrd       (Cpu&, const Insn&);
void op_pinsrq       (Cpu&, const Insn&);
void op_pcmpgtb      (Cpu&, const Insn&);
void op_pcmpgtw      (Cpu&, const Insn&);
void op_pcmpgtd      (Cpu&, const Insn&);
void op_pcmpgtq      (Cpu&, const Insn&);
void op_pmaxub       (Cpu&, const Insn&);
void op_pmaxsw       (Cpu&, const Insn&);
void op_pminub       (Cpu&, const Insn&);
void op_pminsw       (Cpu&, const Insn&);
void op_pabsb        (Cpu&, const Insn&);
void op_pabsw        (Cpu&, const Insn&);
void op_pabsd        (Cpu&, const Insn&);
void op_psadbw       (Cpu&, const Insn&);

// More x87 transcendentals
void op_fsin         (Cpu&, const Insn&);
void op_fcos         (Cpu&, const Insn&);
void op_fpatan       (Cpu&, const Insn&);
void op_fyl2x        (Cpu&, const Insn&);
void op_f2xm1        (Cpu&, const Insn&);
void op_fscale       (Cpu&, const Insn&);
void op_frndint      (Cpu&, const Insn&);

// BMI1 / BMI2
void op_andn         (Cpu&, const Insn&);   // dst = ~vvvv & r/m
void op_bextr        (Cpu&, const Insn&);   // dst = (r/m >> (vvvv&0xFF)) & ((1<<(vvvv>>8 &0xFF))-1)
void op_blsi         (Cpu&, const Insn&);   // dst = r/m & -r/m
void op_blsr         (Cpu&, const Insn&);   // dst = r/m & (r/m - 1)
void op_blsmsk       (Cpu&, const Insn&);   // dst = r/m ^ (r/m - 1)
void op_bzhi         (Cpu&, const Insn&);   // dst = r/m & ((1 << (vvvv & 0xFF)) - 1)
void op_mulx         (Cpu&, const Insn&);
void op_pdep         (Cpu&, const Insn&);
void op_pext         (Cpu&, const Insn&);
void op_rorx         (Cpu&, const Insn&);
void op_sarx         (Cpu&, const Insn&);
void op_shlx         (Cpu&, const Insn&);
void op_shrx         (Cpu&, const Insn&);

// VEX-encoded 3-operand SSE
void op_vpshufb      (Cpu&, const Insn&);
void op_vaesenc      (Cpu&, const Insn&);
void op_vaesenclast  (Cpu&, const Insn&);
void op_vaesdec      (Cpu&, const Insn&);
void op_vaesdeclast  (Cpu&, const Insn&);
void op_vpcmpgtq     (Cpu&, const Insn&);
void op_vpclmulqdq   (Cpu&, const Insn&);

// SHA-NI
void op_sha1rnds4    (Cpu&, const Insn&);
void op_sha1nexte    (Cpu&, const Insn&);
void op_sha1msg1     (Cpu&, const Insn&);
void op_sha1msg2     (Cpu&, const Insn&);
void op_sha256rnds2  (Cpu&, const Insn&);
void op_sha256msg1   (Cpu&, const Insn&);
void op_sha256msg2   (Cpu&, const Insn&);

// SSE4.2 string compare
void op_pcmpestri    (Cpu&, const Insn&);
void op_pcmpestrm    (Cpu&, const Insn&);
void op_pcmpistri    (Cpu&, const Insn&);
void op_pcmpistrm    (Cpu&, const Insn&);

// AVX-512 (EVEX). All honor masking via insn.pad: bits 0-2 = opmask reg,
// bit 3 = zeroing(z=1)-vs-merging(z=0).
void op_vpxord       (Cpu&, const Insn&);   // 32-bit lane XOR
void op_vpxorq       (Cpu&, const Insn&);   // 64-bit lane XOR
void op_vpaddb_evex  (Cpu&, const Insn&);
void op_vpaddw_evex  (Cpu&, const Insn&);
void op_vpaddd_evex  (Cpu&, const Insn&);
void op_vpaddq_evex  (Cpu&, const Insn&);
void op_vpsubb_evex  (Cpu&, const Insn&);
void op_vpsubw_evex  (Cpu&, const Insn&);
void op_vpsubd_evex  (Cpu&, const Insn&);
void op_vpsubq_evex  (Cpu&, const Insn&);
void op_vpandd       (Cpu&, const Insn&);
void op_vpandq       (Cpu&, const Insn&);
void op_vpord        (Cpu&, const Insn&);
void op_vporq        (Cpu&, const Insn&);
void op_vmovdqu32    (Cpu&, const Insn&);
void op_vmovdqu64    (Cpu&, const Insn&);
void op_vmovdqa32    (Cpu&, const Insn&);
void op_vmovdqa64    (Cpu&, const Insn&);

// AVX-512 floating-point (PS=4-byte lane, PD=8-byte lane); masking + broadcast.
void op_vaddps_evex  (Cpu&, const Insn&);
void op_vaddpd_evex  (Cpu&, const Insn&);
void op_vsubps_evex  (Cpu&, const Insn&);
void op_vsubpd_evex  (Cpu&, const Insn&);
void op_vmulps_evex  (Cpu&, const Insn&);
void op_vmulpd_evex  (Cpu&, const Insn&);
void op_vdivps_evex  (Cpu&, const Insn&);
void op_vdivpd_evex  (Cpu&, const Insn&);
void op_vsqrtps_evex (Cpu&, const Insn&);
void op_vsqrtpd_evex (Cpu&, const Insn&);
void op_vminps_evex  (Cpu&, const Insn&);
void op_vminpd_evex  (Cpu&, const Insn&);
void op_vmaxps_evex  (Cpu&, const Insn&);
void op_vmaxpd_evex  (Cpu&, const Insn&);

// AVX2 GATHER (VEX C4, three-byte 0F 38 map).
void op_vpgatherdd   (Cpu&, const Insn&);
void op_vpgatherdq   (Cpu&, const Insn&);
void op_vpgatherqd   (Cpu&, const Insn&);
void op_vpgatherqq   (Cpu&, const Insn&);
void op_vgatherdps   (Cpu&, const Insn&);
void op_vgatherdpd   (Cpu&, const Insn&);
void op_vgatherqps   (Cpu&, const Insn&);
void op_vgatherqpd   (Cpu&, const Insn&);

// AVX-512 SCATTER (EVEX) -- store lanes to indexed mem; opmask gates lanes.
void op_vpscatterdd  (Cpu&, const Insn&);
void op_vpscatterdq  (Cpu&, const Insn&);
void op_vpscatterqd  (Cpu&, const Insn&);
void op_vpscatterqq  (Cpu&, const Insn&);
void op_vscatterdps  (Cpu&, const Insn&);
void op_vscatterdpd  (Cpu&, const Insn&);
void op_vscatterqps  (Cpu&, const Insn&);
void op_vscatterqpd  (Cpu&, const Insn&);

} // namespace handlers
} // namespace emu
