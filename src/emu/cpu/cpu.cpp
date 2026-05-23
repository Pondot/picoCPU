// CPU state + reset + condition-code evaluation.
//
// Flag computation helpers live in flags.cpp.

#include "emu/cpu.h"

#include <cstring>

namespace emu {

void Cpu::snapshot(Snapshot& out) const noexcept {
    sync_flags_();
    std::memcpy(out.gpr,    gpr_,   sizeof(gpr_));
    std::memcpy(out.dr,     dr_,    sizeof(dr_));
    std::memcpy(out.zmm,    zmm_,   sizeof(zmm_));
    std::memcpy(out.k,      k_,     sizeof(k_));
    std::memcpy(out.fpu_st, fpu_st_, sizeof(fpu_st_));
    out.fpu_cw   = fpu_cw_;
    out.fpu_sw   = fpu_sw_;
    out.fpu_top  = fpu_top_;
    out.rip      = rip_;
    out.rflags   = rflags_;
    out.fs_base  = fs_base_;
    out.gs_base  = gs_base_;
    out.cc_op    = cc_op_;
    out.cc_size  = cc_size_;
    out.cc_dst   = cc_dst_;
    out.cc_src   = cc_src_;
    out.cc_src2  = cc_src2_;
    out.fault    = fault_;
    out.halted   = halted_;
    out.branch_taken = branch_taken_;
}

void Cpu::restore(const Snapshot& s) noexcept {
    std::memcpy(gpr_,   s.gpr,    sizeof(gpr_));
    std::memcpy(dr_,    s.dr,     sizeof(dr_));
    std::memcpy(zmm_,   s.zmm,    sizeof(zmm_));
    std::memcpy(k_,     s.k,      sizeof(k_));
    std::memcpy(fpu_st_,s.fpu_st, sizeof(fpu_st_));
    fpu_cw_   = s.fpu_cw;
    fpu_sw_   = s.fpu_sw;
    fpu_top_  = s.fpu_top;
    rip_      = s.rip;
    rflags_   = s.rflags;
    fs_base_  = s.fs_base;
    gs_base_  = s.gs_base;
    cc_op_    = s.cc_op;
    cc_size_  = s.cc_size;
    cc_dst_   = s.cc_dst;
    cc_src_   = s.cc_src;
    cc_src2_  = s.cc_src2;
    fault_    = s.fault;
    halted_   = s.halted;
    branch_taken_ = s.branch_taken;
}

void Cpu::reset() noexcept {
    for (auto& r : gpr_) r = 0;
    rip_          = 0;
    rflags_       = rflags::ALWAYS_ONE;
    fs_base_      = 0;
    gs_base_      = 0;
    halted_       = false;
    branch_taken_ = false;
}

bool Cpu::cond_holds(Cc cc) const noexcept {
    const bool c = cf();
    const bool z = zf();
    const bool s = sf();
    const bool o = of();
    const bool p = pf();
    switch (cc) {
        case Cc::O:   return  o;
        case Cc::NO:  return !o;
        case Cc::B:   return  c;
        case Cc::NB:  return !c;
        case Cc::Z:   return  z;
        case Cc::NZ:  return !z;
        case Cc::BE:  return  c || z;
        case Cc::NBE: return !c && !z;
        case Cc::S:   return  s;
        case Cc::NS:  return !s;
        case Cc::P:   return  p;
        case Cc::NP:  return !p;
        case Cc::L:   return  s != o;
        case Cc::NL:  return  s == o;
        case Cc::LE:  return  z || (s != o);
        case Cc::NLE: return !z && (s == o);
        case Cc::None: return true;
    }
    return false;
}

} // namespace emu
