// IR peephole passes.
//
// Phase 6 (partial): junk-fold -- replace identity-style instructions with
// NOPs so the dispatcher's hot loop pays a near-zero cost for them. Examples:
//   mov  rax, rax              -> NOP
//   xchg rax, rax              -> NOP    (this is the canonical encoding of NOP itself)
//   lea  rax, [rax + 0]        -> NOP    (no SIB, base==dst, disp==0)
//
// Future passes will fold opaque predicates, propagate constants forward,
// and de-duplicate semantically-equivalent blocks.

#pragma once

#include "emu/ir.h"

namespace emu {

// Apply all peephole passes to a freshly-decoded block in place. Safe to call
// once at insertion time (block-cache lifetime).
void peephole_block(DecodedBlock& block) noexcept;

// Diagnostics: how many insns were folded across the lifetime of the
// process. Cheap (atomic counter); used by tests.
[[nodiscard]] u64 peephole_folded_count() noexcept;

} // namespace emu
