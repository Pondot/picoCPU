// Block cache: PC -> DecodedBlock.
//
// A block is decoded once and replayed many times. With this in place the
// dispatcher's hot loop reduces to:
//   for (const Insn& i : block->insns) i.handler(cpu, i);
//
// Phase 4a scope: simple unordered_map keyed by start PC. SMC invalidation
// is exposed via `invalidate_range` but not auto-wired into the write path
// (Phase 6 will plumb that).

#pragma once

#include "emu/error.h"
#include "emu/ir.h"
#include "emu/memory.h"
#include "emu/types.h"

#include <memory>
#include <unordered_map>

namespace emu {

class BlockCache {
public:
    [[nodiscard]] DecodedBlock* lookup(GuestAddr pc) noexcept;
    DecodedBlock*               insert(GuestAddr pc, std::unique_ptr<DecodedBlock> block) noexcept;
    void                        invalidate_range(GuestAddr addr, usize size) noexcept;
    void                        clear() noexcept { blocks_.clear(); hits_ = misses_ = 0; semantic_hits_ = 0; }

    [[nodiscard]] usize size()        const noexcept { return blocks_.size(); }
    [[nodiscard]] u64   hits()        const noexcept { return hits_; }
    [[nodiscard]] u64   misses()      const noexcept { return misses_; }
    [[nodiscard]] u64   semantic_hits() const noexcept { return semantic_hits_; }
    void                reset_stats() noexcept { hits_ = misses_ = semantic_hits_ = 0; }
    void                count_semantic_hit() noexcept { ++semantic_hits_; }

private:
    std::unordered_map<GuestAddr, std::unique_ptr<DecodedBlock>> blocks_;
    u64 hits_          = 0;
    u64 misses_        = 0;
    u64 semantic_hits_ = 0;
};

// Compute a 64-bit hash over a decoded block that ignores operand register
// IDs and immediate values -- only `OpKind`, `op_size`, and `OperandKind` are
// hashed. Polymorphic variants (different reg allocation, different junk
// immediates) collide on this hash; tests use that to assert structural
// equivalence.
[[nodiscard]] u64 semantic_block_hash(const DecodedBlock& block) noexcept;

// Decode a basic block starting at `pc`. Stops *after* the first instruction
// with INSN_FLAG_CONTROL_FLOW (JMP/Jcc/CALL/RET/LOOP), or after `max_insns`.
// `out.pc`, `out.byte_size`, `out.insns` are populated on success.
[[nodiscard]] Status decode_block(MemoryProvider& mem, GuestAddr pc,
                                  DecodedBlock& out, u32 max_insns = 1024) noexcept;

} // namespace emu
