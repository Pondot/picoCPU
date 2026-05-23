// Block cache + decode_block.

#include "emu/block_cache.h"

#include "emu/decoder.h"
#include "emu/logger.h"
#include "emu/peephole.h"

namespace emu {

DecodedBlock* BlockCache::lookup(GuestAddr pc) noexcept {
    auto it = blocks_.find(pc);
    if (it == blocks_.end()) { ++misses_; return nullptr; }
    ++hits_;
    return it->second.get();
}

DecodedBlock* BlockCache::insert(GuestAddr pc, std::unique_ptr<DecodedBlock> block) noexcept {
    auto [it, _] = blocks_.insert_or_assign(pc, std::move(block));
    return it->second.get();
}

void BlockCache::invalidate_range(GuestAddr addr, usize size) noexcept {
    if (size == 0) return;
    const GuestAddr end = addr + size;
    for (auto it = blocks_.begin(); it != blocks_.end();) {
        const GuestAddr blk_pc  = it->first;
        const GuestAddr blk_end = blk_pc + it->second->byte_size;
        if (blk_pc < end && blk_end > addr) {
            it = blocks_.erase(it);
        } else {
            ++it;
        }
    }
}

u64 semantic_block_hash(const DecodedBlock& block) noexcept {
    // FNV-1a 64-bit over a per-insn signature that ignores reg numbers,
    // immediate values, and `imm_extra`. Captures op kind + op size + the
    // *shape* of each operand.
    u64 h = 0xcbf29ce484222325ull;
    auto mix = [&h](u64 v) {
        h ^= v;
        h *= 0x100000001b3ull;
    };
    for (const Insn& i : block.insns) {
        mix(static_cast<u64>(i.kind));
        mix(static_cast<u64>(i.op_size));
        mix(static_cast<u64>(i.dst.kind));
        mix(static_cast<u64>(i.src.kind));
        // Memory operand shape: which fields are populated (not their values).
        if (i.dst.kind == OperandKind::Mem || i.dst.kind == OperandKind::RipRelMem) {
            mix(static_cast<u64>(i.dst.scale) | (i.dst.imm != 0 ? (1ull << 32) : 0));
        }
        if (i.src.kind == OperandKind::Mem || i.src.kind == OperandKind::RipRelMem) {
            mix(static_cast<u64>(i.src.scale) | (i.src.imm != 0 ? (1ull << 32) : 0));
        }
    }
    return h;
}

Status decode_block(MemoryProvider& mem, GuestAddr pc, DecodedBlock& out, u32 max_insns) noexcept {
    out.pc        = pc;
    out.byte_size = 0;
    out.insns.clear();

    GuestAddr cur = pc;
    for (u32 i = 0; i < max_insns; ++i) {
        Insn insn{};
        Status s = decode_one(mem, cur, insn);
        if (fail(s)) {
            if (out.insns.empty()) return s;
            // Already have a partial block -- treat as a soft end.
            LOG_DEBUG("decode_block: trailing decode at 0x%llx failed (%.*s); block ends with %u insns",
                      (unsigned long long)cur,
                      static_cast<int>(to_string(s).size()),
                      to_string(s).data(),
                      static_cast<unsigned>(out.insns.size()));
            break;
        }
        out.insns.push_back(insn);
        out.byte_size += insn.len;
        cur           += insn.len;
        if (insn.flags & INSN_FLAG_CONTROL_FLOW) break;
    }
    peephole_block(out);
    return Status::Ok;
}

} // namespace emu
