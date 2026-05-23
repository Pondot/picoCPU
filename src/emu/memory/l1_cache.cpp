// L1Cache: 64-byte direct-mapped cache lines.
//
// Hot path: line lookup is one shift + mask + tag compare. On miss we pull
// a full 64-byte line from `next_` and re-serve the read out of it.
//
// Writes go *through* -- we always update next_ first, then refresh (or
// invalidate) the cached line so we don't observe a stale read after a write.

#include "emu/cache.h"

#include "emu/logger.h"

#include <cstring>

namespace emu {

namespace {
constexpr u64 LINE_MASK = ~u64{63};
}

void L1Cache::flush() noexcept {
    for (auto& l : lines_) { l.valid = false; l.tag = 0; }
}

void L1Cache::invalidate(GuestAddr addr, usize size) noexcept {
    if (size == 0) return;
    const GuestAddr start = addr & LINE_MASK;
    const GuestAddr end   = (addr + size - 1) & LINE_MASK;
    for (GuestAddr line = start; ; line += 64) {
        const u32 idx = static_cast<u32>((line >> 6) & (ENTRIES - 1));
        if (lines_[idx].valid && lines_[idx].tag == line) {
            lines_[idx].valid = false;
        }
        if (line == end) break;
    }
}

Status L1Cache::read(GuestAddr addr, usize size, void* out) noexcept {
    if (size == 0) return Status::Ok;
    if (next_ == nullptr) return Status::ProviderFailure;

    auto* dst = static_cast<u8*>(out);

    GuestAddr cur = addr;
    usize remaining = size;
    while (remaining > 0) {
        const GuestAddr line_base = cur & LINE_MASK;
        const u32  off    = static_cast<u32>(cur - line_base);
        const usize n     = (64 - off) < remaining ? (64 - off) : remaining;
        const u32  idx    = static_cast<u32>((line_base >> 6) & (ENTRIES - 1));
        Line& slot = lines_[idx];

        if (slot.valid && slot.tag == line_base) {
            ++hits_;
        } else {
            ++misses_;
            if (Status s = next_->read(line_base, 64, slot.data); fail(s)) {
                slot.valid = false;
                return s;
            }
            slot.tag   = line_base;
            slot.valid = true;
        }

        std::memcpy(dst, slot.data + off, n);
        dst       += n;
        cur       += n;
        remaining -= n;
    }
    return Status::Ok;
}

Status L1Cache::write(GuestAddr addr, usize size, const void* data) noexcept {
    if (size == 0) return Status::Ok;
    if (next_ == nullptr) return Status::ProviderFailure;

    // Write-through with in-place update: persist downward, then patch any
    // covered cached line so subsequent reads stay hits.
    if (Status s = next_->write(addr, size, data); fail(s)) return s;

    const auto* src = static_cast<const u8*>(data);
    GuestAddr cur = addr;
    usize remaining = size;
    while (remaining > 0) {
        const GuestAddr line_base = cur & LINE_MASK;
        const u32  off    = static_cast<u32>(cur - line_base);
        const usize n     = (64 - off) < remaining ? (64 - off) : remaining;
        const u32  idx    = static_cast<u32>((line_base >> 6) & (ENTRIES - 1));
        Line& slot = lines_[idx];
        if (slot.valid && slot.tag == line_base) {
            std::memcpy(slot.data + off, src, n);
        }
        src       += n;
        cur       += n;
        remaining -= n;
    }
    return Status::Ok;
}

} // namespace emu
