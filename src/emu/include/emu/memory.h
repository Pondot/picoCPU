// Emulator: memory provider interface + built-in backends.
//
// `MemoryProvider` is the boundary between the emulator core and whatever
// supplies guest bytes. Two backends in this phase:
//   - FakeMemoryProvider:    vector-backed, for unit tests.
//   - WinProcessMemoryProvider (defined in src/host/): wraps OpenProcess +
//     ReadProcessMemory.
//
// Future phases stack L1/L2/L3 caches *in front* of a provider; the cached
// layer also implements MemoryProvider and wraps the underlying one.

#pragma once

#include "emu/error.h"
#include "emu/types.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace emu {

class MemoryProvider {
public:
    virtual ~MemoryProvider() = default;

    // Read `size` bytes at `addr` into `out`. Returns Status::Ok on full read,
    // Status::UnmappedRead / ProviderFailure otherwise. `out` is unmodified on failure.
    [[nodiscard]] virtual Status read(GuestAddr addr, usize size, void* out) noexcept = 0;

    // Write `size` bytes from `data` to `addr`. Default = ProtectionViolation
    // (read-only providers). ShadowPages overrides to actually persist; the
    // cache layers forward writes to their next layer and invalidate their
    // own entries.
    [[nodiscard]] virtual Status write(GuestAddr addr, usize size, const void* data) noexcept {
        (void)addr; (void)size; (void)data;
        return Status::ProtectionViolation;
    }

    // Drop any cached state covering [addr, addr+size). Default no-op.
    virtual void invalidate(GuestAddr addr, usize size) noexcept {
        (void)addr; (void)size;
    }

    // Best-effort: does this provider claim this address is mapped?
    [[nodiscard]] virtual bool is_mapped(GuestAddr addr) const noexcept { (void)addr; return false; }
};

// In-memory provider for tests. Maps fixed-base regions with raw bytes.
class FakeMemoryProvider final : public MemoryProvider {
public:
    // Register a region [base, base+bytes.size()). Overlap is rejected.
    [[nodiscard]] Status add_region(GuestAddr base, std::vector<u8> bytes) noexcept;

    [[nodiscard]] Status read(GuestAddr addr, usize size, void* out) noexcept override;
    [[nodiscard]] bool is_mapped(GuestAddr addr) const noexcept override;

private:
    struct Region {
        GuestAddr        base;
        std::vector<u8>  bytes;
    };
    // Sorted by base for simple lookup. Fine for tests; do not use in hot paths.
    std::vector<Region> regions_;
};

} // namespace emu
