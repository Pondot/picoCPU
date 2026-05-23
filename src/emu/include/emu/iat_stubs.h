// IAT stub layer.
//
// When emulating a PE file, calls to imported functions normally go through
// the Import Address Table: `call [rip + offset_to_iat_entry]` reads a u64
// from the IAT and jumps to it. In our setup the imported DLLs aren't
// loaded, so the IAT entries either still point at IMAGE_IMPORT_BY_NAME
// records (unbound) or at addresses in DLLs we don't have mapped (bound).
//
// This layer intercepts the indirect call by:
//   1. Allocating a "stub region" in guest memory (private region in
//      ShadowPages).
//   2. For each imported function, patching its IAT entry to point at a
//      unique address inside the stub region.
//   3. Hooking the dispatcher so that when RIP enters the stub region, we
//      look up the (dll, name) for that address, invoke the registered
//      stub handler, and synthesize a RET (pop return address from the
//      guest stack).
//
// Stubs are simple C++ functions that read args from the CPU register file,
// do whatever Win32-equivalent computation is needed, write the result to
// RAX, and return. They never decode any guest code.

#pragma once

#include "emu/emu.h"
#include "emu/types.h"
#include "emu/shadow_pages.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace emu {

class IatStubs final {
public:
    using StubFn = void (*)(Cpu&, const char* dll, const char* name);

    // The stub region lives at a fixed high address that no normal binary
    // claims. 64 KiB -> up to 8 KiB stubs (one per import).
    static constexpr GuestAddr kStubBase = 0x0000'7FFE'F000'0000ull;
    static constexpr u64       kStubSize = 64 * 1024;
    static constexpr u64       kStubStride = 8;

    // Register the stub region as a private (zero-filled) region in
    // `shadow`. Must be called once before any imports are bound.
    void install(ShadowPages& shadow) noexcept;

    // Bind one import: write the synthetic stub address into the IAT entry
    // at `iat_addr`, and remember the (dll, name) -> stub_addr mapping so
    // dispatch can look it up. Returns the synthetic address.
    GuestAddr bind(ShadowPages& shadow, GuestAddr iat_addr,
                   std::string dll, std::string name) noexcept;

    // Register a stub handler. Lookup is case-insensitive on dll (we keep
    // names case-sensitive -- they're C symbols).
    void set_stub(const char* dll, const char* name, StubFn fn) noexcept;

    // Default catch-all: invoked when no specific stub is registered for
    // (dll, name). Defaults to "return 0 in RAX" (success-ish).
    void set_default_stub(StubFn fn) noexcept { default_stub_ = fn; }

    // Called by the dispatcher when RIP enters the stub region. Looks up
    // the (dll, name), invokes the stub (or default), then synthesizes a
    // RET by popping the return address from RSP and setting RIP. Returns
    // false on stack/memory failure (cpu is faulted).
    bool dispatch(Cpu& cpu, GuestAddr rip) noexcept;

    [[nodiscard]] bool in_region(GuestAddr rip) const noexcept {
        return rip >= kStubBase && rip < kStubBase + kStubSize;
    }

    [[nodiscard]] std::size_t count() const noexcept { return entries_.size(); }

private:
    struct Entry { std::string dll; std::string name; };
    std::vector<Entry> entries_;     // index = (stub_addr - kStubBase) / kStubStride

    struct StubKey { std::string dll; std::string name; };
    struct StubKeyHash { std::size_t operator()(const std::pair<std::string,std::string>& p) const noexcept; };
    std::unordered_map<std::string, StubFn> stubs_;     // key = "dll|name"
    StubFn default_stub_ = nullptr;
};

} // namespace emu
