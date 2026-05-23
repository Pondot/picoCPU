// PE loader -- open a .exe/.dll from disk, map it into a flat image buffer,
// parse the export table, and expose a MemoryProvider that serves the
// mapped image. Lets the emulator run static binaries without needing a
// live process.
//
// Scope: x86-64 PE32+ images. No relocations (we keep the preferred base),
// no import resolution (functions that call out to ntdll/kernel32 will hit
// unmapped reads -- those need a hook layer to handle). Read-only mapping is
// fine for emulation since ShadowPages will sit on top and absorb writes.

#pragma once

#include "emu/error.h"
#include "emu/memory.h"
#include "emu/types.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace emu::host {

struct PeImport {
    std::string dll;        // lowercased so lookups are case-insensitive on Windows
    std::string name;
    GuestAddr   iat_addr;   // absolute address of this IAT slot (= base + FirstThunk_RVA + i*8)
};

struct PeLoadedImage {
    GuestAddr        base       = 0;     // image base address (ImageBase from header)
    u64              size       = 0;     // SizeOfImage from header
    u32              entry_rva  = 0;     // AddressOfEntryPoint (RVA)
    std::vector<u8>  mapped;             // image as it would appear in memory (size == `size`)
    std::unordered_map<std::string, u32> exports;   // name -> RVA
    std::vector<PeImport> imports;       // every named/ordinal import in the IAT
};

// Load a PE file from disk. On success, `out.mapped` holds a buffer of size
// `out.size` containing each section copied to its VirtualAddress (relative
// to start of buffer). The header bytes occupy the first SizeOfHeaders.
//
// `forced_base` of 0 = use the PE's ImageBase. Non-zero overrides it (no
// relocations applied, so absolute pointers in the image will still refer
// to the original base -- fine for many leaf functions, broken for any code
// that takes the address of a global).
Status pe_load_file(const char* path, GuestAddr forced_base, PeLoadedImage& out) noexcept;

// Look up an exported function by name. Returns the absolute address
// (base + RVA), or 0 if not found.
GuestAddr pe_export_addr(const PeLoadedImage& img, const char* name) noexcept;

// A MemoryProvider that serves bytes from a PeLoadedImage at its base.
// Reads outside [base, base+size) return UnmappedRead. Writes are rejected
// (callers should layer ShadowPages on top to absorb writes).
class PeImageProvider final : public MemoryProvider {
public:
    explicit PeImageProvider(const PeLoadedImage& img) noexcept;

    [[nodiscard]] Status read (GuestAddr addr, usize size, void* out)        noexcept override;
    [[nodiscard]] Status write(GuestAddr addr, usize size, const void* data) noexcept override;
    [[nodiscard]] bool   is_mapped(GuestAddr addr) const noexcept override;

private:
    GuestAddr     base_;
    u64           size_;
    const u8*     data_;
};

} // namespace emu::host
