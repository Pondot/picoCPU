// PE loader implementation.

#include "pe_loader.h"

#include "emu/logger.h"

#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace emu::host {

namespace {

// Tiny in-buffer reader.
struct Reader {
    const u8* data;
    usize     size;
    bool read(usize off, void* out, usize n) const noexcept {
        if (off > size || size - off < n) return false;
        std::memcpy(out, data + off, n);
        return true;
    }
};

// Read an exported function's RVA by 0-based EAT index (NameOrdinals[i]).
// Honors "forwarder" entries (RVA inside the export directory) by returning 0.
u32 read_export_rva(const Reader& r, u32 export_dir_rva, u32 export_dir_size,
                    u32 eat_index) noexcept {
    u32 num_funcs = 0;
    u32 funcs_rva = 0;
    if (!r.read(export_dir_rva + 0x14, &num_funcs, 4)) return 0;
    if (!r.read(export_dir_rva + 0x1C, &funcs_rva, 4)) return 0;
    if (eat_index >= num_funcs) return 0;
    u32 rva = 0;
    if (!r.read(funcs_rva + 4 * eat_index, &rva, 4)) return 0;
    if (rva >= export_dir_rva && rva < export_dir_rva + export_dir_size) return 0;
    return rva;
}

} // namespace

// ---------------------------------------------------------------------------
// PeImageProvider

PeImageProvider::PeImageProvider(const PeLoadedImage& img) noexcept
    : base_(img.base), size_(img.size), data_(img.mapped.data()) {}

Status PeImageProvider::read(GuestAddr addr, usize size, void* out) noexcept {
    if (addr < base_ || addr + size > base_ + size_) {
        return Status::UnmappedRead;
    }
    std::memcpy(out, data_ + (addr - base_), size);
    return Status::Ok;
}

Status PeImageProvider::write(GuestAddr, usize, const void*) noexcept {
    return Status::ProtectionViolation;
}

bool PeImageProvider::is_mapped(GuestAddr addr) const noexcept {
    return addr >= base_ && addr < base_ + size_;
}

// ---------------------------------------------------------------------------
// Loader

Status pe_load_file(const char* path, GuestAddr forced_base, PeLoadedImage& out) noexcept {
    // 1) Slurp the file into memory.
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        LOG_ERROR("pe_load_file: cannot open '%s'", path);
        return Status::ProviderFailure;
    }
    const std::streamsize sz = f.tellg();
    if (sz < 0x100) return Status::Internal;
    std::vector<u8> raw(static_cast<usize>(sz));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(raw.data()), sz);
    if (!f) return Status::ProviderFailure;

    Reader r{raw.data(), raw.size()};

    // 2) DOS / NT headers.
    u8 mz[2] = {};
    if (!r.read(0, mz, 2) || mz[0] != 'M' || mz[1] != 'Z') return Status::Internal;
    u32 e_lfanew = 0;
    if (!r.read(0x3C, &e_lfanew, 4)) return Status::Internal;
    u8 sig[4] = {};
    if (!r.read(e_lfanew, sig, 4) || sig[0] != 'P' || sig[1] != 'E') return Status::Internal;

    const usize nt = e_lfanew;
    u16 machine = 0;
    u16 num_sections = 0;
    u16 opt_size = 0;
    if (!r.read(nt + 4, &machine,     2)) return Status::Internal;
    if (!r.read(nt + 6, &num_sections, 2)) return Status::Internal;
    if (!r.read(nt + 20, &opt_size,   2)) return Status::Internal;
    if (machine != 0x8664) {
        LOG_ERROR("pe_load_file: not x86-64 (machine=0x%x)", machine);
        return Status::NotImplemented;
    }
    const usize opt = nt + 24;
    u16 opt_magic = 0;
    if (!r.read(opt, &opt_magic, 2)) return Status::Internal;
    if (opt_magic != 0x20B) return Status::NotImplemented;  // 0x20B = PE32+

    u32 entry_rva     = 0;
    u32 size_of_image = 0;
    u32 hdr_size      = 0;
    u64 image_base    = 0;
    if (!r.read(opt + 16, &entry_rva,    4)) return Status::Internal;
    if (!r.read(opt + 24, &image_base,   8)) return Status::Internal;
    if (!r.read(opt + 56, &size_of_image,4)) return Status::Internal;
    if (!r.read(opt + 60, &hdr_size,     4)) return Status::Internal;

    // DataDirectory[0] = Export, [1] = Import, [5] = Base relocations.
    const usize dd0 = opt + 112;        // OptionalHeader.DataDirectory[0]
    u32 export_rva = 0, export_size = 0;
    u32 import_rva = 0, import_size = 0;
    u32 reloc_rva  = 0, reloc_size  = 0;
    if (!r.read(dd0 + 0,  &export_rva,  4)) return Status::Internal;
    if (!r.read(dd0 + 4,  &export_size, 4)) return Status::Internal;
    if (!r.read(dd0 + 8,  &import_rva,  4)) return Status::Internal;
    if (!r.read(dd0 + 12, &import_size, 4)) return Status::Internal;
    if (!r.read(dd0 + 40, &reloc_rva,   4)) return Status::Internal;
    if (!r.read(dd0 + 44, &reloc_size,  4)) return Status::Internal;

    // 3) Allocate flat mapped buffer.
    if (size_of_image < hdr_size || size_of_image > (1u << 30)) return Status::Internal;
    std::vector<u8> mapped(size_of_image, 0);

    // Copy headers (raw file bytes 0..hdr_size).
    if (hdr_size > raw.size()) return Status::Internal;
    std::memcpy(mapped.data(), raw.data(), hdr_size);

    // 4) Walk section headers, copy each section's raw bytes to its VA.
    const usize sec_table = opt + opt_size;
    for (u16 i = 0; i < num_sections; ++i) {
        const usize s = sec_table + i * 40;
        u32 virt_size = 0, virt_addr = 0, raw_size = 0, raw_ptr = 0;
        if (!r.read(s + 8,  &virt_size, 4)) return Status::Internal;
        if (!r.read(s + 12, &virt_addr, 4)) return Status::Internal;
        if (!r.read(s + 16, &raw_size,  4)) return Status::Internal;
        if (!r.read(s + 20, &raw_ptr,   4)) return Status::Internal;
        if (virt_addr >= mapped.size()) continue;
        const u32 copy = (raw_size < virt_size) ? raw_size : virt_size;
        const u32 fit  = static_cast<u32>(mapped.size() - virt_addr);
        const u32 n    = (copy < fit) ? copy : fit;
        if (raw_ptr + n > raw.size()) continue;
        std::memcpy(mapped.data() + virt_addr, raw.data() + raw_ptr, n);
    }

    // 4b) Apply base relocations if we're loading at a non-preferred base.
    //
    // The .reloc section is a sequence of IMAGE_BASE_RELOCATION blocks:
    //   +0  PageRVA   (4 bytes) -- RVA of the 4 KiB page the entries apply to
    //   +4  BlockSize (4 bytes) -- total bytes including the header (8 + 2*N)
    //   then N u16 entries, each: high 4 bits = type, low 12 bits = offset
    // Types we handle:
    //   0  IMAGE_REL_BASED_ABSOLUTE     -- pad, ignore
    //   3  IMAGE_REL_BASED_HIGHLOW      -- 32-bit fixup (rare in x64)
    //   10 IMAGE_REL_BASED_DIR64        -- 64-bit fixup (x64 norm)
    const u64 chosen_base = (forced_base != 0) ? forced_base : image_base;
    const i64 delta = static_cast<i64>(chosen_base) - static_cast<i64>(image_base);
    if (reloc_size > 0 && reloc_rva != 0 && delta != 0) {
        const u32 reloc_end = reloc_rva + reloc_size;
        u32 cur = reloc_rva;
        while (cur + 8 <= reloc_end) {
            u32 page_rva = 0, block_size = 0;
            if (cur + 4 > mapped.size()) break;
            std::memcpy(&page_rva,   mapped.data() + cur + 0, 4);
            std::memcpy(&block_size, mapped.data() + cur + 4, 4);
            if (block_size < 8 || cur + block_size > reloc_end) break;
            const u32 num_entries = (block_size - 8) / 2;
            for (u32 i = 0; i < num_entries; ++i) {
                u16 entry = 0;
                std::memcpy(&entry, mapped.data() + cur + 8 + i * 2, 2);
                const u8  type   = static_cast<u8>(entry >> 12);
                const u32 offset = entry & 0xFFFu;
                const u32 target_rva = page_rva + offset;
                if (target_rva + 8 > mapped.size()) continue;
                if (type == 10) {           // IMAGE_REL_BASED_DIR64
                    u64 v = 0;
                    std::memcpy(&v, mapped.data() + target_rva, 8);
                    v = static_cast<u64>(static_cast<i64>(v) + delta);
                    std::memcpy(mapped.data() + target_rva, &v, 8);
                } else if (type == 3) {     // IMAGE_REL_BASED_HIGHLOW
                    if (target_rva + 4 > mapped.size()) continue;
                    u32 v = 0;
                    std::memcpy(&v, mapped.data() + target_rva, 4);
                    v = static_cast<u32>(static_cast<i32>(v) + static_cast<i32>(delta));
                    std::memcpy(mapped.data() + target_rva, &v, 4);
                }
                // type 0 (ABSOLUTE) and any other are silently ignored.
            }
            cur += block_size;
        }
    }

    // 5) Parse export table -- name -> RVA.
    Reader m{mapped.data(), mapped.size()};
    if (export_size > 0 && export_rva != 0) {
        u32 num_names    = 0;
        u32 names_rva    = 0;
        u32 ordinals_rva = 0;
        m.read(export_rva + 0x18, &num_names,    4);
        m.read(export_rva + 0x20, &names_rva,    4);
        m.read(export_rva + 0x24, &ordinals_rva, 4);
        for (u32 i = 0; i < num_names; ++i) {
            u32 name_rva = 0;
            u16 ordinal  = 0;
            if (!m.read(names_rva + 4 * i, &name_rva, 4)) break;
            if (!m.read(ordinals_rva + 2 * i, &ordinal, 2)) break;
            // Read NUL-terminated name.
            std::string name;
            for (u32 j = 0; j < 256; ++j) {
                u8 c = 0;
                if (!m.read(name_rva + j, &c, 1)) { name.clear(); break; }
                if (c == 0) break;
                name.push_back(static_cast<char>(c));
            }
            if (name.empty()) continue;
            // `ordinal` here is already the 0-based EAT index (it's
            // NameOrdinals[i], NOT including ExportDirectory.Base).
            const u32 fn_rva = read_export_rva(m, export_rva, export_size, ordinal);
            if (fn_rva != 0) out.exports.emplace(std::move(name), fn_rva);
        }
    }

    // 6) Parse the import table.
    //
    // Each IMAGE_IMPORT_DESCRIPTOR is 20 bytes:
    //   +0  OriginalFirstThunk (ILT)   -- RVA of u64 array
    //   +4  TimeDateStamp
    //   +8  ForwarderChain
    //   +12 Name                       -- RVA of imported DLL name (NUL-term)
    //   +16 FirstThunk (IAT)           -- RVA of u64 array (loader-written addrs)
    // Array terminated by an all-zero descriptor.
    if (import_size > 0 && import_rva != 0) {
        const u64 base_for_iat = (forced_base != 0) ? forced_base : image_base;
        for (u32 d = 0; ; ++d) {
            const u32 desc = import_rva + d * 20;
            u32 ilt = 0, name_rva = 0, iat = 0;
            if (!m.read(desc + 0,  &ilt,      4)) break;
            if (!m.read(desc + 12, &name_rva, 4)) break;
            if (!m.read(desc + 16, &iat,      4)) break;
            if (ilt == 0 && name_rva == 0 && iat == 0) break;       // sentinel
            // Read DLL name.
            std::string dll;
            for (u32 j = 0; j < 64; ++j) {
                u8 c = 0; if (!m.read(name_rva + j, &c, 1)) { dll.clear(); break; }
                if (c == 0) break;
                dll.push_back(static_cast<char>(std::tolower(c)));
            }
            if (dll.empty()) continue;
            const u32 names_array = (ilt != 0) ? ilt : iat;       // bound IAT also has names initially
            for (u32 i = 0; ; ++i) {
                u64 thunk = 0;
                if (!m.read(names_array + i * 8, &thunk, 8)) break;
                if (thunk == 0) break;                              // end of imports for this DLL
                PeImport imp{};
                imp.dll      = dll;
                imp.iat_addr = base_for_iat + iat + i * 8;
                if (thunk & (u64{1} << 63)) {
                    // Import by ordinal: low 16 bits = ordinal.
                    char buf[32];
                    std::snprintf(buf, sizeof(buf), "#%u",
                                  static_cast<unsigned>(thunk & 0xFFFFu));
                    imp.name = buf;
                } else {
                    // Import by name: u32 = RVA of IMAGE_IMPORT_BY_NAME (u16 hint + name).
                    const u32 ibn_rva = static_cast<u32>(thunk & 0x7FFFFFFFu);
                    std::string name;
                    for (u32 j = 0; j < 128; ++j) {
                        u8 c = 0;
                        if (!m.read(ibn_rva + 2 + j, &c, 1)) { name.clear(); break; }
                        if (c == 0) break;
                        name.push_back(static_cast<char>(c));
                    }
                    if (name.empty()) continue;
                    imp.name = std::move(name);
                }
                out.imports.push_back(std::move(imp));
            }
        }
    }

    out.base      = (forced_base != 0) ? forced_base : image_base;
    out.size      = size_of_image;
    out.entry_rva = entry_rva;
    out.mapped    = std::move(mapped);
    return Status::Ok;
}

GuestAddr pe_export_addr(const PeLoadedImage& img, const char* name) noexcept {
    auto it = img.exports.find(name);
    if (it == img.exports.end()) return 0;
    return img.base + it->second;
}

} // namespace emu::host
