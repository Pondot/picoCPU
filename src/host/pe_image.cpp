// PE image inspector -- .pdata lookup.

#include "pe_image.h"

#include "emu/logger.h"

#include <cstring>

namespace emu::host {

namespace {

struct DataDirectory { emu::u32 rva; emu::u32 size; };

// Read PE DataDirectory[3] (exception directory, .pdata).
emu::Status read_pdata_dir(emu::MemoryProvider& mem, emu::GuestAddr base,
                           DataDirectory& out) noexcept {
    // DOS header: at base+0, "MZ" signature; e_lfanew at offset 0x3C.
    emu::u8 mz[2] = {};
    if (emu::Status s = mem.read(base, 2, mz); emu::fail(s)) return s;
    if (mz[0] != 'M' || mz[1] != 'Z') return emu::Status::Internal;

    emu::u32 e_lfanew = 0;
    if (emu::Status s = mem.read(base + 0x3C, 4, &e_lfanew); emu::fail(s)) return s;

    const emu::GuestAddr nt = base + e_lfanew;
    // Signature: "PE\0\0"
    emu::u8 sig[4] = {};
    if (emu::Status s = mem.read(nt, 4, sig); emu::fail(s)) return s;
    if (sig[0] != 'P' || sig[1] != 'E' || sig[2] != 0 || sig[3] != 0) return emu::Status::Internal;

    // FileHeader.Machine: 0x8664 for x64. We trust the caller; just sanity-check size.
    emu::u16 machine = 0;
    if (emu::Status s = mem.read(nt + 4, 2, &machine); emu::fail(s)) return s;
    if (machine != 0x8664) return emu::Status::NotImplemented;

    // OptionalHeader starts at nt+24. For PE32+, DataDirectory[] is at offset
    // 24 + 112 = 136 from the start of NtHeaders (OptionalHeader.DataDirectory).
    // Index 3 = exception directory (.pdata). Each entry is 8 bytes (rva + size).
    const emu::GuestAddr exc_dir = nt + 24 + 112 + 3 * 8;
    emu::u8 dir[8] = {};
    if (emu::Status s = mem.read(exc_dir, 8, dir); emu::fail(s)) return s;
    std::memcpy(&out.rva,  dir + 0, 4);
    std::memcpy(&out.size, dir + 4, 4);
    return emu::Status::Ok;
}

} // namespace

emu::Status pdata_lookup(emu::MemoryProvider& mem,
                         emu::GuestAddr image_base,
                         emu::GuestAddr rip,
                         RuntimeFunction& out) noexcept {
    if (rip < image_base) return emu::Status::FunctionNotFound;
    const emu::u32 target_rva = static_cast<emu::u32>(rip - image_base);

    DataDirectory dir{};
    if (emu::Status s = read_pdata_dir(mem, image_base, dir); emu::fail(s)) return s;
    if (dir.size == 0) return emu::Status::FunctionNotFound;

    const emu::GuestAddr pdata = image_base + dir.rva;
    const emu::u32 count = dir.size / 12;     // each RUNTIME_FUNCTION = 12 bytes

    // .pdata is sorted by begin_rva -- binary search.
    emu::u32 lo = 0, hi = count;
    while (lo < hi) {
        const emu::u32 mid = lo + (hi - lo) / 2;
        emu::u8 rec[12] = {};
        if (emu::Status s = mem.read(pdata + 12ull * mid, 12, rec); emu::fail(s)) return s;
        emu::u32 b = 0, e = 0, u = 0;
        std::memcpy(&b, rec + 0, 4);
        std::memcpy(&e, rec + 4, 4);
        std::memcpy(&u, rec + 8, 4);
        if (target_rva < b)      hi = mid;
        else if (target_rva >= e) lo = mid + 1;
        else {
            out.begin_rva  = b;
            out.end_rva    = e;
            out.unwind_rva = u;
            return emu::Status::Ok;
        }
    }
    return emu::Status::FunctionNotFound;
}

// ---- SEH UNWIND_INFO walker -------------------------------------------------

namespace {

// UNWIND_OP codes (Win64 SEH).
constexpr emu::u8 UWOP_PUSH_NONVOL     = 0;
constexpr emu::u8 UWOP_ALLOC_LARGE     = 1;
constexpr emu::u8 UWOP_ALLOC_SMALL     = 2;
constexpr emu::u8 UWOP_SET_FPREG       = 3;
constexpr emu::u8 UWOP_SAVE_NONVOL     = 4;
constexpr emu::u8 UWOP_SAVE_NONVOL_FAR = 5;
constexpr emu::u8 UWOP_SAVE_XMM128     = 8;
constexpr emu::u8 UWOP_SAVE_XMM128_FAR = 9;
constexpr emu::u8 UWOP_PUSH_MACHFRAME  = 10;

// Each unwind op consumes 1 to 3 UNWIND_CODE slots.
emu::u8 unwind_op_size(emu::u8 op, emu::u8 info) noexcept {
    switch (op) {
        case UWOP_PUSH_NONVOL:     return 1;
        case UWOP_ALLOC_LARGE:     return (info == 0) ? 2 : 3;
        case UWOP_ALLOC_SMALL:     return 1;
        case UWOP_SET_FPREG:       return 1;
        case UWOP_SAVE_NONVOL:     return 2;
        case UWOP_SAVE_NONVOL_FAR: return 3;
        case UWOP_SAVE_XMM128:     return 2;
        case UWOP_SAVE_XMM128_FAR: return 3;
        case UWOP_PUSH_MACHFRAME:  return 1;
        default:                   return 1;
    }
}

} // namespace

emu::Status seh_unwind_frame(emu::MemoryProvider& mem,
                             emu::GuestAddr image_base,
                             emu::GuestAddr rip,
                             emu::GuestAddr rsp,
                             UnwoundFrame& out) noexcept {
    RuntimeFunction rf{};
    if (emu::Status s = pdata_lookup(mem, image_base, rip, rf); emu::fail(s)) return s;

    // Follow chained UNWIND_INFO if present. The chain link is the top bit
    // of unwind_rva: when set, that "unwind RVA" actually points to another
    // RUNTIME_FUNCTION whose own unwind info covers the same range. We walk
    // up to 16 hops to bound runaway.
    emu::u32 ui_rva = rf.unwind_rva;
    for (int hop = 0; hop < 16; ++hop) {
        if (!(ui_rva & 1u)) break;
        emu::u8 parent[12] = {};
        if (emu::Status s = mem.read(image_base + (ui_rva & ~u64{1}), 12, parent); emu::fail(s)) return s;
        emu::u32 next_unwind = 0;
        std::memcpy(&next_unwind, parent + 8, 4);
        ui_rva = next_unwind;
    }
    const emu::GuestAddr ui_addr = image_base + ui_rva;
    emu::u8 header[4] = {};
    if (emu::Status s = mem.read(ui_addr, 4, header); emu::fail(s)) return s;
    const emu::u8 version_flags    = header[0];
    const emu::u8 count_of_codes   = header[2];
    (void)version_flags;
    (void)header;   // size_of_prolog (header[1]) not needed for unwinding

    // Distance into the function -- used to decide which unwind codes apply.
    const emu::u32 offset_in_func = static_cast<emu::u32>(rip - (image_base + rf.begin_rva));

    // Walk codes. Each UNWIND_CODE is 2 bytes; entries are in reverse-prolog
    // order (newest first). For unwinding we apply codes whose code_offset is
    // ≤ current offset_in_func (i.e. the corresponding prolog op has already
    // executed by the time we're at `rip`).
    const emu::GuestAddr codes_addr = ui_addr + 4;
    emu::u64 working_rsp = rsp;

    emu::u8 idx = 0;
    while (idx < count_of_codes) {
        emu::u8 code[2] = {};
        if (emu::Status s = mem.read(codes_addr + idx * 2, 2, code); emu::fail(s)) return s;
        const emu::u8 code_offset    = code[0];
        const emu::u8 op             = static_cast<emu::u8>(code[1] & 0x0Fu);
        const emu::u8 op_info        = static_cast<emu::u8>(code[1] >> 4);
        const emu::u8 slots          = unwind_op_size(op, op_info);

        // Apply only if this prolog op has already run.
        const bool applies = (code_offset <= offset_in_func);
        if (applies) {
            switch (op) {
                case UWOP_PUSH_NONVOL:
                    working_rsp += 8;
                    break;
                case UWOP_ALLOC_SMALL:
                    working_rsp += (static_cast<emu::u64>(op_info) * 8 + 8);
                    break;
                case UWOP_ALLOC_LARGE: {
                    if (op_info == 0) {
                        emu::u8 extra[2] = {};
                        if (emu::Status s = mem.read(codes_addr + (idx + 1) * 2, 2, extra); emu::fail(s)) return s;
                        const emu::u32 sz = (static_cast<emu::u32>(extra[0])
                                            | (static_cast<emu::u32>(extra[1]) << 8)) * 8u;
                        working_rsp += sz;
                    } else {
                        emu::u8 extra[4] = {};
                        if (emu::Status s = mem.read(codes_addr + (idx + 1) * 2, 4, extra); emu::fail(s)) return s;
                        emu::u32 sz = 0;
                        std::memcpy(&sz, extra, 4);
                        working_rsp += sz;
                    }
                    break;
                }
                case UWOP_SET_FPREG:
                    // The function set a frame register; without that register's
                    // value we can't precisely undo this. Best-effort: leave
                    // working_rsp alone (it'll be wrong if the prolog allocated
                    // after this point, but most simple functions don't).
                    break;
                case UWOP_SAVE_NONVOL:
                case UWOP_SAVE_NONVOL_FAR:
                case UWOP_SAVE_XMM128:
                case UWOP_SAVE_XMM128_FAR:
                    // These save registers to slots within the already-allocated
                    // stack frame; they don't change RSP.
                    break;
                case UWOP_PUSH_MACHFRAME:
                    // Hardware-pushed exception frame: SS, OLD-RSP, RFLAGS, CS,
                    // RIP. info=0 -> 40 bytes, info=1 -> 48 bytes (with error code).
                    working_rsp += (op_info == 0) ? 40 : 48;
                    return emu::Status::NotImplemented;   // exception frames need richer flow
                default:
                    return emu::Status::NotImplemented;
            }
        }
        idx = static_cast<emu::u8>(idx + slots);
    }

    // After undoing all prolog stack movements, top of stack = return address.
    emu::u8 ret_bytes[8] = {};
    if (emu::Status s = mem.read(working_rsp, 8, ret_bytes); emu::fail(s)) return s;
    emu::u64 ret_addr = 0;
    for (int i = 0; i < 8; ++i) ret_addr |= (emu::u64{ret_bytes[i]} << (8 * i));

    out.caller_rip = ret_addr;
    out.caller_rsp = working_rsp + 8;
    return emu::Status::Ok;
}

emu::Status seh_exception_handler(emu::MemoryProvider& mem,
                                  emu::GuestAddr image_base,
                                  emu::GuestAddr rip,
                                  ExceptionInfo& out) noexcept {
    RuntimeFunction rf{};
    if (emu::Status s = pdata_lookup(mem, image_base, rip, rf); emu::fail(s)) return s;

    // Resolve chained UNWIND_INFO (top bit set => chain to another RUNTIME_FUNCTION).
    emu::u32 ui_rva = rf.unwind_rva;
    for (int chain_depth = 0; chain_depth < 16; ++chain_depth) {
        if (!(ui_rva & 1u)) break;   // not chained
        // The "RUNTIME_FUNCTION at ui_rva & ~1" is the parent.
        emu::u8 parent[12] = {};
        if (emu::Status s = mem.read(image_base + (ui_rva & ~u64{1}), 12, parent); emu::fail(s)) return s;
        emu::u32 next_unwind = 0;
        std::memcpy(&next_unwind, parent + 8, 4);
        ui_rva = next_unwind;
    }

    emu::u8 header[4] = {};
    if (emu::Status s = mem.read(image_base + ui_rva, 4, header); emu::fail(s)) return s;
    const emu::u8 flags = static_cast<emu::u8>(header[0] >> 3);
    const emu::u8 count_of_codes = header[2];

    out.is_exception_handler = (flags & 0x1) != 0;
    out.is_unwind_handler    = (flags & 0x2) != 0;
    if (!out.is_exception_handler && !out.is_unwind_handler) {
        return emu::Status::FunctionNotFound;
    }

    // After the codes (padded to align to 4 bytes) comes the exception handler
    // RVA, followed by language-specific scope data.
    const emu::u32 codes_bytes  = static_cast<emu::u32>(count_of_codes) * 2u;
    const emu::u32 codes_padded = (codes_bytes + 3u) & ~u32{3u};
    const emu::GuestAddr after_codes = image_base + ui_rva + 4u + codes_padded;

    emu::u32 handler_rva = 0;
    if (emu::Status s = mem.read(after_codes, 4, &handler_rva); emu::fail(s)) return s;
    out.handler_rva     = handler_rva;
    out.scope_data_rva  = static_cast<emu::u32>(ui_rva + 4u + codes_padded + 4u);
    return emu::Status::Ok;
}

} // namespace emu::host
