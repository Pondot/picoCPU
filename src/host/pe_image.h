// PE image inspector.
//
// Reads PE headers + the .pdata RUNTIME_FUNCTION array via a MemoryProvider
// (so it works equally well over RPM for a live target or a FakeMemoryProvider
// for tests). The .pdata array is the foundation of Windows x64 SEH: each
// 12-byte RUNTIME_FUNCTION covers a contiguous RIP range and points to the
// .xdata UNWIND_INFO for that range.
//
// Phase 9 scope: lookup (which RUNTIME_FUNCTION covers `rip`?). Walking the
// UNWIND_INFO codes to actually unwind a frame is deferred -- the lookup is
// the foundation everything else builds on.

#pragma once

#include "emu/error.h"
#include "emu/memory.h"
#include "emu/types.h"

namespace emu::host {

struct RuntimeFunction {
    emu::u32 begin_rva;    // function start, RVA from image base
    emu::u32 end_rva;      // exclusive end
    emu::u32 unwind_rva;   // RVA of UNWIND_INFO in .xdata
};

// Look up the RUNTIME_FUNCTION whose [begin, end) range contains the absolute
// address `rip` inside the module at `image_base`. Returns Status::Ok and
// fills `out` on hit; Status::FunctionNotFound on miss.
emu::Status pdata_lookup(emu::MemoryProvider& mem,
                         emu::GuestAddr image_base,
                         emu::GuestAddr rip,
                         RuntimeFunction& out) noexcept;

// Result of an SEH frame unwind.
struct UnwoundFrame {
    emu::GuestAddr caller_rip;
    emu::GuestAddr caller_rsp;
};

// Win64 SEH unwind: given the current `rip` (inside a function the .pdata
// describes) and `rsp`, walk the UNWIND_INFO codes for that function to
// compute the caller's RIP/RSP. The walker undoes each prolog operation in
// reverse to compute the un-pushed stack pointer at the function's entry,
// then reads the return address from the top of that stack.
//
// Read access to memory uses `mem`. Returns Status::FunctionNotFound if the
// RIP isn't covered by .pdata; Status::NotImplemented for unwind codes we
// don't yet support (UWOP_SAVE_XMM128_FAR, UWOP_PUSH_MACHFRAME, chained-info
// when we're not the leaf frame).
emu::Status seh_unwind_frame(emu::MemoryProvider& mem,
                             emu::GuestAddr image_base,
                             emu::GuestAddr rip,
                             emu::GuestAddr rsp,
                             UnwoundFrame& out) noexcept;

// Exception-handler info from UNWIND_INFO. Set when UNW_FLAG_EHANDLER (1) or
// UNW_FLAG_UHANDLER (2) is present in the version_flags byte. `handler_rva`
// is the language-specific handler's RVA from the image base; `scope_data`
// points (RVA) at language-specific opaque bytes (e.g. MSVC's
// FuncInfo32 / SCOPE_TABLE).
struct ExceptionInfo {
    emu::u32 handler_rva;
    emu::u32 scope_data_rva;
    bool     is_exception_handler;   // UNW_FLAG_EHANDLER bit 0
    bool     is_unwind_handler;      // UNW_FLAG_UHANDLER bit 1
};

// Find the language-specific exception handler for the function containing
// `rip`. Returns Status::Ok if a handler exists, Status::FunctionNotFound if
// the .pdata entry has no E/UHANDLER flag set, or the underlying read error.
emu::Status seh_exception_handler(emu::MemoryProvider& mem,
                                  emu::GuestAddr image_base,
                                  emu::GuestAddr rip,
                                  ExceptionInfo& out) noexcept;

} // namespace emu::host
