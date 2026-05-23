// Minimal Win64 SEH RaiseException + unwind driver.
//
// Scope: take a fault (kind + code + faulting addr), find the language
// handler covering the current RIP via .pdata + UNW_FLAG_EHANDLER, set up
// the Win64 ABI call frame for that handler, and prime the dispatcher to
// run it. The handler runs as ordinary emulated code; its return value (in
// EAX) is one of the EXCEPTION_DISPOSITION codes (Continue=0, Search=1).
//
// Out of scope: collided unwinds, nested exceptions, the FuncInfo32 +
// scope-table walk used by `__try`/`__except` (that's MSVC-specific glue
// that sits on top of the basic RaiseException flow). Functions that
// install a custom `__C_specific_handler` and read scope tables to find
// `__except` filters still work -- they just have to read the scope table
// themselves; we don't decode it.

#pragma once

#include "emu/emu.h"
#include "emu/types.h"
#include "emu/memory.h"

namespace emu {

// Win64 EXCEPTION_RECORD layout (152 bytes total in Win SDK; we use a
// trimmed copy with the fields the handler typically reads).
constexpr u32 EXCEPTION_NONCONTINUABLE  = 0x00000001;
constexpr u32 EXCEPTION_MAXIMUM_PARAMETERS = 15;

struct GuestExceptionRecord {
    u32 ExceptionCode;
    u32 ExceptionFlags;
    u64 ExceptionRecord;            // pointer to nested ER (we set to 0)
    u64 ExceptionAddress;
    u32 NumberParameters;
    u32 _pad;
    u64 ExceptionInformation[EXCEPTION_MAXIMUM_PARAMETERS];
};
static_assert(sizeof(GuestExceptionRecord) == 152, "ExceptionRecord layout drift");

// Raise an exception at the current RIP. Walks .pdata + .xdata; if a
// language handler is found, allocates a GuestExceptionRecord on the guest
// stack, sets RCX/RDX/R8/R9 per Win64 SEH ABI, and points RIP at the
// handler so the dispatcher runs it on the next iteration.
//
// Returns:
//   Status::Ok                -- handler is set up; caller's run loop will
//                                continue into the handler.
//   Status::FunctionNotFound  -- no .pdata coverage for the current RIP.
//   Status::NotImplemented    -- .pdata says no E/UHANDLER flag is set.
//
// `image_base` is the module base the .pdata belongs to (PE base).
Status emu_raise_exception(Emulator& e, GuestAddr image_base,
                           u32 exception_code, GuestAddr exception_addr) noexcept;

} // namespace emu
