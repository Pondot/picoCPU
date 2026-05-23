// Host-side TEB lookup for a target process.
//
// On Windows x64 each thread's TEB is pointed at by GS. To service GS-relative
// loads in the emulator we set `Cpu::gs_base` to the target thread's TEB
// address -- this is exactly the host-side equivalent of "what would the real
// CPU's GS register hold while that thread ran".

#pragma once

#include "emu/error.h"
#include "emu/types.h"

namespace emu::host {

// Look up the TEB base address for a thread inside the target process.
//
// If `tid == 0`, the thread with the lowest TID in `pid` is used (Win32
// convention: the main thread has the lowest TID of its process). Otherwise
// the named thread is queried.
//
// Returns Status::Ok and fills `out_teb`; HostQueryFailed otherwise.
Status query_thread_teb(u32 pid, u32 tid, GuestAddr& out_teb) noexcept;

} // namespace emu::host
