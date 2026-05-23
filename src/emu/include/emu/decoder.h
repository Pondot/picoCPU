// x86-64 instruction decoder.
//
// Reads up to 15 bytes of guest code at `rip`, returns an `Insn` with its
// `handler` field already bound. Memory reads happen through the caller-
// supplied `MemoryProvider` (so the decoder can equally well work over a
// FakeMemoryProvider or a real WinProcessMemoryProvider with caches in front).

#pragma once

#include "emu/error.h"
#include "emu/ir.h"
#include "emu/memory.h"
#include "emu/types.h"

namespace emu {

// Decode exactly one instruction. On success, fills `out` and returns Ok.
// `out.rip` is set to `rip`, `out.len` to the total byte count consumed.
//
// `mem` is used to fetch the instruction bytes. The decoder reads in 1-byte
// chunks (sub-optimal but correct); when L1/L2 caches are stacked over the
// provider the reads will coalesce.
Status decode_one(MemoryProvider& mem, GuestAddr rip, Insn& out) noexcept;

// Decode a raw byte buffer (used by unit tests). `bytes` must be at least
// `len` long; decoder will not read past `len`. Returns Status::TruncatedInstruction
// if a complete instruction doesn't fit in `len`.
Status decode_bytes(const u8* bytes, usize len, GuestAddr rip, Insn& out) noexcept;

} // namespace emu
