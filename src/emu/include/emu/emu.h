// Top-level emulator API.
//
// `Emulator` holds a `Cpu` and runs instructions one at a time:
//   1. Decode the instruction at RIP (through the `MemoryProvider`).
//   2. Call its handler.
//   3. If the handler took a branch, use the new RIP; else advance by len.
//   4. Stop on halt / fault / instruction-count limit.

#pragma once

#include "emu/block_cache.h"
#include "emu/branch_target_cache.h"
#include "emu/cpu.h"
#include "emu/decoder.h"
#include "emu/error.h"
#include "emu/hooks.h"
#include "emu/ir.h"
#include "emu/memory.h"
#include "emu/types.h"

#include <vector>

namespace emu {

class IatStubs;     // forward decl

struct RunResult {
    Status    status        = Status::Ok;
    u64       insns_executed = 0;
    GuestAddr final_rip      = 0;
    u64       final_rax      = 0;
};

class Emulator {
public:
    Emulator() noexcept { cpu_.attach_hooks(&hooks_); }

    [[nodiscard]] Cpu&       cpu()       noexcept { return cpu_; }
    [[nodiscard]] const Cpu& cpu() const noexcept { return cpu_; }

    void set_memory_read (MemoryProvider* mp) noexcept { cpu_.attach_memory_read (mp); }
    void set_memory_write(MemoryProvider* mp) noexcept { cpu_.attach_memory_write(mp); }

    // Per-instruction trace callback. nullptr = no trace.
    using TraceCb = void (*)(void* user, const Cpu& cpu, const Insn& insn);
    void set_trace(TraceCb cb, void* user) noexcept { trace_cb_ = cb; trace_user_ = user; }

    // Fault recovery: called when a handler latches a fault that would
    // otherwise halt execution. Return true to consume the fault and
    // continue running (caller is responsible for updating Cpu state --
    // typically by jumping to an SEH handler). Return false to halt as
    // before. nullptr = no recovery (current behavior).
    using FaultRecoveryCb = bool (*)(void* user, Emulator& e, const Fault& f);
    void set_fault_recovery(FaultRecoveryCb cb, void* user) noexcept {
        fault_recovery_cb_   = cb;
        fault_recovery_user_ = user;
    }

    // Register an "if-RIP-lands-here, halt cleanly" exit point. The tester
    // uses this to terminate when the outermost RET pops the sentinel
    // return address pushed before entry.
    void add_stop_addr(GuestAddr addr) noexcept;
    void clear_stop_addrs() noexcept;

    // Run from `entry` until halt/fault, stop-addr hit, or `max_insns` exhausted.
    RunResult run(GuestAddr entry, u64 max_insns = 1'000'000) noexcept;

    // Multi-Cpu cooperative scheduling: run the same Emulator infrastructure
    // (block cache, branch-target cache, IatStubs, hooks) against a caller-
    // owned Cpu. Lets the user drive multiple emulated threads on the same
    // shared memory, switching contexts manually between calls.
    RunResult run_with_cpu(Cpu& cpu, GuestAddr entry, u64 max_insns = 1'000'000) noexcept;

    // Block-cache access for diagnostics & invalidation.
    [[nodiscard]] BlockCache&       block_cache()       noexcept { return block_cache_; }
    [[nodiscard]] const BlockCache& block_cache() const noexcept { return block_cache_; }

    // Hook system access.
    [[nodiscard]] HookManager&       hooks()       noexcept { return hooks_; }
    [[nodiscard]] const HookManager& hooks() const noexcept { return hooks_; }

    // Branch target cache (per-emulator).
    [[nodiscard]] BranchTargetCache&       btc()       noexcept { return btc_; }
    [[nodiscard]] const BranchTargetCache& btc() const noexcept { return btc_; }

    // IAT stub dispatcher. Owned by the caller (e.g. tester). Set to nullptr
    // to disable. When non-null, the run loop intercepts RIPs in the stub
    // region and routes to the registered handlers.
    void attach_iat_stubs(IatStubs* st) noexcept { iat_stubs_ = st; }
    [[nodiscard]] IatStubs* iat_stubs() noexcept { return iat_stubs_; }

private:
    [[nodiscard]] bool is_stop_(GuestAddr a) const noexcept;

    Cpu                    cpu_;
    HookManager            hooks_;
    BlockCache             block_cache_;
    BranchTargetCache      btc_;
    TraceCb                trace_cb_   = nullptr;
    void*                  trace_user_ = nullptr;
    FaultRecoveryCb        fault_recovery_cb_   = nullptr;
    void*                  fault_recovery_user_ = nullptr;
    IatStubs*              iat_stubs_  = nullptr;
    std::vector<GuestAddr> stop_addrs_;
};

} // namespace emu
