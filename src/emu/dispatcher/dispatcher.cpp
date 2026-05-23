// Phase 1 dispatcher: single-instruction decode + execute.
//
// No block cache yet -- each iteration re-decodes the instruction at RIP. The
// cost is fine for the gate (one ~17-instruction function). Phase 4a swaps
// this for a block-keyed cache.

#include "emu/emu.h"

#include "emu/hooks.h"
#include "emu/iat_stubs.h"
#include "emu/logger.h"

#include <algorithm>

namespace emu {

void Emulator::add_stop_addr(GuestAddr addr) noexcept {
    stop_addrs_.push_back(addr);
}

void Emulator::clear_stop_addrs() noexcept {
    stop_addrs_.clear();
}

bool Emulator::is_stop_(GuestAddr a) const noexcept {
    return std::find(stop_addrs_.begin(), stop_addrs_.end(), a) != stop_addrs_.end();
}

RunResult Emulator::run(GuestAddr entry, u64 max_insns) noexcept {
    return run_with_cpu(cpu_, entry, max_insns);
}

RunResult Emulator::run_with_cpu(Cpu& cpu, GuestAddr entry, u64 max_insns) noexcept {
    RunResult res{};

    cpu.unhalt();
    cpu.clear_branch();
    cpu.set_rip(entry);

    auto* mp = cpu.mem_read();
    if (mp == nullptr) {
        res.status = Status::ProviderFailure;
        return res;
    }

    u64 executed = 0;
    while (executed < max_insns) {
        if (cpu.halted()) break;

        const GuestAddr pc = cpu.rip();

        // IAT stub interception: if RIP enters the stub region, the dispatch
        // helper runs the registered handler and synthesizes a RET. We then
        // continue from the new RIP without decoding anything at `pc`.
        if (iat_stubs_ != nullptr && iat_stubs_->in_region(pc)) {
            if (!iat_stubs_->dispatch(cpu, pc)) {
                res.status         = cpu.fault().reason;
                res.final_rip      = pc;
                res.final_rax      = cpu.r(reg::RAX);
                res.insns_executed = executed;
                return res;
            }
            if (is_stop_(cpu.rip())) break;
            continue;
        }

        DecodedBlock*   block = block_cache_.lookup(pc);
        if (block == nullptr) {
            auto fresh = std::make_unique<DecodedBlock>();
            Status s = decode_block(*mp, pc, *fresh);
            if (fail(s)) {
                LOG_WARN("decode_block failed at 0x%llx: %.*s",
                         (unsigned long long)pc,
                         static_cast<int>(to_string(s).size()),
                         to_string(s).data());
                res.status         = s;
                res.final_rip      = pc;
                res.final_rax      = cpu.r(reg::RAX);
                res.insns_executed = executed;
                return res;
            }
            block = block_cache_.insert(pc, std::move(fresh));
        }

        // Fire BLOCK hooks before the first instruction of this block.
        if (hooks_.any(HookKind::Block)) {
            hooks_.for_each(HookKind::Block, [&](const Hook& h) {
                if (hook_in_range(h, pc)) {
                    reinterpret_cast<HookBlockCb>(h.callback)(h.user_data, cpu, pc, block->byte_size);
                }
            });
        }

        // Replay the block.
        bool branch_or_halt = false;
        for (const Insn& insn : block->insns) {
            if (cpu.halted()) { branch_or_halt = true; break; }
            cpu.clear_branch();

            // Hardware breakpoint (DR0-DR3, execute type) at this RIP.
            if (cpu.exec_breakpoint_at(insn.rip)) {
                cpu.set_fault(FaultKind::Breakpoint, insn.rip, Status::Ok,
                               "DR execute breakpoint");
                branch_or_halt = true;
                break;
            }

            // Fire CODE hooks (per-instruction).
            if (hooks_.any(HookKind::Code)) {
                hooks_.for_each(HookKind::Code, [&](const Hook& h) {
                    if (hook_in_range(h, insn.rip)) {
                        reinterpret_cast<HookCodeCb>(h.callback)(h.user_data, cpu, insn.rip, insn.len);
                    }
                });
            }

            // Fire INSN hooks (filtered by OpKind).
            if (hooks_.any(HookKind::Insn)) {
                hooks_.for_each(HookKind::Insn, [&](const Hook& h) {
                    if ((h.filter_op == OpKind::Invalid || h.filter_op == insn.kind)
                        && hook_in_range(h, insn.rip)) {
                        reinterpret_cast<HookInsnCb>(h.callback)(h.user_data, cpu, insn);
                    }
                });
            }

            if (trace_cb_) {
                // Refresh cpu.rip() so the trace shows the right value before
                // a control-flow insn rewrites it.
                cpu.set_rip(insn.rip);
                trace_cb_(trace_user_, cpu, insn);
            }

            insn.handler(cpu, insn);
            ++executed;

            if (cpu.halted()) {
                if (cpu.fault().is_set() && fault_recovery_cb_ != nullptr) {
                    // Give the recovery handler a chance to consume the fault
                    // (e.g. by jumping to a registered SEH handler). If it
                    // returns true, we unhalt and continue from the new RIP.
                    if (fault_recovery_cb_(fault_recovery_user_, *this, cpu.fault())) {
                        cpu.clear_fault();
                        cpu.unhalt();
                        branch_or_halt = true;
                        break;
                    }
                }
                res.status         = Status::Ok;
                res.final_rip      = cpu.rip();
                res.final_rax      = cpu.r(reg::RAX);
                res.insns_executed = executed;
                if (cpu.fault().is_set()) {
                    res.status = cpu.fault().reason != Status::Ok
                               ? cpu.fault().reason
                               : Status::Internal;
                    LOG_WARN("fault %s at rip=0x%llx addr=0x%llx (%.*s)",
                             fault_kind_name(cpu.fault().kind),
                             (unsigned long long)cpu.fault().rip,
                             (unsigned long long)cpu.fault().addr,
                             static_cast<int>(to_string(res.status).size()),
                             to_string(res.status).data());
                    if (auto cb = cpu.fault_hook()) {
                        cb(cpu.fault_hook_user(), cpu, cpu.fault());
                    }
                    if (hooks_.any(HookKind::Fault)) {
                        hooks_.for_each(HookKind::Fault, [&](const Hook& h) {
                            reinterpret_cast<HookFaultCb>(h.callback)(h.user_data, cpu, cpu.fault());
                        });
                    }
                }
                return res;
            }

            if (cpu.branch_taken()) {
                // Record / verify the branch target cache entry.
                const GuestAddr predicted = btc_.predict(insn.rip);
                if (predicted == cpu.rip()) {
                    btc_.count_hit();
                } else {
                    btc_.count_miss();
                    btc_.record(insn.rip, cpu.rip());
                }
                branch_or_halt = true;
                break;
            }

            // Straight-line advance.
            cpu.set_rip(insn.rip + insn.len);

            if (is_stop_(cpu.rip())) {
                res.status         = Status::Ok;
                res.final_rip      = cpu.rip();
                res.final_rax      = cpu.r(reg::RAX);
                res.insns_executed = executed;
                return res;
            }

            if (executed >= max_insns) { branch_or_halt = true; break; }
        }

        // If the block ended without a branch (max_insns exhausted mid-block
        // or fall-through), the outer loop's next iteration picks up the new
        // RIP normally.
        if (branch_or_halt && is_stop_(cpu.rip())) {
            res.status         = Status::Ok;
            res.final_rip      = cpu.rip();
            res.final_rax      = cpu.r(reg::RAX);
            res.insns_executed = executed;
            return res;
        }
    }

    res.status         = Status::Ok;
    res.final_rip      = cpu.rip();
    res.final_rax      = cpu.r(reg::RAX);
    res.insns_executed = executed;
    return res;
}

} // namespace emu
