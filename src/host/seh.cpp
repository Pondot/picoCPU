// Minimal SEH RaiseException driver.

#include "emu/seh.h"
#include "emu/cpu.h"
#include "emu/logger.h"

#include "pe_image.h"

#include <cstring>

namespace emu {

Status emu_raise_exception(Emulator& e, GuestAddr image_base,
                           u32 code, GuestAddr addr) noexcept {
    Cpu& cpu = e.cpu();
    auto* mp_r = cpu.mem_read();
    auto* mp_w = cpu.mem_write();
    if (!mp_r || !mp_w) return Status::ProviderFailure;

    // 1. Find the handler covering current RIP.
    host::ExceptionInfo info{};
    Status s = host::seh_exception_handler(*mp_r, image_base, cpu.rip(), info);
    if (fail(s)) return s;
    if (!info.is_exception_handler) return Status::NotImplemented;
    const GuestAddr handler_rip = image_base + info.handler_rva;

    // 2. Build the EXCEPTION_RECORD on the guest stack.
    //    Win64 stack layout (RSP grows down):
    //        [rsp+0]   = ret addr (we push a sentinel so RET halts)
    //        [rsp+8]   ... shadow space for handler's home slots (32 bytes)
    //        ... GuestExceptionRecord ...
    //        ... ContextRecord (we use a stub; handler typically reads only ER)
    GuestExceptionRecord er{};
    er.ExceptionCode    = code;
    er.ExceptionFlags   = 0;
    er.ExceptionRecord  = 0;
    er.ExceptionAddress = (addr != 0) ? addr : cpu.rip();

    constexpr u64 kERSize   = sizeof(GuestExceptionRecord);    // 152
    constexpr u64 kCtxStub  = 0x4D0;                            // CONTEXT.x64 is 0x4D0 bytes
    constexpr u64 kDcSize   = 0x40;                             // DispatcherContext

    u64 rsp = cpu.r(reg::RSP);
    // Reserve and 16-byte-align.
    auto sub = [&](u64 n) { rsp = (rsp - n) & ~u64{0xF}; };

    sub(kCtxStub);
    const GuestAddr ctx_addr = rsp;
    {
        u8 zeros[16] = {};
        for (u64 off = 0; off < kCtxStub; off += 16) {
            (void)mp_w->write(ctx_addr + off, 16, zeros);
        }
    }

    sub(kERSize);
    const GuestAddr er_addr = rsp;
    (void)mp_w->write(er_addr, kERSize, &er);

    sub(kDcSize);
    const GuestAddr dc_addr = rsp;
    {
        u8 zeros[64] = {};
        (void)mp_w->write(dc_addr, kDcSize, zeros);
    }

    // 3. Reserve shadow space (32 bytes) + sentinel return address.
    rsp -= 8;
    {
        constexpr u64 kSentinel = 0xDEAD'BEEF'DEAD'BEEFull;
        u8 buf[8];
        for (int i = 0; i < 8; ++i) buf[i] = static_cast<u8>((kSentinel >> (8 * i)) & 0xFFu);
        (void)mp_w->write(rsp, 8, buf);
    }
    rsp -= 32;     // home/shadow space for callee
    cpu.set_r64(reg::RSP, rsp);

    // 4. Win64 SEH handler ABI:
    //      RCX = ExceptionRecord*
    //      RDX = EstablisherFrame (frame pointer at fault site; we use RBP)
    //      R8  = ContextRecord*
    //      R9  = DispatcherContext*
    cpu.set_r64(reg::RCX, er_addr);
    cpu.set_r64(reg::RDX, cpu.r(reg::RBP));
    cpu.set_r64(reg::R8,  ctx_addr);
    cpu.set_r64(reg::R9,  dc_addr);

    // 5. Jump to the handler. The dispatcher's next iteration runs it.
    cpu.set_rip(handler_rip);

    LOG_INFO("raise: code=0x%08x at 0x%llx -> handler RVA 0x%x (abs 0x%llx)",
             code, (unsigned long long)cpu.rip(), info.handler_rva,
             (unsigned long long)handler_rip);
    (void)mp_r;
    return Status::Ok;
}

} // namespace emu
