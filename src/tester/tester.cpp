// Phase-1 tester. Two modes:
//
//   tester --pid PID --addr 0xADDR --bytes N
//      Read N bytes via RPM at ADDR, hex-dump (Phase 0 behavior).
//
//   tester --pid PID --addr 0xADDR --emulate --seed N [--bytes 256] [--trace]
//      Read function bytes via RPM, emulate `target_fn(seed)`, print RAX.
//      Compares against the seed mixer expected value if --expected is given.

#include "emu/cache.h"
#include "emu/cpu.h"
#include "emu/emu.h"
#include "emu/iat_stubs.h"
#include "emu/ir.h"
#include "emu/logger.h"
#include "emu/memory.h"
#include "emu/shadow_pages.h"
#include "emu/stub_heap.h"

#include "win_process.h"
#include "win_tls.h"
#include "pe_loader.h"

#include "emu/seh.h"

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

struct Args {
    emu::u32       pid           = 0;
    emu::u32       tid           = 0;    // 0 = pick the main thread heuristically
    emu::u64       bench         = 0;    // 0 = single run; N = run target N times for perf
    emu::GuestAddr addr          = 0;
    emu::usize     bytes         = 256;
    bool           emulate       = false;
    bool           trace         = false;
    bool           gs_test       = false;
    bool           stack_test    = false;       // Phase 3: use guest stack + sentinel RET
    bool           no_cache      = false;       // bypass L1/L2/L3 (perf comparison)
    bool           entity_test   = false;       // walk an array of structs in target memory
    bool           selfmod_test  = false;       // execute self-modifying code
    emu::GuestAddr arr_addr      = 0;           // entity array base (for --entity-test)
    emu::u64       arr_count     = 0;           // entity count
    emu::GuestAddr thunk_addr    = 0;           // RWX thunk page (for --selfmod-test)
    emu::u64       thunk_size    = 0;           // thunk page size
    std::string    pe_path;                     // PE file to load (instead of live process)
    std::string    fn_name;                     // exported function name to run
    emu::GuestAddr pe_base       = 0;           // override ImageBase (triggers relocations)
    emu::GuestAddr seh_image_base = 0;          // module base for .pdata lookup on live-process targets
    std::string    input_str;                   // NUL-terminated input; written to 0x10000000, RCX=that
    std::string    input_str2;                  // second buffer at 0x10001000; RDX points there
    emu::u64       input_count   = 0;           // R8 = count (for memcmp etc.)
    emu::u64       rdx_value     = 0;           // explicit RDX (e.g. strchr char arg)
    emu::u64       r9_value      = 0;           // explicit R9 (e.g. qsort comparator ptr)
    bool           rdx_set       = false;
    bool           r9_set        = false;
    std::string    write_bytes_arg;             // "ADDR=HH HH HH ..." -- pre-write bytes
    emu::u64       seed          = 0;
    emu::u64       expected      = 0;
    bool           have_expected = false;
    bool           valid         = false;
};

bool parse_u64(const char* s, emu::u64& out) noexcept {
    if (!s || !*s) return false;
    char* end = nullptr;
    int base = 10;
    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) base = 16;
    out = std::strtoull(s, &end, base);
    return end != s && *end == '\0';
}

Args parse_args(int argc, char** argv) noexcept {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const char* k = argv[i];
        auto need = [&](emu::u64& dst) -> bool {
            if (i + 1 >= argc) return false;
            return parse_u64(argv[++i], dst);
        };
        if (std::strcmp(k, "--pid") == 0) {
            emu::u64 v = 0;
            if (!need(v)) return a;
            a.pid = static_cast<emu::u32>(v);
        } else if (std::strcmp(k, "--tid") == 0) {
            emu::u64 v = 0;
            if (!need(v)) return a;
            a.tid = static_cast<emu::u32>(v);
        } else if (std::strcmp(k, "--addr") == 0) {
            if (!need(a.addr)) return a;
        } else if (std::strcmp(k, "--bytes") == 0) {
            emu::u64 v = 0;
            if (!need(v)) return a;
            a.bytes = static_cast<emu::usize>(v);
        } else if (std::strcmp(k, "--seed") == 0) {
            if (!need(a.seed)) return a;
        } else if (std::strcmp(k, "--expected") == 0) {
            if (!need(a.expected)) return a;
            a.have_expected = true;
        } else if (std::strcmp(k, "--emulate") == 0) {
            a.emulate = true;
        } else if (std::strcmp(k, "--trace") == 0) {
            a.trace = true;
        } else if (std::strcmp(k, "--gs-test") == 0) {
            a.gs_test = true;
            a.emulate = true;   // implies emulation mode
        } else if (std::strcmp(k, "--stack-test") == 0) {
            a.stack_test = true;
            a.emulate = true;
        } else if (std::strcmp(k, "--entity-test") == 0) {
            a.entity_test = true;
            a.emulate = true;
        } else if (std::strcmp(k, "--selfmod-test") == 0) {
            a.selfmod_test = true;
            a.emulate = true;
        } else if (std::strcmp(k, "--arr") == 0) {
            if (!need(a.arr_addr)) return a;
        } else if (std::strcmp(k, "--count") == 0) {
            if (!need(a.arr_count)) return a;
        } else if (std::strcmp(k, "--thunk") == 0) {
            if (!need(a.thunk_addr)) return a;
        } else if (std::strcmp(k, "--thunk-size") == 0) {
            if (!need(a.thunk_size)) return a;
        } else if (std::strcmp(k, "--pe-file") == 0) {
            if (i + 1 >= argc) return a;
            a.pe_path = argv[++i];
            a.emulate = true;
        } else if (std::strcmp(k, "--fn") == 0) {
            if (i + 1 >= argc) return a;
            a.fn_name = argv[++i];
        } else if (std::strcmp(k, "--base") == 0) {
            if (!need(a.pe_base)) return a;
        } else if (std::strcmp(k, "--seh-image-base") == 0) {
            if (!need(a.seh_image_base)) return a;
        } else if (std::strcmp(k, "--input-str") == 0) {
            if (i + 1 >= argc) return a;
            a.input_str = argv[++i];
        } else if (std::strcmp(k, "--input-str2") == 0) {
            if (i + 1 >= argc) return a;
            a.input_str2 = argv[++i];
        } else if (std::strcmp(k, "--input-count") == 0) {
            if (!need(a.input_count)) return a;
        } else if (std::strcmp(k, "--rdx") == 0) {
            if (!need(a.rdx_value)) return a;
            a.rdx_set = true;
        } else if (std::strcmp(k, "--r9") == 0) {
            if (!need(a.r9_value)) return a;
            a.r9_set = true;
        } else if (std::strcmp(k, "--write-bytes") == 0) {
            if (i + 1 >= argc) return a;
            a.write_bytes_arg = argv[++i];
        } else if (std::strcmp(k, "--no-cache") == 0) {
            a.no_cache = true;
        } else if (std::strcmp(k, "--bench") == 0) {
            emu::u64 v = 0;
            if (!need(v)) return a;
            a.bench = v;
            a.emulate = true;
        } else {
            std::fprintf(stderr, "tester: unknown arg '%s'\n", k);
            return a;
        }
    }
    if (!a.pe_path.empty()) {
        // PE-file mode: pid not required; we resolve addr from --fn at run time.
        a.valid = (!a.fn_name.empty()) || (a.addr != 0);
        if (a.bytes == 0) a.bytes = 4096;     // big-enough default for one fn
    } else {
        a.valid = (a.pid != 0) && (a.addr != 0) && (a.bytes > 0) && (a.bytes <= 65536);
    }
    return a;
}

void hex_dump(emu::GuestAddr base, const emu::u8* data, emu::usize n) noexcept {
    constexpr emu::usize W = 16;
    for (emu::usize off = 0; off < n; off += W) {
        emu::usize row = (n - off) < W ? (n - off) : W;
        std::printf("%016llx  ", static_cast<unsigned long long>(base + off));
        for (emu::usize i = 0; i < W; ++i) {
            if (i < row) std::printf("%02x ", data[off + i]);
            else         std::printf("   ");
            if (i == 7) std::putchar(' ');
        }
        std::printf(" |");
        for (emu::usize i = 0; i < row; ++i) {
            const emu::u8 c = data[off + i];
            std::putchar(std::isprint(c) ? c : '.');
        }
        std::printf("|\n");
    }
}

void usage() {
    std::fputs(
        "usage:\n"
        "  tester --pid PID --addr 0xHEX [--bytes 256]                       # dump only\n"
        "  tester --pid PID --addr 0xHEX --emulate --seed N                  # emulate target_fn(seed)\n"
        "         [--bytes 256] [--expected 0xHEX] [--trace] [--no-cache]\n"
        "  tester --pid PID --addr 0xHEX --gs-test [--expected 0xTEB]        # emulate target_get_teb\n"
        "  tester --pid PID --addr 0xHEX --stack-test --seed N [--expected]  # emulate target_stack_fn (uses stack)\n"
        "  tester --pid PID --addr 0xHEX --entity-test --arr 0xHEX --count N [--expected]  # walk entity array\n"
        "  tester --pid PID --addr 0xHEX --selfmod-test --seed N [--expected]              # execute self-modifying code\n"
        "  tester --pe-file PATH --fn NAME --seed N [--expected]                           # load static PE, emulate exported fn\n",
        stderr);
}

void trace_callback(void* /*user*/, const emu::Cpu& cpu, const emu::Insn& insn) {
    std::printf("  [trace] %016llx  %-7s  op_size=%u  rax=%016llx rcx=%016llx flags=%llx\n",
                static_cast<unsigned long long>(insn.rip),
                emu::op_kind_name(insn.kind),
                static_cast<unsigned>(insn.op_size),
                static_cast<unsigned long long>(cpu.r(emu::reg::RAX)),
                static_cast<unsigned long long>(cpu.r(emu::reg::RCX)),
                static_cast<unsigned long long>(cpu.rflags()));
}

} // namespace

int main(int argc, char** argv) {
    emu::log::init(emu::log::Level::Info);
    Args a = parse_args(argc, argv);
    if (!a.valid) { usage(); emu::log::shutdown(); return 2; }

    // PE-file mode: load the image from disk, resolve --fn, and run with the
    // PeImageProvider as the backing memory. No live process required.
    emu::host::PeLoadedImage pe;
    std::unique_ptr<emu::host::PeImageProvider> pe_mp;
    emu::host::Process proc;
    emu::host::WinProcessMemoryProvider* mp_live = nullptr;
    emu::MemoryProvider* mp = nullptr;
    if (!a.pe_path.empty()) {
        if (emu::Status st = emu::host::pe_load_file(a.pe_path.c_str(), a.pe_base, pe); emu::fail(st)) {
            std::fprintf(stderr, "tester: pe_load_file('%s') failed (%.*s)\n",
                         a.pe_path.c_str(),
                         static_cast<int>(emu::to_string(st).size()),
                         emu::to_string(st).data());
            emu::log::shutdown();
            return 3;
        }
        std::printf("pe: base=0x%016llx  size=%llu  exports=%zu\n",
                    (unsigned long long)pe.base, (unsigned long long)pe.size,
                    pe.exports.size());
        if (a.fn_name == "?list" || a.fn_name.rfind("?list:", 0) == 0) {
            std::string filter;
            if (a.fn_name.size() > 6) filter = a.fn_name.substr(6);
            int n = 0;
            for (auto& kv : pe.exports) {
                if (!filter.empty() && kv.first.find(filter) == std::string::npos) continue;
                std::printf("  %s -> 0x%016llx\n", kv.first.c_str(),
                            (unsigned long long)(pe.base + kv.second));
                ++n;
            }
            std::printf("(%d matches of %zu total exports)\n", n, pe.exports.size());
            emu::log::shutdown();
            return 0;
        }
        if (!a.fn_name.empty()) {
            emu::GuestAddr fn_addr = emu::host::pe_export_addr(pe, a.fn_name.c_str());
            if (fn_addr == 0) {
                std::fprintf(stderr, "tester: export '%s' not found\n", a.fn_name.c_str());
                emu::log::shutdown();
                return 5;
            }
            a.addr = fn_addr;
            std::printf("pe: '%s' resolved to 0x%016llx\n",
                        a.fn_name.c_str(), (unsigned long long)fn_addr);
        }
        pe_mp = std::make_unique<emu::host::PeImageProvider>(pe);
        mp = pe_mp.get();
    } else {
        if (emu::fail(proc.open(a.pid))) {
            std::fprintf(stderr, "tester: failed to open pid %u\n", a.pid);
            emu::log::shutdown();
            return 3;
        }
        mp_live = new emu::host::WinProcessMemoryProvider(proc);
        mp = mp_live;
    }

    if (!a.emulate) {
        // Dump-only mode: read N bytes and hex-dump.
        std::vector<emu::u8> code(a.bytes);
        if (emu::Status st = mp->read(a.addr, a.bytes, code.data()); emu::fail(st)) {
            std::fprintf(stderr, "tester: read failed (%.*s)\n",
                         static_cast<int>(emu::to_string(st).size()),
                         emu::to_string(st).data());
            emu::log::shutdown();
            return 4;
        }
        hex_dump(a.addr, code.data(), a.bytes);
        std::fflush(stdout);
        emu::log::flush();
        emu::log::shutdown();
        return 0;
    }

    // Emulation path: read chain  L1 -> L2 -> L3 -> ShadowPages -> RPM.
    // Write chain: straight into ShadowPages (cache layers handle invalidate).
    // Stack lives in a private region of ShadowPages -- never hits the live process.

    emu::ShadowPages shadow(mp);

    constexpr emu::GuestAddr kStackTop  = 0x0000'7000'0000'0000ull;
    constexpr emu::u64       kStackSize = 64 * 1024;
    const     emu::GuestAddr kStackBase = kStackTop - kStackSize;
    shadow.add_private_region(kStackBase, kStackSize);

    // ---- Minimal Win64 TEB / PEB / per-thread state -----------------------
    //
    // Set up enough of the TEB layout that ucrtbase's per-thread state walks
    // don't dereference garbage. Offsets per the documented NtCurrentTeb()
    // layout; we only populate what observed code paths read.
    //
    //   TEB                      @ 0x60000   (4 KiB, all zero except below)
    //     +0x30  NtTib.Self      -> 0x60000 (self-pointer)
    //     +0x58  TLS pointer     -> 0x62000 (TLS array, 64 KiB)
    //     +0x60  PEB pointer     -> 0x65000 (stub PEB)
    //   PEB stub                 @ 0x65000   (4 KiB, all zero)
    //   Per-thread state         @ 0x66000   (16 KiB) -- returned by FlsGetValue
    constexpr emu::GuestAddr kTebBase     = 0x0000'0000'0006'0000ull;
    constexpr emu::GuestAddr kTlsArrayBase= 0x0000'0000'0006'2000ull;
    constexpr emu::GuestAddr kPebBase     = 0x0000'0000'0006'5000ull;
    constexpr emu::GuestAddr kAcrtPtdBase = 0x0000'0000'0006'6000ull;
    shadow.add_private_region(kTebBase,      0x1000);
    shadow.add_private_region(kTlsArrayBase, 0x3000);
    shadow.add_private_region(kPebBase,      0x1000);
    shadow.add_private_region(kAcrtPtdBase,  0x4000);

    // Populate TEB.
    auto write_u64 = [&](emu::GuestAddr addr, emu::u64 v) {
        emu::u8 buf[8];
        for (int i = 0; i < 8; ++i) buf[i] = static_cast<emu::u8>((v >> (8 * i)) & 0xFFu);
        (void)shadow.write(addr, 8, buf);
    };
    write_u64(kTebBase + 0x30, kTebBase);            // NtTib.Self
    write_u64(kTebBase + 0x58, kTlsArrayBase);       // ThreadLocalStoragePointer
    write_u64(kTebBase + 0x60, kPebBase);            // ProcessEnvironmentBlock
    // GS base is set on `e.cpu()` further below, after `e` is constructed.

    // Input buffer for --input-str: lives at 0x10000000 as a private region.
    constexpr emu::GuestAddr kInputBase  = 0x0000'0000'1000'0000ull;
    constexpr emu::GuestAddr kInputBase2 = 0x0000'0000'1000'1000ull;
    if (!a.input_str.empty()) {
        shadow.add_private_region(kInputBase, 4096);
        // +1 for the trailing NUL so strlen has a stop condition.
        (void)shadow.write(kInputBase, a.input_str.size() + 1,
                           reinterpret_cast<const emu::u8*>(a.input_str.c_str()));
    }
    if (!a.input_str2.empty()) {
        shadow.add_private_region(kInputBase2, 4096);
        (void)shadow.write(kInputBase2, a.input_str2.size() + 1,
                           reinterpret_cast<const emu::u8*>(a.input_str2.c_str()));
    }

    emu::IatStubs iat;
    if (!a.pe_path.empty() && !pe.imports.empty()) {
        iat.install(shadow);
    }

    // --write-bytes "ADDR=HH HH HH ..." -- pre-write a small region of bytes
    // (e.g. an emulated comparator thunk). Allocates a private page covering
    // the addr if needed.
    if (!a.write_bytes_arg.empty()) {
        const std::string& s = a.write_bytes_arg;
        const auto eq = s.find('=');
        if (eq != std::string::npos) {
            emu::u64 wb_addr = std::strtoull(s.c_str(), nullptr, 0);
            (void)wb_addr; wb_addr = std::strtoull(s.c_str(), nullptr, 0);
            std::vector<emu::u8> bytes;
            for (size_t p = eq + 1; p < s.size(); ) {
                while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) ++p;
                if (p + 2 > s.size()) break;
                const char hex[3] = { s[p], s[p+1], 0 };
                bytes.push_back(static_cast<emu::u8>(std::strtoul(hex, nullptr, 16)));
                p += 2;
            }
            const emu::GuestAddr page = wb_addr & ~emu::u64{0xFFF};
            shadow.add_private_region(page, 4096);
            (void)shadow.write(wb_addr, bytes.size(), bytes.data());
            std::printf("write-bytes: %zu bytes at 0x%llx\n", bytes.size(),
                        (unsigned long long)wb_addr);
        }
    }

    emu::L3Cache l3(&shadow);
    emu::L2Cache l2(&l3);
    emu::L1Cache l1(&l2);

    emu::Emulator e;
    if (a.no_cache) {
        e.set_memory_read (&shadow);
        e.set_memory_write(&shadow);
    } else {
        // Write-through: routing writes through L1 makes each cache layer
        // forward + invalidate its own entries on the way down to shadow.
        e.set_memory_read (&l1);
        e.set_memory_write(&l1);
    }
    if (a.trace) e.set_trace(&trace_callback, nullptr);

    // Fault -> SEH RaiseException wiring. The recovery cb maps each fault
    // kind to a Windows exception code, then asks emu_raise_exception to
    // find the language handler covering the faulting RIP and prime it.
    // image_base for the .pdata lookup: pe.base in PE-file mode, or the
    // explicit --seh-image-base in live-process mode (Phase 9's GetModule
    // base trick belongs there but we keep it CLI-driven for now).
    struct SehUserData {
        emu::Emulator* e;
        emu::GuestAddr image_base;
    };
    static SehUserData seh_ud;
    seh_ud.e          = &e;
    seh_ud.image_base = !a.pe_path.empty() ? pe.base : a.seh_image_base;
    if (seh_ud.image_base != 0) {
        e.set_fault_recovery(+[](void* u, emu::Emulator& em, const emu::Fault& f) noexcept -> bool {
            auto* sd = static_cast<SehUserData*>(u);
            emu::u32 code = 0;
            switch (f.kind) {
                case emu::FaultKind::DivideError:      code = 0xC0000094; break;  // INT_DIVIDE_BY_ZERO
                case emu::FaultKind::Breakpoint:       code = 0x80000003; break;  // BREAKPOINT
                case emu::FaultKind::InvalidOpcode:    code = 0xC000001D; break;  // ILLEGAL_INSTRUCTION
                case emu::FaultKind::PageFault:        code = 0xC0000005; break;  // ACCESS_VIOLATION
                case emu::FaultKind::GeneralProtection: code = 0xC0000005; break;
                default: return false;
            }
            const emu::GuestAddr fault_rip = f.rip;
            em.cpu().set_rip(fault_rip);   // raise() reads cpu.rip()
            const emu::Status s = emu::emu_raise_exception(em, sd->image_base, code, fault_rip);
            return emu::ok(s);
        }, &seh_ud);
    }

    // Bind the TEB we set up above as this thread's GS base. ucrtbase needs
    // this to walk per-thread state via `mov rax, gs:[0x30]` and similar.
    if (a.gs_test) {
        // gs-test mode already sets gs_base from the live TEB.
    } else {
        e.cpu().set_gs_base(kTebBase);
    }

    // IAT stubs: bind every named import in the PE to a synthetic stub
    // address and register handlers for the most common ntdll/kernel32
    // calls. The dispatcher will route RIPs in the stub region here.
    if (!a.pe_path.empty() && !pe.imports.empty()) {
        // Reserve a 16 MiB bump-heap region (shared by all alloc stubs) so
        // pointers returned to the emulated code are actually readable.
        constexpr emu::GuestAddr kBumpHeapBase = 0x0000'0000'2000'0000ull;
        constexpr emu::u64       kBumpHeapSize = 16ull * 1024 * 1024;
        shadow.add_private_region(kBumpHeapBase, kBumpHeapSize);

        iat.set_default_stub(+[](emu::Cpu& cpu, const char* dll, const char* name) noexcept {
            LOG_INFO("iat-stub default: %s!%s -> RAX=0", dll, name);
            cpu.set_r64(emu::reg::RAX, 0);
        });

        // Register stubs by *name only* (empty dll) so they cover both the
        // ntdll.dll and api-ms-*.dll redirector variants ucrtbase imports.
        iat.set_stub("", "GetLastError",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.set_r64(emu::reg::RAX, 0);
            });
        iat.set_stub("", "SetLastError",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.set_r64(emu::reg::RAX, 0);
            });
        iat.set_stub("", "RtlSetLastWin32Error",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.set_r64(emu::reg::RAX, 0);
            });
        // Free-list heap shared by HeapAlloc/HeapFree/RtlAllocateHeap/RtlFreeHeap.
        // Caller passes size in R8 (Win64 ABI 3rd arg); we ignore the heap
        // handle and flags. HeapFree gets the pointer in R8.
        static emu::StubHeap s_heap;
        s_heap.install(kBumpHeapBase, kBumpHeapSize);
        iat.set_stub("", "HeapAlloc",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                const emu::u64 sz = cpu.r(emu::reg::R8);
                cpu.set_r64(emu::reg::RAX, s_heap.alloc(sz));
            });
        iat.set_stub("", "RtlAllocateHeap",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                const emu::u64 sz = cpu.r(emu::reg::R8);
                cpu.set_r64(emu::reg::RAX, s_heap.alloc(sz));
            });
        iat.set_stub("", "HeapFree",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                const emu::GuestAddr p = cpu.r(emu::reg::R8);
                cpu.set_r64(emu::reg::RAX, s_heap.free(p) ? 1 : 0);
            });
        iat.set_stub("", "RtlFreeHeap",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                const emu::GuestAddr p = cpu.r(emu::reg::R8);
                cpu.set_r64(emu::reg::RAX, s_heap.free(p) ? 1 : 0);
            });
        // HeapReAlloc / RtlReAllocateHeap: simplest semantics -- allocate a
        // fresh block and "copy" by leaving the bytes alone (we don't know
        // the old size). Real code shouldn't depend on this for correctness
        // but should work for typical ucrtbase paths that don't expand.
        iat.set_stub("", "HeapReAlloc",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                // ucrt calls HeapReAlloc(heap, flags, p, new_size); new_size in R9.
                const emu::u64 sz = cpu.r(emu::reg::R9);
                cpu.set_r64(emu::reg::RAX, s_heap.alloc(sz));
            });
        iat.set_stub("", "RtlReAllocateHeap",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                const emu::u64 sz = cpu.r(emu::reg::R9);
                cpu.set_r64(emu::reg::RAX, s_heap.alloc(sz));
            });
        iat.set_stub("", "HeapSize",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                // We don't track size-per-pointer in this minimal heap; return
                // 0xFFFFFFFF (invalid per docs) to signal "unknown".
                cpu.set_r64(emu::reg::RAX, 0xFFFFFFFFFFFFFFFFull);
            });
        iat.set_stub("", "GetProcessHeap",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.set_r64(emu::reg::RAX, 0xDEAD'F00Dull);   // opaque handle
            });
        // Identity / process / thread.
        iat.set_stub("", "GetCurrentProcessId",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.set_r64(emu::reg::RAX, 0x1234);
            });
        iat.set_stub("", "GetCurrentThreadId",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.set_r64(emu::reg::RAX, 0x5678);
            });
        iat.set_stub("", "GetCurrentProcess",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.set_r64(emu::reg::RAX, 0xFFFFFFFFFFFFFFFFull);   // pseudo-handle -1
            });
        iat.set_stub("", "GetCurrentThread",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.set_r64(emu::reg::RAX, 0xFFFFFFFFFFFFFFFEull);   // pseudo-handle -2
            });
        // Time. Deterministic counters so emulations are reproducible.
        iat.set_stub("", "GetSystemTimeAsFileTime",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                static emu::u64 t = 132'000'000'000'000'000ull;   // ~2019 in FILETIME
                t += 100'000;                                     // +10ms per call
                const emu::GuestAddr out = cpu.r(emu::reg::RCX);
                if (auto* mp = cpu.mem_write(); mp != nullptr) {
                    emu::u8 buf[8];
                    for (int i = 0; i < 8; ++i) buf[i] = static_cast<emu::u8>((t >> (8 * i)) & 0xFFu);
                    (void)mp->write(out, 8, buf);
                }
                cpu.set_r64(emu::reg::RAX, 0);
            });
        iat.set_stub("", "GetTickCount",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                static emu::u32 ticks = 0;
                ticks += 16;     // ~60 Hz
                cpu.set_r64(emu::reg::RAX, ticks);
            });
        iat.set_stub("", "GetTickCount64",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                static emu::u64 ticks = 0;
                ticks += 16;
                cpu.set_r64(emu::reg::RAX, ticks);
            });
        iat.set_stub("", "QueryPerformanceCounter",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                static emu::u64 ctr = 0;
                ctr += 1000;
                const emu::GuestAddr out = cpu.r(emu::reg::RCX);
                if (auto* mp = cpu.mem_write(); mp != nullptr) {
                    emu::u8 buf[8];
                    for (int i = 0; i < 8; ++i) buf[i] = static_cast<emu::u8>((ctr >> (8 * i)) & 0xFFu);
                    (void)mp->write(out, 8, buf);
                }
                cpu.set_r64(emu::reg::RAX, 1);     // BOOL TRUE
            });
        iat.set_stub("", "QueryPerformanceFrequency",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                const emu::GuestAddr out = cpu.r(emu::reg::RCX);
                if (auto* mp = cpu.mem_write(); mp != nullptr) {
                    const emu::u64 freq = 10'000'000ull;
                    emu::u8 buf[8];
                    for (int i = 0; i < 8; ++i) buf[i] = static_cast<emu::u8>((freq >> (8 * i)) & 0xFFu);
                    (void)mp->write(out, 8, buf);
                }
                cpu.set_r64(emu::reg::RAX, 1);
            });
        // TLS -- tiny in-emulator slot table. Each stub gets its own static.
        iat.set_stub("", "TlsAlloc",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                static emu::u32 next_slot = 0;
                cpu.set_r64(emu::reg::RAX, next_slot++);
            });
        iat.set_stub("", "TlsFree",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.set_r64(emu::reg::RAX, 1);
            });
        // TLS get/set use a static array keyed by (slot, fixed thread).
        // For slot 0 we return the kAcrtPtdBase region so __acrt_getptd has a
        // non-null per-thread state pointer to work with.
        iat.set_stub("", "TlsGetValue",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                static emu::u64 slots[256] = {};
                const emu::u32 slot = static_cast<emu::u32>(cpu.r(emu::reg::RCX));
                emu::u64 v = (slot < 256) ? slots[slot] : 0;
                if (v == 0 && slot < 4) v = kAcrtPtdBase;   // sentinel default
                cpu.set_r64(emu::reg::RAX, v);
            });
        iat.set_stub("", "TlsSetValue",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                static emu::u64 slots[256] = {};
                const emu::u32 slot = static_cast<emu::u32>(cpu.r(emu::reg::RCX));
                const emu::u64 val  = cpu.r(emu::reg::RDX);
                if (slot < 256) slots[slot] = val;
                cpu.set_r64(emu::reg::RAX, 1);
            });
        iat.set_stub("", "FlsGetValue",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                static emu::u64 slots[256] = {};
                const emu::u32 slot = static_cast<emu::u32>(cpu.r(emu::reg::RCX));
                emu::u64 v = (slot < 256) ? slots[slot] : 0;
                if (v == 0 && slot < 4) v = kAcrtPtdBase;   // __acrt_getptd target
                cpu.set_r64(emu::reg::RAX, v);
            });
        iat.set_stub("", "FlsSetValue",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                static emu::u64 slots[256] = {};
                const emu::u32 slot = static_cast<emu::u32>(cpu.r(emu::reg::RCX));
                const emu::u64 val  = cpu.r(emu::reg::RDX);
                if (slot < 256) slots[slot] = val;
                cpu.set_r64(emu::reg::RAX, 1);
            });
        iat.set_stub("", "FlsAlloc",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                static emu::u32 next = 0;
                cpu.set_r64(emu::reg::RAX, next++);
            });
        // Module / proc lookup -- return non-zero handles, NULL function ptrs.
        iat.set_stub("", "GetModuleHandleW",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.set_r64(emu::reg::RAX, 0xCAFE'BABE'0001ull);
            });
        iat.set_stub("", "GetModuleHandleA",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.set_r64(emu::reg::RAX, 0xCAFE'BABE'0002ull);
            });
        iat.set_stub("", "GetProcAddress",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.set_r64(emu::reg::RAX, 0);     // signals "not found"
            });
        iat.set_stub("", "LoadLibraryW",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.set_r64(emu::reg::RAX, 0xCAFE'BABE'0003ull);
            });
        iat.set_stub("", "LoadLibraryA",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.set_r64(emu::reg::RAX, 0xCAFE'BABE'0004ull);
            });
        // Randomness -- deterministic xorshift, fills the buffer in RCX.
        iat.set_stub("", "BCryptGenRandom",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                static emu::u64 s = 0x9E3779B97F4A7C15ull;
                const emu::GuestAddr buf = cpu.r(emu::reg::RDX);
                const emu::u32       len = static_cast<emu::u32>(cpu.r(emu::reg::R8));
                if (auto* mp = cpu.mem_write(); mp != nullptr) {
                    emu::u8 b[64];
                    for (emu::u32 off = 0; off < len; ) {
                        const emu::u32 chunk = (len - off) < 64 ? (len - off) : 64;
                        for (emu::u32 i = 0; i < chunk; ++i) {
                            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                            b[i] = static_cast<emu::u8>(s & 0xFFu);
                        }
                        (void)mp->write(buf + off, chunk, b);
                        off += chunk;
                    }
                }
                cpu.set_r64(emu::reg::RAX, 0);     // STATUS_SUCCESS
                (void)buf;
            });
        iat.set_stub("", "RtlGenRandom",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                static emu::u64 s = 0xCBF29CE484222325ull;
                const emu::GuestAddr buf = cpu.r(emu::reg::RCX);
                const emu::u32       len = static_cast<emu::u32>(cpu.r(emu::reg::RDX));
                if (auto* mp = cpu.mem_write(); mp != nullptr) {
                    emu::u8 b[64];
                    for (emu::u32 off = 0; off < len; ) {
                        const emu::u32 chunk = (len - off) < 64 ? (len - off) : 64;
                        for (emu::u32 i = 0; i < chunk; ++i) {
                            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                            b[i] = static_cast<emu::u8>(s & 0xFFu);
                        }
                        (void)mp->write(buf + off, chunk, b);
                        off += chunk;
                    }
                }
                cpu.set_r64(emu::reg::RAX, 1);   // TRUE
            });
        // ucrtbase internal initialization no-ops.
        iat.set_stub("", "InitializeSListHead",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.set_r64(emu::reg::RAX, 0);
            });
        iat.set_stub("", "InitializeCriticalSectionEx",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.set_r64(emu::reg::RAX, 1);
            });
        iat.set_stub("", "InitializeCriticalSection",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.set_r64(emu::reg::RAX, 0);
            });
        iat.set_stub("", "EnterCriticalSection",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.set_r64(emu::reg::RAX, 0);
            });
        iat.set_stub("", "LeaveCriticalSection",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.set_r64(emu::reg::RAX, 0);
            });
        iat.set_stub("", "DeleteCriticalSection",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.set_r64(emu::reg::RAX, 0);
            });
        iat.set_stub("", "EncodePointer",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.set_r64(emu::reg::RAX, cpu.r(emu::reg::RCX));   // identity
            });
        iat.set_stub("", "DecodePointer",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.set_r64(emu::reg::RAX, cpu.r(emu::reg::RCX));
            });
        // IsProcessorFeaturePresent -- say "yes" for everything to keep
        // ucrtbase from gating SSE2-fastpath behind a runtime check.
        iat.set_stub("", "IsProcessorFeaturePresent",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.set_r64(emu::reg::RAX, 1);
            });
        iat.set_stub("", "IsDebuggerPresent",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.set_r64(emu::reg::RAX, 0);
            });
        // Process control -- these halt cleanly so the emulation can wrap up.
        iat.set_stub("", "ExitProcess",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.halt();
            });
        iat.set_stub("", "TerminateProcess",
            +[](emu::Cpu& cpu, const char*, const char*) noexcept {
                cpu.halt();
            });

        for (const auto& imp : pe.imports) {
            (void)iat.bind(shadow, imp.iat_addr, imp.dll, imp.name);
        }
        e.attach_iat_stubs(&iat);
        std::printf("iat: bound %zu imports\n", pe.imports.size());
    }

    if (!a.no_cache) {
        // One bulk RPM into L3 for the 2 MiB region containing the entry.
        (void)l3.prefetch_region(a.addr);
    }

    if (a.gs_test) {
        emu::GuestAddr teb = 0;
        if (emu::Status st = emu::host::query_thread_teb(a.pid, a.tid, teb); emu::fail(st)) {
            std::fprintf(stderr, "tester: query_thread_teb failed (%.*s)\n",
                         static_cast<int>(emu::to_string(st).size()),
                         emu::to_string(st).data());
            emu::log::shutdown();
            return 8;
        }
        LOG_INFO("gs-test: TEB base = 0x%llx", static_cast<unsigned long long>(teb));
        e.cpu().set_gs_base(teb);
    } else if (a.entity_test) {
        // Pass array base in RCX, count in RDX (Win64 ABI second arg).
        e.cpu().set_r64(emu::reg::RCX, a.arr_addr);
        e.cpu().set_r64(emu::reg::RDX, a.arr_count);
        e.cpu().set_r64(emu::reg::RAX, 0);
    } else if (!a.input_str.empty()) {
        // RCX points at the buffer we just wrote; optional RDX/R8/R9 for multi-arg fns.
        e.cpu().set_r64(emu::reg::RCX, kInputBase);
        if (a.rdx_set)             e.cpu().set_r64(emu::reg::RDX, a.rdx_value);
        else if (!a.input_str2.empty())
                                   e.cpu().set_r64(emu::reg::RDX, kInputBase2);
        if (a.input_count != 0)    e.cpu().set_r64(emu::reg::R8,  a.input_count);
        if (a.r9_set)              e.cpu().set_r64(emu::reg::R9,  a.r9_value);
        e.cpu().set_r64(emu::reg::RAX, 0);
    } else {
        e.cpu().set_r64(emu::reg::RCX, a.seed);
        e.cpu().set_r64(emu::reg::RAX, 0);
    }

    // SMC support: when running selfmod-test, register both the function
    // bytes AND the RWX thunk page as code ranges, then hook ShadowPages
    // writes to invalidate the block cache on any overlapping store. The
    // thunk page is the one actually being self-modified.
    if (a.selfmod_test) {
        shadow.add_code_range(a.addr, a.bytes);
        if (a.thunk_addr != 0) {
            const emu::u64 tsz = (a.thunk_size > 0) ? a.thunk_size : 4096;
            shadow.add_code_range(a.thunk_addr, tsz);
        }
        shadow.set_write_listener(
            +[](void* user, emu::GuestAddr w_addr, emu::usize w_size) noexcept {
                auto* emul = reinterpret_cast<emu::Emulator*>(user);
                emul->block_cache().invalidate_range(w_addr, w_size);
            },
            &e);
    }

    // Set up a guest stack. We push a sentinel return address so the
    // outermost RET cleanly halts the dispatcher.
    constexpr emu::GuestAddr kSentinel = 0xDEAD'BEEF'DEAD'BEEFull;
    emu::GuestAddr rsp = kStackTop;
    rsp -= 8;
    {
        emu::u8 buf[8];
        for (int i = 0; i < 8; ++i) buf[i] = static_cast<emu::u8>((kSentinel >> (8 * i)) & 0xFFu);
        (void)shadow.write(rsp, 8, buf);
    }
    e.cpu().set_r64(emu::reg::RSP, rsp);
    e.add_stop_addr(kSentinel);

    emu::RunResult r{};
    if (a.bench == 0) {
        r = e.run(a.addr, /*max_insns=*/16'000'000);
    } else {
        // Bench mode: re-run the entry function N times, resetting RSP and arg
        // before each call. Reports ns/call so we can verify the sub-ms claim.
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (emu::u64 i = 0; i < a.bench; ++i) {
            // Reset stack frame.
            e.cpu().set_r64(emu::reg::RSP, rsp);
            e.cpu().set_r64(emu::reg::RCX, a.seed);
            e.cpu().set_r64(emu::reg::RAX, 0);
            e.cpu().unhalt();
            e.cpu().clear_branch();
            r = e.run(a.addr, 65536);
            if (emu::fail(r.status)) break;
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        std::printf("bench: %llu calls in %lld ns  (%.1f ns/call)\n",
                    static_cast<unsigned long long>(a.bench),
                    static_cast<long long>(ns),
                    static_cast<double>(ns) / static_cast<double>(a.bench));
    }

    std::printf("emu: status=%.*s insns=%llu rip=0x%llx rax=0x%016llx\n",
                static_cast<int>(emu::to_string(r.status).size()),
                emu::to_string(r.status).data(),
                static_cast<unsigned long long>(r.insns_executed),
                static_cast<unsigned long long>(r.final_rip),
                static_cast<unsigned long long>(r.final_rax));
    std::printf("block_cache: %llu hit / %llu miss / %llu blocks\n",
                (unsigned long long)e.block_cache().hits(),
                (unsigned long long)e.block_cache().misses(),
                (unsigned long long)e.block_cache().size());
    std::printf("btc: %llu hit / %llu miss\n",
                (unsigned long long)e.btc().hits(),
                (unsigned long long)e.btc().misses());
    if (!a.no_cache) {
        std::printf("cache: L1 %llu hit / %llu miss   L2 %llu hit / %llu miss   L3 %llu hit / %llu miss   L3 fetches=%llu\n",
                    (unsigned long long)l1.hits(), (unsigned long long)l1.misses(),
                    (unsigned long long)l2.hits(), (unsigned long long)l2.misses(),
                    (unsigned long long)l3.hits(), (unsigned long long)l3.misses(),
                    (unsigned long long)l3.fetches());
    }

    int rc = 0;
    if (a.have_expected) {
        const bool ok = (r.final_rax == a.expected);
        std::printf("gate: %s  (expected=0x%016llx, got=0x%016llx)\n",
                    ok ? "PASS" : "FAIL",
                    static_cast<unsigned long long>(a.expected),
                    static_cast<unsigned long long>(r.final_rax));
        if (!ok) rc = 6;
    }
    if (emu::fail(r.status)) rc = 7;

    delete mp_live;
    emu::log::flush();
    emu::log::shutdown();
    return rc;
}
