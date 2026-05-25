# picoCPU

x86-64 emulator. Reads a function's bytes from another process, runs them in
its own CPU, returns what the function would've returned. No injection, no
debugger attach, no writes to the target.


## What it can do

- Pull a function out of a running process and run it in isolation
- Same for static .exe / .dll files via the PE loader path
- Survive polymorphic and obfuscated code (control-flow flattening, opaque
  predicates, junk instruction padding)
- Run self-modifying code (block cache invalidates on writes to executable
  regions)
- ~250 instruction handlers covering ALU, bit ops, shifts, control flow,
  string ops, SSE, SSE2, SSSE3, SSE4.x, AES-NI, SHA-NI, AVX, AVX2, AVX-512
  (with masking + broadcast), BMI1/2, x87 + 80-bit soft-float add/sub/mul/div
- Hooks (block, code, memory r/w, instruction, fault)
- Hardware breakpoints (DR0-7 + DR6 status bits)
- Lazy EFLAGS (cc_op model)
- SEH walking via .pdata + UNWIND_INFO
- IAT stub layer for emulating functions that call into ntdll/kernel32

## Use cases

- Reverse engineering a single routine (RNG, hash, crypto, license check)
  without attaching a debugger
- Running suspicious code from a malware sample in isolation
- CTF / crackme work
- Reading game state by emulating the game's own getters externally
- Predicting RNG outputs by running the PRNG step outside the target
- Validating crypto test vectors pulled straight from a binary
- Studying VM-protected code (VMProtect/Themida-style dispatch loops) at
  the IR level


## Performance

Bench numbers from the integration tests:

```
target_fn (17 insns, simple mixer)
  100,000 calls in 51 ms      0.52 us/call

target_obfuscated_fn (189 insns, control-flow flattened)
  100,000 calls in 807 ms     8.07 us/call

Ackermann A(3,6) (172,000 recursive calls, ~2.4M emulated insns)
  142 ms,  17M insns/sec

Riemann zeta(2) with N=10000 (10K SSE FP loop iterations)
  31 ms,  ~7M insns/sec, returns bit-exact 0x3FFA513D881EF17A
```

L3 RPM fetches are usually 3 to 6 per function call. After that, every byte
the emulator reads comes out of L1.

build.bat finds these automatically via `vswhere` and `where`. If you need
to override, set `VCVARS`, `CMAKE`, or `NINJA` env vars before running it.

## License

MIT. See LICENSE.
