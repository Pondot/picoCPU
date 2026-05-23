# math_test

Standalone target for benchmarking an emulator. Two functions:

- `target_ackermann_fn(packed)` - deep recursion. RCX gets `(m << 32) | n`.
- `target_zeta_fn(N)` - SSE FP loop. RCX gets N (terms in sum of 1/i^2).

Both return their result in RAX (zeta returns the IEEE-754 bit pattern of
the double).

## Build

```
build.bat
```

Produces math_target.exe.

## Run

```
math_target.exe
```

It prints PID, function addresses, sizes, and the expected return values,
then sleeps 10 minutes so an external emulator has time to attach.

Example output:

```
pid=12345
ackermann_fn: addr=0x... size=80 input=0x0000000300000006 expected=0x00000000000001fd
zeta_fn: addr=0x... size=192 input=N=10000 expected=0x3ffa513d881ef17a
```

## What to verify

| Function | Input (RCX) | Expected RAX |
|---|---|---|
| ackermann | `0x0000000300000006` (= A(3,6)) | `0x00000000000001fd` (= 509) |
| zeta | `0x0000000000002710` (= N=10000) | `0x3ffa513d881ef17a` (~1.6448340718) |

Bit-exact match = pass.

## What it tests

Ackermann: CALL/RET, deep recursion (~172k nested calls), stack
push/pop. Takes ~2.4M emulated insns to complete.

Zeta: cvtsi2sd, mulsd, divsd, addsd in a tight 10k-iteration loop. ~220k
emulated insns. The "bit-exact double" requirement means the emulator's
SSE2 FP semantics have to match real hardware to the LSB.
