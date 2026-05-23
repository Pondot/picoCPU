#!/usr/bin/env bash
# Real-binary validation suite -- emulate exports of C:\Windows\System32\ucrtbase.dll.
#
# Each gate calls a real CRT function from a real Windows DLL, on disk, with
# the emulator's PE-loader. No live target.exe needed.

PE=/c/Windows/System32/ucrtbase.dll
TESTER=./build/Release/bin/tester.exe

run() {
    local name="$1"; shift
    local out
    out=$("$TESTER" --pe-file "$PE" "$@" 2>&1)
    if echo "$out" | grep -q "gate: PASS"; then
        echo "  PASS  $name"
    elif echo "$out" | grep -q "gate: FAIL"; then
        echo "  FAIL  $name"
        echo "$out" | grep -E "(gate|status=)" | sed 's/^/    /'
    else
        # No --expected given; just print RAX.
        local rax
        rax=$(echo "$out" | grep -oE "rax=0x[0-9a-fA-F]+" | head -1)
        echo "  OK?   $name  ($rax)"
    fi
}

echo "=== Real-binary gates against ucrtbase.dll ==="
run "strlen('Hello, Emulator!')"   --fn strlen  --input-str "Hello, Emulator!"          --expected 16
run "strlen('')"                   --fn strlen  --input-str ""                          --expected 0
run "strlen('A')"                  --fn strlen  --input-str "A"                         --expected 1
run "strlen(short=6)"              --fn strlen  --input-str "Hello!"                    --expected 6
run "strlen(56-char string)"       --fn strlen  --input-str "The quick brown fox jumps over the lazy dog. 0123456789." --expected 56
run "strchr('Hello', 'l')"         --fn strchr  --input-str "Hello"    --rdx 0x6C       --expected 0x10000002
run "strchr('Hello', 'X') (none)"  --fn strchr  --input-str "Hello"    --rdx 0x58       --expected 0
run "strchr('Hello', NUL)"         --fn strchr  --input-str "Hello"    --rdx 0          --expected 0x10000005
run "strstr('Hello, World!', 'World')"  --fn strstr --input-str "Hello, World!" --input-str2 "World"      --expected 0x10000007
run "memchr('Hello, World!', 'o', 13)"  --fn memchr --input-str "Hello, World!" --rdx 0x6F --input-count 13 --expected 0x10000004
run "memcmp(equal)"                --fn memcmp  --input-str "Hello" --input-str2 "Hello"    --input-count 5 --expected 0
run "strcmp(equal)"                --fn strcmp  --input-str "Hello" --input-str2 "Hello"                    --expected 0
run "tolower('Z')"                 --fn tolower --seed 0x5A                              --expected 0x7A
run "tolower('A')"                 --fn tolower --seed 0x41                              --expected 0x61
run "atoi('12345')"                --fn atoi    --input-str "12345"                      --expected 12345
run "abs(-42)"                     --fn abs     --seed 0xFFFFFFFFFFFFFFD6                --expected 0x2A
run "abs(42)"                      --fn abs     --seed 0x2A                              --expected 0x2A
run "abs(0)"                       --fn abs     --seed 0                                 --expected 0
run "abs(INT_MIN)"                 --fn abs     --seed 0x80000000                        --expected 0x80000000

echo
echo "=== Relocated base ==="
run "strlen at 0x300000000"        --base 0x300000000 --fn strlen --input-str "Hello!"   --expected 6
run "strchr at 0x440000000"        --base 0x440000000 --fn strchr --input-str "Hello" --rdx 0x6C --expected 0x10000002

echo
echo "=== IAT-stub-dependent ==="
run "malloc(64) -> stub HeapAlloc"  --fn malloc  --seed 64
run "calloc(8,16)"                 --fn calloc  --seed 8 --rdx 16
run "_strdup('clone me')"          --fn _strdup --input-str "clone me"
