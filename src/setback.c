/*
 * setback.c - the ONLY translation unit that touches setjmp/longjmp.
 *
 * The caller-facing contract lives in `protect`/`recover` (lib.rs). This file
 * only upholds the invariants that make setjmp/longjmp safe to drive from C:
 *
 *  * setjmp() runs in C, never Rust, and only as a controlling
 *    expression (C11 7.13.1.1p4), its result is never stored.
 *  * No local of the setjmp frame is written after setjmp returns or read on
 *    the resume path, so none can come back indeterminate (C11 7.13.2.1p3).
 *  * longjmp() unwinds only this C frame back to its setjmp; the abandoned Rust
 *    frames above it are leaked by `protect`'s contract.
 *  * The mark is armed only while the setjmp frame and the recovery stack
 *    below it are both live.
 */

#include <setjmp.h>
#include <stddef.h>
#include <stdint.h>

/* Returned to Rust by setback_call as an int32_t: OK if the trampoline
 * completed, RECOVERED if a longjmp came back. `int` is 16 bits on some
 * targets, so the width is pinned to match Rust's `i32` everywhere.
 * The cause code travels out of band in the Rust Mark. */
#define SETBACK_OK 0
#define SETBACK_RECOVERED 1

/* Must equal RECOVERY_STACK_BYTES in lib.rs. */
#define SETBACK_RECOVERY_STACK_BYTES 64
/* The strictest SP alignment among the supported ABIs. */
#define SETBACK_RECOVERY_STACK_ALIGN 16

_Static_assert(SETBACK_RECOVERY_STACK_BYTES % SETBACK_RECOVERY_STACK_ALIGN == 0,
               "the recovery stack top must stay SP-aligned");

struct setback_recovery_stack {
  _Alignas(SETBACK_RECOVERY_STACK_ALIGN) uint8_t bytes[SETBACK_RECOVERY_STACK_BYTES];
};

size_t setback_jmpbuf_size(void) { return sizeof(jmp_buf); }
size_t setback_jmpbuf_align(void) { return _Alignof(jmp_buf); }

/*
 * Lay the recovery stack down below the setjmp mark, arm the mark with its top,
 * run the closure, disarm. noinline: the frame holding `rs` must be established
 * after setback_call's setjmp so it sits below the mark.
 *
 * The probe touches the lowest byte while still unarmed, so a guard that faults
 * precisely (MPU, PSPLIM, PMP) catches short headroom before there is a mark to
 * jump into.
 */
__attribute__((noinline)) static void
setback_run_below_recovery_stack(void (*tramp)(void *), void *data,
                                 uintptr_t *top) {
  volatile struct setback_recovery_stack rs;
  rs.bytes[0] = 0;
  /* C11 does not order a volatile access against a relaxed atomic. */
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
  __atomic_store_n(top, (uintptr_t)&rs.bytes[SETBACK_RECOVERY_STACK_BYTES],
                   __ATOMIC_RELAXED);
  /* Keeps arm and disarm on their side of the call if `tramp` gets inlined. */
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
  tramp(data);
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
  /* Disarm while the setjmp frame is still live. */
  __atomic_store_n(top, 0, __ATOMIC_RELAXED);
}

/*
 * Set the recovery mark, then call the Rust trampoline below the recovery stack.
 *
 * jb    : Rust-owned storage of >= setback_jmpbuf_size() bytes.
 * top   : Rust-owned `uintptr_t`: 0 while unarmed, else the recovery stack top.
 * tramp : extern "C" Rust fn running the closure.
 * data  : opaque payload threaded to the trampoline.
 *
 * Returns SETBACK_OK on completion, SETBACK_RECOVERED if a longjmp came back
 * here. noinline so the Rust call site cannot be reordered in a way that defeats
 * the returns_twice handling.
 */
__attribute__((noinline)) int32_t setback_call(void *jb,
                                               uintptr_t *top,
                                               void (*tramp)(void *),
                                               void *data) {
  jmp_buf *env = (jmp_buf *)jb;

  /* setjmp as an `if` controlling expression (legal per C11 7.13.1.1p4). We only
   * need first return (0) vs longjmp-resume (nonzero). The cause travels in the
   * Mark. */
  if (setjmp(*env) == 0) {
    setback_run_below_recovery_stack(tramp, data, top);
    return SETBACK_OK;
  }

  /* Reached via longjmp; the abandoned Rust frames are leaked by contract. */
  return SETBACK_RECOVERED;
}

__attribute__((noreturn)) void setback_longjmp(void *jb) {
  jmp_buf *env = (jmp_buf *)jb;
  longjmp(*env, SETBACK_RECOVERED);
}
