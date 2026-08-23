/*
 * setback.c - the ONLY translation unit that touches setjmp/longjmp.
 *
 * The caller-facing contract lives in `protect`/`recover` (lib.rs). This file
 * only upholds the invariants that make setjmp/longjmp safe to drive from C:
 *
 *  * setjmp() runs in C, never Rust, and only as a controlling
 *    expression (C11 7.13.1.1p4), its result is never stored.
 *  * No local of the setjmp frame is written after the mark is armed or read on
 *    the resume path, so none can come back indeterminate (C11 7.13.2.1p3).
 *  * longjmp() unwinds only this C frame back to its setjmp; the abandoned Rust
 *    frames above it are leaked by `protect`'s contract.
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

/* Stack reserved below the setjmp mark before the closure runs, so a fault
 * handler has room to run `recover` on abandoned frames - see `protect`'s
 * recovery-stack guarantee. Must equal RECOVERY_GAP_BYTES in lib.rs, multiple of 8. */
#define SETBACK_RECOVERY_GAP_BYTES 64

size_t setback_jmpbuf_size(void) { return sizeof(jmp_buf); }
size_t setback_jmpbuf_align(void) { return _Alignof(jmp_buf); }

/*
 * Run the closure with SETBACK_RECOVERY_GAP_BYTES reserved below the setjmp mark.
 *
 * Must be a separate noinline function: its frame (holding `gap`) is laid down
 * when it is called, after setback_call armed the mark - that ordering is what
 * puts the gap below the mark. `gap` is volatile and touched on both sides of
 * the call so the reservation materializes and stays live (no tail call pops it
 * early); the leading touch faults here, during setup, if headroom is already
 * short on a platform with a stack monitor or guard page.
 */
__attribute__((noinline)) static void
setback_run_with_gap(void (*tramp)(void *), void *data) {
  volatile unsigned char gap[SETBACK_RECOVERY_GAP_BYTES];
  gap[0] = 0;
  tramp(data);
  (void)gap[0];
}

/*
 * Arm the recovery mark, then call the Rust trampoline.
 *
 * jb    : Rust-owned storage of >= setback_jmpbuf_size() bytes.
 * armed : Rust-owned `uint8_t` set to 1 once the mark is usable. Exactly 8
 *         bits wide with no padding and alignment 1, so it matches Rust's
 *         `u8`/`AtomicU8` on every target where `uint8_t` exists at all.
 * tramp : extern "C" Rust fn running the closure.
 * data  : opaque payload threaded to the trampoline.
 *
 * Returns SETBACK_OK on completion, SETBACK_RECOVERED if a longjmp came back
 * here. noinline so the Rust call site cannot be reordered in a way that defeats
 * the returns_twice handling.
 */
__attribute__((noinline)) int32_t setback_call(void *jb,
                                               uint8_t *armed,
                                               void (*tramp)(void *),
                                               void *data) {
  jmp_buf *env = (jmp_buf *)jb;

  /* setjmp as an `if` controlling expression (legal per C11 7.13.1.1p4). We only
   * need armed (0) vs longjmp-resume (nonzero). The cause travels in the Mark. */
  if (setjmp(*env) == 0) {
    /* First return: publish the mark, only now usable, to the fault handler.
     * Then run the closure inside the gap frame, established after setjmp. */
    __atomic_store_n(armed, 1, __ATOMIC_RELEASE);
    setback_run_with_gap(tramp, data);
    return SETBACK_OK;
  }

  /* Reached via longjmp; the abandoned Rust frames are leaked by contract. */
  return SETBACK_RECOVERED;
}

__attribute__((noreturn)) void setback_longjmp(void *jb) {
  jmp_buf *env = (jmp_buf *)jb;
  longjmp(*env, SETBACK_RECOVERED);
}
