/*
 * setback.c - the ONLY translation unit that touches setjmp/longjmp.
 *
 * The caller-facing contract lives in `protect`/`recover` (lib.rs). This file
 * only upholds the invariants that make setjmp/longjmp safe to drive from C:
 *
 *  * setjmp() runs in C, never Rust, and only as a controlling
 *    expression (C11 7.13.1.1p4), its result is never stored.
 *  * Anything read after the jump is held in `volatile` temporaries, so longjmp
 *    cannot leave it in a clobbered caller-saved register.
 *  * longjmp() unwinds only this C frame back to its setjmp; the abandoned Rust
 *    frames above it are leaked by `protect`'s contract.
 */

#include <setjmp.h>
#include <stddef.h>

/* Returned to Rust by setback_call: OK if the trampoline completed, RECOVERED
 * if a longjmp came back. The cause code travels out of band in the Rust Mark. */
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
 * tramp : extern "C" Rust fn running the closure.
 * data  : opaque payload threaded to the trampoline.
 *
 * Returns SETBACK_OK on completion, SETBACK_RECOVERED if a longjmp came back
 * here. noinline so the Rust call site cannot be reordered in a way that defeats
 * the returns_twice handling.
 */
__attribute__((noinline)) int setback_call(void *jb,
                                           void (*tramp)(void *),
                                           void *data) {
  jmp_buf *env = (jmp_buf *)jb;

  /* volatile so a longjmp back into this frame finds these unchanged, rather
   * than in a caller-saved register longjmp clobbered. */
  void (*volatile vtramp)(void *) = tramp;
  void *volatile vdata = data;

  /* setjmp as an `if` controlling expression (legal per C11 7.13.1.1p4). We only
   * need armed (0) vs longjmp-resume (nonzero). The cause travels in the Mark. */
  if (setjmp(*env) == 0) {
    /* First return: mark armed. Run the closure inside the gap frame, which is
     * established now, after setjmp. */
    setback_run_with_gap(vtramp, vdata);
    return SETBACK_OK;
  }

  /* Reached via longjmp; the abandoned Rust frames are leaked by contract. */
  return SETBACK_RECOVERED;
}

__attribute__((noreturn)) void setback_longjmp(void *jb) {
  jmp_buf *env = (jmp_buf *)jb;
  longjmp(*env, SETBACK_RECOVERED);
}
