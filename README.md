# setback

`no_std` failure recovery for Rust built on C `setjmp`/`longjmp`, with all of
the jump confined to C frames. Set a mark, run a closure, and get `Ok(value)`
back - or, if a stack overflow, OOM, or explicit trigger fires while it runs,
get `Err(RecoveryError)` with the abandoned stack leaked (no destructors run).

It targets mostly-pure computations - parsers, interpreters, deserializers -
that might recurse too deep or exhaust the heap on hostile input, where you want
a clean error on the faulting task instead of a system reset. Built for
bare-metal / RTOS targets, runs on the host too, though modern machines with
MMUs should not need this crate.

## Capabilities

- One call protects a scope. `protect(tid, f)` runs `f` and returns
  `Result<R, RecoveryError>`. Nesting works and recovery resolves to the
  innermost scope for a thread.
- One shared fault handler recovers any task by id: `recover(tid, cause)` jumps
  into that thread's innermost active scope. `cause` is any `i32` whose
  meaning you choose. It resurfaces as `RecoveryError::cause`.
- `no_std` and `no_alloc`. A single `static` intrusive list keyed by thread id
  holds the active marks, based on [`critical-section`].
- A recovery-stack gap (`RECOVERY_GAP_BYTES`) is reserved below each mark so a
  fault handler always has stack to run `recover` on.

[`critical-section`]: https://docs.rs/critical-section/latest/critical_section/

## How it works

All `setjmp`/`longjmp` lives in `src/setback.c`. A `longjmp` unwinds that one C
frame back to its `setjmp`, abandoning (and leaking) the Rust frames above it.
The full rationale, contract, and recovery-stack guarantee are in the
API docs - see `protect` and `recover`.

## Usage

```rust
use setback::{protect, ThreadId, RecoveryError};

// Config is large and deeply nested. Hostile JSON can recurse serde past the
// stack limit or exhaust the heap.
fn parse_config(
    tid: ThreadId,
    bytes: &[u8],           // &[u8] is UnwindSafe, nothing escapes
) -> Result<Result<Config, serde_json::Error>, RecoveryError> {
    unsafe { protect(tid, || serde_json::from_slice(bytes)) }
}
```

## Caveats

- `recover` must be called on (or for) the faulting thread, whose `protect`
  frame must still be live.
- On embedded targets, recovering from a fault handler into thread-mode code needs
  glue to resume at a thunk that runs the jump in thread mode (`longjmp` is a plain
  branch, not an exception return). `recover` finds the mark, the mode
  transition is yours.

## Simplified wiring example (you provide the fault handler, stack guard, and CS impl)

```rust
use setback::{protect, recover, ThreadId};

const STACK_OVERFLOW: i32 = 1;
const OOM: i32 = 2;

// 1. cortex-m provides the critical-section impl (single-core):
//    cortex-m = { version = "0.7", features = ["critical-section-single-core"] }
// 2. Put an MPU guard region (or PSPLIM on Armv8-M) below each task stack so an
//    overflow faults instead of corrupting RAM.

// OOM fires in THREAD mode: call recover directly.
#[alloc_error_handler]
fn oom(_: core::alloc::Layout) -> ! {
    unsafe { let _ = recover(rtos_current_task() as ThreadId, OOM); }
    loop {}
}

// Stack overflow fires in HANDLER MODE on a broken stack. The handler only
// redirects: rewrite the stacked PSP frame so the exception return resumes at
// overflow_trampoline in thread mode (see `recover` Safety for the full recipe).
unsafe fn on_stack_overflow() {
    let f = __get_PSP() as *mut u32;
    *f.add(0) = rtos_current_task() as u32;                        // r0 = tid
    *f.add(1) = STACK_OVERFLOW as u32;                             // r1 = cause
    *f.add(2) = (rtos_stack_bottom() + RECOVERY_GAP_BYTES) as u32; // r2 = safe SP
    *f.add(6) = overflow_trampoline as u32;                        // PC
}

// Reset SP (the overflowed stack is unusable) then enter thread-mode Rust.
#[naked]
extern "C" fn overflow_trampoline(tid: ThreadId, cause: i32) -> ! {
    unsafe { core::arch::asm!("mov sp, r2", "b overflow_enter", options(noreturn)) }
}

extern "C" fn overflow_enter(tid: ThreadId, cause: i32) -> ! {
    unsafe { let _ = recover(tid, cause); }
    loop { cortex_m::asm::bkpt() }
}
```

## Testing

- Host tests: `cargo test --features std` (pulls in critical-section's
  std-backed impl so the registry locking links).
