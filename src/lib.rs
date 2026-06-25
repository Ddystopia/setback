#![no_std]
#![allow(unsafe_op_in_unsafe_fn)]

/*!
# `setback`: setjmp/longjmp failure recovery, confined to C

[`protect`] runs a closure and returns `Ok(value)` on normal completion, or
`Err(RecoveryError)` if a `longjmp` - triggered by a stack-overflow fault
handler, an out-of-memory handler, or explicit user code via [`recover`] -
abandons the closure's stack. Everything on the abandoned stack is leaked: no
`Drop` runs. See [`protect`] for the full safety contract.

## How it works

All `setjmp`/`longjmp` lives in a tiny C file (`setback.c`): rustc does not support
`setjmp`/`longjmp`, so calling `setjmp` from Rust risks miscompilation. Rust hands
C a data pointer and an `extern "C"` trampoline, C arms the mark and calls the
trampoline, which runs the closure. A `longjmp` resets the stack pointer to
that `setjmp`, jumping over every live Rust frame above it - the trampoline, the
closure, and its whole call tree - and abandons them where they sit. The jump
stops at the C frame, and [`protect`] returns `Err(RecoveryError)`.

An uncaught panic crossing the `extern "C"` trampoline aborts (Rust 1.81+)
rather than entering C.

## One global registry, keyed by thread id

The crate owns a single `static` intrusive doubly-linked list of active marks.
Each [`protect`] call links one node, tagged with the caller's [`ThreadId`], and
unlinks it on exit. One shared fault handler, given the *faulting* thread's id,
calls [`recover`] to find that thread's innermost active mark and jump into it.
The link/unlink runs inside a [`critical_section`], the protected closure runs
outside it. You supply the [`critical-section`] impl in the final binary.

[`critical-section`]: https://docs.rs/critical-section/latest/critical_section/

*/

#[cfg(target_family = "wasm")]
compile_error!("`setback` does not support wasm targets");

use core::cell::UnsafeCell;
use core::convert::Infallible;
use core::error::Error;
use core::ffi::c_void;
use core::mem::{ManuallyDrop, MaybeUninit};
use core::panic::UnwindSafe;
use core::ptr;

/// Identifier the caller uses to tag a `protect` scope and that the fault
/// handler uses to find it again. Cast your RTOS task handle / index to `usize`.
pub type ThreadId = usize;

/// Wrap a capture (or a whole closure) to assert it is unwind-safe if needed,
/// satisfying the [`UnwindSafe`] bound on [`protect`]. Safe in itself, you
/// should still fulfill the safety contract of [`protect`] when the closure runs.
pub use core::panic::AssertUnwindSafe;

/// Returned by [`protect`] when the closure's stack was abandoned by a `longjmp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecoveryError {
    /// The code the caller of [`recover`] chose for this abandonment.
    /// `setback` assigns it no meaning. You decide what each value stands for.
    pub cause: i32,
}

/// Returned by [`recover`] when the given `tid` has no active [`protect`] scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecoveryFailure;

unsafe extern "C" {
    fn setback_jmpbuf_size() -> usize;
    fn setback_jmpbuf_align() -> usize;
    fn setback_call(
        jb: *mut c_void,
        tramp: unsafe extern "C" fn(*mut c_void),
        data: *mut c_void,
    ) -> i32;
    fn setback_longjmp(jb: *mut c_void) -> !;
}

const SETBACK_OK: i32 = 0;

/// Bytes of stack that [`protect`] reserves below the recovery mark before it
/// runs the closure - the gap a fault handler may rely on when choosing where
/// to run [`recover`]. See the "Recovery-stack guarantee" on [`protect`].
//
// Must stay equal to `SETBACK_RECOVERY_GAP_BYTES` in `setback.c`.
pub const RECOVERY_GAP_BYTES: usize = 64;

/// Backing storage for one C `jmp_buf`. 512 bytes / 16-byte alignment covers
/// every mainstream target. The constructor asserts it.
#[repr(C, align(16))]
struct JmpBufStorage {
    bytes: UnsafeCell<MaybeUninit<[u8; 512]>>,
}

struct Mark {
    tid: ThreadId,
    jmpbuf: JmpBufStorage,
    prev: *mut Mark,
    next: *mut Mark,
    /// Written by [`recover`] just before its `longjmp`. Read by [`protect`] on
    /// the recovery path. Uninitialized until then (never read on the Ok path).
    cause: MaybeUninit<i32>,
}

struct Registry {
    head: UnsafeCell<*mut Mark>,
}

// SAFETY: every access runs inside critical_section::with, which the provided
// impl makes mutually exclusive across all threads and cores.
unsafe impl Sync for Registry {}

struct CallPayload<F, R> {
    func: ManuallyDrop<F>,
    result: MaybeUninit<R>,
}

static REGISTRY: Registry = Registry {
    head: UnsafeCell::new(ptr::null_mut()),
};

/// Run `f` under recovery protection, tagging this scope with `tid`.
///
/// Returns `Ok(value)` on normal completion, or `Err(RecoveryError)` if
/// [`recover`] (from the fault/OOM handler) jumped into this scope. On the
/// `Err` path everything `f` had on the stack is leaked: no destructors run.
/// Nesting is supported (the handler resolves to the innermost scope for `tid`).
/// Note that nesting different `tid`s will lead to UB.
///
/// ## The [`UnwindSafe`] bound
///
/// `protect` requires `F: UnwindSafe` for the reason `std::panic::catch_unwind`
/// does: a closure abandoned mid-mutation can leave a value torn, so the bound
/// makes the usual offenders (`&mut T` captures, `Cell`/`RefCell`/`Mutex`) fail
/// at the call site instead of passing silently. It is advisory -
/// [`AssertUnwindSafe`] satisfies it unconditionally and safely. The obligations
/// the type system cannot express are in `# Safety` below, which is why
/// `protect` is `unsafe`.
///
/// ## Recovery-stack guarantee
///
/// Before calling `f`, `protect` reserves at least [`RECOVERY_GAP_BYTES`] of
/// stack between the closure and the recovery mark (the `setjmp` point) and
/// holds it reserved for the whole run, so `f` never touches it. This gives a
/// fault handler somewhere to stand: to turn a fault into an `Err`, the handler
/// resumes the faulting thread and calls [`recover`], which must not overwrite
/// the mark, the saved `jmp_buf`, or any frame at or before the `protect` call.
/// Those all sit at or before the mark, and the reserved gap guarantees room
/// below it - so a handler may land `recover` at the bottom of the thread's
/// stack and run entirely on abandoned frames.
///
/// Gap isn't designed to always be a place to run the handler, but it gives you
/// a guarantee the you can go off [`RECOVERY_GAP_BYTES`] bytes before the stack
/// bottom.
///
/// # Safety
///
/// Recovery rewinds the stack pointer and runs no destructors: every frame `f`
/// pushed is leaked in place and its storage is reused by later calls. The
/// caller must ensure nothing depends on those frames living on, or on their
/// `Drop` running. This is non-exhaustive - among the things it breaks:
///
/// - `Pin`'s drop guarantee for stack-pinned `!Unpin` values
///   (`core::pin::pin!`, an on-stack address-sensitive future, an intrusive
///   node): the storage is invalidated and reused with no `Drop`. (`Pin<Box<T>>`
///   is safe - heap storage is only leaked.)
/// - Raw pointers into the frames dangle after `Err`: fine to hold, UB to
///   dereference.
/// - References into the frames dangle too, and a reference can be UB just by
///   staying live across recovery (using it retags it), not only when read.
/// - Scope-based APIs (such as `thread::scope`) are bypassed.
/// - `Drop`-based invariants (lock guards, `RAII cleanup) do not run.
/// - Interior-mutable state shared outward can be left torn if `f` was
///   abandoned mid-mutation.
///
/// ...and anything else that assumed the stack above the mark stayed valid.
pub unsafe fn protect<F, R>(tid: ThreadId, f: F) -> Result<R, RecoveryError>
where
    F: FnOnce() -> R + UnwindSafe,
{
    let mut payload = CallPayload::<F, R> {
        func: ManuallyDrop::new(f),
        result: MaybeUninit::uninit(),
    };
    let mut mark = Mark {
        tid,
        jmpbuf: JmpBufStorage::new(),
        prev: ptr::null_mut(),
        next: ptr::null_mut(),
        cause: MaybeUninit::uninit(),
    };
    let mark_ptr: *mut Mark = &mut mark;
    let jb = JmpBufStorage::raw(&raw const (*mark_ptr).jmpbuf);

    critical_section::with(|_cs| registry_push(mark_ptr));

    let outcome = setback_call(
        jb,
        trampoline::<F, R>,
        &mut payload as *mut CallPayload<F, R> as *mut c_void,
    );

    critical_section::with(|_cs| registry_unlink(mark_ptr));

    if outcome == SETBACK_OK {
        // SAFETY: success path wrote the result.
        Ok(payload.result.assume_init())
    } else {
        // SAFETY: a nonzero outcome means `recover` longjmp'd back here, and it
        // wrote `cause` into this mark before jumping.
        Err(RecoveryError {
            cause: (*mark_ptr).cause.assume_init(),
        })
    }
}

unsafe extern "C" fn trampoline<F, R>(data: *mut c_void)
where
    F: FnOnce() -> R,
{
    // SAFETY: `data` is the &mut CallPayload<F,R> passed into setback_call.
    let payload = unsafe { &mut *(data as *mut CallPayload<F, R>) };
    // SAFETY: `payload.func` is a live closure, and we are calling it exactly once.
    let f = unsafe { ManuallyDrop::take(&mut payload.func) };
    payload.result.write(f());
}

/// From the shared fault/OOM handler: recover the thread identified by `tid` by
/// jumping into its innermost active [`protect`] scope, reporting `cause`.
///
/// Diverges on success: the matching [`protect`] returns
/// `Err(RecoveryError { cause })`. Returns `Err(RecoveryFailure)` if `tid` has no
/// active scope, so the caller can halt or escalate.
///
/// # Safety
/// - `tid` must identify the thread on whose stack the matching `protect` is
///   still live.
/// - Must be called from the same thread as `tid`, not from the other thread,
///   context, or the fault handler.
/// - All leak / `protect` `# Safety` obligations apply to everything between
///   the fault point and the mark.
pub unsafe fn recover(tid: ThreadId, cause: i32) -> Result<Infallible, RecoveryFailure> {
    let jb = critical_section::with(|_cs| {
        let mark = registry_find(tid);
        if mark.is_null() {
            return ptr::null_mut();
        }
        // Stash the cause while the node is locked-live, the matching `protect`
        // reads it back after the jump. `recover` runs on the faulting thread
        // and `protect` resumes on it, so the write and read do not race.
        (*mark).cause = MaybeUninit::new(cause);
        JmpBufStorage::raw(&raw const (*mark).jmpbuf)
    });
    if jb.is_null() {
        return Err(RecoveryFailure);
    }
    setback_longjmp(jb)
}

unsafe fn registry_push(node: *mut Mark) {
    let head = *REGISTRY.head.get();
    (*node).next = head;
    (*node).prev = ptr::null_mut();
    if !head.is_null() {
        (*head).prev = node;
    }
    *REGISTRY.head.get() = node;
}

unsafe fn registry_unlink(node: *mut Mark) {
    let prev = (*node).prev;
    let next = (*node).next;
    if prev.is_null() {
        *REGISTRY.head.get() = next;
    } else {
        (*prev).next = next;
    }
    if !next.is_null() {
        (*next).prev = prev;
    }
}

unsafe fn registry_find(tid: ThreadId) -> *mut Mark {
    let mut p = *REGISTRY.head.get();
    while !p.is_null() {
        if (*p).tid == tid {
            return p;
        }
        p = (*p).next;
    }
    ptr::null_mut()
}

impl JmpBufStorage {
    #[inline]
    fn new() -> Self {
        let need = unsafe { setback_jmpbuf_size() };
        let align = unsafe { setback_jmpbuf_align() };
        assert!(need <= 512, "setback: jmp_buf larger than reserved storage");
        assert!(
            align <= 16,
            "setback: jmp_buf alignment exceeds storage alignment"
        );
        JmpBufStorage {
            bytes: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    #[inline]
    unsafe fn raw(this: *const JmpBufStorage) -> *mut c_void {
        UnsafeCell::raw_get(&raw const (*this).bytes) as *mut c_void
    }
}

impl Error for RecoveryFailure {}
impl core::fmt::Display for RecoveryFailure {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "setback recovery failure (no active scope)")
    }
}
