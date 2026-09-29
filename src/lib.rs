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
calls [`recover`] to find that thread's innermost active mark and jump into it,
or [`can_recover`] to ask whether such a mark exists without jumping.
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
use core::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

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
        top: *mut usize,
        tramp: unsafe extern "C" fn(*mut c_void),
        data: *mut c_void,
    ) -> i32;
    fn setback_longjmp(jb: *mut c_void) -> !;
}

const SETBACK_OK: i32 = 0;

/// Bytes of stack [`protect`] reserves below the mark for a fault handler to
/// run [`recover`] on. See the "Recovery-stack guarantee" on [`protect`].
//
// Must equal `SETBACK_RECOVERY_STACK_BYTES` in `setback.c`.
pub const RECOVERY_STACK_BYTES: usize = 64;

/// Backing storage for one C `jmp_buf`. 512 bytes / 16-byte alignment covers
/// every mainstream target. The constructor asserts it.
#[repr(C, align(16))]
struct JmpBufStorage {
    bytes: UnsafeCell<MaybeUninit<[u8; 512]>>,
}

struct Mark {
    tid: ThreadId,
    accepts: Option<i32>,
    /// Recovery stack top while armed, else `0`. Written by `setback.c` only.
    recovery_stack_top: AtomicUsize,
    jmpbuf: JmpBufStorage,
    prev: *mut Mark,
    /// Atomic because a walk from a fault handler may run concurrently with a
    /// link/unlink: a critical section cannot exclude a context that preempts
    /// it, such as an NMI. `prev` stays plain - only the mutators read it, and
    /// they exclude each other.
    next: AtomicPtr<Mark>,
    cause: MaybeUninit<i32>,
}

struct CallPayload<F, R> {
    func: ManuallyDrop<F>,
    result: MaybeUninit<R>,
}

/// Head of the intrusive list of active marks, most recently linked first.
static REGISTRY_HEAD: AtomicPtr<Mark> = AtomicPtr::new(ptr::null_mut());

/// Run `f` under recovery protection, tagging this scope with `tid`. Catches
/// any cause; see [`protect_cause`] to recover from a single cause only.
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
/// To turn a fault into an `Err`, a fault handler resumes the faulting thread
/// and calls [`recover`], which must not overwrite the mark (the `setjmp`
/// point), the saved `jmp_buf`, or any frame at or before the `protect` call.
///
/// Before calling `f`, `protect` lays down [`RECOVERY_STACK_BYTES`] of stack
/// below the mark, touches its lowest byte, then arms the scope until `f`
/// returns. Given a guard that faults precisely (MPU, `PSPLIM`, PMP), a thread
/// short on headroom therefore faults while still unarmed. A handler may load
/// [`recovery_stack_top`] into SP and run `recover` there: the recovery stack
/// and the abandoned frames of `f` below it are all free. The top is 16-byte
/// aligned as at a call site, so on x86 a handler that enters a function
/// directly leaves the return-address slot below it.
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
    protect_inner(tid, None, f)
}

/// Like [`protect`], but only recovers when [`recover`]'s `cause` equals
/// `cause`; any other cause skips this scope.
///
/// # Safety
///
/// The contract of [`protect`] applies in full.
pub unsafe fn protect_cause<F, R>(
    tid: ThreadId,
    cause: i32,
    f: F,
) -> Result<R, RecoveryError>
where
    F: FnOnce() -> R + UnwindSafe,
{
    protect_inner(tid, Some(cause), f)
}

// SAFETY: the contract is `protect`'s; the wrappers only pick `accepts`.
unsafe fn protect_inner<F, R>(
    tid: ThreadId,
    accepts: Option<i32>,
    f: F,
) -> Result<R, RecoveryError>
where
    F: FnOnce() -> R + UnwindSafe,
{
    let mut payload = CallPayload::<F, R> {
        func: ManuallyDrop::new(f),
        result: MaybeUninit::uninit(),
    };
    let mut mark = Mark {
        tid,
        accepts,
        recovery_stack_top: AtomicUsize::new(0),
        jmpbuf: JmpBufStorage::new(),
        prev: ptr::null_mut(),
        next: AtomicPtr::new(ptr::null_mut()),
        cause: MaybeUninit::uninit(),
    };
    let mark_ptr: *mut Mark = &mut mark;
    let jb = JmpBufStorage::raw(&raw const (*mark_ptr).jmpbuf);
    let top = (&raw mut (*mark_ptr).recovery_stack_top).cast::<usize>();

    critical_section::with(|_cs| registry_push(mark_ptr));

    let outcome = setback_call(
        jb,
        top,
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
/// jumping into its innermost active scope that accepts `cause`, reporting it.
/// [`protect`] scopes accept any cause; [`protect_cause`] scopes accept one.
///
/// Diverges on success: the matching [`protect`] returns
/// `Err(RecoveryError { cause })`. Returns `Err(RecoveryFailure)` if no active
/// scope for `tid` accepts `cause`, so the caller can halt or escalate, leaving
/// every scope live.
///
/// Scopes for `tid` nested inside the one it jumps into never return: the jump
/// abandons their frames and drops their marks from the registry.
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
        let mark = registry_find(tid, cause);
        if mark.is_null() {
            return ptr::null_mut();
        }
        // The jump abandons every scope for `tid` nested inside `mark`; their
        // marks leave the list here, while it can still be walked safely.
        registry_unlink_nested(tid, mark);
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

/// Whether [`recover`] would find a scope: `true` when `tid` has an active
/// [`protect`] scope that accepts `cause`. Shorthand for
/// [`recovery_stack_top`]`(tid, cause).is_some()`, safe in the same contexts.
pub fn can_recover(tid: ThreadId, cause: i32) -> bool {
    recovery_stack_top(tid, cause).is_some()
}

/// Top of the recovery stack of the scope [`recover`] would jump into, or
/// `None` when no active scope for `tid` accepts `cause`. A fault handler
/// loads it into SP before resuming the thread at code that calls `recover`
/// with the same `tid` and `cause`, see the "Recovery-stack guarantee" on
/// [`protect`]. Valid until that scope ends. Safe to call from a
/// fault handler, including one that preempts a critical section.
pub fn recovery_stack_top(tid: ThreadId, cause: i32) -> Option<usize> {
    // SAFETY: `registry_find` needs every node it walks to stay alive, and the
    // critical section keeps every mutator out for the duration. A caller that
    // preempts the critical section instead of taking it - a fault handler -
    // has the mutator stopped mid-`protect`, so its mark cannot go away either.
    critical_section::with(|_cs| unsafe {
        let mark = registry_find(tid, cause);
        if mark.is_null() {
            return None;
        }
        // Another core may have disarmed the scope since the find.
        match (*mark).recovery_stack_top.load(Ordering::Relaxed) {
            0 => None,
            top => Some(top),
        }
    })
}

unsafe fn registry_push(node: *mut Mark) {
    let head = REGISTRY_HEAD.load(Ordering::Relaxed);
    (*node).next.store(head, Ordering::Relaxed);
    (*node).prev = ptr::null_mut();
    if !head.is_null() {
        (*head).prev = node;
    }
    // Release, paired with the load in `registry_find`: the node becomes
    // reachable only once its `tid`, `accepts` and `next` are visible, so a walk
    // that reaches it never reads them half-written or follows a stale `next`.
    REGISTRY_HEAD.store(node, Ordering::Release);
}

unsafe fn registry_unlink(node: *mut Mark) {
    let prev = (*node).prev;
    let next = (*node).next.load(Ordering::Relaxed);
    if prev.is_null() {
        REGISTRY_HEAD.store(next, Ordering::Release);
    } else {
        (*prev).next.store(next, Ordering::Relaxed);
    }
    if !next.is_null() {
        (*next).prev = prev;
    }
}

/// Unlink every mark for `tid` that sits ahead of `target` in the list.
///
/// A `longjmp` into `target` abandons those scopes' frames without returning
/// through their `protect`, so nothing else would ever unlink them. For one
/// `tid` the marks form a LIFO sub-stack, so every mark ahead of `target` is a
/// scope nested inside it - armed or still arming, both are abandoned by the
/// jump. Marks for other `tid`s live on other stacks and are left alone.
///
/// # Safety
///
/// `target` must be a node in the list, and every node this walks must stay
/// alive for the walk, so the caller must hold the critical section - it keeps
/// the other mutators out, and this one splices nodes rather than only reading
/// them, so it cannot run from a context that merely preempts them.
unsafe fn registry_unlink_nested(tid: ThreadId, target: *mut Mark) {
    let mut p = REGISTRY_HEAD.load(Ordering::Acquire);
    while !p.is_null() && p != target {
        // Read `next` before the splice, so the walk does not rest on what
        // `registry_unlink` leaves behind in the node it removes.
        let next = (*p).next.load(Ordering::Relaxed);
        if (*p).tid == tid {
            registry_unlink(p);
        }
        p = next;
    }
}

/// Innermost mark for `tid` that accepts `cause`, or null.
///
/// # Safety
///
/// Every node this walks must stay alive for the walk. Marks live on the
/// protected thread's stack, so the caller must either hold the critical
/// section, which keeps every mutator out, or run in a context that cannot be
/// preempted by one - a fault handler, whose interrupted mutator is stopped
/// mid-list and cannot return out of its `protect` frame.
unsafe fn registry_find(tid: ThreadId, cause: i32) -> *mut Mark {
    // Acquire, paired with the stores in `registry_push` / `registry_unlink`.
    // The links are read atomically because this may interrupt a mutator: a
    // critical section does not exclude the contexts that call `recover` and
    // `can_recover`. Walking head -> tail keeps that sound. `registry_push`
    // publishes the head last, and `registry_unlink` only re-points its
    // neighbours, so a walk in progress sees either list, never a dangling link.
    let mut p = REGISTRY_HEAD.load(Ordering::Acquire);
    while !p.is_null() {
        if (*p).recovery_stack_top.load(Ordering::Relaxed) != 0
            && (*p).tid == tid
            && (*p).accepts.is_none_or(|c| c == cause)
        {
            return p;
        }
        p = (*p).next.load(Ordering::Relaxed);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_linked_mark_is_ignored_until_it_is_armed() {
        const TID: ThreadId = 1234;
        const CAUSE: i32 = 9;

        let mut mark = Mark {
            tid: TID,
            accepts: None,
            recovery_stack_top: AtomicUsize::new(0),
            jmpbuf: JmpBufStorage::new(),
            prev: ptr::null_mut(),
            next: AtomicPtr::new(ptr::null_mut()),
            cause: MaybeUninit::uninit(),
        };
        let mark_ptr: *mut Mark = &mut mark;

        let found = || critical_section::with(|_cs| !unsafe { registry_find(TID, CAUSE) }.is_null());

        unsafe {
            critical_section::with(|_cs| registry_push(mark_ptr));
            assert!(!found());

            (*mark_ptr).recovery_stack_top.store(0x1000, Ordering::Relaxed);
            assert!(found());

            critical_section::with(|_cs| registry_unlink(mark_ptr));
            assert!(!found());
        }
    }

    #[test]
    fn a_recovery_disarms_the_mark_it_lands_in() {
        unsafe extern "C" fn jump(jb: *mut c_void) {
            unsafe { setback_longjmp(jb) }
        }

        let jmpbuf = JmpBufStorage::new();
        let top = AtomicUsize::new(0);

        let outcome = unsafe {
            let jb = JmpBufStorage::raw(&raw const jmpbuf);
            setback_call(jb, top.as_ptr(), jump, jb)
        };

        assert_ne!(outcome, SETBACK_OK);
        assert_eq!(top.load(Ordering::Relaxed), 0);
    }
}
