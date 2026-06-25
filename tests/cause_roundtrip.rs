//! End-to-end: a cause chosen at `recover` must survive the `longjmp` (carried
//! in the `Mark`) and resurface as `RecoveryError::cause`.
//!
//! Run with `cargo test --features std` (the std `critical-section` impl backs
//! the registry locking on the host).

use setback::{protect, recover, RecoveryError, RecoveryFailure, ThreadId};

/// Arm a scope under `tid`, then recover it with `cause` from inside the closure
/// (same thread, so the `longjmp` lands back in `protect`).
fn roundtrip(tid: ThreadId, cause: i32) -> Result<(), RecoveryError> {
    unsafe {
        protect(tid, move || {
            // Diverges via `longjmp`; the line never falls through.
            let _ = recover(tid, cause);
        })
    }
}

#[test]
fn arbitrary_causes_survive_the_jump() {
    for cause in [1, 2, 8, 9, -1, i32::MIN, i32::MAX] {
        assert_eq!(roundtrip(11, cause), Err(RecoveryError { cause }));
    }
}

#[test]
fn recover_without_active_scope_errs() {
    // No `protect` is live for this tid, so `recover` reports the miss instead
    // of jumping.
    let outcome = unsafe { recover(987, 1) };
    assert_eq!(outcome, Err(RecoveryFailure));
}
