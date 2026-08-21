//! `can_recover` answers the lookup `recover` performs, without jumping: it is
//! `true` exactly when a `recover` for the same `(tid, cause)` would land in a
//! scope, and `false` when that `recover` would report `RecoveryFailure`. A
//! fault handler asks it first when reaching `recover` at all costs something
//! it cannot undo.
//!
//! Each test uses its own `tid`s, so a `recover` only ever jumps within its own
//! thread's stack while the global mark list interleaves marks across tests.
//!
//! Run with `cargo test --features std`.

use setback::{can_recover, protect, protect_cause, recover, RecoveryError, RecoveryFailure};

const OOM: i32 = 2;
const STACK_OVERFLOW: i32 = 1;

/// Outside any scope there is nothing to jump into, and inside one there is.
#[test]
fn tracks_the_scope() {
    const TID: usize = 91;

    assert!(!can_recover(TID, OOM));

    let r: Result<(), RecoveryError> = unsafe {
        protect(TID, || {
            assert!(can_recover(TID, OOM));
        })
    };

    assert_eq!(r, Ok(()));
    assert!(!can_recover(TID, OOM), "the mark is unlinked on the way out");
}

/// A scope belongs to one `tid`: it does not answer for another thread.
#[test]
fn other_tid_is_not_covered() {
    const TID: usize = 92;
    const OTHER: usize = 93;

    let r: Result<(), RecoveryError> = unsafe {
        protect(TID, || {
            assert!(can_recover(TID, OOM));
            assert!(!can_recover(OTHER, OOM));
        })
    };

    assert_eq!(r, Ok(()));
}

/// It applies `protect_cause`'s filter, so it stays in step with `recover`:
/// a rejected cause reports `false`, and the `recover` it predicts fails.
#[test]
fn honours_the_cause_filter() {
    const TID: usize = 94;

    let r: Result<&str, RecoveryError> = unsafe {
        protect_cause(TID, OOM, || {
            assert!(can_recover(TID, OOM));
            assert!(!can_recover(TID, STACK_OVERFLOW));

            let miss = recover(TID, STACK_OVERFLOW);
            assert_eq!(miss, Err(RecoveryFailure));
            "ran to completion"
        })
    };

    assert_eq!(r, Ok("ran to completion"));
}

/// An outer catch-all scope keeps the answer `true` for a cause the inner scope
/// rejects, matching the scope `recover` resolves to.
#[test]
fn sees_past_a_rejecting_inner_scope() {
    const TID: usize = 95;

    let outer: Result<(), RecoveryError> = unsafe {
        protect(TID, || {
            let _inner: Result<(), RecoveryError> = protect_cause(TID, OOM, || {
                assert!(can_recover(TID, STACK_OVERFLOW));
                // Skips the inner scope and jumps to the outer one.
                let _ = recover(TID, STACK_OVERFLOW);
            });
            unreachable!("recover jumped to the outer scope, not back here");
        })
    };

    assert_eq!(outer, Err(RecoveryError { cause: STACK_OVERFLOW }));
}

/// Recovery unlinks the abandoned mark, so the answer goes back to `false`.
#[test]
fn false_again_after_recovery() {
    const TID: usize = 96;

    let r: Result<(), RecoveryError> = unsafe {
        protect(TID, || {
            let _ = recover(TID, OOM);
        })
    };

    assert_eq!(r, Err(RecoveryError { cause: OOM }));
    assert!(!can_recover(TID, OOM));
}
