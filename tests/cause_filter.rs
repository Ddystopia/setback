//! `protect_cause` narrows a scope to one cause: `recover` jumps in only when
//! the cause matches, otherwise the scope is skipped just like a foreign-`tid`
//! mark. A matching cause lands here; a non-matching one resolves to the next
//! outer accepting scope, or reports `RecoveryFailure` when none accepts it.
//! A skipped scope on the jumping thread's own stack is abandoned by the jump,
//! so it leaves the mark list as well.
//!
//! Each test uses its own `tid`s, so a `recover` only ever jumps within its own
//! thread's stack while the global mark list interleaves marks across tests.
//!
//! Run with `cargo test --features std`.

use setback::{can_recover, protect, protect_cause, recover, RecoveryError, RecoveryFailure};

const OOM: i32 = 2;
const STACK_OVERFLOW: i32 = 1;

/// The matching cause jumps into the scope and resurfaces as `RecoveryError`.
#[test]
fn matching_cause_recovers() {
    const TID: usize = 71;

    let r: Result<(), RecoveryError> = unsafe {
        protect_cause(TID, OOM, || {
            // Matches `OOM`; diverges via longjmp, never falls through.
            let _ = recover(TID, OOM);
        })
    };

    assert_eq!(r, Err(RecoveryError { cause: OOM }));
}

/// A cause the scope does not accept is not recovered here: `recover` finds no
/// accepting scope, returns `RecoveryFailure`, and the closure completes.
#[test]
fn non_matching_cause_is_skipped() {
    const TID: usize = 72;

    let r: Result<&str, RecoveryError> = unsafe {
        protect_cause(TID, OOM, || {
            let miss = recover(TID, STACK_OVERFLOW);
            assert_eq!(miss, Err(RecoveryFailure));
            "ran to completion"
        })
    };

    assert_eq!(r, Ok("ran to completion"));
}

/// An inner `protect_cause` that rejects the cause is skipped, so recovery
/// resolves to the outer catch-all scope below it.
#[test]
fn non_matching_inner_resolves_to_outer() {
    const TID: usize = 73;

    let outer: Result<(), RecoveryError> = unsafe {
        protect(TID, || {
            let _inner: Result<(), RecoveryError> = protect_cause(TID, OOM, || {
                // Inner accepts only `OOM`; this `STACK_OVERFLOW` skips it and
                // jumps past to the outer scope, abandoning the inner frame.
                let _ = recover(TID, STACK_OVERFLOW);
            });
            unreachable!("recover jumped to the outer scope, not back here");
        })
    };

    assert_eq!(outer, Err(RecoveryError { cause: STACK_OVERFLOW }));
}

/// The jump that skips an inner scope abandons its frame, so the scope's mark
/// must leave the global list with it: after the outer scope catches, nothing
/// for this `tid` is left to find.
#[test]
fn a_skipped_scope_leaves_the_list_when_the_jump_passes_it() {
    const TID: usize = 74;

    let outer: Result<(), RecoveryError> = unsafe {
        protect(TID, || {
            let _inner: Result<(), RecoveryError> = protect_cause(TID, OOM, || {
                let _ = recover(TID, STACK_OVERFLOW);
            });
            unreachable!("recover jumped to the outer scope, not back here");
        })
    };

    assert_eq!(outer, Err(RecoveryError { cause: STACK_OVERFLOW }));
    // The inner mark would still match here if the jump had left it linked,
    // and its `jmp_buf` now sits in reusable stack memory.
    assert!(!can_recover(TID, OOM));
    assert!(!can_recover(TID, STACK_OVERFLOW));
}

/// Two nested scopes reject the cause and are both jumped over: neither mark
/// survives the jump.
#[test]
fn every_skipped_scope_leaves_the_list_not_just_the_innermost() {
    const TID: usize = 75;
    const TIMEOUT: i32 = 3;

    let outer: Result<(), RecoveryError> = unsafe {
        protect(TID, || {
            let _middle: Result<(), RecoveryError> = protect_cause(TID, OOM, || {
                let _inner: Result<(), RecoveryError> = protect_cause(TID, TIMEOUT, || {
                    let _ = recover(TID, STACK_OVERFLOW);
                });
                unreachable!("the jump passed both inner scopes");
            });
            unreachable!("recover jumped to the outer scope, not back here");
        })
    };

    assert_eq!(outer, Err(RecoveryError { cause: STACK_OVERFLOW }));
    assert!(!can_recover(TID, OOM));
    assert!(!can_recover(TID, TIMEOUT));
    assert!(!can_recover(TID, STACK_OVERFLOW));
}
