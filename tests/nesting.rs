//! Nesting: for one `tid`, `protect` scopes form a LIFO sub-stack inside the
//! single global mark list, so `recover` always resolves to the *innermost*
//! live scope for that `tid`. This falls straight out of the registry's
//! linked-list discipline:
//!
//! * `registry_push` links each new mark at the **head** (LIFO), so the
//!   innermost scope for a `tid` sits nearest the head.
//! * `registry_find` walks **head -> tail** and returns the **first** `tid`
//!   match -- i.e. that innermost scope.
//! * `registry_unlink` splices a node out from **anywhere** in O(1) (the list
//!   is doubly linked), which is what lets an inner scope recover and leave
//!   while the outer one -- now in the middle of the list once other threads
//!   are present -- stays linked.
//!
//! The harness runs each `#[test]` on its own thread, so the protected frames
//! of different tests live on different stacks. Every test therefore uses its
//! own `tid`s: a `recover` only ever jumps within its own thread's stack, even
//! while the global list interleaves marks from tests running in parallel.
//!
//! Run with `cargo test --features std`.

use setback::{protect, recover, AssertUnwindSafe, RecoveryError, RecoveryFailure};

/// Recovering from the inner scope lands in the *inner* `protect` (it returns
/// `Err`) and leaves the outer scope free to run to completion: the jump
/// resolved to the head-most mark for the tid, not the outer one below it.
#[test]
fn recovery_into_inner_leaves_the_outer_scope_intact() {
    const TID: usize = 41;
    let c = 7;

    let outer: Result<&str, RecoveryError> = unsafe {
        protect(TID, || {
            let inner: Result<(), RecoveryError> = protect(TID, || {
                // Diverges via longjmp into this innermost mark; never falls through.
                let _ = recover(TID, c);
            });
            assert_eq!(inner, Err(RecoveryError { cause: c }));
            "outer ran to completion"
        })
    };

    assert_eq!(outer, Ok("outer ran to completion"));
}

/// Three scopes share one tid. Each `recover` resolves to the *current*
/// innermost live scope, so the causes come back inner -> middle -> outer:
/// every recovery unlinks the head-most mark, advancing the head to the next
/// scope out.
#[test]
fn recovery_peels_the_nest_innermost_first() {
    const TID: usize = 52;
    let (c_inner, c_middle, c_outer) = (1, 2, 3);

    let outer: Result<(), RecoveryError> = unsafe {
        protect(TID, || {
            let middle: Result<(), RecoveryError> = protect(TID, || {
                let inner: Result<(), RecoveryError> = protect(TID, || {
                    let _ = recover(TID, c_inner);
                });
                assert_eq!(inner, Err(RecoveryError { cause: c_inner }));
                let _ = recover(TID, c_middle);
            });
            assert_eq!(middle, Err(RecoveryError { cause: c_middle }));
            let _ = recover(TID, c_outer);
        })
    };

    assert_eq!(outer, Err(RecoveryError { cause: c_outer }));
}

/// With two nested marks live, recovering an *absent* tid walks the whole list
/// and reports the miss without disturbing either scope -- both then complete
/// normally, proving the failed walk left the list intact.
#[test]
fn find_reports_a_miss_while_nested_marks_are_live() {
    const OUTER_TID: usize = 63;
    const INNER_TID: usize = 64;
    const ABSENT_TID: usize = 6399;

    let outer: Result<&str, RecoveryError> = unsafe {
        protect(OUTER_TID, || {
            let inner: Result<&str, RecoveryError> = protect(INNER_TID, || {
                let miss = recover(ABSENT_TID, 1);
                assert_eq!(miss, Err(RecoveryFailure));
                "inner ran to completion"
            });
            assert_eq!(inner, Ok("inner ran to completion"));
            "outer ran to completion"
        })
    };

    assert_eq!(outer, Ok("outer ran to completion"));
}

/// The reason the list is doubly linked and keyed by `tid`: a second thread's
/// mark sits in the one global list, and `recover` on this thread must skip it
/// (it is nearer the head, pushed later) to find this thread's mark -- which
/// then unlinks from the *middle* of the list in O(1). The foreign scope is
/// untouched and finishes normally on its own stack.
#[test]
fn find_skips_another_threads_mark_in_the_global_list() {
    use std::sync::mpsc;
    use std::thread;

    const MAIN_TID: usize = 100;
    const FOREIGN_TID: usize = 200;

    // main -> foreign: MAIN is linked, you may enter now (forces FOREIGN to be
    // pushed *after* MAIN, so FOREIGN ends up at the head and must be skipped).
    let (go_tx, go_rx) = mpsc::channel::<()>();
    // foreign -> main: FOREIGN is now linked at the head.
    let (ready_tx, ready_rx) = mpsc::channel::<()>();
    // main -> foreign: recovery is done, you may finish.
    let (wake_tx, wake_rx) = mpsc::channel::<()>();

    let foreign = thread::spawn(move || unsafe {
        go_rx.recv().unwrap();
        // Channels aren't `UnwindSafe`; this scope completes normally (never
        // recovered), so asserting unwind-safety here is sound.
        protect(
            FOREIGN_TID,
            AssertUnwindSafe(move || {
                ready_tx.send(()).unwrap(); // FOREIGN now linked at the head
                wake_rx.recv().unwrap(); // park, keeping this frame -- and its mark -- live
                "foreign ran to completion"
            }),
        )
    });

    let c = 42;
    let main: Result<(), RecoveryError> = unsafe {
        protect(
            MAIN_TID,
            AssertUnwindSafe(|| {
                go_tx.send(()).unwrap(); // MAIN is linked; release foreign
                ready_rx.recv().unwrap(); // FOREIGN is now linked above MAIN
                // Walk: head FOREIGN (skip) -> MAIN (match). Jumps within this
                // thread's stack only; MAIN unlinks from the middle on return.
                let _ = recover(MAIN_TID, c);
            }),
        )
    };
    assert_eq!(main, Err(RecoveryError { cause: c }));

    wake_tx.send(()).unwrap();
    assert_eq!(foreign.join().unwrap(), Ok("foreign ran to completion"));
}
