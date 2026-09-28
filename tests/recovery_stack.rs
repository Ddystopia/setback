//! `protect` lays down a recovery stack of `RECOVERY_STACK_BYTES` below the
//! mark and publishes its top for a fault handler to use as SP.
//!
//! Run with `cargo test --features std`.

use core::hint::black_box;

use setback::{
    protect, protect_cause, recovery_stack_top, RECOVERY_STACK_BYTES,
};

const OOM: i32 = 2;
const STACK_OVERFLOW: i32 = 1;

const SP_ALIGN: usize = 16;

/// Address of a local in a fresh, un-inlined frame: a stand-in for "the stack
/// pointer here". `black_box` keeps the probe and its address from folding away.
#[inline(never)]
fn stack_addr() -> usize {
    let probe = 0u8;
    black_box(&probe) as *const u8 as usize
}

#[test]
fn recovery_stack_size_is_aligned_and_nonzero() {
    assert!(RECOVERY_STACK_BYTES >= SP_ALIGN);
    assert_eq!(RECOVERY_STACK_BYTES % SP_ALIGN, 0);
}

#[test]
fn closure_runs_below_the_recovery_stack() {
    const TID: usize = 301;

    let anchor = stack_addr();
    let (top, inner) = unsafe {
        protect(TID, || {
            let top = recovery_stack_top(TID, OOM).expect("scope is armed");
            (top, stack_addr())
        })
    }
    .unwrap();

    // Assumes a downward-growing stack.
    assert!(anchor > top, "top above the caller (anchor={anchor:#x}, top={top:#x})");
    assert!(
        top - inner >= RECOVERY_STACK_BYTES,
        "closure ran {} bytes below the top; the recovery stack guarantees >= {}",
        top - inner,
        RECOVERY_STACK_BYTES
    );
    assert_eq!(top % SP_ALIGN, 0, "top={top:#x}");
}

#[test]
fn none_outside_a_scope() {
    const TID: usize = 302;

    assert_eq!(recovery_stack_top(TID, OOM), None);
    unsafe { protect(TID, || ()) }.unwrap();
    assert_eq!(recovery_stack_top(TID, OOM), None, "the scope is gone");
}

#[test]
fn top_belongs_to_the_scope_recover_would_pick() {
    const TID: usize = 303;

    let (outer_top, (inner_top, picked_for_oom)) = unsafe {
        protect(TID, || {
            let outer_top = recovery_stack_top(TID, OOM).unwrap();
            let inner = protect_cause(TID, STACK_OVERFLOW, || {
                (
                    recovery_stack_top(TID, STACK_OVERFLOW).unwrap(),
                    recovery_stack_top(TID, OOM).unwrap(),
                )
            })
            .unwrap();
            (outer_top, inner)
        })
    }
    .unwrap();

    assert!(outer_top > inner_top, "outer={outer_top:#x} inner={inner_top:#x}");
    assert_eq!(picked_for_oom, outer_top);
}
