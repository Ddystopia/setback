//! `protect` must reserve at least `RECOVERY_GAP_BYTES` of stack between the
//! closure and the recovery mark (the "Recovery-stack guarantee").
//!
//! Run with `cargo test --features std`.

use core::hint::black_box;

/// Address of a local in a fresh, un-inlined frame: a stand-in for "the stack
/// pointer here". `black_box` keeps the probe and its address from folding away.
#[inline(never)]
fn stack_addr() -> usize {
    let probe = 0u8;
    black_box(&probe) as *const u8 as usize
}

#[test]
fn gap_size_is_aligned_and_nonzero() {
    let gap = setback::RECOVERY_GAP_BYTES;
    assert!(gap >= 8, "gap must be at least a word: {gap}");
    assert_eq!(gap % 8, 0, "gap must keep the stack 8-aligned: {gap}");
}

#[test]
fn closure_runs_below_the_reserved_gap() {
    let gap = setback::RECOVERY_GAP_BYTES;

    let anchor = stack_addr();
    let inner = unsafe { setback::protect(1, || stack_addr()) }.unwrap();

    // Assumes a downward-growing stack. The closure runs below `anchor` by at
    // least the reserved gap, plus the intervening protect / setback_call /
    // gap-runner frames.
    assert!(
        anchor > inner,
        "expected a downward-growing stack (anchor={anchor:#x}, inner={inner:#x})"
    );
    assert!(
        anchor - inner >= gap,
        "closure ran {} bytes below the anchor; gap guarantees >= {}",
        anchor - inner,
        gap
    );
}
