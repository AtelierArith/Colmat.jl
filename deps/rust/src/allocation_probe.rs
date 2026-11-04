//! Test-only calling-thread allocation counter; no production allocator change.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static COUNT: Cell<usize> = const { Cell::new(0) };
}
struct Probe;
fn record() {
    let _ = ACTIVE.try_with(|a| {
        if a.get() {
            let _ = COUNT.try_with(|n| n.set(n.get() + 1));
        }
    });
}
// SAFETY: the allocator forwards every layout/pointer unchanged to System;
// the thread-local counters do not allocate or retain those pointers.
unsafe impl GlobalAlloc for Probe {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        record();
        // SAFETY: GlobalAlloc supplies a valid layout; forwarding it unchanged.
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        record();
        // SAFETY: GlobalAlloc supplies a valid layout; forwarding it unchanged.
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        record();
        // SAFETY: GlobalAlloc supplies the live pointer/layout and valid new size.
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: GlobalAlloc supplies the matching live pointer and layout.
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static ALLOCATOR: Probe = Probe;
pub(crate) fn count(f: impl FnOnce()) -> usize {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            ACTIVE.with(|a| a.set(false));
        }
    }
    COUNT.with(|n| n.set(0));
    ACTIVE.with(|a| {
        assert!(!a.replace(true));
    });
    let guard = Guard;
    f();
    drop(guard);
    COUNT.with(Cell::get)
}
