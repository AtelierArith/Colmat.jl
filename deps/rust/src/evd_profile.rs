//! Opt-in benchmark counters. Default builds contain only zero-sized scopes.
//! Counts Rust GlobalAlloc requests, not Julia allocations, RSS or OS stacks.
#[cfg(feature = "profile-general-evd")]
pub const STAGES: usize = 15;
#[cfg(feature = "profile-general-evd")]
pub const WORDS: usize = 9 + STAGES * 5;

#[cfg(not(feature = "profile-general-evd"))]
pub struct Scope;
#[cfg(not(feature = "profile-general-evd"))]
impl Scope {
    #[inline(always)]
    pub fn new(_: usize) -> Self {
        Self
    }
}

#[cfg(feature = "profile-general-evd")]
mod enabled {
    use super::{STAGES, WORDS};
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{
        AtomicU64, AtomicUsize,
        Ordering::{Relaxed, SeqCst},
    };
    use std::time::Instant;

    // 0=idle, 1=reset/snapshot, 2=capturing. A single external caller owns a
    // capture. Native worker allocations use the same active phase.
    static STATE: AtomicUsize = AtomicUsize::new(0);
    static RECORDING: AtomicUsize = AtomicUsize::new(0);
    static PHASE: AtomicUsize = AtomicUsize::new(STAGES - 1);
    static LIVE: AtomicU64 = AtomicU64::new(0);
    static BASE: AtomicU64 = AtomicU64::new(0);
    static PEAK: AtomicU64 = AtomicU64::new(0);
    struct Counters {
        allocations: AtomicU64,
        bytes: AtomicU64,
        frees: AtomicU64,
        freed_bytes: AtomicU64,
        nanos: AtomicU64,
    }
    impl Counters {
        const fn new() -> Self {
            Self {
                allocations: AtomicU64::new(0),
                bytes: AtomicU64::new(0),
                frees: AtomicU64::new(0),
                freed_bytes: AtomicU64::new(0),
                nanos: AtomicU64::new(0),
            }
        }
        fn clear(&self) {
            for value in [
                &self.allocations,
                &self.bytes,
                &self.frees,
                &self.freed_bytes,
                &self.nanos,
            ] {
                value.store(0, Relaxed);
            }
        }
        fn values(&self) -> [u64; 5] {
            [
                self.allocations.load(Relaxed),
                self.bytes.load(Relaxed),
                self.frees.load(Relaxed),
                self.freed_bytes.load(Relaxed),
                self.nanos.load(Relaxed),
            ]
        }
    }
    static COUNTS: [Counters; STAGES] = [const { Counters::new() }; STAGES];
    struct Recording;
    impl Recording {
        fn enter() -> Option<Self> {
            if STATE.load(SeqCst) != 2 {
                return None;
            }
            RECORDING.fetch_add(1, SeqCst);
            if STATE.load(SeqCst) != 2 {
                RECORDING.fetch_sub(1, SeqCst);
                return None;
            }
            Some(Self)
        }
    }
    impl Drop for Recording {
        fn drop(&mut self) {
            RECORDING.fetch_sub(1, SeqCst);
        }
    }
    fn allocated(bytes: usize) {
        let live = LIVE.fetch_add(bytes as u64, Relaxed) + bytes as u64;
        if let Some(_recording) = Recording::enter() {
            PEAK.fetch_max(live, Relaxed);
            let counters = &COUNTS[PHASE.load(Relaxed).min(STAGES - 1)];
            counters.allocations.fetch_add(1, Relaxed);
            counters.bytes.fetch_add(bytes as u64, Relaxed);
        }
    }
    fn freed(bytes: usize) {
        LIVE.fetch_sub(bytes as u64, Relaxed);
        if let Some(_recording) = Recording::enter() {
            let counters = &COUNTS[PHASE.load(Relaxed).min(STAGES - 1)];
            counters.frees.fetch_add(1, Relaxed);
            counters.freed_bytes.fetch_add(bytes as u64, Relaxed);
        }
    }
    struct MeasuredAllocator;
    #[global_allocator]
    static ALLOCATOR: MeasuredAllocator = MeasuredAllocator;
    unsafe impl GlobalAlloc for MeasuredAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ptr = System.alloc(layout);
            if !ptr.is_null() {
                allocated(layout.size());
            }
            ptr
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let ptr = System.alloc_zeroed(layout);
            if !ptr.is_null() {
                allocated(layout.size());
            }
            ptr
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            System.dealloc(ptr, layout);
            freed(layout.size());
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            let next = System.realloc(ptr, layout, size);
            if !next.is_null() {
                freed(layout.size());
                allocated(size);
            }
            next
        }
    }
    pub struct Scope {
        phase: usize,
        previous: usize,
        start: Option<Instant>,
    }
    impl Scope {
        #[inline]
        pub fn new(phase: usize) -> Self {
            if STATE.load(SeqCst) != 2 {
                return Self {
                    phase,
                    previous: STAGES - 1,
                    start: None,
                };
            }
            let previous = PHASE.swap(phase, Relaxed);
            Self {
                phase,
                previous,
                start: Some(Instant::now()),
            }
        }
    }
    impl Drop for Scope {
        fn drop(&mut self) {
            if let Some(start) = self.start {
                COUNTS[self.phase]
                    .nanos
                    .fetch_add(start.elapsed().as_nanos() as u64, Relaxed);
                PHASE.store(self.previous, Relaxed);
            }
        }
    }
    #[no_mangle]
    pub extern "C" fn colmat_evd_profile_begin() -> i32 {
        if STATE.compare_exchange(0, 1, SeqCst, SeqCst).is_err() {
            return -1;
        }
        for counters in &COUNTS {
            counters.clear();
        }
        PHASE.store(STAGES - 1, Relaxed);
        let base = LIVE.load(Relaxed);
        BASE.store(base, Relaxed);
        PEAK.store(base, Relaxed);
        STATE.store(2, SeqCst);
        0
    }
    /// Header: schema, stages, allocations, allocated bytes, frees, freed bytes,
    /// baseline live bytes, ending live bytes, extra peak live bytes. Then five
    /// words per phase: allocation count/bytes, free count/bytes, inclusive ns.
    #[no_mangle]
    pub unsafe extern "C" fn colmat_evd_profile_end(out: *mut u64, len: usize) -> i32 {
        if out.is_null() || len < WORDS {
            return -1;
        }
        if STATE.compare_exchange(2, 1, SeqCst, SeqCst).is_err() {
            return -2;
        }
        while RECORDING.load(SeqCst) != 0 {
            std::hint::spin_loop();
        }
        let totals = COUNTS.iter().fold([0u64; 4], |mut sum, counters| {
            for (slot, value) in sum.iter_mut().zip(counters.values()) {
                *slot += value;
            }
            sum
        });
        let header = [
            1,
            STAGES as u64,
            totals[0],
            totals[1],
            totals[2],
            totals[3],
            BASE.load(Relaxed),
            LIVE.load(Relaxed),
            PEAK.load(Relaxed).saturating_sub(BASE.load(Relaxed)),
        ];
        for (i, value) in header.into_iter().enumerate() {
            *out.add(i) = value;
        }
        for (stage, counters) in COUNTS.iter().enumerate() {
            for (j, value) in counters.values().into_iter().enumerate() {
                *out.add(9 + 5 * stage + j) = value;
            }
        }
        STATE.store(0, SeqCst);
        0
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn measures_allocation_zeroed_reallocation_and_peak_without_allocating_output() {
            assert_eq!(colmat_evd_profile_begin(), 0);
            assert_eq!(colmat_evd_profile_begin(), -1);
            let mut result = [0u64; WORDS];
            unsafe {
                let _phase = Scope::new(4);
                let layout = Layout::from_size_align(32, 8).unwrap();
                let pointer = std::alloc::alloc_zeroed(layout);
                assert!(!pointer.is_null());
                std::hint::black_box(std::slice::from_raw_parts(pointer, 32));
                let pointer = std::alloc::realloc(pointer, layout, 64);
                assert!(!pointer.is_null());
                std::alloc::dealloc(pointer, Layout::from_size_align(64, 8).unwrap());
            }
            assert_eq!(
                unsafe { colmat_evd_profile_end(result.as_mut_ptr(), WORDS - 1) },
                -1
            );
            assert_eq!(
                unsafe { colmat_evd_profile_end(result.as_mut_ptr(), WORDS) },
                0
            );
            assert_eq!(result[0..2], [1, STAGES as u64]);
            assert!(result[2] >= 2 && result[3] >= 96 && result[8] >= 64);
            assert!(result[9 + 5 * 4] >= 2);
            assert_eq!(
                unsafe { colmat_evd_profile_end(result.as_mut_ptr(), WORDS) },
                -2
            );
        }
    }
}
#[cfg(feature = "profile-general-evd")]
pub use enabled::Scope;
