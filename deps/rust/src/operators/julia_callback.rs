//! Callback identities persist; Julia function pointers are borrowed only in a
//! synchronous scope on the original calling thread, never by a worker thread.
use super::{Base, Data, Operator, OwnedLinearOperator};
use crate::guarded;
use faer::dyn_stack::{MemStack, StackReq};
use faer::mat::AsMatMut;
use faer::matrix_free::{BiLinOp, LinOp};
use faer::prelude::{Reborrow, ReborrowMut};
use faer::traits::{math_utils::*, ComplexField};
use faer::{MatMut, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;
use std::cell::Cell;
type Dispatcher = extern "C" fn(
    usize,
    i32,
    i32,
    usize,
    usize,
    usize,
    *const u8,
    isize,
    isize,
    *mut u8,
    isize,
    isize,
) -> i32;
thread_local! { static DISPATCH: Cell<Option<Dispatcher>> = const { Cell::new(None) }; }
struct Scope(Option<Dispatcher>);
impl Drop for Scope {
    fn drop(&mut self) {
        DISPATCH.with(|d| d.set(self.0));
    }
}
/// Execute a synchronous Julia operation with its callback dispatcher installed.
/// No borrowed function pointer is retained after this function returns.
#[julia]
pub fn with_operator_callbacks(
    dispatcher: *const u8,
    body: extern "C" fn() -> i32,
) -> Result<(), String> {
    guarded(|| {
        if dispatcher.is_null() {
            return Err("Julia callback dispatcher must be nonnull".into());
        }
        // SAFETY: Julia supplies its process-lived @cfunction entrypoint with
        // precisely Dispatcher's ABI, rooted in the loaded module. The scope
        // never transfers the pointer to workers and restores it on all exits.
        let callback: Dispatcher = unsafe { std::mem::transmute(dispatcher) };
        let guard = Scope(DISPATCH.with(|d| d.replace(Some(callback))));
        let status = body();
        drop(guard);
        if status != 1 {
            return Err("Julia callback operation failed".into());
        }
        Ok(())
    })
}
#[derive(Clone, Copy, Debug)]
pub(super) struct JuliaCallback {
    pub id: usize,
    pub m: usize,
    pub n: usize,
    pub tag: i32,
}
fn fail(message: &'static str) -> ! {
    // A failed callback is an expected adapter error. Resume unwinding to the
    // existing guarded boundary without printing the global panic hook.
    std::panic::resume_unwind(Box::new(message))
}
impl JuliaCallback {
    fn invoke<T: ComplexField>(&self, out: MatMut<'_, T>, b: MatRef<'_, T>, adjoint: bool) {
        let callback = DISPATCH.with(Cell::get).unwrap_or_else(|| {
            fail("Julia operator application requires a callback scope on the calling thread")
        });
        assert_eq!(b.nrows(), if adjoint { self.m } else { self.n });
        assert_eq!(
            out.shape(),
            (if adjoint { self.n } else { self.m }, b.ncols())
        );
        // Both references are valid for this synchronous call. Julia copies
        // inputs into owned arrays and fills a separate owned output before
        // writing all native output entries; it never exposes borrowed arrays.
        let status = callback(
            self.id,
            self.tag,
            i32::from(adjoint),
            out.nrows(),
            b.nrows(),
            b.ncols(),
            b.as_ptr().cast(),
            b.row_stride(),
            b.col_stride(),
            out.as_ptr_mut().cast(),
            out.row_stride(),
            out.col_stride(),
        );
        if status != 1 {
            fail("Julia operator callback failed");
        }
    }
    fn scratch<T: ComplexField>(&self, k: usize) -> StackReq {
        if self.m == 0 || self.n == 0 || k == 0 {
            return StackReq::EMPTY;
        }
        faer::linalg::temp_mat_scratch::<T>(self.m, k)
            .and(faer::linalg::temp_mat_scratch::<T>(self.n, k))
    }
    fn run<T: ComplexField, const ADJ: bool, const C: bool>(
        &self,
        mut out: MatMut<'_, T>,
        b: MatRef<'_, T>,
        stack: &mut MemStack,
    ) {
        if self.m == 0 || self.n == 0 || b.ncols() == 0 {
            out.fill(zero::<T>());
            return;
        }
        if !C {
            self.invoke(out, b, ADJ);
            return;
        }
        // SAFETY: input is fully initialized by the conjugating copy. Output
        // is fully initialized by a successful dispatcher before being read;
        // callback failure unwinds without reading uninitialized output.
        let (mut input, stack) =
            unsafe { faer::linalg::temp_mat_uninit::<T, _, _>(b.nrows(), b.ncols(), stack) };
        // SAFETY: a successful dispatcher initializes every result entry;
        // a failed dispatch unwinds before this temporary can be read.
        let (mut result, _) =
            unsafe { faer::linalg::temp_mat_uninit::<T, _, _>(out.nrows(), out.ncols(), stack) };
        let mut input = input.as_mat_mut();
        let mut result = result.as_mat_mut();
        for j in 0..b.ncols() {
            for i in 0..b.nrows() {
                input[(i, j)] = conj(&b[(i, j)]);
            }
        }
        self.invoke(result.rb_mut(), input.rb(), ADJ);
        for j in 0..out.ncols() {
            for i in 0..out.nrows() {
                out[(i, j)] = conj(&result[(i, j)]);
            }
        }
    }
}
impl<T: ComplexField> LinOp<T> for JuliaCallback {
    fn nrows(&self) -> usize {
        self.m
    }
    fn ncols(&self) -> usize {
        self.n
    }
    fn apply_scratch(&self, k: usize, _: ParShape) -> StackReq {
        self.scratch::<T>(k)
    }
    fn apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, _: Par<'_>, s: &mut MemStack) {
        self.run::<T, false, false>(o, b, s)
    }
    fn conj_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, _: Par<'_>, s: &mut MemStack) {
        self.run::<T, false, true>(o, b, s)
    }
}
impl<T: ComplexField> BiLinOp<T> for JuliaCallback {
    fn transpose_apply_scratch(&self, k: usize, _: ParShape) -> StackReq {
        self.scratch::<T>(k)
    }
    fn transpose_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, _: Par<'_>, s: &mut MemStack) {
        self.run::<T, true, true>(o, b, s)
    }
    fn adjoint_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, _: Par<'_>, s: &mut MemStack) {
        self.run::<T, true, false>(o, b, s)
    }
}
#[julia]
impl OwnedLinearOperator {
    #[julia]
    pub fn operator_callback(
        &mut self,
        id: usize,
        m: usize,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| {
            if id == 0 {
                return Err("Julia callback identity must be nonzero".into());
            }
            let a = JuliaCallback { id, m, n, tag };
            self.data = Some(match tag {
                0 => Data::F32(Operator::new(Base::Callback(a))),
                1 => Data::F64(Operator::new(Base::Callback(a))),
                2 => Data::C32(Operator::new(Base::Callback(a))),
                3 => Data::C64(Operator::new(Base::Callback(a))),
                _ => return Err("unsupported Julia callback scalar tag".into()),
            });
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern "C" fn first(
        _: usize,
        _: i32,
        _: i32,
        _: usize,
        _: usize,
        _: usize,
        _: *const u8,
        _: isize,
        _: isize,
        _: *mut u8,
        _: isize,
        _: isize,
    ) -> i32 {
        1
    }
    extern "C" fn second(
        _: usize,
        _: i32,
        _: i32,
        _: usize,
        _: usize,
        _: usize,
        _: *const u8,
        _: isize,
        _: isize,
        _: *mut u8,
        _: isize,
        _: isize,
    ) -> i32 {
        1
    }
    extern "C" fn active() -> i32 {
        i32::from(DISPATCH.with(|d| d.get().is_some()))
    }
    extern "C" fn reject() -> i32 {
        0
    }
    extern "C" fn nested() -> i32 {
        let old = DISPATCH.with(Cell::get).map(|f| f as usize);
        let worker_clear = std::thread::spawn(|| DISPATCH.with(|d| d.get().is_none()))
            .join()
            .unwrap_or(false);
        let ok = with_operator_callbacks(second as *const u8, active).is_ok();
        i32::from(ok && worker_clear && old == DISPATCH.with(Cell::get).map(|f| f as usize))
    }
    #[test]
    fn callback_scope_restores_nested_failed_and_foreign_thread_state() {
        assert!(DISPATCH.with(|d| d.get().is_none()));
        assert!(with_operator_callbacks(first as *const u8, nested).is_ok());
        assert!(DISPATCH.with(|d| d.get().is_none()));
        assert!(with_operator_callbacks(first as *const u8, reject).is_err());
        assert!(DISPATCH.with(|d| d.get().is_none()));
        assert!(with_operator_callbacks(std::ptr::null(), active).is_err());
    }
    #[test]
    fn callback_identity_never_invokes_julia_without_a_scope() {
        let a = JuliaCallback {
            id: 1,
            m: 2,
            n: 2,
            tag: 1,
        };
        let b = faer::Mat::<f64>::zeros(2, 1);
        let mut out = faer::Mat::<f64>::zeros(2, 1);
        let mut mem = faer::dyn_stack::MemBuffer::new(a.scratch::<f64>(1));
        let result = guarded(|| {
            a.run::<f64, false, false>(out.as_mut(), b.as_ref(), MemStack::new(&mut mem));
            Ok(())
        });
        assert!(result.is_err());
        assert_eq!(out, faer::Mat::zeros(2, 1));
    }
}
