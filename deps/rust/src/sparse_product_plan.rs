//! Reusable sparse x sparse product plans.
//!
//! faer splits `sparse_sparse_matmul` into a symbolic phase (result pattern and
//! a flop-count partition) and a numeric phase that writes into a destination
//! whose pattern is exactly that result. A plan runs the symbolic phase once
//! and keeps copies of both operand patterns, so every numeric call can verify
//! (without allocating) that the operands and the destination still have the
//! planned structure before any value is written. The numeric phase needs a
//! dense work column per thread, supplied by the caller's scratch.
//!
//! CSR operands are handled as column-major transposes: `Cᵀ = Bᵀ Aᵀ`, so a CSR
//! plan is the column-major plan of `(Bᵀ, Aᵀ)` and its result is `Cᵀ`.
//!
//! The plan follows the storage format of `A`. A `B` stored in the other
//! format is transposed once at planning time: the plan keeps the transposed
//! pattern and a value permutation, and each numeric call gathers `B`'s
//! values through it into a buffer taken from the caller's scratch.
use crate::borrowed_sparse::{
    numeric as borrowed_numeric, numeric_mut as borrowed_numeric_mut,
    symbolic as borrowed_symbolic, Raw,
};
use crate::guarded;
use crate::sparse::{SparseData, SparseOwner, Storage};
use crate::teams::{with_team, OwnedTeam};
use crate::workspace::OwnedScratch;
use faer::dyn_stack::StackReq;
use faer::reborrow::ReborrowMut;
use faer::sparse::linalg::matmul::{
    sparse_sparse_matmul_numeric, sparse_sparse_matmul_numeric_scratch,
    sparse_sparse_matmul_symbolic, SparseMatMulInfo,
};
use faer::sparse::{
    SparseColMat, SparseColMatMut, SparseColMatRef, SymbolicSparseColMat, SymbolicSparseColMatRef,
};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Accum, Par, ParShape};
use rustcall_julia_macros::julia;

/// Status codes of a numeric call; nonzero codes leave the destination intact.
pub(crate) const OK: usize = 0;
pub(crate) const LHS_PATTERN: usize = 1;
pub(crate) const RHS_PATTERN: usize = 2;
pub(crate) const DST_PATTERN: usize = 3;
pub(crate) const FORMAT: usize = 4;
pub(crate) const SCALAR: usize = 5;
pub(crate) const SCRATCH: usize = 6;

/// An owned copy of one column-major pattern.
struct Pattern {
    nrows: usize,
    ncols: usize,
    ptr: Vec<usize>,
    idx: Vec<usize>,
}
impl Pattern {
    fn of(s: SymbolicSparseColMatRef<'_, usize>) -> Self {
        let n = s.ncols();
        let end = s.col_ptr()[n];
        Self {
            nrows: s.nrows(),
            ncols: n,
            ptr: s.col_ptr().to_vec(),
            idx: s.row_idx()[..end].to_vec(),
        }
    }
    fn matches(&self, s: SymbolicSparseColMatRef<'_, usize>) -> bool {
        s.nrows() == self.nrows
            && s.ncols() == self.ncols
            && s.col_nnz().is_none()
            && s.col_ptr() == self.ptr.as_slice()
            && s.row_idx().get(..self.idx.len()) == Some(self.idx.as_slice())
    }
    fn view(&self) -> SymbolicSparseColMatRef<'_, usize> {
        // SAFETY: copied from a validated faer pattern.
        unsafe {
            SymbolicSparseColMatRef::new_unchecked(
                self.nrows, self.ncols, &self.ptr, None, &self.idx,
            )
        }
    }
}

/// The transpose of a stored column-major pattern, with `perm[q]` the stored
/// value index of transposed entry `q`.
struct Transposed {
    pattern: Pattern,
    perm: Vec<usize>,
}
impl Transposed {
    fn of(s: SymbolicSparseColMatRef<'_, usize>) -> Self {
        let (m, n) = (s.nrows(), s.ncols());
        let (ptr, idx) = (s.col_ptr(), s.row_idx());
        let k = ptr[n];
        let mut tptr = vec![0usize; m + 1];
        for &i in &idx[..k] {
            tptr[i + 1] += 1;
        }
        for i in 0..m {
            tptr[i + 1] += tptr[i];
        }
        let mut next = tptr[..m].to_vec();
        let mut tidx = vec![0usize; k];
        let mut perm = vec![0usize; k];
        for j in 0..n {
            for p in ptr[j]..ptr[j + 1] {
                let q = &mut next[idx[p]];
                tidx[*q] = j;
                perm[*q] = p;
                *q += 1;
            }
        }
        Self {
            pattern: Pattern {
                nrows: n,
                ncols: m,
                ptr: tptr,
                idx: tidx,
            },
            perm,
        }
    }
}

struct Plan {
    csr: bool,
    /// Storage format of `B` (differs from `csr` when `b_transposed` is set).
    b_csr: bool,
    /// Column-major patterns of the stored `A` and `B` (`Aᵀ`, `Bᵀ` for CSR).
    a: Pattern,
    b: Pattern,
    /// For a `B` stored in the other format: the transpose of its stored
    /// column-major pattern, i.e. the column-major view the plan needs.
    b_transposed: Option<Transposed>,
    /// Column-major result: `C`, or `Cᵀ` for CSR.
    result: SymbolicSparseColMat<usize>,
    info: SparseMatMulInfo,
}
impl Plan {
    /// `a`/`b` are the stored column-major patterns of `A` (format `csr`) and
    /// `B` (format `b_csr`).
    fn build(
        a: SymbolicSparseColMatRef<'_, usize>,
        b: SymbolicSparseColMatRef<'_, usize>,
        csr: bool,
        b_csr: bool,
    ) -> Result<Self, String> {
        let b_transposed = (csr != b_csr).then(|| Transposed::of(b));
        let bv = b_transposed.as_ref().map_or(b, |t| t.pattern.view());
        let (lhs, rhs) = if csr { (bv, a) } else { (a, bv) };
        if lhs.ncols() != rhs.nrows() {
            return Err("sparse product dimensions must match".into());
        }
        let (result, info) =
            sparse_sparse_matmul_symbolic(lhs, rhs).map_err(|e| format!("{e:?}"))?;
        Ok(Self {
            csr,
            b_csr,
            a: Pattern::of(a),
            b: Pattern::of(b),
            b_transposed,
            result,
            info,
        })
    }
    /// `(rows, cols)` of the user-level product `C`.
    fn dims(&self) -> (usize, usize) {
        let (m, n) = (self.result.nrows(), self.result.ncols());
        if self.csr {
            (n, m)
        } else {
            (m, n)
        }
    }
    fn requirement<T: ComplexField>(&self, par: ParShape) -> StackReq {
        let gathered = self
            .b_transposed
            .as_ref()
            .map_or(StackReq::EMPTY, |t| StackReq::new::<T>(t.perm.len()));
        gathered.and(sparse_sparse_matmul_numeric_scratch::<usize, T>(
            self.result.as_ref(),
            par,
        ))
    }
    /// Validate every pattern, then `C = alpha * A * B + beta * C` in place
    /// (`beta == None` replaces). The arguments are column-major views of the
    /// stored operands; nothing is written unless every check passes.
    #[allow(clippy::too_many_arguments)]
    fn numeric<T: ComplexField>(
        &self,
        dst: SparseColMatMut<'_, usize, T>,
        a: SparseColMatRef<'_, usize, T>,
        b: SparseColMatRef<'_, usize, T>,
        alpha: T,
        beta: Option<T>,
        par: Par<'_>,
        scratch: &mut OwnedScratch,
    ) -> Result<usize, String> {
        if !self.a.matches(a.symbolic()) {
            return Ok(LHS_PATTERN);
        }
        if !self.b.matches(b.symbolic()) {
            return Ok(RHS_PATTERN);
        }
        let r = self.result.as_ref();
        let d = dst.symbolic();
        if d.nrows() != r.nrows()
            || d.ncols() != r.ncols()
            || d.col_nnz().is_some()
            || d.col_ptr() != r.col_ptr()
            || d.row_idx() != r.row_idx()
        {
            return Ok(DST_PATTERN);
        }
        let stack = match scratch.stack(self.requirement::<T>(par.shape())) {
            Ok(stack) => stack,
            Err(_) => return Ok(SCRATCH),
        };
        let mut dst = dst;
        let accum = match beta {
            None => Accum::Replace,
            Some(beta) => {
                if beta != one::<T>() {
                    for v in dst.rb_mut().val_mut() {
                        *v = mul(&beta, v);
                    }
                }
                Accum::Add
            }
        };
        let Some(t) = &self.b_transposed else {
            let (lhs, rhs) = if self.csr { (b, a) } else { (a, b) };
            sparse_sparse_matmul_numeric(dst, accum, lhs, rhs, alpha, &self.info, par, stack);
            return Ok(OK);
        };
        let stored = b.val();
        let (vals, stack) = stack.make_with::<T>(t.perm.len(), |q| copy(&stored[t.perm[q]]));
        let b = SparseColMatRef::new(t.pattern.view(), &vals);
        let (lhs, rhs) = if self.csr { (b, a) } else { (a, b) };
        sparse_sparse_matmul_numeric(dst, accum, lhs, rhs, alpha, &self.info, par, stack);
        Ok(OK)
    }
}

/// Column-major pattern of a stored owner and whether it is CSR.
fn pattern_of(d: &SparseData) -> (SymbolicSparseColMatRef<'_, usize>, bool) {
    fn of<T>(a: &Storage<T>) -> (SymbolicSparseColMatRef<'_, usize>, bool) {
        match a {
            Storage::Csc(a) => (a.symbolic(), false),
            Storage::Csr(a) => (a.symbolic().transpose(), true),
        }
    }
    match d {
        SparseData::F32(a) => of(a),
        SparseData::F64(a) => of(a),
        SparseData::C32(a) => of(a),
        SparseData::C64(a) => of(a),
    }
}
fn column_major<T>(a: &Storage<T>) -> SparseColMatRef<'_, usize, T> {
    match a {
        Storage::Csc(a) => a.as_ref(),
        Storage::Csr(a) => a.as_ref().transpose(),
    }
}
fn column_major_mut<T>(a: &mut Storage<T>) -> (SparseColMatMut<'_, usize, T>, bool) {
    match a {
        Storage::Csc(a) => (a.as_dyn_mut(), false),
        Storage::Csr(a) => (a.transpose_mut(), true),
    }
}
// SAFETY: alpha/beta address one `T` each; beta is ignored unless `accumulate`.
unsafe fn scalars<T: ComplexField>(
    alpha: *const u8,
    beta: *const u8,
    accumulate: bool,
) -> (T, Option<T>) {
    let alpha = copy(&*alpha.cast::<T>());
    let beta = accumulate.then(|| copy(&*beta.cast::<T>()));
    (alpha, beta)
}
fn write_req(req: StackReq, out: *mut usize) {
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
}
fn requirement_for(plan: &Plan, tag: i32, par: ParShape) -> Result<StackReq, String> {
    Ok(match tag {
        0 => plan.requirement::<f32>(par),
        1 => plan.requirement::<f64>(par),
        2 => plan.requirement::<faer::c32>(par),
        3 => plan.requirement::<faer::c64>(par),
        _ => return Err("invalid sparse product scalar".into()),
    })
}

#[julia]
pub struct SparseProductPlan {
    plan: Option<Plan>,
}
impl SparseProductPlan {
    fn plan(&self) -> Result<&Plan, String> {
        self.plan
            .as_ref()
            .ok_or_else(|| "sparse product plan is closed or uninitialized".into())
    }
}

/// Owned operands: `dst = alpha * lhs * rhs (+ beta * dst)` in one storage format.
#[allow(clippy::too_many_arguments)]
fn owned_mul(
    plan: &Plan,
    dst: &mut SparseOwner,
    lhs: &SparseOwner,
    rhs: &SparseOwner,
    alpha: *const u8,
    beta: *const u8,
    accumulate: bool,
    team: Option<&OwnedTeam>,
    scratch: &mut OwnedScratch,
) -> Result<usize, String> {
    fn run<T: ComplexField>(
        plan: &Plan,
        d: &mut Storage<T>,
        a: &Storage<T>,
        b: &Storage<T>,
        (alpha, beta): (T, Option<T>),
        team: Option<&OwnedTeam>,
        scratch: &mut OwnedScratch,
    ) -> Result<usize, String> {
        let (a, b) = (column_major(a), column_major(b));
        let (d, csr) = column_major_mut(d);
        if csr != plan.csr {
            return Ok(FORMAT);
        }
        with_team(team, |par| plan.numeric(d, a, b, alpha, beta, par, scratch))
    }
    // Operand formats are checked against the plan before any value is read.
    let (_, lcsr) = pattern_of(lhs.data()?);
    let (_, rcsr) = pattern_of(rhs.data()?);
    if lcsr != plan.csr || rcsr != plan.b_csr {
        return Ok(FORMAT);
    }
    let a = lhs.data()?;
    let b = rhs.data()?;
    let d = dst
        .data
        .as_mut()
        .ok_or("sparse matrix is not initialized")?;
    // SAFETY: Julia roots the scalar boxes for this call.
    unsafe {
        match (d, a, b) {
            (SparseData::F32(d), SparseData::F32(a), SparseData::F32(b)) => run(
                plan,
                d,
                a,
                b,
                scalars(alpha, beta, accumulate),
                team,
                scratch,
            ),
            (SparseData::F64(d), SparseData::F64(a), SparseData::F64(b)) => run(
                plan,
                d,
                a,
                b,
                scalars(alpha, beta, accumulate),
                team,
                scratch,
            ),
            (SparseData::C32(d), SparseData::C32(a), SparseData::C32(b)) => run(
                plan,
                d,
                a,
                b,
                scalars(alpha, beta, accumulate),
                team,
                scratch,
            ),
            (SparseData::C64(d), SparseData::C64(a), SparseData::C64(b)) => run(
                plan,
                d,
                a,
                b,
                scalars(alpha, beta, accumulate),
                team,
                scratch,
            ),
            _ => Ok(SCALAR),
        }
    }
}

/// Raw column-major operands of scalar `tag` (borrowed Julia CSC matrices or
/// raw parts of owners) into an owner of the plan's format.
#[allow(clippy::too_many_arguments)]
unsafe fn borrowed_mul(
    plan: &Plan,
    dst: &mut SparseOwner,
    a: Raw,
    b: Raw,
    tag: i32,
    alpha: *const u8,
    beta: *const u8,
    accumulate: bool,
    team: Option<&OwnedTeam>,
    scratch: &mut OwnedScratch,
) -> Result<usize, String> {
    unsafe fn run<T: ComplexField>(
        plan: &Plan,
        d: &mut Storage<T>,
        a: Raw,
        b: Raw,
        (alpha, beta): (T, Option<T>),
        team: Option<&OwnedTeam>,
        scratch: &mut OwnedScratch,
    ) -> Result<usize, String> {
        let (d, csr) = column_major_mut(d);
        if csr != plan.csr {
            return Ok(FORMAT);
        }
        let (a, b) = (borrowed_numeric::<T>(a)?, borrowed_numeric::<T>(b)?);
        with_team(team, |par| plan.numeric(d, a, b, alpha, beta, par, scratch))
    }
    let d = dst
        .data
        .as_mut()
        .ok_or("sparse matrix is not initialized")?;
    match (d, tag) {
        (SparseData::F32(d), 0) => run(
            plan,
            d,
            a,
            b,
            scalars(alpha, beta, accumulate),
            team,
            scratch,
        ),
        (SparseData::F64(d), 1) => run(
            plan,
            d,
            a,
            b,
            scalars(alpha, beta, accumulate),
            team,
            scratch,
        ),
        (SparseData::C32(d), 2) => run(
            plan,
            d,
            a,
            b,
            scalars(alpha, beta, accumulate),
            team,
            scratch,
        ),
        (SparseData::C64(d), 3) => run(
            plan,
            d,
            a,
            b,
            scalars(alpha, beta, accumulate),
            team,
            scratch,
        ),
        _ => Ok(SCALAR),
    }
}

/// Raw CSC operands into a borrowed Julia CSC destination `dst` whose values
/// live at `dval`. Julia rejects destinations that might alias an operand.
#[allow(clippy::too_many_arguments)]
unsafe fn borrowed_into(
    plan: &Plan,
    dst: Raw,
    dval: *mut u8,
    a: Raw,
    b: Raw,
    tag: i32,
    alpha: *const u8,
    beta: *const u8,
    accumulate: bool,
    team: Option<&OwnedTeam>,
    scratch: &mut OwnedScratch,
) -> Result<usize, String> {
    #[allow(clippy::too_many_arguments)]
    unsafe fn run<T: ComplexField>(
        plan: &Plan,
        dst: Raw,
        dval: *mut u8,
        a: Raw,
        b: Raw,
        (alpha, beta): (T, Option<T>),
        team: Option<&OwnedTeam>,
        scratch: &mut OwnedScratch,
    ) -> Result<usize, String> {
        if plan.csr {
            return Ok(FORMAT);
        }
        let d = borrowed_numeric_mut::<T>(dst, dval)?;
        let (a, b) = (borrowed_numeric::<T>(a)?, borrowed_numeric::<T>(b)?);
        with_team(team, |par| plan.numeric(d, a, b, alpha, beta, par, scratch))
    }
    let s = (alpha, beta, accumulate);
    match tag {
        0 => run::<f32>(plan, dst, dval, a, b, scalars(s.0, s.1, s.2), team, scratch),
        1 => run::<f64>(plan, dst, dval, a, b, scalars(s.0, s.1, s.2), team, scratch),
        2 => run::<faer::c32>(plan, dst, dval, a, b, scalars(s.0, s.1, s.2), team, scratch),
        3 => run::<faer::c64>(plan, dst, dval, a, b, scalars(s.0, s.1, s.2), team, scratch),
        _ => Ok(SCALAR),
    }
}

#[allow(clippy::too_many_arguments)]
fn raw(m: usize, n: usize, ptr: *const usize, idx: *const usize, val: *const u8, k: usize) -> Raw {
    Raw {
        m,
        n,
        ptr,
        idx,
        val,
        k,
    }
}

#[julia]
impl SparseProductPlan {
    #[julia]
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_plan_mul_into_borrowed(
        &self,
        dm: usize,
        dn: usize,
        dptr: *const usize,
        didx: *const usize,
        dval: *mut u8,
        dk: usize,
        m: usize,
        n: usize,
        ptr: *const usize,
        idx: *const usize,
        val: *const u8,
        k: usize,
        m2: usize,
        n2: usize,
        ptr2: *const usize,
        idx2: *const usize,
        val2: *const u8,
        k2: usize,
        tag: i32,
        alpha: *const u8,
        beta: *const u8,
        accumulate: bool,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
    ) -> Result<usize, String> {
        guarded(|| unsafe {
            borrowed_into(
                self.plan()?,
                raw(dm, dn, dptr, didx, dval.cast_const(), dk),
                dval,
                raw(m, n, ptr, idx, val, k),
                raw(m2, n2, ptr2, idx2, val2, k2),
                tag,
                alpha,
                beta,
                accumulate,
                team,
                scratch,
            )
        })
    }
}

#[julia]
impl SparseProductPlan {
    #[julia]
    pub fn new() -> Self {
        Self { plan: None }
    }
    /// Symbolic phase of `A * B` for two owners; the plan follows `A`'s format.
    #[julia]
    pub fn sparse_product_plan_build(
        &mut self,
        lhs: &SparseOwner,
        rhs: &SparseOwner,
    ) -> Result<(), String> {
        guarded(|| {
            let (a, acsr) = pattern_of(lhs.data()?);
            let (b, bcsr) = pattern_of(rhs.data()?);
            self.plan = Some(Plan::build(a, b, acsr, bcsr)?);
            Ok(())
        })
    }
    /// Symbolic phase of `A * B` from raw column-major patterns (borrowed
    /// Julia CSC matrices or `sparse_csc_raw_parts` of owners); `acsr`/`bcsr`
    /// say whether each operand is a CSR owner read as its transpose.
    #[julia]
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_product_plan_build_borrowed(
        &mut self,
        m: usize,
        n: usize,
        ptr: *const usize,
        idx: *const usize,
        k: usize,
        m2: usize,
        n2: usize,
        ptr2: *const usize,
        idx2: *const usize,
        k2: usize,
        acsr: bool,
        bcsr: bool,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let raw = |m, n, ptr, idx, k| Raw {
                m,
                n,
                ptr,
                idx,
                val: std::ptr::null(),
                k,
            };
            let a = borrowed_symbolic(raw(m, n, ptr, idx, k))?;
            let b = borrowed_symbolic(raw(m2, n2, ptr2, idx2, k2))?;
            self.plan = Some(Plan::build(a, b, acsr, bcsr)?);
            Ok(())
        })
    }
    /// Independent plan with the same operand patterns (symbolic phase rerun).
    #[julia]
    pub fn sparse_product_plan_clone_from(
        &mut self,
        other: &SparseProductPlan,
    ) -> Result<(), String> {
        guarded(|| {
            let p = other.plan()?;
            self.plan = Some(Plan::build(p.a.view(), p.b.view(), p.csr, p.b_csr)?);
            Ok(())
        })
    }
    /// Write `[rows, cols, nnz, csr, b_csr]` of the planned product.
    #[julia]
    pub fn sparse_product_plan_info(&self, out: *mut usize) -> Result<(), String> {
        guarded(|| {
            let p = self.plan()?;
            let (m, n) = p.dims();
            unsafe {
                out.write(m);
                out.add(1).write(n);
                out.add(2).write(p.result.row_idx().len());
                out.add(3).write(p.csr as usize);
                out.add(4).write(p.b_csr as usize);
            }
            Ok(())
        })
    }
    /// Write sequential `(bytes, alignment)` of the numeric phase for `tag`.
    #[julia]
    pub fn sparse_product_plan_requirements(
        &self,
        tag: i32,
        out: &mut [usize],
    ) -> Result<(), String> {
        let out = crate::workspace::requirement_out(out)?;
        guarded(|| {
            write_req(requirement_for(self.plan()?, tag, ParShape::Seq)?, out);
            Ok(())
        })
    }
    #[julia]
    pub fn sparse_product_plan_close(&mut self) -> Result<(), String> {
        guarded(|| {
            self.plan = None;
            Ok(())
        })
    }
}
#[julia]
impl OwnedTeam {
    /// Numeric-phase `(bytes, alignment)` under this team's execution shape.
    #[julia]
    pub fn sparse_product_plan_team_requirements(
        &self,
        plan: &SparseProductPlan,
        tag: i32,
        out: &mut [usize],
    ) -> Result<(), String> {
        let out = crate::workspace::requirement_out(out)?;
        guarded(|| {
            let p = plan.plan()?;
            let req = self.execute(|par| requirement_for(p, tag, par.shape()))??;
            write_req(req, out);
            Ok(())
        })
    }
}
#[julia]
impl SparseOwner {
    /// Zero-valued owner with the planned result pattern and format.
    #[julia]
    pub fn sparse_from_product_plan(
        &mut self,
        plan: &SparseProductPlan,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| {
            let p = plan.plan()?;
            fn zeros<T: ComplexField>(p: &Plan) -> Storage<T> {
                let c = SparseColMat::new(
                    p.result.clone(),
                    vec![zero::<T>(); p.result.row_idx().len()],
                );
                if p.csr {
                    Storage::Csr(c.into_transpose())
                } else {
                    Storage::Csc(c)
                }
            }
            self.data = Some(match tag {
                0 => SparseData::F32(zeros(p)),
                1 => SparseData::F64(zeros(p)),
                2 => SparseData::C32(zeros(p)),
                3 => SparseData::C64(zeros(p)),
                _ => return Err("invalid sparse product scalar".into()),
            });
            Ok(())
        })
    }
    #[julia]
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_plan_mul(
        &mut self,
        plan: &SparseProductPlan,
        lhs: &SparseOwner,
        rhs: &SparseOwner,
        alpha: *const u8,
        beta: *const u8,
        accumulate: bool,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
    ) -> Result<usize, String> {
        guarded(|| {
            owned_mul(
                plan.plan()?,
                self,
                lhs,
                rhs,
                alpha,
                beta,
                accumulate,
                team,
                scratch,
            )
        })
    }

    #[julia]
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_plan_mul_borrowed(
        &mut self,
        plan: &SparseProductPlan,
        m: usize,
        n: usize,
        ptr: *const usize,
        idx: *const usize,
        val: *const u8,
        k: usize,
        m2: usize,
        n2: usize,
        ptr2: *const usize,
        idx2: *const usize,
        val2: *const u8,
        k2: usize,
        tag: i32,
        alpha: *const u8,
        beta: *const u8,
        accumulate: bool,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
    ) -> Result<usize, String> {
        guarded(|| unsafe {
            let a = Raw {
                m,
                n,
                ptr,
                idx,
                val,
                k,
            };
            let b = Raw {
                m: m2,
                n: n2,
                ptr: ptr2,
                idx: idx2,
                val: val2,
                k: k2,
            };
            borrowed_mul(
                plan.plan()?,
                self,
                a,
                b,
                tag,
                alpha,
                beta,
                accumulate,
                team,
                scratch,
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::c64;
    use faer::sparse::Triplet;

    fn storage(m: usize, n: usize, seed: usize, csr: bool) -> Storage<c64> {
        let mut t = Vec::new();
        for j in 0..n {
            for i in 0..m {
                if (i * 7 + j * 3 + seed) % 4 == 0 {
                    t.push(Triplet::new(
                        i,
                        j,
                        c64::new((i + seed) as f64, j as f64 - 1.0),
                    ));
                }
            }
        }
        let a = SparseColMat::<usize, c64>::try_new_from_triplets(m, n, &t).unwrap();
        if csr {
            Storage::Csr(a.to_row_major().unwrap())
        } else {
            Storage::Csc(a)
        }
    }
    fn owner(s: Storage<c64>) -> SparseOwner {
        SparseOwner {
            data: Some(SparseData::C64(s)),
        }
    }
    fn dense(o: &SparseOwner) -> faer::Mat<c64> {
        let Some(SparseData::C64(s)) = &o.data else {
            unreachable!()
        };
        let c = s.csc().unwrap();
        c.to_dense()
    }
    fn plan_for(a: &SparseOwner, b: &SparseOwner) -> SparseProductPlan {
        let mut p = SparseProductPlan::new();
        p.sparse_product_plan_build(a, b).unwrap();
        p
    }
    fn scratch_for(p: &SparseProductPlan) -> OwnedScratch {
        let mut req = [0usize; 2];
        p.sparse_product_plan_requirements(3, &mut req[..])
            .unwrap();
        let mut s = OwnedScratch::new();
        s.scratch_reserve(req[0], req[1]).unwrap();
        s
    }

    #[test]
    fn planned_products_match_dense_and_do_not_allocate() {
        for (csr, b_csr) in [(false, false), (true, true), (false, true), (true, false)] {
            let a = owner(storage(6, 5, 1, csr));
            let b = owner(storage(5, 7, 2, b_csr));
            let p = plan_for(&a, &b);
            let mut c = SparseOwner::new();
            c.sparse_from_product_plan(&p, 3).unwrap();
            let mut s = scratch_for(&p);
            let (alpha, beta) = (c64::new(2.0, -1.0), c64::new(0.5, 0.25));
            let gold = &dense(&a) * &dense(&b);
            let mut code = usize::MAX;
            let count = crate::allocation_probe::count(|| {
                code = c
                    .sparse_plan_mul(
                        &p,
                        &a,
                        &b,
                        (&alpha as *const c64).cast(),
                        (&beta as *const c64).cast(),
                        false,
                        &mut s,None
                    )
                    .unwrap();
            });
            assert_eq!((code, count), (OK, 0), "csr={csr} b_csr={b_csr}");
            let first = dense(&c);
            assert!((&first - &gold * faer::Scale(alpha)).norm_max() < 1e-10);
            // Accumulate: C = alpha*A*B + beta*C.
            let count = crate::allocation_probe::count(|| {
                code = c
                    .sparse_plan_mul(
                        &p,
                        &a,
                        &b,
                        (&alpha as *const c64).cast(),
                        (&beta as *const c64).cast(),
                        true,
                        &mut s,None
                    )
                    .unwrap();
            });
            assert_eq!((code, count), (OK, 0), "csr={csr} b_csr={b_csr}");
            assert!(
                (dense(&c) - (&gold * faer::Scale(alpha) + &first * faer::Scale(beta))).norm_max()
                    < 1e-10
            );
        }
    }

    #[test]
    fn mismatched_patterns_and_formats_leave_the_destination_intact() {
        let a = owner(storage(6, 5, 1, false));
        let b = owner(storage(5, 7, 2, false));
        let p = plan_for(&a, &b);
        let mut c = SparseOwner::new();
        c.sparse_from_product_plan(&p, 3).unwrap();
        let mut s = scratch_for(&p);
        let one = c64::new(1.0, 0.0);
        let one = (&one as *const c64).cast();
        assert_eq!(
            c.sparse_plan_mul(&p, &a, &b, one, one, false, &mut s, None)
                .unwrap(),
            OK
        );
        let before = dense(&c);
        let other = owner(storage(6, 5, 3, false));
        let wide = owner(storage(5, 7, 0, false));
        assert_eq!(
            c.sparse_plan_mul(&p, &other, &b, one, one, false, &mut s, None)
                .unwrap(),
            LHS_PATTERN
        );
        assert_eq!(
            c.sparse_plan_mul(&p, &a, &wide, one, one, false, &mut s, None)
                .unwrap(),
            RHS_PATTERN
        );
        let csr = owner(storage(6, 5, 1, true));
        assert_eq!(
            c.sparse_plan_mul(&p, &csr, &b, one, one, false, &mut s, None)
                .unwrap(),
            FORMAT
        );
        let b_csr = owner(storage(5, 7, 2, true));
        assert_eq!(
            c.sparse_plan_mul(&p, &a, &b_csr, one, one, false, &mut s, None)
                .unwrap(),
            FORMAT
        );
        // A plan over a CSR `B` rejects the CSC `B` and too small a scratch.
        let mixed = plan_for(&a, &b_csr);
        let mut short = OwnedScratch::new();
        let mut req = [0usize; 2];
        p.sparse_product_plan_requirements(3, &mut req[..])
            .unwrap();
        short.scratch_reserve(req[0], req[1]).unwrap();
        assert_eq!(
            c.sparse_plan_mul(&mixed, &a, &b, one, one, false, &mut s, None)
                .unwrap(),
            FORMAT
        );
        assert_eq!(
            c.sparse_plan_mul(&mixed, &a, &b_csr, one, one, false, &mut short, None)
                .unwrap(),
            SCRATCH
        );
        let mut wrong = owner(storage(6, 7, 0, false));
        assert_eq!(
            wrong
                .sparse_plan_mul(&p, &a, &b, one, one, false, &mut s, None)
                .unwrap(),
            DST_PATTERN
        );
        let mut small = OwnedScratch::new();
        small.scratch_reserve(0, 1).unwrap();
        assert_eq!(
            c.sparse_plan_mul(&p, &a, &b, one, one, false, &mut small, None)
                .unwrap(),
            SCRATCH
        );
        assert_eq!(dense(&c), before);
    }

    #[test]
    fn team_products_and_clones_match_the_sequential_plan() {
        let a = owner(storage(300, 200, 1, false));
        let b = owner(storage(200, 250, 2, false));
        let p = plan_for(&a, &b);
        let mut q = SparseProductPlan::new();
        q.sparse_product_plan_clone_from(&p).unwrap();
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        let mut req = [0usize; 2];
        team.sparse_product_plan_team_requirements(&q, 3, &mut req[..])
            .unwrap();
        let mut s = OwnedScratch::new();
        s.scratch_reserve(req[0], req[1]).unwrap();
        let mut c = SparseOwner::new();
        c.sparse_from_product_plan(&q, 3).unwrap();
        let one = c64::new(1.0, 0.0);
        let one = (&one as *const c64).cast();
        assert_eq!(
            c.sparse_plan_mul(&q, &a, &b, one, one, false, &mut s, Some(&team))
                .unwrap(),
            OK
        );
        assert!((dense(&c) - &dense(&a) * &dense(&b)).norm_max() < 1e-8);
    }
}
