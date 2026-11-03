//! Sparse symbolic analyses and independent reusable numeric factorizations.
use crate::permutations::OwnedPermutation;
use crate::sparse::{SparseData, SparseOwner, Storage};
use crate::sparsity::SparsityOwner;
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, OwnedMatrix};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::solvers::Solve;
use faer::sparse::{
    linalg::{
        cholesky::{self, LltRef, SymbolicCholesky, SymmetricOrdering},
        solvers::{Lu, SymbolicLu},
        LuError, SupernodalThreshold,
    },
    SymbolicSparseColMat,
};
use faer::traits::{
    math_utils::{copy, zero},
    ComplexField,
};
use faer::{c32, c64, Conj, Mat, MatRef, Par, Side};
use rustcall_julia_macros::julia;
use std::sync::Arc;

#[derive(Clone)]
enum Analysis {
    SimplicialLu(faer::perm::Perm<usize>),
    SimplicialQr(crate::sparse_qr_components::Plan),
    Lu(SymbolicLu<usize>),
    Qr(Arc<faer::sparse::linalg::qr::SymbolicQr<usize>>),
    Llt(Arc<SymbolicCholesky<usize>>),
}
#[derive(Clone)]
struct Plan {
    analysis: Analysis,
    pattern: SymbolicSparseColMat<usize>,
    lower: bool,
}
#[julia]
#[derive(Default)]
pub struct SparseSymbolic {
    plan: Option<Plan>,
}
impl SparseSymbolic {
    fn plan(&self) -> Result<&Plan, String> {
        self.plan
            .as_ref()
            .ok_or_else(|| "sparse symbolic analysis is empty".into())
    }
}
#[julia]
impl SparseSymbolic {
    #[julia]
    pub fn new() -> Self {
        Self::default()
    }
    #[julia]
    pub fn sparse_analyze(
        &mut self,
        pattern: &SparsityOwner,
        kind: i32,
        lower: bool,
        ordering: i32,
        permutation: *const u8,
        dense: f64,
        aggressive: bool,
        supernodal: i32,
    ) -> Result<(), String> {
        guarded(|| {
            let pattern = pattern.data()?.csc()?;
            let (m, n) = (pattern.nrows(), pattern.ncols());
            if kind != 1 && m != n {
                return Err("sparse LU/Cholesky requires a square matrix".into());
            }
            if kind == 1 && m < n {
                return Err("sparse QR requires at least as many rows as columns".into());
            }
            let analysis = match kind {
                0 if supernodal == 1 => {
                    let p = if ordering == 2 {
                        unsafe { &*permutation.cast::<OwnedPermutation>() }
                            .data()?
                            .clone()
                    } else {
                        let p: Vec<usize> = (0..n).collect();
                        faer::perm::Perm::new_checked(
                            p.clone().into_boxed_slice(),
                            p.into_boxed_slice(),
                            n,
                        )
                    };
                    if p.len() != n {
                        return Err("LU column permutation length mismatch".into());
                    }
                    Analysis::SimplicialLu(p)
                }
                0 => Analysis::Lu(
                    SymbolicLu::try_new(pattern.as_ref()).map_err(|e| format!("{e:?}"))?,
                ),
                1 if supernodal == 1 => {
                    let owned;
                    let cols = if ordering == 2 {
                        unsafe { &*permutation.cast::<OwnedPermutation>() }
                            .data()?
                            .as_ref()
                    } else {
                        owned = faer::perm::Perm::new_checked(
                            (0..n).collect::<Vec<_>>().into_boxed_slice(),
                            (0..n).collect::<Vec<_>>().into_boxed_slice(),
                            n,
                        );
                        owned.as_ref()
                    };
                    if cols.len() != n {
                        return Err("QR column permutation length mismatch".into());
                    }
                    Analysis::SimplicialQr(crate::sparse_qr_components::Plan::new(
                        pattern.as_ref(),
                        cols,
                    )?)
                }
                1 => Analysis::Qr(Arc::new(
                    faer::sparse::linalg::qr::factorize_symbolic_qr(
                        pattern.as_ref(),
                        faer::sparse::linalg::qr::QrSymbolicParams {
                            supernodal_flop_ratio_threshold: if supernodal == 2 {
                                SupernodalThreshold::FORCE_SUPERNODAL
                            } else {
                                SupernodalThreshold::AUTO
                            },
                            ..Default::default()
                        },
                    )
                    .map_err(|e| format!("{e:?}"))?,
                )),
                2 => {
                    if !dense.is_finite() {
                        return Err("AMD threshold must be finite".into());
                    }
                    let ord = match ordering {
                        0 => SymmetricOrdering::Amd,
                        1 => SymmetricOrdering::Identity,
                        2 => {
                            let p = unsafe { &*permutation.cast::<OwnedPermutation>() }.data()?;
                            if p.len() != n {
                                return Err("Cholesky permutation length mismatch".into());
                            }
                            SymmetricOrdering::Custom(p.as_ref())
                        }
                        _ => return Err("unknown Cholesky ordering".into()),
                    };
                    let threshold = match supernodal {
                        0 => SupernodalThreshold::AUTO,
                        1 => SupernodalThreshold::FORCE_SIMPLICIAL,
                        2 => SupernodalThreshold::FORCE_SUPERNODAL,
                        _ => return Err("unknown sparse factor algorithm".into()),
                    };
                    Analysis::Llt(Arc::new(
                        cholesky::factorize_symbolic_cholesky(
                            pattern.as_ref(),
                            if lower { Side::Lower } else { Side::Upper },
                            ord,
                            cholesky::CholeskySymbolicParams {
                                amd_params: faer::sparse::linalg::amd::Control {
                                    dense,
                                    aggressive,
                                },
                                supernodal_flop_ratio_threshold: threshold,
                                ..Default::default()
                            },
                        )
                        .map_err(|e| format!("{e:?}"))?,
                    ))
                }
                _ => return Err("unknown sparse analysis kind".into()),
            };
            self.plan = Some(Plan {
                analysis,
                pattern,
                lower,
            });
            Ok(())
        })
    }
    #[julia]
    pub fn sparse_symbolic_clone(&mut self, source: &Self) -> Result<(), String> {
        guarded(|| {
            self.plan = Some(source.plan()?.clone());
            Ok(())
        })
    }
}
#[derive(Clone)]
struct SparseLlt<T> {
    symbolic: Arc<SymbolicCholesky<usize>>,
    values: Vec<T>,
}
#[derive(Clone)]
enum Numeric<T> {
    SimplicialLu(crate::sparse_lu_components::SimplicialFactors<T>),
    SimplicialQr(crate::sparse_qr_components::Factors<T>),
    Lu(Lu<usize, T>),
    Qr(crate::sparse_qr_owned::Factors<T>),
    Llt(SparseLlt<T>),
}
#[derive(Clone)]
enum Factors {
    F32(Numeric<f32>),
    F64(Numeric<f64>),
    C32(Numeric<c32>),
    C64(Numeric<c64>),
}
#[julia]
#[derive(Default)]
pub struct SparseFactor {
    factors: Option<Factors>,
}
impl SparseFactor {
    fn factors(&self) -> Result<&Factors, String> {
        self.factors
            .as_ref()
            .ok_or_else(|| "sparse factorization is unsuccessful or empty".into())
    }
}
fn numeric<T: ComplexField>(p: &Plan, a: &Storage<T>) -> Result<Result<Numeric<T>, usize>, String> {
    let a = a.csc()?;
    let s = a.symbolic();
    if (s.nrows(), s.ncols()) != (p.pattern.nrows(), p.pattern.ncols())
        || s.col_ptr() != p.pattern.col_ptr()
        || s.row_idx() != p.pattern.row_idx()
    {
        return Err(
            "numeric matrix sparsity pattern must match the symbolic analysis exactly".into(),
        );
    }
    Ok(Ok(match &p.analysis {
        Analysis::SimplicialLu(cols) => {
            match crate::sparse_lu_components::SimplicialFactors::new(a.as_ref(), cols.as_ref()) {
                Ok(f) => Numeric::SimplicialLu(f),
                Err(LuError::SymbolicSingular { index } | LuError::NumericSingular { index }) => {
                    return Ok(Err(index + 1))
                }
                Err(e) => return Err(format!("{e:?}")),
            }
        }
        Analysis::Lu(sym) => match Lu::try_new_with_symbolic(sym.clone(), a.as_ref(), Par::Seq) {
            Ok(f) => Numeric::Lu(f),
            Err(LuError::SymbolicSingular { index } | LuError::NumericSingular { index }) => {
                return Ok(Err(index + 1))
            }
            Err(e) => return Err(format!("{e:?}")),
        },
        Analysis::SimplicialQr(sym) => {
            Numeric::SimplicialQr(crate::sparse_qr_components::Factors::new(sym, a.as_ref())?)
        }
        Analysis::Qr(sym) => Numeric::Qr(crate::sparse_qr_owned::Factors::new(
            sym.clone(),
            a.as_ref(),
        )?),
        Analysis::Llt(sym) => {
            // faer's identity-order numeric path can pass the original matrix
            // straight to its triangle-only kernel. Strip the ignored half
            // explicitly so stored entries there never contaminate workspace.
            let entries: Vec<_> = a
                .as_ref()
                .triplet_iter()
                .filter(|e| {
                    if p.lower {
                        e.row >= e.col
                    } else {
                        e.row <= e.col
                    }
                })
                .map(|e| faer::sparse::Triplet {
                    row: e.row,
                    col: e.col,
                    val: copy(e.val),
                })
                .collect();
            let triangle =
                faer::sparse::SparseColMat::try_new_from_triplets(a.nrows(), a.ncols(), &entries)
                    .map_err(|e| format!("{e:?}"))?;
            let mut values = vec![zero::<T>(); sym.len_val()];
            let par = Par::Seq;
            let mut mem = MemBuffer::try_new(
                sym.factorize_numeric_llt_scratch::<T>(par.shape(), Default::default()),
            )
            .map_err(|e| format!("{e:?}"))?;
            match sym.factorize_numeric_llt(
                &mut values,
                triangle.as_ref(),
                if p.lower { Side::Lower } else { Side::Upper },
                Default::default(),
                par,
                MemStack::new(&mut mem),
                Default::default(),
            ) {
                Ok(_) => Numeric::Llt(SparseLlt {
                    symbolic: sym.clone(),
                    values,
                }),
                Err(faer::linalg::cholesky::llt::factor::LltError::NonPositivePivot { index }) => {
                    // faer's sparse simplicial kernel reports a 1-based pivot;
                    // its supernodal kernel forwards the dense 0-based index.
                    let info = match sym.raw() {
                        cholesky::SymbolicCholeskyRaw::Simplicial(_) => index,
                        cholesky::SymbolicCholeskyRaw::Supernodal(_) => index + 1,
                    };
                    return Ok(Err(info));
                }
            }
        }
    }))
}
#[julia]
impl SparseFactor {
    #[julia]
    pub fn new() -> Self {
        Self::default()
    }
    #[julia]
    pub fn sparse_factorize(
        &mut self,
        analysis: &SparseSymbolic,
        matrix: &SparseOwner,
    ) -> Result<usize, String> {
        guarded(|| {
            let plan = analysis.plan()?;
            let a = matrix.data()?;
            let result = match a {
                SparseData::F32(a) => numeric(plan, a)?.map(Factors::F32),
                SparseData::F64(a) => numeric(plan, a)?.map(Factors::F64),
                SparseData::C32(a) => numeric(plan, a)?.map(Factors::C32),
                SparseData::C64(a) => numeric(plan, a)?.map(Factors::C64),
            };
            match result {
                Ok(f) => {
                    self.factors = Some(f);
                    Ok(0)
                }
                Err(info) => {
                    self.factors = None;
                    Ok(info)
                }
            }
        })
    }
    #[julia]
    pub fn sparse_factor_clone(&mut self, source: &Self) -> Result<(), String> {
        guarded(|| {
            self.factors = Some(source.factors()?.clone());
            Ok(())
        })
    }
}
fn solve<T: ComplexField>(
    f: &Numeric<T>,
    b: MatRef<'_, T>,
    trans: bool,
    adj: bool,
) -> Result<Mat<T>, String> {
    macro_rules! run {
        ($f:expr) => {
            if trans {
                if adj {
                    $f.solve_adjoint(b, Par::Seq)
                } else {
                    $f.solve_transpose(b, Par::Seq)
                }
            } else {
                $f.solve(b, Par::Seq)
            }
        };
    }
    Ok(match f {
        Numeric::Lu(f) => run!(f),
        Numeric::SimplicialLu(f) => f.solve(b, trans, trans && adj),
        Numeric::Llt(f) => {
            let mut out = b.to_owned();
            let par = Par::Seq;
            let mut mem = MemBuffer::try_new(
                f.symbolic
                    .solve_in_place_scratch::<T>(b.ncols(), par.shape()),
            )
            .map_err(|e| format!("{e:?}"))?;
            LltRef::new(&f.symbolic, &f.values).solve_in_place_with_conj(
                if trans && !adj { Conj::Yes } else { Conj::No },
                out.as_mut(),
                par,
                MemStack::new(&mut mem),
            );
            out
        }
        Numeric::SimplicialQr(f) => {
            if trans {
                return f.solve_transpose(b, adj);
            }
            f.solve(b)?
        }
        Numeric::Qr(f) => {
            if trans {
                return Err("transpose/adjoint sparse QR solve is not provided by faer".into());
            }
            f.solve(b)?
        }
    })
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn sparse_factor_solve(
        &mut self,
        factor: &SparseFactor,
        rhs: &OwnedMatrix,
        trans: bool,
        adj: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let b = rhs;
            match factor.factors()? {
                Factors::F64(f) => self.matrix = solve(f, b.matrix.as_ref(), trans, adj)?,
                _ => return Err("sparse factor and RHS scalar types must match".into()),
            }
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn sparse_factor_solve(
        &mut self,
        factor: &SparseFactor,
        rhs: &TypedMatrix,
        trans: bool,
        adj: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let b = rhs.data()?;
            self.data = Some(match (factor.factors()?, b) {
                (Factors::F32(f), MatrixData::F32(b)) => {
                    MatrixData::F32(solve(f, b.as_ref(), trans, adj)?)
                }
                (Factors::C32(f), MatrixData::C32(b)) => {
                    MatrixData::C32(solve(f, b.as_ref(), trans, adj)?)
                }
                (Factors::C64(f), MatrixData::C64(b)) => {
                    MatrixData::C64(solve(f, b.as_ref(), trans, adj)?)
                }
                _ => return Err("sparse factor and RHS scalar types must match".into()),
            });
            Ok(())
        })
    }
}

#[julia]
impl OwnedPermutation {
    #[julia]
    pub fn sparse_cholesky_permutation(&mut self, analysis: &SparseSymbolic) -> Result<(), String> {
        guarded(|| {
            let p = analysis.plan()?;
            let Analysis::Llt(sym) = &p.analysis else {
                return Err("permutation query currently requires sparse Cholesky".into());
            };
            let n = sym.nrows();
            self.data = Some(match sym.perm() {
                Some(p) => {
                    let (fwd, inv) = p.arrays();
                    faer::perm::Perm::new_checked(
                        fwd.to_vec().into_boxed_slice(),
                        inv.to_vec().into_boxed_slice(),
                        n,
                    )
                }
                None => faer::perm::Perm::new_checked(
                    (0..n).collect::<Vec<_>>().into_boxed_slice(),
                    (0..n).collect::<Vec<_>>().into_boxed_slice(),
                    n,
                ),
            });
            Ok(())
        })
    }
}
#[julia]
impl SparseSymbolic {
    #[julia]
    pub fn sparse_cholesky_algorithm(&self) -> Result<i32, String> {
        guarded(|| {
            let Analysis::Llt(sym) = &self.plan()?.analysis else {
                return Err("algorithm query requires sparse Cholesky".into());
            };
            Ok(match sym.raw() {
                cholesky::SymbolicCholeskyRaw::Simplicial(_) => 1,
                cholesky::SymbolicCholeskyRaw::Supernodal(_) => 2,
            })
        })
    }
}
#[julia]
impl SparseSymbolic {
    #[julia]
    pub fn sparse_cholesky_storage(&self) -> Result<usize, String> {
        guarded(|| {
            let Analysis::Llt(sym) = &self.plan()?.analysis else {
                return Err("storage query requires sparse Cholesky".into());
            };
            Ok(sym.len_val())
        })
    }
}

// Export only mathematical lower-triangle entries. Supernode buffers also
// contain unused upper block entries, which must not become sparse values.
fn cholesky_entries<T: ComplexField>(
    f: &SparseLlt<T>,
) -> Vec<faer::sparse::Triplet<usize, usize, T>> {
    use faer::sparse::Triplet;
    let mut entries = Vec::new();
    match f.symbolic.raw() {
        cholesky::SymbolicCholeskyRaw::Simplicial(sym) => {
            for col in 0..sym.ncols() {
                for k in sym.col_ptr()[col]..sym.col_ptr()[col + 1] {
                    entries.push(Triplet {
                        row: sym.row_idx()[k],
                        col,
                        val: copy(&f.values[k]),
                    });
                }
            }
        }
        cholesky::SymbolicCholeskyRaw::Supernodal(sym) => {
            let factor = cholesky::supernodal::SupernodalLltRef::new(sym, &f.values);
            for s in 0..sym.n_supernodes() {
                let node = factor.supernode(s);
                let block = node.val();
                let width = block.ncols();
                let start = node.start();
                for j in 0..width {
                    for i in j..width {
                        entries.push(Triplet {
                            row: start + i,
                            col: start + j,
                            val: copy(&block[(i, j)]),
                        });
                    }
                    for (i, &row) in node.pattern().iter().enumerate() {
                        entries.push(Triplet {
                            row,
                            col: start + j,
                            val: copy(&block[(width + i, j)]),
                        });
                    }
                }
            }
        }
    }
    entries
}
fn cholesky_component<T: ComplexField>(
    f: &SparseLlt<T>,
    upper: bool,
) -> Result<Storage<T>, String> {
    let mut entries = cholesky_entries(f);
    if upper {
        for e in &mut entries {
            std::mem::swap(&mut e.row, &mut e.col);
            e.val = faer::traits::math_utils::conj(&e.val);
        }
    }
    let n = f.symbolic.nrows();
    Ok(Storage::Csc(
        faer::sparse::SparseColMat::try_new_from_triplets(n, n, &entries)
            .map_err(|e| format!("{e:?}"))?,
    ))
}
#[julia]
impl SparseOwner {
    #[julia]
    pub fn sparse_cholesky_component(
        &mut self,
        factor: &SparseFactor,
        upper: bool,
    ) -> Result<(), String> {
        guarded(|| {
            self.data = Some(match factor.factors()? {
                Factors::F32(Numeric::Llt(f)) => SparseData::F32(cholesky_component(f, upper)?),
                Factors::F64(Numeric::Llt(f)) => SparseData::F64(cholesky_component(f, upper)?),
                Factors::C32(Numeric::Llt(f)) => SparseData::C32(cholesky_component(f, upper)?),
                Factors::C64(Numeric::Llt(f)) => SparseData::C64(cholesky_component(f, upper)?),
                _ => return Err("component export requires sparse Cholesky".into()),
            });
            Ok(())
        })
    }
}
fn cholesky_diagonal<T: ComplexField>(f: &SparseLlt<T>) -> Vec<T> {
    let mut out = Vec::with_capacity(f.symbolic.nrows());
    match f.symbolic.raw() {
        cholesky::SymbolicCholeskyRaw::Simplicial(sym) => {
            for col in 0..sym.ncols() {
                out.push(copy(&f.values[sym.col_ptr()[col]]));
            }
        }
        cholesky::SymbolicCholeskyRaw::Supernodal(sym) => {
            let factor = cholesky::supernodal::SupernodalLltRef::new(sym, &f.values);
            for s in 0..sym.n_supernodes() {
                let node = factor.supernode(s);
                for j in 0..node.val().ncols() {
                    out.push(copy(&node.val()[(j, j)]));
                }
            }
        }
    }
    out
}
#[julia]
impl SparseFactor {
    #[julia]
    pub fn sparse_cholesky_diagonal(&self, out: *mut u8, n: usize) -> Result<(), String> {
        guarded(|| {
            fn write<T: ComplexField>(
                f: &SparseLlt<T>,
                out: *mut u8,
                n: usize,
            ) -> Result<(), String> {
                if n != f.symbolic.nrows() {
                    return Err("Cholesky diagonal length mismatch".into());
                }
                for (i, value) in cholesky_diagonal(f).into_iter().enumerate() {
                    unsafe {
                        out.cast::<T>().add(i).write(value);
                    }
                }
                Ok(())
            }
            match self.factors()? {
                Factors::F32(Numeric::Llt(f)) => write(f, out, n),
                Factors::F64(Numeric::Llt(f)) => write(f, out, n),
                Factors::C32(Numeric::Llt(f)) => write(f, out, n),
                Factors::C64(Numeric::Llt(f)) => write(f, out, n),
                _ => Err("diagonal export requires sparse Cholesky".into()),
            }
        })
    }
}

fn lu_component<T: ComplexField>(
    f: &crate::sparse_lu_components::SimplicialFactors<T>,
    upper: bool,
) -> Result<Storage<T>, String> {
    let a = if upper {
        f.numeric.u_factor_unsorted()
    } else {
        f.numeric.l_factor_unsorted()
    };
    let entries: Vec<_> = a
        .triplet_iter()
        .map(|e| faer::sparse::Triplet {
            row: e.row,
            col: e.col,
            val: copy(e.val),
        })
        .collect();
    Ok(Storage::Csc(
        faer::sparse::SparseColMat::try_new_from_triplets(a.nrows(), a.ncols(), &entries)
            .map_err(|e| format!("{e:?}"))?,
    ))
}
#[julia]
impl SparseOwner {
    #[julia]
    pub fn sparse_lu_component(
        &mut self,
        source: &SparseFactor,
        upper: bool,
    ) -> Result<(), String> {
        guarded(|| {
            self.data = Some(match source.factors()? {
                Factors::F32(Numeric::SimplicialLu(f)) => SparseData::F32(lu_component(f, upper)?),
                Factors::F64(Numeric::SimplicialLu(f)) => SparseData::F64(lu_component(f, upper)?),
                Factors::C32(Numeric::SimplicialLu(f)) => SparseData::C32(lu_component(f, upper)?),
                Factors::C64(Numeric::SimplicialLu(f)) => SparseData::C64(lu_component(f, upper)?),
                _ => return Err("LU components require algorithm=:simplicial".into()),
            });
            Ok(())
        })
    }
}
#[julia]
impl OwnedPermutation {
    #[julia]
    pub fn sparse_lu_permutation(
        &mut self,
        source: &SparseFactor,
        column: bool,
    ) -> Result<(), String> {
        guarded(|| {
            macro_rules! get {
                ($f:expr) => {
                    if column {
                        $f.cols.clone()
                    } else {
                        $f.rows.clone()
                    }
                };
            }
            self.data = Some(match source.factors()? {
                Factors::F32(Numeric::SimplicialLu(f)) => get!(f),
                Factors::F64(Numeric::SimplicialLu(f)) => get!(f),
                Factors::C32(Numeric::SimplicialLu(f)) => get!(f),
                Factors::C64(Numeric::SimplicialLu(f)) => get!(f),
                _ => return Err("LU permutations require algorithm=:simplicial".into()),
            });
            Ok(())
        })
    }
}

#[julia]
impl SparseOwner {
    #[julia]
    pub fn sparse_qr_r(&mut self, factor: &SparseFactor) -> Result<(), String> {
        guarded(|| {
            fn export<T: ComplexField>(
                f: &crate::sparse_qr_components::Factors<T>,
            ) -> Result<Storage<T>, String> {
                let entries: Vec<_> =
                    f.r.as_ref()
                        .triplet_iter()
                        .map(|e| faer::sparse::Triplet {
                            row: e.row,
                            col: e.col,
                            val: copy(e.val),
                        })
                        .collect();
                Ok(Storage::Csc(
                    faer::sparse::SparseColMat::try_new_from_triplets(
                        f.r.nrows(),
                        f.r.ncols(),
                        &entries,
                    )
                    .map_err(|e| format!("{e:?}"))?,
                ))
            }
            self.data = Some(match factor.factors()? {
                Factors::F32(Numeric::SimplicialQr(f)) => SparseData::F32(export(f)?),
                Factors::F64(Numeric::SimplicialQr(f)) => SparseData::F64(export(f)?),
                Factors::C32(Numeric::SimplicialQr(f)) => SparseData::C32(export(f)?),
                Factors::C64(Numeric::SimplicialQr(f)) => SparseData::C64(export(f)?),
                _ => return Err("QR components require algorithm=:simplicial".into()),
            });
            Ok(())
        })
    }
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn sparse_qr_q(&mut self, factor: &SparseFactor) -> Result<(), String> {
        guarded(|| {
            let Factors::F64(Numeric::SimplicialQr(f)) = factor.factors()? else {
                return Err("QR Q requires Float64 simplicial factors".into());
            };
            self.matrix = f.q();
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn sparse_qr_q(&mut self, factor: &SparseFactor) -> Result<(), String> {
        guarded(|| {
            self.data = Some(match factor.factors()? {
                Factors::F32(Numeric::SimplicialQr(f)) => MatrixData::F32(f.q()),
                Factors::F64(Numeric::SimplicialQr(f)) => MatrixData::F64(f.q()),
                Factors::C32(Numeric::SimplicialQr(f)) => MatrixData::C32(f.q()),
                Factors::C64(Numeric::SimplicialQr(f)) => MatrixData::C64(f.q()),
                _ => return Err("QR Q requires simplicial factors".into()),
            });
            Ok(())
        })
    }
}
#[julia]
impl OwnedPermutation {
    #[julia]
    pub fn sparse_qr_permutation(&mut self, factor: &SparseFactor) -> Result<(), String> {
        guarded(|| {
            self.data = Some(match factor.factors()? {
                Factors::F32(Numeric::SimplicialQr(f)) => f.plan.cols.clone(),
                Factors::F64(Numeric::SimplicialQr(f)) => f.plan.cols.clone(),
                Factors::C32(Numeric::SimplicialQr(f)) => f.plan.cols.clone(),
                Factors::C64(Numeric::SimplicialQr(f)) => f.plan.cols.clone(),
                _ => return Err("QR permutation requires simplicial factors".into()),
            });
            Ok(())
        })
    }
}

#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn sparse_qr_reconstruct(&mut self, factor: &SparseFactor) -> Result<(), String> {
        guarded(|| {
            let Factors::F64(Numeric::SimplicialQr(f)) = factor.factors()? else {
                return Err("QR reconstruction requires Float64 simplicial factors".into());
            };
            self.matrix = f.reconstruct();
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn sparse_qr_reconstruct(&mut self, factor: &SparseFactor) -> Result<(), String> {
        guarded(|| {
            self.data = Some(match factor.factors()? {
                Factors::F32(Numeric::SimplicialQr(f)) => MatrixData::F32(f.reconstruct()),
                Factors::F64(Numeric::SimplicialQr(f)) => MatrixData::F64(f.reconstruct()),
                Factors::C32(Numeric::SimplicialQr(f)) => MatrixData::C32(f.reconstruct()),
                Factors::C64(Numeric::SimplicialQr(f)) => MatrixData::C64(f.reconstruct()),
                _ => return Err("QR reconstruction requires simplicial factors".into()),
            });
            Ok(())
        })
    }
}

#[julia]
impl SparseFactor {
    #[julia]
    pub fn sparse_qr_has_components(&self) -> Result<bool, String> {
        guarded(|| {
            Ok(matches!(
                self.factors()?,
                Factors::F32(Numeric::SimplicialQr(_))
                    | Factors::F64(Numeric::SimplicialQr(_))
                    | Factors::C32(Numeric::SimplicialQr(_))
                    | Factors::C64(Numeric::SimplicialQr(_))
            ))
        })
    }
}
