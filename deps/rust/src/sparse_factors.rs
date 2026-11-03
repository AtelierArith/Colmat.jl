//! Sparse symbolic analyses and independent reusable numeric factorizations.
use crate::permutations::OwnedPermutation;
use crate::sparse::{SparseData, SparseOwner, Storage};
use crate::sparsity::SparsityOwner;
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, OwnedMatrix};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::solvers::{Solve, SolveLstsq};
use faer::sparse::{
    linalg::{
        cholesky::{self, LltRef, SymbolicCholesky, SymmetricOrdering},
        solvers::{Lu, Qr, SymbolicLu, SymbolicQr},
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
    Lu(SymbolicLu<usize>),
    Qr(SymbolicQr<usize>),
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
        pattern: *const u8,
        kind: i32,
        lower: bool,
        ordering: i32,
        permutation: *const u8,
        dense: f64,
        aggressive: bool,
        supernodal: i32,
    ) -> Result<(), String> {
        guarded(|| {
            let pattern = unsafe { &*pattern.cast::<SparsityOwner>() }.data()?.csc()?;
            let (m, n) = (pattern.nrows(), pattern.ncols());
            if kind != 1 && m != n {
                return Err("sparse LU/Cholesky requires a square matrix".into());
            }
            if kind == 1 && m < n {
                return Err("sparse QR requires at least as many rows as columns".into());
            }
            let analysis = match kind {
                0 => Analysis::Lu(
                    SymbolicLu::try_new(pattern.as_ref()).map_err(|e| format!("{e:?}"))?,
                ),
                1 => Analysis::Qr(
                    SymbolicQr::try_new(pattern.as_ref()).map_err(|e| format!("{e:?}"))?,
                ),
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
    pub fn sparse_symbolic_clone(&mut self, source: *const u8) -> Result<(), String> {
        guarded(|| {
            self.plan = Some(unsafe { &*source.cast::<Self>() }.plan()?.clone());
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
    Lu(Lu<usize, T>),
    Qr(Qr<usize, T>),
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
        Analysis::Lu(sym) => match Lu::try_new_with_symbolic(sym.clone(), a.as_ref(), Par::Seq) {
            Ok(f) => Numeric::Lu(f),
            Err(LuError::SymbolicSingular { index } | LuError::NumericSingular { index }) => {
                return Ok(Err(index + 1))
            }
            Err(e) => return Err(format!("{e:?}")),
        },
        Analysis::Qr(sym) => Numeric::Qr(
            Qr::try_new_with_symbolic(sym.clone(), a.as_ref(), Par::Seq)
                .map_err(|e| format!("{e:?}"))?,
        ),
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
                Err(e) => return Err(format!("{e:?}")),
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
        analysis: *const u8,
        matrix: *const u8,
    ) -> Result<usize, String> {
        guarded(|| {
            let plan = unsafe { &*analysis.cast::<SparseSymbolic>() }.plan()?;
            let a = unsafe { &*matrix.cast::<SparseOwner>() }.data()?;
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
    pub fn sparse_factor_clone(&mut self, source: *const u8) -> Result<(), String> {
        guarded(|| {
            self.factors = Some(unsafe { &*source.cast::<Self>() }.factors()?.clone());
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
        Numeric::Qr(f) => {
            if trans {
                return Err("transpose/adjoint sparse QR solve is not provided by faer".into());
            }
            f.solve_lstsq(b, Par::Seq)
        }
    })
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn sparse_factor_solve(
        &mut self,
        factor: *const u8,
        rhs: *const u8,
        trans: bool,
        adj: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let b = unsafe { &*rhs.cast::<OwnedMatrix>() };
            match unsafe { &*factor.cast::<SparseFactor>() }.factors()? {
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
        factor: *const u8,
        rhs: *const u8,
        trans: bool,
        adj: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let b = unsafe { &*rhs.cast::<TypedMatrix>() }.data()?;
            self.data = Some(
                match (unsafe { &*factor.cast::<SparseFactor>() }.factors()?, b) {
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
                },
            );
            Ok(())
        })
    }
}

#[julia]
impl OwnedPermutation {
    #[julia]
    pub fn sparse_cholesky_permutation(&mut self, analysis: *const u8) -> Result<(), String> {
        guarded(|| {
            let p = unsafe { &*analysis.cast::<SparseSymbolic>() }.plan()?;
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
