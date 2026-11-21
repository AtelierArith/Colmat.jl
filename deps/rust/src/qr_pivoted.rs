//! Column-pivoted QR adapter primitives. Pivot choices use fresh residual
//! column norms rather than translating a diagonal-based native rank estimate.
//! Large factorizations use the blocked `?laqps` panel scheme so each trailing
//! update is one GEMM; small ones keep the unblocked level-2 path.
use crate::householder::ReflectorScalar;
use crate::kernels::{nrm2, Scalar};
use colmat::Par;

pub(crate) struct PivotedQr {
    pub permutation: Vec<usize>,
    pub rank: usize,
    pub residual: f64,
    pub initial: f64,
    pub exception: usize,
}

/// Panel width for the blocked factorization. The trailing update of a panel is
/// one GEMM, which is where the parallel speedup comes from.
const PANEL: usize = 16;
/// Narrower panels for factorizations whose trailing matrix is still small: the
/// panel's `F` bookkeeping and the per-column rank-1 update dominate there, and
/// a 16-wide panel only pulls ahead for large trailing matrices.
const SMALL_PANEL: usize = 8;
/// Scaled `m*n` below which the narrow panel is the faster one: 8 wins for real
/// matrices up to about 245 wide and for complex ones up to about 200 wide,
/// while a 256-wide factorization is already faster with 16. The complex limit
/// is larger because the weight below already counts five real updates per
/// complex one.
const SMALL_PANEL_WORK_REAL: usize = 60_000;
const SMALL_PANEL_WORK_COMPLEX: usize = 200_000;
/// Real element updates below which a pivoted-QR panel stays off the resident
/// team. A 256-wide real panel (about 1M updates) is 6% faster threaded at four
/// and eight threads, while the last, smaller panel of the same factorization
/// is faster sequential; 400k keeps the large panels threaded.
const PIVOTED_PANEL_WORK: usize = 400_000;
/// Real work at or below which trailing updates stay on the level-2 path. On
/// the current Householder kernels the unblocked path wins to 128 wide real and
/// 80 wide complex; the panel only pulls ahead once its trailing GEMM is large.
const UNBLOCKED_WORK_REAL: usize = 16_384;
const UNBLOCKED_WORK_COMPLEX: usize = 32_000;

unsafe fn swap_columns<T: Scalar>(m: usize, a: *mut T, lda: isize, i: usize, j: usize) {
    if i == j {
        return;
    }
    for row in 0..m {
        std::ptr::swap(
            a.offset(row as isize + i as isize * lda),
            a.offset(row as isize + j as isize * lda),
        );
    }
}

/// Panel width policy. Blocking only pays for itself when the panel leaves a
/// trailing matrix worth multiplying. Small real factorizations (up to 64x64)
/// stay on the level-2 path, which avoids the panel's extra GEMVs; complex
/// arithmetic does four times the work per element, so even a 32x32 complex
/// factorization is faster blocked. The panel narrows from 16 to 8 while the
/// trailing matrix is small, where the wider panel's bookkeeping outweighs the
/// GEMM it feeds.
fn panel_width(m: usize, n: usize, complex: bool, k: usize) -> usize {
    let work = m
        .saturating_mul(n)
        .saturating_mul(if complex { 5 } else { 1 });
    let small = if complex {
        SMALL_PANEL_WORK_COMPLEX
    } else {
        SMALL_PANEL_WORK_REAL
    };
    let unblocked = if complex {
        UNBLOCKED_WORK_COMPLEX
    } else {
        UNBLOCKED_WORK_REAL
    };
    if k <= 8 || work <= unblocked {
        1
    } else if work < small {
        SMALL_PANEL.min(k)
    } else {
        PANEL.min(k)
    }
}

pub(crate) unsafe fn pivoted_qr<T: ReflectorScalar>(
    m: usize,
    n: usize,
    nrhs: usize,
    a: *mut T,
    lda: isize,
    tau: *mut T,
    fixed: &[bool],
    max_rank: usize,
    abs_tol: f64,
    rel_tol: f64,
) -> PivotedQr {
    let k = m.min(n).min(max_rank);
    let panel = panel_width(m, n, T::COMPLEX, k);
    pivoted_qr_panel(
        m, n, nrhs, a, lda, tau, fixed, max_rank, abs_tol, rel_tol, panel,
    )
}

/// Blocked column-pivoted QR in the style of LAPACK's `?laqps`. `panel` is the
/// maximum number of reflectors per panel; `panel == 1` is the unblocked
/// algorithm. Pivot search stays global: the panel keeps the row of each
/// reflector in the `F` factor so the residual norm of every trailing column is
/// downdated at each step, exactly as the unblocked code does.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn pivoted_qr_panel<T: ReflectorScalar>(
    m: usize,
    n: usize,
    nrhs: usize,
    a: *mut T,
    lda: isize,
    tau: *mut T,
    fixed: &[bool],
    max_rank: usize,
    abs_tol: f64,
    rel_tol: f64,
    panel: usize,
) -> PivotedQr {
    let mut permutation: Vec<usize> = (0..n).collect();
    // Stable fixed-column order is part of JPVT's public contract.
    let order: Vec<usize> = (0..n)
        .filter(|&j| fixed[j])
        .chain((0..n).filter(|&j| !fixed[j]))
        .collect();
    for (target, &original) in order.iter().enumerate() {
        let source = permutation.iter().position(|&p| p == original).unwrap();
        if source != target {
            for i in 0..m {
                std::ptr::swap(
                    a.offset(i as isize + target as isize * lda),
                    a.offset(i as isize + source as isize * lda),
                );
            }
            permutation.swap(target, source);
        }
    }
    let fixed_count = fixed.iter().filter(|&&x| x).count();
    let mut initial = 0.0_f64;
    let mut exception = 0;
    // Residual column norms are maintained between steps, as in LAPACK's GEQP3.
    // Recomputing every remaining column norm from scratch at each step makes
    // the factorization O(n^3) in norm passes alone (and was 12-30x slower than
    // the reference backends).
    let mut norms: Vec<f64> = Vec::with_capacity(n);
    for j in 0..n {
        let norm = nrm2(m as isize, a.offset(j as isize * lda), 1);
        if norm.is_nan() {
            return PivotedQr {
                permutation,
                rank: 0,
                residual: f64::NAN,
                initial: f64::NAN,
                exception: j + 1,
            };
        }
        if norm.is_infinite() && exception == 0 {
            exception = n + j + 1;
        }
        initial = initial.max(norm);
        norms.push(norm);
    }
    let mut result = PivotedQr {
        permutation,
        rank: 0,
        residual: initial,
        initial,
        exception,
    };
    let k = m.min(n).min(max_rank);
    let total = n + nrhs;
    let panel = panel.max(1).min(k.max(1));
    // `F` holds the row of each panel reflector for every remaining column:
    // `F[column + step * total]`. It is the `F` of LAPACK's `?laqps`, which keeps
    // the column norms exact while the panel update stays a single GEMM.
    let mut f = vec![T::z(); total.max(1) * panel];
    let mut auxv = vec![T::z(); panel];
    let mut row = vec![T::z(); panel];
    // Contiguous staging for the row of `A` a panel step updates. The axpy that
    // updates it vectorizes in place, and the norm downdate then reads the row
    // sequentially instead of stepping through one cache line per column.
    let mut rowfull = vec![T::z(); total.max(1)];
    let mut work = vec![T::z(); total.max(1)];
    let adj = if T::COMPLEX { b'C' } else { b'T' };
    let recompute_tol = T::EPS.sqrt();
    let mut start = 0usize;
    while start < k {
        let width = panel.min(k - start);
        // One team acquisition per panel: the panel issues one large GEMV per
        // reflector plus the trailing GEMM, and paying the pool handshake per
        // call made the threaded factorization slower than the sequential one.
        // Small panels stay sequential: entering the resident team costs about
        // 8 us, more than a panel of a small factorization does.
        let outcome = if panel > 1 {
            let panel_work =
                crate::threading::real_work(m - start, total - start, width, T::COMPLEX);
            crate::threading::with_par_if(panel_work, PIVOTED_PANEL_WORK, |par| {
                run_panel(
                    m,
                    n,
                    total,
                    a,
                    lda,
                    tau,
                    start,
                    width,
                    fixed_count,
                    initial,
                    abs_tol,
                    rel_tol,
                    recompute_tol,
                    adj,
                    &mut norms,
                    &mut f,
                    &mut auxv,
                    &mut row,
                    &mut rowfull,
                    &mut work,
                    &mut result,
                    par,
                )
            })
        } else {
            run_panel(
                m,
                n,
                total,
                a,
                lda,
                tau,
                start,
                width,
                fixed_count,
                initial,
                abs_tol,
                rel_tol,
                recompute_tol,
                adj,
                &mut norms,
                &mut f,
                &mut auxv,
                &mut row,
                &mut rowfull,
                &mut work,
                &mut result,
                Par::Seq,
            )
        };
        if outcome.fatal > 0 {
            result.residual = f64::NAN;
            result.exception = outcome.fatal;
            return result;
        }
        if outcome.stopped || outcome.done == 0 || outcome.done < width {
            break;
        }
        start += outcome.done;
    }
    result.residual = 0.0;
    if result.rank < m.min(n) {
        for col in result.rank..n {
            result.residual = result.residual.max(nrm2(
                (m - result.rank) as isize,
                a.offset(result.rank as isize + col as isize * lda),
                1,
            ));
        }
    }
    result
}

struct PanelOutcome {
    fatal: usize,
    done: usize,
    stopped: bool,
}

/// One panel of the blocked factorization, from the first pivot search to the
/// trailing GEMM. Returns the number of completed reflectors and whether the
/// rank tolerance stopped the panel early; a nonzero `fatal` is an INFO value
/// (NaN in a column norm or in a reflector).
#[allow(clippy::too_many_arguments)]
unsafe fn run_panel<T: ReflectorScalar>(
    m: usize,
    n: usize,
    total: usize,
    a: *mut T,
    lda: isize,
    tau: *mut T,
    start: usize,
    width: usize,
    fixed_count: usize,
    initial: f64,
    abs_tol: f64,
    rel_tol: f64,
    recompute_tol: f64,
    adj: u8,
    norms: &mut [f64],
    f: &mut [T],
    auxv: &mut [T],
    row: &mut [T],
    rowfull: &mut [T],
    work: &mut [T],
    result: &mut PivotedQr,
    mut par: Par<'_>,
) -> PanelOutcome {
    let relfree = n - start;
    let relall = total - start;
    let mut done = 0usize;
    let mut stopped = false;
    for step in 0..width {
        let j = start + step;
        let mut pivot = step;
        let mut best = -1.0_f64;
        for col in step..relfree {
            let norm = norms[start + col];
            if norm.is_nan() {
                return PanelOutcome {
                    fatal: start + col + 1,
                    done,
                    stopped,
                };
            }
            if norm > best {
                best = norm;
                pivot = col;
            }
        }
        result.residual = best;
        if best == 0.0
            || (abs_tol >= 0.0 && best <= abs_tol)
            || (rel_tol >= 0.0 && (if j == 0 { 1.0 } else { best / initial }) <= rel_tol)
        {
            stopped = true;
            break;
        }
        if j < fixed_count {
            pivot = step;
        }
        if pivot != step {
            swap_columns(m, a, lda, j, start + pivot);
            result.permutation.swap(j, start + pivot);
            norms.swap(j, start + pivot);
            // The incoming column already has an `F` row from previous steps.
            for col in 0..step {
                f.swap(step + col * total, pivot + col * total);
            }
        }
        let column = a.offset(j as isize * (lda + 1));
        if width == 1 {
            // Single reflector: apply it to the whole trailing matrix at once,
            // as the unblocked algorithm does. Building `F` for one panel
            // column would repeat that work.
            crate::householder::generate(
                (m - j) as isize,
                column,
                column.add(1),
                1,
                tau.add(j),
                false,
            );
            if (*tau.add(j)).magnitude().is_nan() {
                return PanelOutcome {
                    fatal: j + 1,
                    done,
                    stopped,
                };
            }
            let trailing = total - j - 1;
            if trailing > 0 {
                let beta = *column;
                *column = T::o();
                crate::householder::apply(
                    b'L',
                    (m - j) as isize,
                    trailing as isize,
                    column,
                    1,
                    (*tau.add(j)).cj(),
                    a.offset(j as isize + (j + 1) as isize * lda),
                    lda,
                    work.as_mut_ptr(),
                );
                *column = beta;
            }
            for col in 0..relall {
                rowfull[col] = *a.offset(j as isize + (start + col) as isize * lda);
            }
        } else {
            // Apply the reflectors already generated in this panel to the new
            // column, using their stored `F` coefficients.
            if step > 0 {
                for col in 0..step {
                    row[col] = f[step + col * total].cj();
                }
                crate::kernels::col_axpy::<T, false>(
                    column,
                    a.offset(j as isize + start as isize * lda),
                    lda,
                    row.as_ptr(),
                    m - j,
                    step,
                    -T::o(),
                    Par::Seq,
                );
            }
            crate::householder::generate(
                (m - j) as isize,
                column,
                column.add(1),
                1,
                tau.add(j),
                false,
            );
            if (*tau.add(j)).magnitude().is_nan() {
                return PanelOutcome {
                    fatal: j + 1,
                    done,
                    stopped,
                };
            }
            let beta = *column;
            *column = T::o();
            let trailing = total - j - 1;
            // F[step+1:, step] = tau * A[j:m, j+1:]^H * A[j:m, j]. This is the
            // panel's dominant non-GEMM cost: it reads the whole trailing matrix
            // once per step, so it uses the panel's team and the specialized
            // transposed GEMV.
            if trailing > 0 {
                crate::kernels::conj_gemv(
                    f.as_mut_ptr().add(step + 1 + step * total),
                    a.offset(j as isize + (j + 1) as isize * lda),
                    lda,
                    column,
                    m - j,
                    trailing,
                    *tau.add(j),
                    par.reborrow(),
                );
            }
            // F[0:step+1, step] = 0, then add the accumulated part so the
            // incoming column's row stays consistent.
            for col in 0..=step {
                f[col + step * total] = T::z();
            }
            if step > 0 {
                crate::kernels::conj_gemv(
                    auxv.as_mut_ptr(),
                    a.offset(j as isize + start as isize * lda),
                    lda,
                    column,
                    m - j,
                    step,
                    -(*tau.add(j)),
                    Par::Seq,
                );
                crate::kernels::col_axpy::<T, false>(
                    f.as_mut_ptr().add(step * total),
                    f.as_ptr(),
                    total as isize,
                    auxv.as_ptr(),
                    relall,
                    step,
                    T::o(),
                    par.reborrow(),
                );
            }
            // A[j, j+1:] -= A[j, start:j] * F[j+1:, 0:step+1]^H. The row is
            // staged in `rowfull`, so the 1-by-`n` GEMM writes contiguous memory
            // and the downdate below reads the same buffer.
            let len = step + 1;
            for col in 0..relall {
                rowfull[col] = *a.offset(j as isize + (start + col) as isize * lda);
            }
            for col in 0..len {
                row[col] = rowfull[col];
            }
            if relall > len {
                crate::kernels::col_axpy::<T, true>(
                    rowfull.as_mut_ptr().add(len),
                    f.as_ptr().add(len),
                    total as isize,
                    row.as_ptr(),
                    relall - len,
                    len,
                    -T::o(),
                    par.reborrow(),
                );
                for col in len..relall {
                    *a.offset(j as isize + (start + col) as isize * lda) = rowfull[col];
                }
            }
            *column = beta;
        }
        // Downdate the residual norms of the trailing columns, falling back to
        // a fresh norm when cancellation makes the update unreliable.
        if step + 1 < relfree {
            let remaining = (m - j - 1) as isize;
            // Downdate every residual norm from the removed component, without a
            // branch or a square root per column, so the pass vectorizes. A
            // column whose update lost too much accuracy is marked with NaN.
            for col in (step + 1)..relfree {
                let target = start + col;
                let (re, im) = rowfull[col].reim();
                let norm = norms[target];
                let r2 = re * re + im * im;
                let temp = 1.0 - r2 / (norm * norm);
                let bad = !norm.is_finite() || r2.is_nan() || !(temp > recompute_tol);
                norms[target] = if bad { f64::NAN } else { norm * temp.sqrt() };
            }
            // The rare column that needs a fresh norm gets the same `?nrm2` pass
            // LAPACK uses after its panel.
            for col in (step + 1)..relfree {
                let target = start + col;
                if norms[target].is_nan() {
                    norms[target] = nrm2(
                        remaining,
                        a.offset((j + 1) as isize + target as isize * lda),
                        1,
                    );
                }
            }
        }
        done = step + 1;
        result.rank = j + 1;
    }
    // Apply the panel's block reflector to the trailing matrix with one GEMM:
    // A[rk:m, rk:] -= A[rk:m, start:rk] * F[rk:, 0:done]^H.
    if width > 1 && done > 0 {
        let rk = start + done;
        if rk < m && rk < total {
            crate::kernels::gemm_with_par(
                b'N',
                adj,
                (m - rk) as isize,
                (total - rk) as isize,
                done as isize,
                -T::o(),
                a.offset(rk as isize + start as isize * lda),
                lda,
                f.as_ptr().add(done),
                total as isize,
                T::o(),
                a.offset(rk as isize + rk as isize * lda),
                lda,
                par,
            );
        }
    }
    PanelOutcome {
        fatal: 0,
        done,
        stopped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    }

    #[test]
    fn panel_width_policy() {
        // The unblocked path wins through 128 wide real and 80 wide complex.
        // Both narrow to 8 while the trailing matrix is small and widen to 16
        // for large ones. `k <= 8` always stays unblocked.
        assert_eq!(panel_width(32, 32, false, 32), 1);
        assert_eq!(panel_width(64, 64, false, 64), 1);
        assert_eq!(panel_width(128, 128, false, 128), 1);
        assert_eq!(panel_width(160, 160, false, 160), 8);
        assert_eq!(panel_width(256, 256, false, 256), 16);
        assert_eq!(panel_width(512, 512, false, 512), 16);
        assert_eq!(panel_width(64, 64, true, 64), 1);
        assert_eq!(panel_width(80, 80, true, 80), 1);
        assert_eq!(panel_width(96, 96, true, 96), 8);
        assert_eq!(panel_width(128, 128, true, 128), 8);
        assert_eq!(panel_width(256, 256, true, 256), 16);
        assert_eq!(panel_width(512, 512, true, 512), 16);
        assert_eq!(panel_width(1000, 8, false, 8), 1);
        assert_eq!(panel_width(1000, 8, true, 8), 1);
        // A rank limit below the panel width still caps the panel.
        assert_eq!(panel_width(128, 128, true, 5), 1);
    }

    // The panel's `F` recurrence must select the same pivots and produce the
    // same factorization as applying one reflector at a time.
    #[test]
    fn blocked_panel_matches_unblocked() {
        unsafe {
            let (m, n) = (24usize, 14usize);
            let mut state = 7u64;
            let mut original = vec![0.0_f64; m * n];
            for j in 0..n {
                for i in 0..m {
                    original[i + j * m] = next(&mut state);
                }
            }
            let fixed = [
                false, true, false, false, true, false, false, false, false, false, false, false,
                false, false,
            ];
            let mut permutations = Vec::new();
            for panel in [1usize, 4, 8] {
                let mut a = original.clone();
                let mut tau = vec![0.0_f64; m.min(n)];
                let result = pivoted_qr_panel(
                    m,
                    n,
                    0,
                    a.as_mut_ptr(),
                    m as isize,
                    tau.as_mut_ptr(),
                    &fixed,
                    m.min(n),
                    -1.0,
                    -1.0,
                    panel,
                );
                assert_eq!(result.rank, n);
                assert_eq!(&result.permutation[..2], &[1, 4]);
                permutations.push(result.permutation.clone());
                let mut r = a.clone();
                for j in 0..n {
                    for i in j + 1..m {
                        r[i + j * m] = 0.0;
                    }
                }
                crate::qr::apply_q(
                    b'L',
                    b'N',
                    m,
                    n,
                    n,
                    a.as_ptr(),
                    m as isize,
                    tau.as_ptr(),
                    r.as_mut_ptr(),
                    m as isize,
                );
                for j in 0..n {
                    let source = result.permutation[j];
                    for i in 0..m {
                        let want = original[i + source * m];
                        assert!(
                            (r[i + j * m] - want).abs() < 1e-10,
                            "panel {panel}: ({i},{j}) {} vs {want}",
                            r[i + j * m]
                        );
                    }
                }
            }
            for other in &permutations[1..] {
                assert_eq!(&permutations[0], other);
            }
        }
    }

    // The trailing right-hand-side columns of `?geqp3rk` are updated by the
    // same panel GEMM and must still equal Q^H * B.
    #[test]
    fn blocked_panel_updates_rhs() {
        unsafe {
            let (m, n, nrhs) = (20usize, 12usize, 3usize);
            let total = n + nrhs;
            let mut state = 11u64;
            let mut original = vec![0.0_f64; m * total];
            for j in 0..total {
                for i in 0..m {
                    original[i + j * m] = next(&mut state);
                }
            }
            let mut a = original.clone();
            let mut tau = vec![0.0_f64; m.min(n)];
            let result = pivoted_qr_panel(
                m,
                n,
                nrhs,
                a.as_mut_ptr(),
                m as isize,
                tau.as_mut_ptr(),
                &[false; 12],
                m.min(n),
                -1.0,
                -1.0,
                5,
            );
            let mut rhs = a[n * m..total * m].to_vec();
            crate::qr::apply_q(
                b'L',
                b'N',
                m,
                nrhs,
                result.rank,
                a.as_ptr(),
                m as isize,
                tau.as_ptr(),
                rhs.as_mut_ptr(),
                m as isize,
            );
            for j in 0..nrhs {
                for i in 0..m {
                    let want = original[i + (n + j) * m];
                    assert!(
                        (rhs[i + j * m] - want).abs() < 1e-10,
                        "rhs ({i},{j}) {} vs {want}",
                        rhs[i + j * m]
                    );
                }
            }
        }
    }

    #[test]
    fn fixed_columns_and_residual_stop() {
        unsafe {
            let original = [1.0_f64, 2.0, 3.0, 7.0, 0.0, 1.0, 0.0, 4.0, 2.0];
            let mut a = original;
            let mut tau = [0.0; 3];
            let result = pivoted_qr(
                3,
                3,
                0,
                a.as_mut_ptr(),
                3,
                tau.as_mut_ptr(),
                &[false, true, false],
                3,
                -1.0,
                -1.0,
            );
            assert_eq!(result.permutation[0], 1);
            let mut r = a;
            for j in 0..3 {
                for i in j + 1..3 {
                    r[i + j * 3] = 0.0;
                }
            }
            crate::qr::apply_q(
                b'L',
                b'N',
                3,
                3,
                result.rank,
                a.as_ptr(),
                3,
                tau.as_ptr(),
                r.as_mut_ptr(),
                3,
            );
            for j in 0..3 {
                for i in 0..3 {
                    assert!((r[i + j * 3] - original[i + result.permutation[j] * 3]).abs() < 1e-12);
                }
            }
            let mut a = original;
            let mut tau = [0.0; 3];
            let result = pivoted_qr(
                3,
                3,
                0,
                a.as_mut_ptr(),
                3,
                tau.as_mut_ptr(),
                &[false; 3],
                3,
                f64::INFINITY,
                -1.0,
            );
            assert_eq!(result.rank, 0);
            assert_eq!(a, original);
            assert_eq!(result.residual, result.initial);
        }
    }
}

use crate::lapack::LapackInt;
use colmat::{c32, c64};
use paste::paste;
unsafe fn complete<T: ReflectorScalar, I: LapackInt>(
    m: usize,
    n: usize,
    a: *mut T,
    lda: isize,
    jpvt: *mut I,
    tau: *mut T,
) {
    let fixed: Vec<bool> = (0..n).map(|j| (*jpvt.add(j)).val() != 0).collect();
    for j in 0..m.min(n) {
        *tau.add(j) = T::z();
    }
    let result = pivoted_qr(m, n, 0, a, lda, tau, &fixed, m.min(n), -1.0, -1.0);
    for j in 0..n {
        *jpvt.add(j) = I::from_i64(result.permutation[j] as i64 + 1);
    }
}
macro_rules! real_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p geqp3 $s>](m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,jpvt:*mut $i,tau:*mut $t,work:*mut $t,lwork:*const $i,info:*mut $i) {
            *info=0;let(m,n,la,lw)=(*m as isize,*n as isize,*lda as isize,*lwork as isize);
            let minwork=if m.min(n)==0 {1} else {3*n+1};
            for(arg,ok)in [(1,m>=0),(2,n>=0),(4,la>=m.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>(minwork);
            if lw!= -1&&lw<minwork {*info=-8;crate::invalid(8);return;}
            if lw== -1 {return;}
            complete(m as usize,n as usize,a,la,jpvt,tau);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p geqpf $s>](m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,jpvt:*mut $i,tau:*mut $t,_work:*mut $t,info:*mut $i) {
            *info=0;let(m,n,la)=(*m as isize,*n as isize,*lda as isize);
            for(arg,ok)in [(1,m>=0),(2,n>=0),(4,la>=m.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            complete(m as usize,n as usize,a,la,jpvt,tau);
        }
    }};
}
macro_rules! complex_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,$s:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p geqp3 $s>](m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,jpvt:*mut $i,tau:*mut $t,work:*mut $t,lwork:*const $i,_rwork:*mut $r,info:*mut $i) {
            *info=0;let(m,n,la,lw)=(*m as isize,*n as isize,*lda as isize,*lwork as isize);
            let minwork=if m.min(n)==0 {1} else {n+1};
            for(arg,ok)in [(1,m>=0),(2,n>=0),(4,la>=m.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>(minwork);
            if lw!= -1&&lw<minwork {*info=-8;crate::invalid(8);return;}
            if lw== -1 {return;}
            complete(m as usize,n as usize,a,la,jpvt,tau);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p geqpf $s>](m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,jpvt:*mut $i,tau:*mut $t,_work:*mut $t,_rwork:*mut $r,info:*mut $i) {
            *info=0;let(m,n,la)=(*m as isize,*n as isize,*lda as isize);
            for(arg,ok)in [(1,m>=0),(2,n>=0),(4,la>=m.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            complete(m as usize,n as usize,a,la,jpvt,tau);
        }
    }};
}
real_exports!(s, f32, i32, _);
real_exports!(d, f64, i32, _);
real_exports!(s, f32, i64, _64_);
real_exports!(d, f64, i64, _64_);
complex_exports!(c, c32, f32, i32, _);
complex_exports!(z, c64, f64, i32, _);
complex_exports!(c, c32, f32, i64, _64_);
complex_exports!(z, c64, f64, i64, _64_);

unsafe fn rank_driver<T: ReflectorScalar, R: Scalar, I: LapackInt>(
    m: I,
    n: I,
    nrhs: I,
    kmax: I,
    abstol: *mut R,
    reltol: *mut R,
    a: *mut T,
    lda: I,
    k: *mut I,
    residual: *mut R,
    relative: *mut R,
    jpiv: *mut I,
    tau: *mut T,
    work: *mut T,
    lwork: I,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let (m, n, r, limit, la, lw) = (
        m.val(),
        n.val(),
        nrhs.val(),
        kmax.val(),
        lda.val(),
        lwork.val(),
    );
    for (arg, ok) in [
        (1, m >= 0),
        (2, n >= 0),
        (3, r >= 0),
        (4, limit >= 0),
        (5, !(*abstol).re().is_nan()),
        (6, !(*reltol).re().is_nan()),
        (8, la >= m.max(1)),
    ] {
        if !ok {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return;
        }
    }
    let minimum = if m.min(n) == 0 {
        1
    } else if T::COMPLEX {
        n + r - 1
    } else {
        3 * n + r - 1
    };
    *work = crate::lapack::workspace::<T>(minimum.max(1));
    if lw != -1 && lw < minimum {
        *info = I::from_i64(-15);
        crate::invalid(15);
        return;
    }
    if lw == -1 {
        return;
    }
    let (m, n, r) = (m as usize, n as usize, r as usize);
    for j in 0..m.min(n) {
        *tau.add(j) = T::z();
    }
    let absolute = if (*abstol).re() >= 0.0 {
        (*abstol).re().max(T::SAFE * T::EPS)
    } else {
        -1.0
    };
    let relative_tol = if (*reltol).re() >= 0.0 {
        (*reltol).re().max(T::EPS / 2.0)
    } else {
        -1.0
    };
    let result = pivoted_qr(
        m,
        n,
        r,
        a,
        la,
        tau,
        &vec![false; n],
        limit as usize,
        absolute,
        relative_tol,
    );
    *k = I::from_i64(result.rank as i64);
    *residual = R::real(result.residual);
    *relative = R::real(if result.residual == 0.0 {
        0.0
    } else if result.rank == 0 && !result.initial.is_nan() {
        1.0
    } else {
        result.residual / result.initial
    });
    *info = I::from_i64(result.exception as i64);
    for j in 0..n {
        *jpiv.add(j) = I::from_i64(result.permutation[j] as i64 + 1);
    }
}
macro_rules! rank_real_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p geqp3rk $s>](m:*const $i,n:*const $i,nrhs:*const $i,kmax:*const $i,abstol:*mut $t,reltol:*mut $t,
            a:*mut $t,lda:*const $i,k:*mut $i,residual:*mut $t,relative:*mut $t,jpiv:*mut $i,tau:*mut $t,work:*mut $t,lwork:*const $i,_iwork:*mut $i,info:*mut $i) {
            rank_driver(*m,*n,*nrhs,*kmax,abstol,reltol,a,*lda,k,residual,relative,jpiv,tau,work,*lwork,info);
        }
    }};
}
macro_rules! rank_complex_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,$s:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p geqp3rk $s>](m:*const $i,n:*const $i,nrhs:*const $i,kmax:*const $i,abstol:*mut $r,reltol:*mut $r,
            a:*mut $t,lda:*const $i,k:*mut $i,residual:*mut $r,relative:*mut $r,jpiv:*mut $i,tau:*mut $t,work:*mut $t,lwork:*const $i,_rwork:*mut $r,_iwork:*mut $i,info:*mut $i) {
            rank_driver(*m,*n,*nrhs,*kmax,abstol,reltol,a,*lda,k,residual,relative,jpiv,tau,work,*lwork,info);
        }
    }};
}
rank_real_exports!(s, f32, i32, _);
rank_real_exports!(d, f64, i32, _);
rank_real_exports!(s, f32, i64, _64_);
rank_real_exports!(d, f64, i64, _64_);
rank_complex_exports!(c, c32, f32, i32, _);
rank_complex_exports!(z, c64, f64, i32, _);
rank_complex_exports!(c, c32, f32, i64, _64_);
rank_complex_exports!(z, c64, f64, i64, _64_);

#[cfg(test)]
mod blocked_c64_tests {
    use super::*;

    fn next(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    }

    /// A blocked complex factorization must reconstruct its input: `Q R = A P`.
    /// The SIMD panel kernels only take part for larger shapes, so this uses a
    /// matrix wide enough to leave the small-block path.
    #[test]
    fn blocked_c64_reconstructs() {
        unsafe {
            for &(n, panel) in &[(97usize, 16usize), (128, 16), (200, 16)] {
                let mut s = 11u64;
                let mut a = vec![c64::new(0.0, 0.0); n * n];
                for v in a.iter_mut() {
                    *v = c64::new(next(&mut s), next(&mut s));
                }
                let orig = a.clone();
                let mut tau = vec![c64::new(0.0, 0.0); n];
                let r = pivoted_qr_panel(
                    n,
                    n,
                    0,
                    a.as_mut_ptr(),
                    n as isize,
                    tau.as_mut_ptr(),
                    &vec![false; n],
                    n,
                    -1.0,
                    -1.0,
                    panel,
                );
                assert_eq!(r.rank, n, "n={n}: rank");
                // Q = H0 H1 ... H_{k-1}, applied to the identity from the left in
                // reverse order.
                let mut q = vec![c64::new(0.0, 0.0); n * n];
                for j in 0..n {
                    q[j + j * n] = c64::new(1.0, 0.0);
                }
                let mut work = vec![c64::new(0.0, 0.0); n];
                for j in (0..n).rev() {
                    let beta = a[j + j * n];
                    a[j + j * n] = c64::new(1.0, 0.0);
                    crate::householder::apply(
                        b'L',
                        (n - j) as isize,
                        n as isize,
                        a.as_ptr().add(j + j * n),
                        1,
                        tau[j],
                        q.as_mut_ptr().add(j),
                        n as isize,
                        work.as_mut_ptr(),
                    );
                    a[j + j * n] = beta;
                }
                // residual = A[:, p] - Q R, with R upper triangular.
                let mut worst = 0.0f64;
                let mut scale = 0.0f64;
                for j in 0..n {
                    let tgt = r.permutation[j];
                    for i in 0..n {
                        let mut acc = orig[i + tgt * n];
                        for c in 0..=j {
                            acc -= q[i + c * n] * a[c + j * n];
                        }
                        worst = worst.max(acc.norm());
                        scale = scale.max(orig[i + tgt * n].norm());
                    }
                }
                let err = worst / (scale * n as f64);
                assert!(err < 1e-13, "n={n} panel={panel}: residual {err:e}");
            }
        }
    }
}
