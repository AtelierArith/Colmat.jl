//! Public pivoted LLT factor kernel, primitive output adapter.
mod workspace;
use crate::guarded;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt_pivoting::factor;
use faer::traits::{math_utils::*, ComplexField};
use faer::{Mat, Par};
use rustcall_julia_macros::julia;
unsafe fn factorize<T: ComplexField>(
    p: *const u8,
    n: usize,
    out: *mut u8,
    perm: *mut usize,
    tol: f64,
) -> Result<usize, String> {
    n.checked_mul(n).ok_or("pivoted LLT shape overflow")?;
    let mut a = Mat::<T>::from_fn(n, n, |i, j| copy(&*p.cast::<T>().add(i + j * n)));
    if !a.is_all_finite() {
        return Err("pivoted LLT input must be finite".into());
    }
    let mut forward: Vec<usize> = (0..n).collect();
    let mut inverse = forward.clone();
    if tol >= 0.0 {
        let threshold = from_f64::<T::Real>(tol);
        let mut l = Mat::<T>::zeros(n, n);
        let mut rank = 0;
        for j in 0..n {
            let mut pivot = j;
            for i in j + 1..n {
                if real(&a[(i, i)]) > real(&a[(pivot, pivot)]) {
                    pivot = i;
                }
            }
            let d = real(&a[(pivot, pivot)]);
            if d <= threshold || d <= zero::<T::Real>() {
                break;
            }
            if pivot != j {
                for col in 0..n {
                    let x = copy(&a[(j, col)]);
                    a[(j, col)] = copy(&a[(pivot, col)]);
                    a[(pivot, col)] = x;
                }
                for row in 0..n {
                    let x = copy(&a[(row, j)]);
                    a[(row, j)] = copy(&a[(row, pivot)]);
                    a[(row, pivot)] = x;
                }
                for col in 0..j {
                    let x = copy(&l[(j, col)]);
                    l[(j, col)] = copy(&l[(pivot, col)]);
                    l[(pivot, col)] = x;
                }
                forward.swap(j, pivot);
            }
            let root = sqrt(&d);
            l[(j, j)] = from_real::<T>(&root);
            for i in j + 1..n {
                l[(i, j)] = mul_real(&a[(i, j)], &recip(&root));
            }
            for col in j + 1..n {
                for row in j + 1..n {
                    a[(row, col)] = sub(&a[(row, col)], &mul(&l[(row, j)], &conj(&l[(col, j)])));
                }
            }
            rank += 1;
        }
        for j in 0..n {
            perm.add(j).write(forward[j]);
            for i in 0..n {
                out.cast::<T>().add(i + j * n).write(copy(&l[(i, j)]));
            }
        }
        return Ok(rank);
    }
    let mut mem = MemBuffer::new(factor::cholesky_in_place_scratch::<usize, T>(
        n,
        Par::Seq.shape(),
        Default::default(),
    ));
    let (info, _) = factor::cholesky_in_place(
        a.as_mut(),
        &mut forward,
        &mut inverse,
        Par::Seq,
        MemStack::new(&mut mem),
        Default::default(),
    )
    .map_err(|e| format!("pivoted LLT: {e:?}"))?;
    for j in 0..n {
        perm.add(j).write(forward[j]);
        for i in 0..n {
            out.cast::<T>()
                .add(i + j * n)
                .write(if i < j || j >= info.rank {
                    zero::<T>()
                } else {
                    copy(&a[(i, j)])
                });
        }
    }
    Ok(info.rank)
}
#[julia]
pub fn pivoted_llt(
    p: *const u8,
    n: usize,
    tag: i32,
    out: *mut u8,
    perm: *mut usize,
    tol: f64,
) -> Result<usize, String> {
    guarded(|| unsafe {
        match tag {
            0 => factorize::<f32>(p, n, out, perm, tol),
            1 => factorize::<f64>(p, n, out, perm, tol),
            2 => factorize::<faer::c32>(p, n, out, perm, tol),
            3 => factorize::<faer::c64>(p, n, out, perm, tol),
            _ => Err("invalid pivoted LLT scalar".into()),
        }
    })
}
unsafe fn apply<T: ComplexField>(
    p: *const u8,
    indices: *const usize,
    n: usize,
    rhs: *const u8,
    k: usize,
    out: *mut u8,
    op: i32,
) -> Result<(), String> {
    use faer::linalg::cholesky::llt_pivoting::{inverse, reconstruct, solve};
    if !(0..=2).contains(&op) {
        return Err("invalid pivoted LLT operation".into());
    }
    let l = Mat::<T>::from_fn(n, n, |i, j| copy(&*p.cast::<T>().add(i + j * n)));
    let forward: Vec<usize> = (0..n).map(|i| indices.add(i).read()).collect();
    let mut backward = vec![usize::MAX; n];
    for (i, &j) in forward.iter().enumerate() {
        if j >= n || backward[j] != usize::MAX {
            return Err("invalid pivoted LLT permutation".into());
        }
        backward[j] = i;
    }
    let perm = faer::perm::PermRef::new_checked(&forward, &backward, n);
    let mut result = if op == 0 {
        Mat::<T>::from_fn(n, k, |i, j| copy(&*rhs.cast::<T>().add(i + j * n)))
    } else {
        Mat::<T>::zeros(n, n)
    };
    let req = match op {
        0 => solve::solve_in_place_scratch::<usize, T>(n, k, Par::Seq.shape()),
        1 => inverse::inverse_scratch::<usize, T>(n, Par::Seq.shape()),
        _ => reconstruct::reconstruct_scratch::<usize, T>(n, Par::Seq.shape()),
    };
    let mut mem = MemBuffer::new(req);
    let stack = MemStack::new(&mut mem);
    match op {
        0 => solve::solve_in_place(l.as_ref(), perm, result.as_mut(), Par::Seq, stack),
        1 => inverse::inverse(result.as_mut(), l.as_ref(), perm, Par::Seq, stack),
        _ => reconstruct::reconstruct(result.as_mut(), l.as_ref(), perm, Par::Seq, stack),
    }
    if op != 0 {
        for j in 0..n {
            for i in 0..j {
                result[(i, j)] = conj(&result[(j, i)]);
            }
        }
    }
    crate::typed::write_matrix(result.as_ref(), out);
    Ok(())
}
#[julia]
pub fn pivoted_llt_apply(
    p: *const u8,
    indices: *const usize,
    n: usize,
    tag: i32,
    rhs: *const u8,
    k: usize,
    out: *mut u8,
    op: i32,
) -> Result<(), String> {
    guarded(|| unsafe {
        match tag {
            0 => apply::<f32>(p, indices, n, rhs, k, out, op),
            1 => apply::<f64>(p, indices, n, rhs, k, out, op),
            2 => apply::<faer::c32>(p, indices, n, rhs, k, out, op),
            3 => apply::<faer::c64>(p, indices, n, rhs, k, out, op),
            _ => Err("invalid pivoted LLT scalar".into()),
        }
    })
}
