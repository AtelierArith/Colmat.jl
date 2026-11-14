//! LAPACK row swaps using native borrowed row views.
use colmat::{perm, MatMut};
use paste::paste;

unsafe fn apply<T>(a: *mut T, m: usize, n: usize, ld: usize, p: Vec<usize>, rows: bool) {
    let mut visited = vec![false; p.len()];
    let mut matrix = MatMut::from_raw_parts_mut(a, m, n, 1, ld as isize);
    for start in 0..p.len() {
        if visited[start] {
            continue;
        }
        let mut current = start;
        visited[current] = true;
        while p[current] != start {
            let next = p[current];
            if rows {
                perm::swap_rows_idx(matrix.as_mut(), current, next);
            } else {
                perm::swap_cols_idx(matrix.as_mut(), current, next);
            }
            current = next;
            visited[current] = true;
        }
    }
}

macro_rules! exports {
    ($p:ident,$t:ty,$i:ty,$suffix:tt)=>{paste!{
        // Fortran LOGICAL follows the integer width of this interface.
        #[no_mangle]
        pub unsafe extern "C" fn [<$p lapmt $suffix>](forward:*const $i,m:*const $i,n:*const $i,
            a:*mut $t,ld:*const $i,k:*mut $i) {
            [<$p permutation $suffix>](forward,m,n,a,ld,k,false);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p lapmr $suffix>](forward:*const $i,m:*const $i,n:*const $i,
            a:*mut $t,ld:*const $i,k:*mut $i) {
            [<$p permutation $suffix>](forward,m,n,a,ld,k,true);
        }
        unsafe fn [<$p permutation $suffix>](forward:*const $i,m:*const $i,n:*const $i,
            a:*mut $t,ld:*const $i,k:*mut $i,rows:bool) {
            let(m,n,ld)=(*m as isize,*n as isize,*ld as isize);
            if m<0 {crate::invalid(2);return;}
            if n<0 {crate::invalid(3);return;}
            if ld<m.max(1) {crate::invalid(5);return;}
            let count=if rows {m}else{n} as usize;
            if m==0 || n==0 || count<=1 {return;}
            let mut p=vec![0;count];
            let mut seen=vec![false;count];
            for j in 0..count {
                let target=*k.add(j) as isize;
                if target<1 || target>count as isize || seen[target as usize-1] {
                    crate::invalid(6);return;
                }
                let target=target as usize-1;
                seen[target]=true;
                if *forward!=0 {p[j]=target;}else{p[target]=j;}
            }
            apply(a,m as usize,n as usize,ld as usize,p,rows);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p laswp $suffix>](n:*const $i,a:*mut $t,lda:*const $i,
            k1:*const $i,k2:*const $i,ipiv:*const $i,incx:*const $i) {
            let (n,ld,first,last,inc)=(*n as isize,*lda as isize,*k1 as isize,*k2 as isize,*incx as isize);
            // INCX=0 and empty columns perform no pivot or matrix accesses.
            if inc==0 || n==0 || last<first {return;}
            if n<0 {crate::invalid(1);return;}
            if first<1 {crate::invalid(4);return;}
            if ld<last.max(1) {crate::invalid(3);return;}
            let Some(step)=inc.checked_abs() else {crate::invalid(7);return;};
            let Some(final_index)=(last-first).checked_mul(step).and_then(|x| x.checked_add(first-1))
                else {crate::invalid(7);return;};
            // Check the pivot range before changing A. Each entry is read at
            // K1+(row-K1)*abs(INCX), independently of the application order.
            for index in (first-1..=final_index).step_by(step as usize) {
                let pivot=*ipiv.offset(index) as isize;
                if pivot<1 || pivot>ld {crate::invalid(6);return;}
            }
            let mut matrix=MatMut::from_raw_parts_mut(a,ld as usize,n as usize,1,ld);
            for offset in 0..=last-first {
                let row=if inc>0 {first+offset}else{last-offset};
                let index=first-1+(row-first)*step;
                let pivot=*ipiv.offset(index) as usize-1;
                perm::swap_rows_idx(matrix.as_mut(),row as usize-1,pivot);
            }
        }
    }};
}
exports!(s, f32, i32, _);
exports!(d, f64, i32, _);
exports!(c, colmat::c32, i32, _);
exports!(z, colmat::c64, i32, _);
exports!(s, f32, i64, _64_);
exports!(d, f64, i64, _64_);
exports!(c, colmat::c32, i64, _64_);
exports!(z, colmat::c64, i64, _64_);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_permutation_leaves_matrix_and_indices_unchanged() {
        unsafe {
            let mut a = [1.0, 2.0, 3.0, 4.0];
            let mut k = [1, 1];
            dlapmt_(&1, &2, &2, a.as_mut_ptr(), &2, k.as_mut_ptr());
            assert_eq!(crate::colmat_blas_last_error(), 6);
            assert_eq!(a, [1.0, 2.0, 3.0, 4.0]);
            assert_eq!(k, [1, 1]);
            // A singleton has no permutation to apply and reads no K/X.
            slapmr_(&1, &1, &5, std::ptr::null_mut(), &1, std::ptr::null_mut());
        }
    }
    #[test]
    fn zero_columns_and_zero_increment_need_no_array_storage() {
        unsafe {
            for (n, inc) in [(0, -3), (7, 0)] {
                dlaswp_(&n, std::ptr::null_mut(), &8, &2, &5, std::ptr::null(), &inc);
            }
            for (n, inc) in [(0_i64, -3_i64), (7, 0)] {
                zlaswp_64_(&n, std::ptr::null_mut(), &8, &2, &5, std::ptr::null(), &inc);
            }
        }
    }
}
