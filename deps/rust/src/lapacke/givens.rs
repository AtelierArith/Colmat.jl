//! LAPACKE LARTGP/LARTGS high/work adapters (no matrix-layout argument).
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d}{lartgp,lartgs}{,_work}.c.
use super::LAPACKE_get_nancheck;

macro_rules! exports {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p lartgp $($suffix)*>](
            f:$t,g:$t,cs:*mut $t,sn:*mut $t,r:*mut $t,
        )->$i {
            if LAPACKE_get_nancheck()!=0 {
                if f!=f {return -1;}
                if g!=g {return -2;}
            }
            crate::givens::[<$p lartgp $fortran>](&f,&g,cs,sn,r);
            0
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p lartgp_work $($suffix)*>](
            f:$t,g:$t,cs:*mut $t,sn:*mut $t,r:*mut $t,
        )->$i {
            crate::givens::[<$p lartgp $fortran>](&f,&g,cs,sn,r);
            0
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p lartgs $($suffix)*>](
            x:$t,y:$t,sigma:$t,cs:*mut $t,sn:*mut $t,
        )->$i {
            if LAPACKE_get_nancheck()!=0 {
                if sigma!=sigma {return -3;}
                if x!=x {return -1;}
                if y!=y {return -2;}
            }
            crate::givens::[<$p lartgs $fortran>](&x,&y,&sigma,cs,sn);
            0
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p lartgs_work $($suffix)*>](
            x:$t,y:$t,sigma:$t,cs:*mut $t,sn:*mut $t,
        )->$i {
            crate::givens::[<$p lartgs $fortran>](&x,&y,&sigma,cs,sn);
            0
        }
    }};
}
exports!(s, f32, i32, [], _);
exports!(d, f64, i32, [], _);
exports!(s, f32, i64, [64_], _64_);
exports!(d, f64, i64, [64_], _64_);
