"""Compare Fortran TZRZF query, execution and error workspace for a square input."""
import argparse
import ctypes as C
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--reference', required=True)
parser.add_argument('--native', default=str(Path(__file__).resolve().parents[1] / 'deps/usr/lib/libcolmat_blas.dylib'))
args = parser.parse_args()
for path in (args.native, args.reference):
    lib = C.CDLL(path)
    for prefix, real, complex_scalar in [('d', C.c_double, False), ('z', C.c_double, True)]:
        scalar = real * 2 if complex_scalar else real
        integer = C.c_int64
        fn = getattr(lib, prefix + 'tzrzf_64_')
        fn.argtypes = [C.POINTER(integer), C.POINTER(integer), C.POINTER(scalar), C.POINTER(integer), C.POINTER(scalar), C.POINTER(scalar), C.POINTER(integer), C.POINTER(integer)]
        fn.restype = None
        for lw in (-1, 0, 3):
            a = (scalar * 16)(*[scalar(777) for _ in range(16)])
            tau = (scalar * 8)(*[scalar(778) for _ in range(8)])
            work = (scalar * 4)(*[scalar(779) for _ in range(4)])
            m, n, lda, lwork, info = integer(3), integer(3), integer(4), integer(lw), integer(-99)
            for j in range(3):
                for i in range(3):
                    a[i + j * 4] = scalar(0 if i > j else 2 if i == j else 1)
            before = bytes(a)
            fn(C.byref(m), C.byref(n), a, C.byref(lda), tau, work, C.byref(lwork), C.byref(info))
            value = [work[0][0], work[0][1]] if complex_scalar else work[0]
            print(json.dumps(dict(library=Path(path).name, scalar=prefix, lwork=lw, info=info.value, work=value, a_unchanged=bytes(a) == before)), flush=True)
