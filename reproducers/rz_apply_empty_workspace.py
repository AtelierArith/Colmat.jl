"""Observe empty ORMRZ/UNMRZ minimum, query and execution workspace separately."""
import argparse
import ctypes as C
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--reference', required=True)
parser.add_argument('--native', required=True)
args = parser.parse_args()
for path, external in ((args.native, False), (args.reference, True)):
    lib = C.CDLL(path)
    for prefix, complex_scalar in [('d', False), ('z', True)]:
        scalar = C.c_double * 2 if complex_scalar else C.c_double
        integer = C.c_int64
        fn = getattr(lib, prefix + ('unmrz' if complex_scalar else 'ormrz') + '_64_')
        signature = [C.POINTER(C.c_char)] * 2 + [C.POINTER(integer)] * 4 + [C.POINTER(scalar), C.POINTER(integer), C.POINTER(scalar), C.POINTER(scalar), C.POINTER(integer), C.POINTER(scalar), C.POINTER(integer), C.POINTER(integer)]
        fn.argtypes = signature + ([C.c_size_t, C.c_size_t] if external else [])
        fn.restype = None
        for mm, nn in ((0, 3), (3, 0)):
            for side in ('L', 'R'):
                for lw in (-1, 0, 1, 3):
                    a = (scalar * 8)(*[scalar(777) for _ in range(8)])
                    tau = (scalar * 8)(*[scalar(778) for _ in range(8)])
                    c = (scalar * 16)(*[scalar(779) for _ in range(16)])
                    work = (scalar * 8)(*[scalar(780) for _ in range(8)])
                    s, tr = C.c_char(side.encode()), C.c_char(b'N')
                    m, n, k, l = integer(mm), integer(nn), integer(0), integer(0)
                    lda, ldc, lwork, info = integer(1), integer(max(1, mm)), integer(lw), integer(-99)
                    before = (bytes(a), bytes(tau), bytes(c))
                    values = [C.byref(s), C.byref(tr), C.byref(m), C.byref(n), C.byref(k), C.byref(l), a, C.byref(lda), tau, c, C.byref(ldc), work, C.byref(lwork), C.byref(info)]
                    fn(*(values + ([1, 1] if external else [])))
                    value = [work[0][0], work[0][1]] if complex_scalar else work[0]
                    print(json.dumps(dict(library=Path(path).name, scalar=prefix, side=side, m=mm, n=nn, lwork=lw, info=info.value, work=value, arrays_unchanged=before == (bytes(a), bytes(tau), bytes(c)))), flush=True)
