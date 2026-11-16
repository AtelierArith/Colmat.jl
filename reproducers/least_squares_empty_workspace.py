"""Compare Fortran least-squares empty/query/error outputs in separate libraries.

Usage: python3 least_squares_empty_workspace.py NATIVE_LIBRARY REFERENCE_LIBRARY
Both libraries must expose the ILP64 Fortran names. Queries are implementation
choices; differing nonempty optimum values are observations, not failures.
"""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    libraries = [("native", C.CDLL(sys.argv[1])), ("reference", C.CDLL(sys.argv[2]))]
    records = []
    integer = C.c_int64
    for prefix, real, complex_type in [("s",C.c_float,False),("d",C.c_double,False),("c",C.c_float,True),("z",C.c_double,True)]:
        class Complex(C.Structure):
            _fields_ = [("re",real),("im",real)]
        scalar = Complex if complex_type else real
        make = (lambda x: scalar(x,0)) if complex_type else (lambda x: scalar(x))
        value = (lambda x: float(x.re)) if complex_type else float
        for family in ("gels","gelsy","gelss","gelsd"):
            for m,n,nrhs in [(0,0,2),(0,3,2),(3,0,2),(3,5,0)]:
                for lwork in (-1,0,65536):
                    for label,library in libraries:
                        mm,nn,rr,lda,ldb,lw = map(integer,(m,n,nrhs,max(1,m),max(1,m,n),lwork))
                        a=(scalar*128)();b=(scalar*128)();work=(scalar*65536)();work[0]=make(777)
                        singular=(real*128)();rwork=(real*65536)();rwork[0]=real(779)
                        iwork=(integer*65536)();iwork[0]=integer(780)
                        pivot=(integer*128)();rank=integer(778);rcond=real(-1);info=integer(-999)
                        byref=C.byref
                        types=[C.POINTER(integer)]*3+[C.POINTER(scalar),C.POINTER(integer),C.POINTER(scalar),C.POINTER(integer)]
                        args=[byref(mm),byref(nn),byref(rr),a,byref(lda),b,byref(ldb)]
                        if family=="gels":
                            trans=C.c_char(b'N');types=[C.POINTER(C.c_char)]+types;args=[byref(trans)]+args
                        elif family=="gelsy":
                            types += [C.POINTER(integer),C.POINTER(real),C.POINTER(integer)];args += [pivot,byref(rcond),byref(rank)]
                        else:
                            types += [C.POINTER(real),C.POINTER(real),C.POINTER(integer)];args += [singular,byref(rcond),byref(rank)]
                        types += [C.POINTER(scalar),C.POINTER(integer)];args += [work,byref(lw)]
                        if complex_type and family!="gels":
                            types += [C.POINTER(real)];args += [rwork]
                        if family=="gelsd":
                            types += [C.POINTER(integer)];args += [iwork]
                        types += [C.POINTER(integer)];args += [byref(info)]
                        if family=="gels":
                            types += [C.c_size_t];args += [C.c_size_t(1)]
                        function=getattr(library,prefix+family+"_64_");function.argtypes=types;function.restype=None
                        flush=C.CDLL(None).fflush;flush.argtypes=[C.c_void_p];flush.restype=C.c_int
                        saved_stdout=os.dup(1)
                        try:
                            flush(None);os.dup2(2,1)
                            function(*args)
                            flush(None)
                        finally:
                            os.dup2(saved_stdout,1);os.close(saved_stdout)
                        if label=="native":
                            diagnostic=library.colmat_blas_last_error;diagnostic.argtypes=[];diagnostic.restype=C.c_int
                            actual=diagnostic();assert actual==(-info.value if info.value<0 else 0),(prefix,family,info.value,actual)
                        records.append(dict(scalar=prefix,family=family,m=m,n=n,nrhs=nrhs,lwork=lwork,library=label,info=info.value,work=value(work[0]),rank=rank.value,rwork=float(rwork[0]),iwork=iwork[0]))
    assert len(records)==384
    print(json.dumps(dict(full_suite_passed=False,library_sha256={label:hashlib.sha256(Path(path).read_bytes()).hexdigest() for (label,_),path in zip(libraries,sys.argv[1:])},observations=records),indent=2))


if __name__=="__main__":
    main()
