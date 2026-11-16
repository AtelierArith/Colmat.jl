"""Execute legal NRHS=0 GELSD/GELSS calls against separate ILP64 libraries.

A=diag(1,2) with a zero third row has known singular values (2,1), rank two.
Usage: python3 gelsd_zero_rhs.py NATIVE_LIBRARY REFERENCE_LIBRARY
"""
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import sys


def main():
    libraries=[("native",C.CDLL(sys.argv[1])),("reference",C.CDLL(sys.argv[2]))]
    observations=[];integer=C.c_int64
    flush=C.CDLL(None).fflush;flush.argtypes=[C.c_void_p];flush.restype=C.c_int
    for prefix,real,complex_type in [("s",C.c_float,False),("d",C.c_double,False),("c",C.c_float,True),("z",C.c_double,True)]:
        class Complex(C.Structure):
            _fields_=[("re",real),("im",real)]
        scalar=Complex if complex_type else real
        make=(lambda x:scalar(x,0)) if complex_type else (lambda x:scalar(x))
        value=(lambda x:float(x.re)) if complex_type else float
        for family in ("gelss","gelsd"):
            for label,library in libraries:
                for lwork in (-1,65536):
                    m,n,r,lda,ldb,lw=map(integer,(3,2,0,5,6,lwork))
                    a=(scalar*16)();a[0]=make(1);a[6]=make(2)
                    b=(scalar*16)();work=(scalar*65536)();rwork=(real*65536)();iwork=(integer*65536)()
                    singular=(real*4)(782,782,782,782);rank=(integer*3)(777,-1,777);rcond=real(.001);info=integer(-999)
                    singular_pointer=C.cast(C.byref(singular,C.sizeof(real)),C.POINTER(real));rank_pointer=C.cast(C.byref(rank,C.sizeof(integer)),C.POINTER(integer))
                    types=[C.POINTER(integer)]*3+[C.POINTER(scalar),C.POINTER(integer),C.POINTER(scalar),C.POINTER(integer),C.POINTER(real),C.POINTER(real),C.POINTER(integer),C.POINTER(scalar),C.POINTER(integer)]
                    args=[C.byref(m),C.byref(n),C.byref(r),a,C.byref(lda),b,C.byref(ldb),singular_pointer,C.byref(rcond),rank_pointer,work,C.byref(lw)]
                    if complex_type:types+=[C.POINTER(real)];args+=[rwork]
                    if family=="gelsd":types+=[C.POINTER(integer)];args+=[iwork]
                    types+=[C.POINTER(integer)];args+=[C.byref(info)]
                    function=getattr(library,prefix+family+"_64_");function.argtypes=types;function.restype=None
                    saved=os.dup(1)
                    try:
                        flush(None);os.dup2(2,1);function(*args);flush(None)
                    finally:os.dup2(saved,1);os.close(saved)
                    assert rank[0]==777 and rank[2]==777 and singular[0]==782 and singular[3]==782
                    if lwork==-1:assert info.value==0 and rank[1]==-1
                    elif label=="reference" and family=="gelsd":assert info.value==-4 and rank[1]==-1
                    else:
                        assert info.value==0 and rank[1]==2
                        assert math.isclose(singular[1],2,rel_tol=1e-6) and math.isclose(singular[2],1,rel_tol=1e-6)
                    observations.append(dict(scalar=prefix,family=family,library=label,nrhs=0,lwork=lwork,info=info.value,rank=rank[1],singular_values=[float(singular[1]),float(singular[2])],query=value(work[0])))
    assert len(observations)==32
    print(json.dumps(dict(full_suite_passed=False,library_sha256={label:hashlib.sha256(Path(path).read_bytes()).hexdigest() for (label,_),path in zip(libraries,sys.argv[1:])},observations=observations),indent=2))


if __name__=="__main__":main()
