"""Compare guarded GGQRF/GGRQF WORK/header outputs in native and reference libraries.

Usage: python3 generalized_qr_workspace.py NATIVE_LIBRARY REFERENCE_LIBRARY
Optimal WORK values are implementation choices. No C adapter is invoked.
"""
import ctypes as c,json,os,sys
N=c.CDLL(sys.argv[1])
R=c.CDLL(sys.argv[2])
last=N.colmat_blas_last_error;last.argtypes=[];last.restype=c.c_int
saved_stdout=os.dup(1);os.dup2(2,1)
out=[]
for prefix in 'sdcz':
 for family in ('ggqrf','ggrqf'):
  for bits in (32,64):
   for case in ('query','insufficient','negative_first','bad_lda'):
    for own in (True,False):
     J=c.c_int32 if own and bits==32 else c.c_int64
     T=c.c_float if prefix in 'sc' else c.c_double
     width=2 if prefix in 'cz' else 1
     f=getattr(N if own else R,prefix+family+('_' if own and bits==32 else '_64_'))
     f.restype=None;f.argtypes=[c.POINTER(J)]*3+[c.c_void_p,c.POINTER(J),c.c_void_p,c.c_void_p,c.POINTER(J),c.c_void_p,c.c_void_p,c.POINTER(J),c.POINTER(J)]
     first,second,third=(J(x) for x in ((-1,3,4) if case=='negative_first' else (2,3,4)))
     la=J(0 if case=='bad_lda' else 5);lb=J(5);lw=J(0 if case in ('insufficient','negative_first') else -1)
     a=(T*256)();b=(T*256)();ta=(T*16)();tb=(T*16)();work=(T*(3*width))(*([777]*(3*width)));info=(J*3)(-999,-999,-999)
     f(c.byref(first),c.byref(second),c.byref(third),a,c.byref(la),ta,b,c.byref(lb),tb,c.byref(work,c.sizeof(T)*width),c.byref(lw),c.cast(c.byref(info,c.sizeof(J)),c.POINTER(J)))
     diagnostic=last() if own else None
     assert list(info)[::2]==[-999,-999]
     assert list(work)[:width]==[777]*width and list(work)[2*width:]==[777]*width
     out.append(dict(scalar=prefix,family=family,bits=bits,case=case,implementation='native' if own else 'reference',info=int(info[1]),work=[float(v) for v in list(work)[width:2*width]],diagnostic=diagnostic))
c.CDLL(None).fflush(None)
os.dup2(saved_stdout,1);os.close(saved_stdout)
print(json.dumps(out,indent=2))
