"""Probe unchanged reference GEMQRT row packing; numerical work is stubbed.

All caller buffers are padded for the observed copy extents. The probe records
copies and forwarded strides, not numerical GEMQRT correctness.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import urllib.request

TYPES = {"s": "float", "d": "double", "c": "lapack_complex_float", "z": "lapack_complex_double"}
VERSION = "v3.12.1"


def main():
    with tempfile.TemporaryDirectory(prefix="gemqrt-packing-") as directory:
        root = Path(directory)
        header = """#include <complex.h>
#include <stdint.h>
#include <stdlib.h>
typedef int64_t lapack_int;
typedef float complex lapack_complex_float;
typedef double complex lapack_complex_double;
#define API_SUFFIX(x) x
#define MAX(a,b) ((a)>(b)?(a):(b))
#define LAPACK_ROW_MAJOR 101
#define LAPACK_COL_MAJOR 102
#define LAPACK_TRANSPOSE_MEMORY_ERROR -1011
#define LAPACKE_malloc(n) calloc(1,n)
#define LAPACKE_free free
void LAPACKE_xerbla(const char*,lapack_int);
"""
        driver = """#include "lapacke_utils.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
static lapack_int fdv,fdt,fdc,rr[4],cc[4];static int copies;
void LAPACKE_xerbla(const char* name,lapack_int info){(void)name;(void)info;assert(0);}
"""
        sources, hashes = [], {}
        for p, scalar in TYPES.items():
            header += f"""
void LAPACKE_{p}ge_trans(int,lapack_int,lapack_int,const {scalar}*,lapack_int,{scalar}*,lapack_int);
void LAPACK_{p}gemqrt(const char*,const char*,const lapack_int*,const lapack_int*,const lapack_int*,const lapack_int*,const {scalar}*,const lapack_int*,const {scalar}*,const lapack_int*,{scalar}*,const lapack_int*,{scalar}*,lapack_int*);
lapack_int LAPACKE_{p}gemqrt_work(int,char,char,lapack_int,lapack_int,lapack_int,lapack_int,const {scalar}*,lapack_int,const {scalar}*,lapack_int,{scalar}*,lapack_int,{scalar}*);
"""
            driver += f"""
void LAPACKE_{p}ge_trans(int layout,lapack_int m,lapack_int n,const {scalar}* a,lapack_int la,{scalar}* b,lapack_int lb){{
    assert(copies<4);rr[copies]=m;cc[copies]=n;copies++;
    for(lapack_int i=0;i<m;i++){{for(lapack_int j=0;j<n;j++){{
        if(layout==101){{b[i+j*lb]=a[i*la+j];}}else{{b[i*lb+j]=a[i+j*la];}}
    }}}}
}}
void LAPACK_{p}gemqrt(const char* side,const char* trans,const lapack_int* m,const lapack_int* n,const lapack_int* k,const lapack_int* nb,const {scalar}* v,const lapack_int* lv,const {scalar}* t,const lapack_int* lt,{scalar}* c,const lapack_int* lc,{scalar}* w,lapack_int* info){{
    (void)side;(void)trans;(void)m;(void)n;(void)k;(void)nb;(void)v;(void)t;(void)c;(void)w;
    fdv=*lv;fdt=*lt;fdc=*lc;*info=0;
}}
"""
            raw = urllib.request.urlopen(
                f"https://raw.githubusercontent.com/Reference-LAPACK/lapack/{VERSION}/LAPACKE/src/lapacke_{p}gemqrt_work.c"
            ).read()
            source = root / f"lapacke_{p}gemqrt_work.c"
            source.write_bytes(raw)
            sources.append(str(source))
            hashes[p + "gemqrt_work"] = hashlib.sha256(raw).hexdigest()
        (root / "lapacke_utils.h").write_text(header)
        driver += "int main(void){\n"
        for p, scalar in TYPES.items():
            driver += f"""
    for(int layout=101;layout<=102;layout++)for(int right=0;right<2;right++)for(int which=0;which<2;which++){{
        lapack_int m=which?5:3,n=which?3:5,k=2,nb=1,order=right?n:m;
        lapack_int lv=layout==101?k+2:order+2,lt=layout==101?k+2:nb+2,lc=layout==101?n+2:m+2;
        {scalar} v[128],t[128],c[128],saved_v[128],saved_t[128],saved_c[128],w[32]={{0}};
        for(int i=0;i<128;i++){{v[i]=({scalar})(i+1);t[i]=({scalar})(i+129);c[i]=({scalar})(i+257);}}
        memcpy(saved_v,v,sizeof(v));memcpy(saved_t,t,sizeof(t));memcpy(saved_c,c,sizeof(c));
        copies=0;fdv=fdt=fdc=-1;for(int i=0;i<4;i++){{rr[i]=cc[i]=-1;}}
        char side=right?'R':'L';lapack_int info=LAPACKE_{p}gemqrt_work(layout,side,'N',m,n,k,nb,v,lv,t,lt,c,lc,w);
        assert(info==0&&memcmp(v,saved_v,sizeof(v))==0&&memcmp(t,saved_t,sizeof(t))==0&&memcmp(c,saved_c,sizeof(c))==0);
        printf("{p} %d %c %lld %lld %lld %lld %lld %lld %lld %lld %lld %d %lld %lld %lld %lld\\n",layout,side,(long long)m,(long long)n,(long long)order,(long long)lv,(long long)lt,(long long)lc,(long long)fdv,(long long)fdt,(long long)fdc,copies,(long long)rr[0],(long long)cc[0],(long long)rr[1],(long long)cc[1]);
    }}
"""
        driver += "return 0;}\n"
        (root / "driver.c").write_text(driver)
        executable = root / "probe"
        subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra", "-Werror",
                        "-I", str(root), *sources, str(root / "driver.c"), "-o", str(executable)], check=True)
        output = subprocess.check_output([str(executable)], text=True)
        observations = []
        for line in output.splitlines():
            p, layout, side, *values = line.split()
            layout = int(layout)
            m, n, order, lv, lt, lc, fdv, fdt, fdc, count, vr, vc, tr, tc = map(int, values)
            assert order == (n if side == "R" else m)
            if layout == 101:
                assert (fdv, fdt, fdc, count) == (lv, lt, m, 4)
                assert (vr, vc, tr, tc) == (lv, 2, lt, 1)
            else:
                assert (fdv, fdt, fdc, count) == (lv, lt, lc, 0)
            observations.append(dict(scalar=p, layout=layout, side=side, m=m, n=n,
                                     q_order=order, caller_ldv=lv, caller_ldt=lt,
                                     forwarded_ldv=fdv, forwarded_ldt=fdt,
                                     forwarded_ldc=fdc, transpose_calls=count,
                                     observed_v_shape=[vr, vc], observed_t_shape=[tr, tc],
                                     logical_v_shape=[order, 2], logical_t_shape=[1, 2]))
        assert len(observations) == 32
        print(json.dumps(dict(reference_version=VERSION,
                              numerical_work="stubbed; packing/header-only probe",
                              source_sha256=hashes, observations=observations), indent=2))


if __name__ == "__main__":
    main()
