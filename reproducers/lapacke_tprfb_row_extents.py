"""Probe unchanged LAPACKE TPRFB row conversion, with numerical work stubbed.

All buffers are padded for every copied extent. This establishes conversion
shapes only, not numerical TPRFB accuracy. Uses Reference LAPACK v3.12.1.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import urllib.request

TYPES = {"s": "float", "d": "double", "c": "float complex", "z": "double complex"}
VERSION = "v3.12.1"


def main():
    with tempfile.TemporaryDirectory(prefix="tprfb-row-extents-") as directory:
        root = Path(directory)
        header = """#include <complex.h>
#include <stdint.h>
#include <stdlib.h>
typedef int64_t lapack_int;
typedef float complex lapack_complex_float;
typedef double complex lapack_complex_double;
#define API_SUFFIX(x) x
#define LAPACK_ROW_MAJOR 101
#define LAPACK_COL_MAJOR 102
#define LAPACK_TRANSPOSE_MEMORY_ERROR -1011
#define LAPACKE_malloc malloc
#define LAPACKE_free free
#define MAX(a,b) ((a)>(b)?(a):(b))
void LAPACKE_xerbla(const char*,lapack_int);
"""
        driver = """#include "lapacke_utils.h"
#include <assert.h>
#include <stdio.h>
static lapack_int dims[6][2]; static int count;
void LAPACKE_xerbla(const char* s,lapack_int i){(void)s;(void)i;assert(0);}
"""
        sources = []
        hashes = {}
        for prefix, scalar in TYPES.items():
            signature = (f"(int,char,char,char,char,lapack_int,lapack_int,lapack_int,lapack_int,"
                         f"const {scalar}*,lapack_int,const {scalar}*,lapack_int,{scalar}*,lapack_int,"
                         f"{scalar}*,lapack_int,{scalar}*,lapack_int)")
            header += f"lapack_int LAPACKE_{prefix}tprfb_work{signature};\n"
            header += (f"void LAPACKE_{prefix}ge_trans(int,lapack_int,lapack_int,const {scalar}*,lapack_int,{scalar}*,lapack_int);\n"
                       f"void LAPACK_{prefix}tprfb(const char*,const char*,const char*,const char*,const lapack_int*,const lapack_int*,const lapack_int*,const lapack_int*,const {scalar}*,const lapack_int*,const {scalar}*,const lapack_int*,{scalar}*,const lapack_int*,{scalar}*,const lapack_int*,{scalar}*,const lapack_int*);\n")
            driver += f"""
void LAPACKE_{prefix}ge_trans(int layout,lapack_int m,lapack_int n,const {scalar}* a,lapack_int lda,{scalar}* b,lapack_int ldb){{
    assert(count<6);dims[count][0]=m;dims[count++][1]=n;
    for(lapack_int i=0;i<m;i++)for(lapack_int j=0;j<n;j++){{
        if(layout==101)b[i+j*ldb]=a[i*lda+j];else b[i*ldb+j]=a[i+j*lda];
    }}
}}
void LAPACK_{prefix}tprfb(const char* side,const char* trans,const char* direct,const char* store,const lapack_int* m,const lapack_int* n,const lapack_int* k,const lapack_int* l,const {scalar}* v,const lapack_int* ldv,const {scalar}* t,const lapack_int* ldt,{scalar}* a,const lapack_int* lda,{scalar}* b,const lapack_int* ldb,{scalar}* work,const lapack_int* ldwork){{
    (void)side;(void)trans;(void)direct;(void)store;(void)m;(void)n;(void)k;(void)l;(void)v;(void)ldv;(void)t;(void)ldt;(void)a;(void)lda;(void)b;(void)ldb;(void)work;(void)ldwork;
}}
"""
            url = f"https://raw.githubusercontent.com/Reference-LAPACK/lapack/{VERSION}/LAPACKE/src/lapacke_{prefix}tprfb_work.c"
            raw = urllib.request.urlopen(url).read()
            source = root / f"{prefix}tprfb.c"
            source.write_bytes(raw)
            sources.append(str(source))
            hashes[prefix] = hashlib.sha256(raw).hexdigest()
        driver += "int main(void){\n"
        for prefix, scalar in TYPES.items():
            driver += f"""
    for(int side=0;side<2;side++)for(int store=0;store<2;store++){{
        {scalar} v[1024]={{0}},t[1024]={{0}},a[1024]={{0}},b[1024]={{0}},w[1024]={{0}};
        count=0;
        lapack_int info=LAPACKE_{prefix}tprfb_work(101,side?'R':'L','N','F',store?'R':'C',3,5,2,1,v,7,t,6,a,8,b,9,w,8);
        assert(info==0 && count==6);
        printf("{prefix} %c %c",side?'R':'L',store?'R':'C');
        for(int j=0;j<6;j++)printf(" %lld %lld",(long long)dims[j][0],(long long)dims[j][1]);
        printf("\\n");
    }}
"""
        driver += "return 0;}\n"
        (root / "lapacke_utils.h").write_text(header)
        (root / "driver.c").write_text(driver)
        executable = root / "probe"
        subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra", "-Werror",
                        "-I", str(root), *sources, str(root / "driver.c"), "-o", str(executable)], check=True)
        records = []
        for line in subprocess.check_output([str(executable)], text=True).splitlines():
            prefix, side, store, *values = line.split()
            dims = [list(map(int, values[i:i+2])) for i in range(0, len(values), 2)]
            assert dims == [[7, 2], [6, 2], [2, 3], [3, 5], [2, 3], [3, 5]], dims
            records.append(dict(prefix=prefix, side=side, storev=store, conversion_shapes=dims))
        assert len(records) == 16
        print(json.dumps(dict(reference_version=VERSION, numerical_work="stubbed; conversion-only probe",
                              source_sha256=hashes, observations=records), indent=2))


if __name__ == "__main__":
    main()
