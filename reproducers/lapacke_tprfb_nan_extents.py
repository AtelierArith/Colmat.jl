"""Execute unchanged TPRFB high wrappers with numerical work stubbed.

Padded buffers cover actual and transposed A extents. This is a NaN-policy
probe, not a numerical oracle. Reference LAPACK v3.12.1 sources are unchanged.
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
    with tempfile.TemporaryDirectory(prefix="tprfb-nan-extents-") as directory:
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
#define LAPACK_WORK_MEMORY_ERROR -1010
#define LAPACKE_malloc malloc
#define LAPACKE_free free
#define MAX(a,b) ((a)>(b)?(a):(b))
void LAPACKE_xerbla(const char*,lapack_int);
int LAPACKE_get_nancheck(void);
int LAPACKE_lsame(char,char);
"""
        driver = """#include "lapacke_utils.h"
#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
static lapack_int arows,acols;static int scans;
int LAPACKE_get_nancheck(void){return 1;}
int LAPACKE_lsame(char a,char b){return toupper((unsigned char)a)==toupper((unsigned char)b);}
void LAPACKE_xerbla(const char* s,lapack_int i){(void)s;(void)i;assert(0);}
"""
        sources, hashes = [], {}
        for prefix, scalar in TYPES.items():
            arguments = (f"int,char,char,char,char,lapack_int,lapack_int,lapack_int,lapack_int,"
                         f"const {scalar}*,lapack_int,const {scalar}*,lapack_int,{scalar}*,lapack_int,{scalar}*,lapack_int")
            header += f"lapack_int LAPACKE_{prefix}tprfb({arguments});\n"
            header += f"lapack_int LAPACKE_{prefix}tprfb_work({arguments},{scalar}*,lapack_int);\n"
            header += f"int LAPACKE_{prefix}ge_nancheck(int,lapack_int,lapack_int,const {scalar}*,lapack_int);\n"
            driver += f"""
int LAPACKE_{prefix}ge_nancheck(int layout,lapack_int m,lapack_int n,const {scalar}* a,lapack_int ld){{
    if(scans++==0){{arows=m;acols=n;}}
    for(lapack_int i=0;i<m;i++)for(lapack_int j=0;j<n;j++){{
        {scalar} x=a[layout==101?i*ld+j:i+j*ld];if(x!=x)return 1;
    }}return 0;
}}
lapack_int LAPACKE_{prefix}tprfb_work(int layout,char side,char trans,char direct,char store,lapack_int m,lapack_int n,lapack_int k,lapack_int l,const {scalar}* v,lapack_int ldv,const {scalar}* t,lapack_int ldt,{scalar}* a,lapack_int lda,{scalar}* b,lapack_int ldb,{scalar}* work,lapack_int ldwork){{
    (void)layout;(void)side;(void)trans;(void)direct;(void)store;(void)m;(void)n;(void)k;(void)l;(void)v;(void)ldv;(void)t;(void)ldt;(void)a;(void)lda;(void)b;(void)ldb;(void)work;(void)ldwork;return 0;
}}
"""
            raw = urllib.request.urlopen(f"https://raw.githubusercontent.com/Reference-LAPACK/lapack/{VERSION}/LAPACKE/src/lapacke_{prefix}tprfb.c").read()
            source = root / f"{prefix}tprfb.c"
            source.write_bytes(raw)
            sources.append(str(source));hashes[prefix] = hashlib.sha256(raw).hexdigest()
        driver += "int main(void){\n"
        for prefix, scalar in TYPES.items():
            driver += f"""
    for(int layout=101;layout<=102;layout++)for(int side=0;side<2;side++)for(int padding=0;padding<2;padding++){{
        {scalar} v[128]={{0}},t[128]={{0}},a[128]={{0}},b[128]={{0}};
        lapack_int m=side?5:3,n=side?3:5,k=2,ld=8;
        lapack_int i=side?(padding?0:4):(padding?4:0),j=side?(padding?4:0):(padding?0:4);
        a[layout==101?i*ld+j:i+j*ld]=({scalar})NAN;
        scans=0;arows=acols=-1;
        lapack_int info=LAPACKE_{prefix}tprfb(layout,side?'R':'L','N','F','C',m,n,k,1,v,ld,t,ld,a,ld,b,ld);
        printf("{prefix} %d %c %d %lld %lld %lld\\n",layout,side?'R':'L',padding,(long long)arows,(long long)acols,(long long)info);
    }}
"""
        driver += "return 0;}\n"
        (root / "lapacke_utils.h").write_text(header);(root / "driver.c").write_text(driver)
        executable = root / "probe"
        subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra", "-Werror", "-I", str(root), *sources, str(root / "driver.c"), "-o", str(executable)], check=True)
        records = []
        for line in subprocess.check_output([str(executable)], text=True).splitlines():
            prefix, layout, side, padding, rows, cols, info = line.split()
            layout,padding,rows,cols,info = map(int,(layout,padding,rows,cols,info))
            assert (rows,cols)==((5,2) if side=='L' else (2,5))
            assert info==(-14 if padding else 0)
            records.append(dict(prefix=prefix,layout=layout,side=side,nan_in_padding=bool(padding),a_rows=rows,a_cols=cols,info=info))
        assert len(records)==32
        print(json.dumps(dict(reference_version=VERSION,numerical_work="stubbed; NaN-policy-only probe",source_sha256=hashes,observations=records),indent=2))


if __name__ == "__main__":
    main()
