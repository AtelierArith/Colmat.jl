"""Execute unchanged Reference LAPACK high wrappers with numerical-work stubs.

This probes NaN-policy dimensions only, not numerical Q application. All test
buffers include enough padding for either observed dimension convention.
Run: uv run --python 3.12 python reproducers/lapacke_q_apply_nan_extent.py
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import urllib.request

VERSION = "v3.12.1"
TYPES = {"s": "float", "d": "double", "c": "lapack_complex_float", "z": "lapack_complex_double"}
FAMILIES = [(p, ("unm" if p in "cz" else "orm") + orientation)
            for p in TYPES for orientation in ("qr", "lq", "ql", "rq")]


def prototype(p, family, high=False):
    scalar = TYPES[p]
    tail = "" if high else f", {scalar}*, lapack_int"
    return (f"lapack_int LAPACKE_{p}{family}{'' if high else '_work'}"
            f"(int, char, char, lapack_int, lapack_int, lapack_int, const {scalar}*, "
            f"lapack_int, const {scalar}*, {scalar}*, lapack_int{tail});")


def main():
    with tempfile.TemporaryDirectory(prefix="lapacke-q-policy-") as directory:
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
#define LAPACKE_malloc malloc
#define LAPACKE_free free
#define LAPACK_WORK_MEMORY_ERROR -1010
#define LAPACK_S2INT(x) ((lapack_int)(x))
#define LAPACK_D2INT(x) ((lapack_int)(x))
#define LAPACK_C2INT(x) ((lapack_int)crealf(x))
#define LAPACK_Z2INT(x) ((lapack_int)creal(x))
int LAPACKE_get_nancheck(void);
int LAPACKE_lsame(char,char);
void LAPACKE_xerbla(const char*,lapack_int);
"""
        for p, scalar in TYPES.items():
            header += f"int LAPACKE_{p}ge_nancheck(int,lapack_int,lapack_int,const {scalar}*,lapack_int);\n"
            header += f"int LAPACKE_{p}_nancheck(lapack_int,const {scalar}*,lapack_int);\n"
        for p, family in FAMILIES:
            header += prototype(p, family) + "\n" + prototype(p, family, True) + "\n"
        (root / "lapacke_utils.h").write_text(header)
        driver = """#include "lapacke_utils.h"
#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
static lapack_int arows,acols; static int scans;
int LAPACKE_get_nancheck(void){return 1;}
int LAPACKE_lsame(char a,char b){return toupper((unsigned char)a)==toupper((unsigned char)b);}
void LAPACKE_xerbla(const char* name,lapack_int info){(void)name;(void)info;assert(0);}
"""
        for p, scalar in TYPES.items():
            driver += f"""
int LAPACKE_{p}ge_nancheck(int layout,lapack_int m,lapack_int n,const {scalar}* a,lapack_int ld){{
    if(scans++==0){{arows=m;acols=n;}}
    for(lapack_int i=0;i<m;i++)for(lapack_int j=0;j<n;j++){{
        {scalar} x=a[layout==101?i*ld+j:i+j*ld];if(x!=x)return 1;
    }}return 0;
}}
int LAPACKE_{p}_nancheck(lapack_int n,const {scalar}* a,lapack_int inc){{
    for(lapack_int i=0;i<n;i++){{{scalar} x=a[i*inc];if(x!=x)return 1;}}return 0;
}}
"""
        sources, hashes = [], {}
        for p, family in FAMILIES:
            name = p + family
            url = f"https://raw.githubusercontent.com/Reference-LAPACK/lapack/{VERSION}/LAPACKE/src/lapacke_{name}.c"
            raw = urllib.request.urlopen(url).read()
            source = root / f"lapacke_{name}.c"
            source.write_bytes(raw)  # Preserve the primary source and its license verbatim.
            sources.append(str(source))
            hashes[name] = hashlib.sha256(raw).hexdigest()
            scalar = TYPES[p]
            driver += f"""
lapack_int LAPACKE_{name}_work(int layout,char side,char trans,lapack_int m,lapack_int n,lapack_int k,const {scalar}* a,lapack_int lda,const {scalar}* tau,{scalar}* c,lapack_int ldc,{scalar}* work,lapack_int lw){{
    (void)layout;(void)side;(void)trans;(void)m;(void)n;(void)k;(void)a;(void)lda;
    (void)tau;(void)c;(void)ldc;(void)lw;work[0]=1;return 0;
}}
"""
        driver += "int main(void){\n"
        for p, family in FAMILIES:
            scalar = TYPES[p]
            row_reflectors = family.endswith(("lq", "rq"))
            driver += f"""
    for(int layout=101;layout<=102;layout++)for(int which=0;which<2;which++){{
        lapack_int m=which?5:3,n=which?3:5,k=2,ld=8;
        {scalar} a[128]={{0}},c[128]={{0}},tau[8]={{0}};
        int i={0 if row_reflectors else 4},j={4 if row_reflectors else 0};
        a[layout==101?i*ld+j:i+j*ld]=({scalar})NAN;scans=0;arows=acols=-1;
        lapack_int info=LAPACKE_{p}{family}(layout,'R','N',m,n,k,a,ld,tau,c,ld);
        printf("{p}{family} %d %lld %lld %lld %lld %lld\\n",layout,(long long)m,(long long)n,(long long)arows,(long long)acols,(long long)info);
    }}
"""
        driver += "return 0;}\n"
        (root / "driver.c").write_text(driver)
        executable = root / "probe"
        subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra", "-Werror",
                        "-I", str(root), *sources, str(root / "driver.c"), "-o", str(executable)], check=True)
        output = subprocess.check_output([str(executable)], text=True)
        records = []
        for line in output.splitlines():
            name, layout, m, n, ar, ac, info = line.split()
            layout, m, n, ar, ac, info = map(int, (layout, m, n, ar, ac, info))
            rows = name.endswith(("lq", "rq"))
            m_scan = name.endswith("rq") or (name[0] in "cz" and name.endswith("lq"))
            order = m if m_scan else n  # SIDE=R in every probe.
            assert (ar, ac) == ((2, order) if rows else (order, 2))
            assert info == (-7 if order > 4 else 0)
            records.append(dict(name=name, layout=layout, m=m, n=n, a_rows=ar, a_cols=ac, info=info))
        assert len(records) == 64
        print(json.dumps(dict(reference_version=VERSION, numerical_work="stubbed; policy-only probe",
                              source_sha256=hashes, observations=records), indent=2))


if __name__ == "__main__":
    main()
