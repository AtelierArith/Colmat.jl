/* Independent C caller for the LAPACKE LACPY/LASET/LASWP adapters. */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int32_t i32;
typedef int64_t i64;
typedef struct {
    float re, im;
} cfloat;
typedef struct {
    double re, im;
} cdouble;

#define DECLARE(T, S, I)                                                        \
    typedef i32 (*S##lacpy_##I)(int, char, I, I, const T *, I, T *, I);         \
    typedef i32 (*S##laset_##I)(int, char, I, I, T, T, T *, I);                 \
    typedef i32 (*S##laswp_##I)(int, I, T *, I, I, I, const I *, I);

DECLARE(float, LAPACKE_s, i32)
DECLARE(double, LAPACKE_d, i32)
DECLARE(cfloat, LAPACKE_c, i32)
DECLARE(cdouble, LAPACKE_z, i32)
DECLARE(float, LAPACKE_s, i64)
DECLARE(double, LAPACKE_d, i64)
DECLARE(cfloat, LAPACKE_c, i64)
DECLARE(cdouble, LAPACKE_z, i64)

static void *load(void *library, const char *name) {
    void *p = dlsym(library, name);
    if (!p) {
        fprintf(stderr, "missing %s: %s\n", name, dlerror());
        exit(1);
    }
    return p;
}

#define CHECK(cond, name)                                                        \
    do {                                                                         \
        if (!(cond)) {                                                           \
            fprintf(stderr, "FAILED %s:%d %s\n", __FILE__, __LINE__, name);      \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

#define INIT_R(a, k, v) ((a)[k] = (v))
#define INIT_C(a, k, v)                                                          \
    do {                                                                         \
        (a)[k].re = (v);                                                         \
        (a)[k].im = 0;                                                           \
    } while (0)

#define RUN(F, T, I, NAME, INIT, EQ)                                             \
    do {                                                                         \
        T a[12];                                                                 \
        T b[12];                                                                 \
        memset(a, 0, sizeof a);                                                  \
        memset(b, 0, sizeof b);                                                  \
        for (int k = 0; k < 12; k++) {                                           \
            INIT(a, k, ((double)k + 1));                                            \
        }                                                                        \
        F f = (F)load(library, NAME);                                            \
        i32 info = f(101, 'A', (I)2, (I)3, a, (I)4, b, (I)4);                   \
        CHECK(info == 0, NAME);                                                  \
        CHECK(EQ(b[0], a[0]) && EQ(b[1], a[1]) && EQ(b[4], a[4]), NAME);        \
    } while (0)

#define REAL_EQ(x, y) ((x) == (y))
#define COMPLEX_EQ(x, y) ((x).re == (y).re && (x).im == (y).im)

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s LIBRARY\n", argv[0]);
        return 2;
    }
    void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        fprintf(stderr, "dlopen: %s\n", dlerror());
        return 1;
    }

    RUN(LAPACKE_slacpy_i32, float, i32, "LAPACKE_slacpy", INIT_R, REAL_EQ);
    RUN(LAPACKE_dlacpy_i32, double, i32, "LAPACKE_dlacpy", INIT_R, REAL_EQ);
    RUN(LAPACKE_clacpy_i32, cfloat, i32, "LAPACKE_clacpy", INIT_C, COMPLEX_EQ);
    RUN(LAPACKE_zlacpy_i32, cdouble, i32, "LAPACKE_zlacpy", INIT_C, COMPLEX_EQ);
    RUN(LAPACKE_slacpy_i64, float, i64, "LAPACKE_slacpy64_", INIT_R, REAL_EQ);
    RUN(LAPACKE_dlacpy_i64, double, i64, "LAPACKE_dlacpy64_", INIT_R, REAL_EQ);
    RUN(LAPACKE_clacpy_i64, cfloat, i64, "LAPACKE_clacpy64_", INIT_C, COMPLEX_EQ);
    RUN(LAPACKE_zlacpy_i64, cdouble, i64, "LAPACKE_zlacpy64_", INIT_C, COMPLEX_EQ);

    for (int wide = 0; wide < 2; wide++) {
        const char *p = wide ? "LAPACKE_dlaset64_" : "LAPACKE_dlaset";
        double a[8] = {0};
        i32 info;
        if (wide) {
            LAPACKE_dlaset_i64 f = (LAPACKE_dlaset_i64)load(library, p);
            info = f(102, 'U', (i64)2, (i64)2, 3.0, 5.0, a, (i64)2);
        } else {
            LAPACKE_dlaset_i32 f = (LAPACKE_dlaset_i32)load(library, p);
            info = f(102, 'U', (i32)2, (i32)2, 3.0, 5.0, a, (i32)2);
        }
        CHECK(info == 0, p);
        CHECK(a[0] == 5.0 && a[2] == 3.0, p);
    }

    for (int wide = 0; wide < 2; wide++) {
        const char *p = wide ? "LAPACKE_dlaswp64_" : "LAPACKE_dlaswp";
        double a[6] = {0, 1, 2, 3, 4, 5};
        i32 info;
        if (wide) {
            i64 piv[1] = {2};
            LAPACKE_dlaswp_i64 f = (LAPACKE_dlaswp_i64)load(library, p);
            info = f(102, (i64)3, a, (i64)2, (i64)1, (i64)1, piv, (i64)1);
        } else {
            i32 piv[1] = {2};
            LAPACKE_dlaswp_i32 f = (LAPACKE_dlaswp_i32)load(library, p);
            info = f(102, (i32)3, a, (i32)2, (i32)1, (i32)1, piv, (i32)1);
        }
        CHECK(info == 0, p);
        CHECK(a[0] == 1.0 && a[2] == 3.0 && a[4] == 5.0, p);
    }

    printf("LAPACKE matrix auxiliary C ABI: ok\n");
    dlclose(library);
    return 0;
}
