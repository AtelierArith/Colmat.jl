/* Independent C caller for the LAPACKE LAPMR/LAPMT adapters. */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef int32_t i32;
typedef int64_t i64;
typedef struct { float re, im; } cfloat;
typedef struct { double re, im; } cdouble;

#define DECLARE(T, S, I)                                                         \
    typedef i32 (*S##lapmr_##I)(int, I, I, I, T *, I, I *);                      \
    typedef i32 (*S##lapmt_##I)(int, I, I, I, T *, I, I *);

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

#define RUN_R(T, S, I, NAME, SUF, LAPMR, LAPMT)                                       \
    do {                                                                         \
        T a[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};                                    \
        I k[3] = {1, 2, 3};                                                      \
        i32 info;                                                                \
        if (LAPMR) {                                                             \
            S##lapmr_##I f = (S##lapmr_##I)load(library, NAME "lapmr" SUF);          \
            info = f(101, (I)1, (I)3, (I)2, a, (I)3, k);                        \
        } else {                                                                 \
            S##lapmt_##I f = (S##lapmt_##I)load(library, NAME "lapmt" SUF);          \
            info = f(101, (I)1, (I)3, (I)2, a, (I)3, k);                        \
        }                                                                        \
        CHECK(info == 0, NAME);                                                  \
        CHECK(k[0] == 1 && k[1] == 2 && k[2] == 3, NAME);                        \
    } while (0)

#define RUN_C(T, S, I, NAME, SUF)                                                     \
    do {                                                                         \
        T a[9] = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}}; \
        I k[3] = {1, 2, 3};                                                      \
        S##lapmr_##I f = (S##lapmr_##I)load(library, NAME "lapmr" SUF);              \
        i32 info = f(101, (I)1, (I)3, (I)2, a, (I)3, k);                        \
        CHECK(info == 0, NAME);                                                  \
        S##lapmt_##I g = (S##lapmt_##I)load(library, NAME "lapmt_work" SUF);         \
        info = g(101, (I)1, (I)3, (I)2, a, (I)3, k);                            \
        CHECK(info == 0, NAME);                                                  \
    } while (0)

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
    RUN_R(float, LAPACKE_s, i32, "LAPACKE_s", "", 1, 0);
    RUN_R(float, LAPACKE_s, i32, "LAPACKE_s", "", 0, 1);
    RUN_R(double, LAPACKE_d, i32, "LAPACKE_d", "", 1, 0);
    RUN_R(double, LAPACKE_d, i32, "LAPACKE_d", "", 0, 1);
    RUN_C(cfloat, LAPACKE_c, i32, "LAPACKE_c", "");
    RUN_C(cdouble, LAPACKE_z, i32, "LAPACKE_z", "");
    RUN_R(float, LAPACKE_s, i64, "LAPACKE_s", "64_", 1, 0);
    RUN_R(float, LAPACKE_s, i64, "LAPACKE_s", "64_", 0, 1);
    RUN_R(double, LAPACKE_d, i64, "LAPACKE_d", "64_", 1, 0);
    RUN_R(double, LAPACKE_d, i64, "LAPACKE_d", "64_", 0, 1);
    RUN_C(cfloat, LAPACKE_c, i64, "LAPACKE_c", "64_");
    RUN_C(cdouble, LAPACKE_z, i64, "LAPACKE_z", "64_");
    printf("LAPACKE matrix permutation C ABI: ok\n");
    dlclose(library);
    return 0;
}
