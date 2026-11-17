/* Independent C caller for the LAPACKE TGEVC high/work adapters. */
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int32_t i32;
typedef int64_t i64;
typedef struct { float re, im; } cfloat;
typedef struct { double re, im; } cdouble;
typedef i32 (*tgevc_i32)(int, char, char, const i32 *, i32, const void *,
                         i32, const void *, i32, void *, i32, void *, i32,
                         i32, i32 *);
typedef i64 (*tgevc_i64)(int, char, char, const i64 *, i64, const void *,
                         i64, const void *, i64, void *, i64, void *, i64,
                         i64, i64 *);

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

static void check_real32(void *library, const char *name, int wide) {
    float s[4] = {1.0f, 0.0f, 2.0f, 3.0f};
    float t[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float vl[4] = {0};
    float vr[4] = {0};
    int info;
    if (wide) {
        i64 select[2] = {1, 1};
        i64 m = -1;
        tgevc_i64 f = (tgevc_i64)load(library, name);
        info = (int)f(102, 'B', 'A', select, 2, s, 2, t, 2, vl, 2, vr, 2, 2, &m);
        CHECK(m == 2 && info == 0, name);
    } else {
        i32 select[2] = {1, 1};
        i32 m = -1;
        tgevc_i32 f = (tgevc_i32)load(library, name);
        info = (int)f(102, 'B', 'A', select, 2, s, 2, t, 2, vl, 2, vr, 2, 2, &m);
        CHECK(m == 2 && info == 0, name);
    }
    CHECK(fabsf(fabsf(vr[0]) - 1.0f) < 1e-5f && fabsf(fabsf(vl[0]) - 1.0f) < 1e-5f, name);
}

static void check_real64(void *library, const char *name, int wide) {
    double s[4] = {1.0, 0.0, 2.0, 3.0};
    double t[4] = {1.0, 0.0, 0.0, 1.0};
    double vl[4] = {0};
    double vr[4] = {0};
    int info;
    if (wide) {
        i64 select[2] = {1, 1};
        i64 m = -1;
        tgevc_i64 f = (tgevc_i64)load(library, name);
        info = (int)f(102, 'B', 'A', select, 2, s, 2, t, 2, vl, 2, vr, 2, 2, &m);
        CHECK(m == 2 && info == 0, name);
    } else {
        i32 select[2] = {1, 1};
        i32 m = -1;
        tgevc_i32 f = (tgevc_i32)load(library, name);
        info = (int)f(102, 'B', 'A', select, 2, s, 2, t, 2, vl, 2, vr, 2, 2, &m);
        CHECK(m == 2 && info == 0, name);
    }
    CHECK(fabs(fabs(vr[0]) - 1.0) < 1e-5 && fabs(fabs(vl[0]) - 1.0) < 1e-5, name);
}

static void check_complex32(void *library, const char *name, int wide) {
    cfloat s[4] = {{1, 1}, {0, 0}, {2, 0}, {3, -1}};
    cfloat t[4] = {{1, 0}, {0, 0}, {0, 0}, {1, 0}};
    cfloat vl[4] = {{0, 0}};
    cfloat vr[4] = {{0, 0}};
    int info;
    if (wide) {
        i64 select[2] = {1, 1};
        i64 m = -1;
        tgevc_i64 f = (tgevc_i64)load(library, name);
        info = (int)f(102, 'B', 'A', select, 2, s, 2, t, 2, vl, 2, vr, 2, 2, &m);
        CHECK(m == 2 && info == 0, name);
    } else {
        i32 select[2] = {1, 1};
        i32 m = -1;
        tgevc_i32 f = (tgevc_i32)load(library, name);
        info = (int)f(102, 'B', 'A', select, 2, s, 2, t, 2, vl, 2, vr, 2, 2, &m);
        CHECK(m == 2 && info == 0, name);
    }
    CHECK(fabsf(fabsf(vr[0].re) - 1.0f) < 1e-5f && fabsf(fabsf(vl[0].re) - 1.0f) < 1e-5f, name);
}

static void check_complex64(void *library, const char *name, int wide) {
    cdouble s[4] = {{1, 1}, {0, 0}, {2, 0}, {3, -1}};
    cdouble t[4] = {{1, 0}, {0, 0}, {0, 0}, {1, 0}};
    cdouble vl[4] = {{0, 0}};
    cdouble vr[4] = {{0, 0}};
    int info;
    if (wide) {
        i64 select[2] = {1, 1};
        i64 m = -1;
        tgevc_i64 f = (tgevc_i64)load(library, name);
        info = (int)f(102, 'B', 'A', select, 2, s, 2, t, 2, vl, 2, vr, 2, 2, &m);
        CHECK(m == 2 && info == 0, name);
    } else {
        i32 select[2] = {1, 1};
        i32 m = -1;
        tgevc_i32 f = (tgevc_i32)load(library, name);
        info = (int)f(102, 'B', 'A', select, 2, s, 2, t, 2, vl, 2, vr, 2, 2, &m);
        CHECK(m == 2 && info == 0, name);
    }
    CHECK(fabs(fabs(vr[0].re) - 1.0) < 1e-5 && fabs(fabs(vl[0].re) - 1.0) < 1e-5, name);
}

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
    check_real32(library, "LAPACKE_stgevc", 0);
    check_real64(library, "LAPACKE_dtgevc", 0);
    check_real32(library, "LAPACKE_stgevc_work", 0);
    check_real64(library, "LAPACKE_dtgevc_work", 0);
    check_real32(library, "LAPACKE_stgevc64_", 1);
    check_real64(library, "LAPACKE_dtgevc64_", 1);
    check_real32(library, "LAPACKE_stgevc_work64_", 1);
    check_real64(library, "LAPACKE_dtgevc_work64_", 1);
    check_complex32(library, "LAPACKE_ctgevc", 0);
    check_complex64(library, "LAPACKE_ztgevc", 0);
    check_complex32(library, "LAPACKE_ctgevc_work", 0);
    check_complex64(library, "LAPACKE_ztgevc_work", 0);
    check_complex32(library, "LAPACKE_ctgevc64_", 1);
    check_complex64(library, "LAPACKE_ztgevc64_", 1);
    check_complex32(library, "LAPACKE_ctgevc_work64_", 1);
    check_complex64(library, "LAPACKE_ztgevc_work64_", 1);
    printf("LAPACKE TGEVC C ABI: ok\n");
    dlclose(library);
    return 0;
}
