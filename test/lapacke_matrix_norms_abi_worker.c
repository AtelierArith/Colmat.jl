/* Independent C caller for the LAPACKE matrix norm adapters. */
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef int32_t i32;
typedef int64_t i64;
typedef struct { float re, im; } cfloat;
typedef struct { double re, im; } cdouble;

#define DECLARE(T, R, S, I)                                                      \
    typedef R (*S##lange_##I)(int, char, I, I, const T *, I);                    \
    typedef R (*S##lansy_##I)(int, char, char, I, const T *, I);                 \
    typedef R (*S##lantr_##I)(int, char, char, char, I, I, const T *, I);

DECLARE(float, float, LAPACKE_s, i32)
DECLARE(double, double, LAPACKE_d, i32)
DECLARE(cfloat, float, LAPACKE_c, i32)
DECLARE(cdouble, double, LAPACKE_z, i32)
DECLARE(float, float, LAPACKE_s, i64)
DECLARE(double, double, LAPACKE_d, i64)
DECLARE(cfloat, float, LAPACKE_c, i64)
DECLARE(cdouble, double, LAPACKE_z, i64)

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

static void check_real(void *library, const char *lange, const char *lansy,
                       const char *lantr, int wide) {
    float a[9] = {1, 2, 3, 4, 0, 0, 0, 0, 0};
    if (wide) {
        CHECK(((LAPACKE_slange_i64)load(library, lange))(102, 'M', 2, 2, a, 2) == 4.0f, lange);
        CHECK(((LAPACKE_slansy_i64)load(library, lansy))(102, 'M', 'U', 2, a, 2) == 4.0f, lansy);
        CHECK(((LAPACKE_slantr_i64)load(library, lantr))(102, 'M', 'U', 'N', 2, 2, a, 2) == 4.0f, lantr);
    } else {
        { float got=((LAPACKE_slange_i32)load(library, lange))(102, 'M', 2, 2, a, 2); if (got != 4.0f) { fprintf(stderr,"slange got %g\n",got); exit(1);} }
        CHECK(((LAPACKE_slansy_i32)load(library, lansy))(102, 'M', 'U', 2, a, 2) == 4.0f, lansy);
        CHECK(((LAPACKE_slantr_i32)load(library, lantr))(102, 'M', 'U', 'N', 2, 2, a, 2) == 4.0f, lantr);
    }
}

static void check_double(void *library, const char *lange, const char *lansy,
                         const char *lantr, int wide) {
    double a[4] = {1, 2, 2, 4};
    if (wide) {
        CHECK(((LAPACKE_dlange_i64)load(library, lange))(102, 'M', 2, 2, a, 2) == 4.0, lange);
        CHECK(((LAPACKE_dlansy_i64)load(library, lansy))(102, 'M', 'L', 2, a, 2) == 4.0, lansy);
        CHECK(((LAPACKE_dlantr_i64)load(library, lantr))(102, 'M', 'L', 'N', 2, 2, a, 2) == 4.0, lantr);
    } else {
        CHECK(((LAPACKE_dlange_i32)load(library, lange))(102, 'M', 2, 2, a, 2) == 4.0, lange);
        CHECK(((LAPACKE_dlansy_i32)load(library, lansy))(102, 'M', 'L', 2, a, 2) == 4.0, lansy);
        CHECK(((LAPACKE_dlantr_i32)load(library, lantr))(102, 'M', 'L', 'N', 2, 2, a, 2) == 4.0, lantr);
    }
}

static void check_complex(void *library, const char *lange, const char *lansy,
                          const char *lantr, int wide) {
    cdouble a[4] = {{1, 1}, {2, 0}, {2, 0}, {4, 0}};
    if (wide) {
        CHECK(((LAPACKE_zlange_i64)load(library, lange))(102, 'M', 2, 2, a, 2) == 4.0, lange);
        CHECK(((LAPACKE_zlansy_i64)load(library, lansy))(102, 'M', 'L', 2, a, 2) == 4.0, lansy);
        CHECK(((LAPACKE_zlantr_i64)load(library, lantr))(102, 'M', 'L', 'N', 2, 2, a, 2) == 4.0, lantr);
    } else {
        CHECK(((LAPACKE_zlange_i32)load(library, lange))(102, 'M', 2, 2, a, 2) == 4.0, lange);
        CHECK(((LAPACKE_zlansy_i32)load(library, lansy))(102, 'M', 'L', 2, a, 2) == 4.0, lansy);
        CHECK(((LAPACKE_zlantr_i32)load(library, lantr))(102, 'M', 'L', 'N', 2, 2, a, 2) == 4.0, lantr);
    }
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
    check_real(library, "LAPACKE_slange", "LAPACKE_slansy", "LAPACKE_slantr", 0);
    check_real(library, "LAPACKE_slange_work", "LAPACKE_slansy_work", "LAPACKE_slantr_work", 0);
    check_real(library, "LAPACKE_slange64_", "LAPACKE_slansy64_", "LAPACKE_slantr64_", 1);
    check_real(library, "LAPACKE_slange_work64_", "LAPACKE_slansy_work64_", "LAPACKE_slantr_work64_", 1);
    check_double(library, "LAPACKE_dlange", "LAPACKE_dlansy", "LAPACKE_dlantr", 0);
    check_double(library, "LAPACKE_dlange_work", "LAPACKE_dlansy_work", "LAPACKE_dlantr_work", 0);
    check_double(library, "LAPACKE_dlange64_", "LAPACKE_dlansy64_", "LAPACKE_dlantr64_", 1);
    check_double(library, "LAPACKE_dlange_work64_", "LAPACKE_dlansy_work64_", "LAPACKE_dlantr_work64_", 1);
    check_complex(library, "LAPACKE_zlange", "LAPACKE_zlansy", "LAPACKE_zlantr", 0);
    check_complex(library, "LAPACKE_zlange_work", "LAPACKE_zlansy_work", "LAPACKE_zlantr_work", 0);
    check_complex(library, "LAPACKE_zlange64_", "LAPACKE_zlansy64_", "LAPACKE_zlantr64_", 1);
    check_complex(library, "LAPACKE_zlange_work64_", "LAPACKE_zlansy_work64_", "LAPACKE_zlantr_work64_", 1);
    printf("LAPACKE matrix norm C ABI: ok\n");
    dlclose(library);
    return 0;
}
