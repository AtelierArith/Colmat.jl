/* Test-only foreign caller. No production code or allocation depends on this. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

typedef void (*potrf64)(const char *, const int64_t *, double *, const int64_t *, int64_t *, size_t);
typedef void (*wrong_potrf32)(const char *, const int64_t *, double *, const int64_t *, int32_t *, size_t);
typedef void (*wrong_info64)(const char *, const int32_t *, double *, const int32_t *, int64_t *, size_t);
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    struct rlimit limit = {0, 0};
    if (setrlimit(RLIMIT_CORE, &limit)) return 3;
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) return 4;
    char *memory = mmap(NULL, 2 * page, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANON, -1, 0);
    if (memory == MAP_FAILED || mprotect(memory + page, page, PROT_NONE)) return 5;
    void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) { fprintf(stderr, "%s\n", dlerror()); return 6; }
    void *entry = dlsym(library, "dpotrf_64_");
    if (!entry) return 7;
    int64_t n = 0, lda = 1;
    double a = 0;
    char uplo = 'L';
    if (!strcmp(argv[2], "valid")) {
        int64_t *info = (int64_t *)(memory + page - sizeof(int64_t));
        *info = -777;
        ((potrf64)entry)(&uplo, &n, &a, &lda, info, 1);
        if (*info != 0) return 8;
        puts("VALID_WIDTH_CONTROL_PASSED");
    } else if (!strcmp(argv[2], "wrong")) {
        int32_t *info = (int32_t *)(memory + page - sizeof(int32_t));
        *info = -777;
        puts("WRONG_WIDTH_CALL_READY");
        fflush(stdout);
        ((wrong_potrf32)entry)(&uplo, &n, &a, &lda, info, 1);
        return 9; /* Returning is a failed crash witness. */
    } else if (!strcmp(argv[2], "narrow")) {
        void *narrow_entry = dlsym(library, "dpotrf_");
        if (!narrow_entry) return 11;
        int32_t n32 = 0, lda32 = 1;
        int64_t *info = (int64_t *)(memory + page - sizeof(int64_t));
        *info = INT64_C(0x1357913579);
        int64_t expected = *info;
        memset(&expected, 0, sizeof(int32_t));
        ((wrong_info64)narrow_entry)(&uplo, &n32, &a, &lda32, info, 1);
        if (*info == 0 || *info != expected) return 12;
        puts("NARROW_INFO_WRITE_DETECTED");
    } else return 10;
    dlclose(library);
    munmap(memory, 2 * page);
    return 0;
}
