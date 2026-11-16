/* Independent C caller: OpenBLAS v0.3.30 cblas.h ABI, loaded directly. */
#include <assert.h>
#include <complex.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define ENTRY(type, name) type name = (type)dlsym(library, #name); assert(name)
typedef float complex (*cdot32)(int32_t,const void *,int32_t,const void *,int32_t);
typedef float complex (*cdot64)(int64_t,const void *,int64_t,const void *,int64_t);
typedef double complex (*zdot32)(int32_t,const void *,int32_t,const void *,int32_t);
typedef double complex (*zdot64)(int64_t,const void *,int64_t,const void *,int64_t);
typedef size_t (*index32)(int32_t,const void *,int32_t);
typedef size_t (*index64)(int64_t,const void *,int64_t);
/* Enums remain C int while dimensions become int64_t. */
typedef void (*hemm64)(int,int,int,int64_t,int64_t,const void *,const void *,int64_t,const void *,int64_t,const void *,void *,int64_t);

int main(int argc,char **argv) {
    if (argc!=2) return 2;
    void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if (!library) {fprintf(stderr,"%s\n",dlerror());return 3;}
    ENTRY(cdot32,cblas_cdotu);
    ENTRY(cdot64,cblas_cdotc64_);
    ENTRY(zdot32,cblas_zdotu);
    ENTRY(zdot64,cblas_zdotc64_);
    ENTRY(index32,cblas_icamax);
    ENTRY(index64,cblas_izamax64_);
    ENTRY(hemm64,cblas_zhemm64_);
    const float complex xf[]={1+2*I,3-I},yf[]={2-I,-1+3*I};
    const double complex xd[]={1+2*I,3-I},yd[]={2-I,-1+3*I};
    assert(cabsf(cblas_cdotu(2,xf,1,yf,1)-(xf[0]*yf[0]+xf[1]*yf[1]))<1e-5f);
    assert(cabsf(cblas_cdotc64_(2,xf,1,yf,1)-(conjf(xf[0])*yf[0]+conjf(xf[1])*yf[1]))<1e-5f);
    assert(cabs(cblas_zdotu(2,xd,1,yd,1)-(xd[0]*yd[0]+xd[1]*yd[1]))<1e-12);
    assert(cabs(cblas_zdotc64_(2,xd,1,yd,1)-(conj(xd[0])*yd[0]+conj(xd[1])*yd[1]))<1e-12);
    assert(cblas_icamax(2,xf,1)==(size_t)1);
    assert(cblas_izamax64_(2,xd,1)==(size_t)1);
    const double complex alpha=1,beta=0;
    const double complex a[]={2,1+2*I,97,NAN,3,97};
    const double complex b[]={1+I,97,2-I,97};
    double complex c[]={NAN,97,NAN,97};
    cblas_zhemm64_(101,141,121,2,1,&alpha,a,3,b,2,&beta,c,2);
    assert(cabs(c[0]-(6+5*I))<1e-12);
    assert(cabs(c[2]-(9-4*I))<1e-12);
    assert(c[1]==97&&c[3]==97);
    assert(dlclose(library)==0);
    puts("CBLAS C ABI: complex returns, size_t indices, ILP64 enums and coefficients passed");
    return 0;
}
