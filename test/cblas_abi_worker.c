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
typedef void (*her64)(int,int,int64_t,double,const void *,int64_t,void *,int64_t);
typedef void (*herk64)(int,int,int,int64_t,int64_t,double,const void *,int64_t,double,void *,int64_t);
typedef void (*her2k64)(int,int,int,int64_t,int64_t,const void *,const void *,int64_t,const void *,int64_t,double,void *,int64_t);
typedef void (*tp64)(int,int,int,int,int64_t,const void *,void *,int64_t);
typedef void (*tb64)(int,int,int,int,int64_t,int64_t,const void *,int64_t,void *,int64_t);
typedef void (*gb64)(int,int,int64_t,int64_t,int64_t,int64_t,const void *,const void *,int64_t,const void *,int64_t,const void *,void *,int64_t);
typedef void (*hb64)(int,int,int64_t,int64_t,const void *,const void *,int64_t,const void *,int64_t,const void *,void *,int64_t);
typedef void (*hp64)(int,int,int64_t,const void *,const void *,const void *,int64_t,const void *,void *,int64_t);
typedef void (*pr64)(int,int,int64_t,double,const void *,int64_t,void *);
typedef void (*pr264)(int,int,int64_t,const void *,const void *,int64_t,const void *,int64_t,void *);
typedef void (*copy64)(int,int,int64_t,int64_t,const void *,const void *,int64_t,void *,int64_t);
typedef void (*icopy64)(int,int,int64_t,int64_t,const void *,void *,int64_t,int64_t);
typedef void (*add64)(int,int64_t,int64_t,const void *,const void *,int64_t,const void *,void *,int64_t);
typedef void (*tg64)(int,int,int,int,int64_t,int64_t,const void *,const void *,int64_t,const void *,int64_t,const void *,void *,int64_t);

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
    ENTRY(her64,cblas_zher64_);
    ENTRY(herk64,cblas_zherk64_);
    ENTRY(her2k64,cblas_zher2k64_);
    ENTRY(tp64,cblas_ztpmv64_);
    ENTRY(tp64,cblas_ztpsv64_);
    ENTRY(tb64,cblas_ztbmv64_);
    ENTRY(tb64,cblas_ztbsv64_);
    ENTRY(gb64,cblas_zgbmv64_);
    ENTRY(hb64,cblas_zhbmv64_);
    ENTRY(hp64,cblas_zhpmv64_);
    ENTRY(pr64,cblas_zhpr64_);
    ENTRY(pr264,cblas_zhpr264_);
    ENTRY(copy64,cblas_zomatcopy64_);
    ENTRY(icopy64,cblas_zimatcopy64_);
    ENTRY(add64,cblas_zgeadd64_);
    ENTRY(tg64,cblas_zgemmt64_);
    ENTRY(tg64,cblas_zgemmtr64_);
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
    double complex updated[]={2,1+2*I,97,89,3,97};
    cblas_zher64_(101,121,2,0.5,b,2,updated,3);
    assert(cabs(updated[0]-3)<1e-12);
    assert(cabs(updated[1]-(1.5+3.5*I))<1e-12);
    assert(cabs(updated[4]-5.5)<1e-12);
    assert(updated[2]==97&&updated[3]==89&&updated[5]==97);
    double complex rank_c[]={NAN,NAN,97,89,NAN,97};
    cblas_zherk64_(101,121,111,2,1,0.5,b,2,0.0,rank_c,3);
    assert(cabs(rank_c[0]-1)<1e-12);
    assert(cabs(rank_c[1]-(0.5+1.5*I))<1e-12);
    assert(cabs(rank_c[4]-2.5)<1e-12);
    assert(rank_c[2]==97&&rank_c[3]==89&&rank_c[5]==97);
    const double complex rank_b[]={2+I,97,1-2*I,97};
    const double complex rank_alpha=0.75+0.25*I;
    double complex rank2_c[]={2,1+2*I,97,89,3,97};
    const double complex rank2_initial[]={2,1+2*I,97,89,3,97};
    cblas_zher2k64_(101,121,111,2,1,&rank_alpha,b,2,rank_b,2,0.5,rank2_c,3);
    for (int row=0;row<2;++row) for (int col=row;col<2;++col) {
        double complex expected=rank_alpha*b[2*row]*conj(rank_b[2*col])
            +conj(rank_alpha)*rank_b[2*row]*conj(b[2*col])
            +0.5*rank2_initial[row*3+col];
        assert(cabs(rank2_c[row*3+col]-expected)<1e-12);
    }
    assert(rank2_c[2]==97&&rank2_c[3]==89&&rank2_c[5]==97);
    const double complex packed[]={2,1+2*I,3};
    const double complex band[]={2,1+2*I,97,3,NAN,97};
    double complex compact_x[]={2-I,1+I};
    cblas_ztpmv64_(101,121,114,131,2,packed,compact_x,-1);
    assert(cabs(compact_x[0]-(6-3*I))<1e-12);
    assert(cabs(compact_x[1]-(2-3*I))<1e-12);
    cblas_ztpsv64_(101,121,114,131,2,packed,compact_x,-1);
    assert(cabs(compact_x[0]-(2-I))<1e-12);
    assert(cabs(compact_x[1]-(1+I))<1e-12);
    const double complex copy_alpha=1+I;
    const double complex copy_source[]={1+I,2-I,3+2*I,97,4-I,5+I,6-2*I,97,97,97,97,97};
    double complex copy_out[12],copy_inplace[12],copy_expected[12];
    for (int pos=0;pos<12;++pos) {copy_out[pos]=97;copy_inplace[pos]=copy_source[pos];copy_expected[pos]=copy_source[pos];}
    cblas_zomatcopy64_(101,113,2,3,&copy_alpha,copy_source,4,copy_out,3);
    for (int row=0;row<3;++row) for (int col=0;col<2;++col) {
        double complex expected=copy_alpha*conj(copy_source[col*4+row]);
        assert(cabs(copy_out[row*3+col]-expected)<1e-12);
        copy_expected[row*3+col]=expected;
    }
    assert(copy_out[2]==97&&copy_out[5]==97&&copy_out[8]==97);
    cblas_zimatcopy64_(101,113,2,3,&copy_alpha,copy_inplace,4,3);
    for (int pos=0;pos<12;++pos) assert(cabs(copy_inplace[pos]-copy_expected[pos])<1e-12);
    double complex add_out[]={NAN,NAN,NAN,97,NAN,NAN,NAN,97};
    cblas_zgeadd64_(101,2,3,&copy_alpha,copy_source,4,&beta,add_out,4);
    for (int row=0;row<2;++row) for (int col=0;col<3;++col) assert(cabs(add_out[row*4+col]-copy_alpha*copy_source[row*4+col])<1e-12);
    assert(add_out[3]==97&&add_out[7]==97);
    const double complex tri_a[]={1+I,2-I,97,3+2*I,4-I,97};
    const double complex tri_b[]={2+I,1-2*I,97,3-I,2+2*I,97};
    tg64 triangular_functions[]={cblas_zgemmt64_,cblas_zgemmtr64_};
    for (int variant=0;variant<2;++variant) {
        double complex tri_c[]={NAN,NAN,97,89,NAN,97};
        triangular_functions[variant](101,121,114,113,2,2,&alpha,tri_a,3,tri_b,3,&beta,tri_c,3);
        for (int row=0;row<2;++row) for (int col=row;col<2;++col) {
            double complex expected=0;
            for (int inner=0;inner<2;++inner) expected+=conj(tri_a[row*3+inner])*conj(tri_b[col*3+inner]);
            assert(cabs(tri_c[row*3+col]-expected)<1e-12);
        }
        assert(tri_c[2]==97&&tri_c[3]==89&&tri_c[5]==97);
    }
    const double complex general_band[]={1+I,2-I,NAN,97,3+2*I,4-I,NAN,97};
    const double complex general_x[]={1+I,2-I};
    double complex general_y[]={NAN,NAN,NAN};
    cblas_zgbmv64_(101,113,2,3,0,1,&alpha,general_band,4,general_x,1,&beta,general_y,1);
    assert(cabs(general_y[0]-2)<1e-12);
    assert(cabs(general_y[1]-(5-4*I))<1e-12);
    assert(cabs(general_y[2]-(9-2*I))<1e-12);
    double complex symmetric_y[]={NAN,NAN};
    cblas_zhbmv64_(101,121,2,1,&alpha,band,3,general_x,1,&beta,symmetric_y,1);
    assert(cabs(symmetric_y[0]-(6+5*I))<1e-12);
    assert(cabs(symmetric_y[1]-(9-4*I))<1e-12);
    double complex packed_updated[]={2,1+2*I,3};
    cblas_zhpr64_(101,121,2,0.5,general_x,1,packed_updated);
    assert(cabs(packed_updated[0]-3)<1e-12);
    assert(cabs(packed_updated[1]-(1.5+3.5*I))<1e-12);
    assert(cabs(packed_updated[2]-5.5)<1e-12);
    const double complex second_x[]={2+I,1-2*I};
    const double complex packed_initial[]={2,1+2*I,3};
    double complex packed_two[]={2,1+2*I,3};
    cblas_zhpr264_(101,121,2,&rank_alpha,general_x,1,second_x,1,packed_two);
    const int packed_rows[]={0,0,1},packed_cols[]={0,1,1};
    for (int pos=0;pos<3;++pos) {
        int row=packed_rows[pos],col=packed_cols[pos];
        double complex expected=packed_initial[pos]
            +rank_alpha*general_x[row]*conj(second_x[col])
            +conj(rank_alpha)*second_x[row]*conj(general_x[col]);
        assert(cabs(packed_two[pos]-expected)<1e-12);
    }
    symmetric_y[0]=NAN;symmetric_y[1]=NAN;
    cblas_zhpmv64_(101,121,2,&alpha,packed,general_x,1,&beta,symmetric_y,1);
    assert(cabs(symmetric_y[0]-(6+5*I))<1e-12);
    assert(cabs(symmetric_y[1]-(9-4*I))<1e-12);
    cblas_ztbmv64_(101,121,114,131,2,1,band,3,compact_x,-1);
    assert(cabs(compact_x[0]-(6-3*I))<1e-12);
    assert(cabs(compact_x[1]-(2-3*I))<1e-12);
    cblas_ztbsv64_(101,121,114,131,2,1,band,3,compact_x,-1);
    assert(cabs(compact_x[0]-(2-I))<1e-12);
    assert(cabs(compact_x[1]-(1+I))<1e-12);
    assert(dlclose(library)==0);
    puts("CBLAS C ABI: complex returns, size_t indices, ILP64 enums and coefficients passed");
    return 0;
}
