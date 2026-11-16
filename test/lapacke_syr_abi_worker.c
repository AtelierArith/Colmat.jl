/* LAPACK v3.12.1 CSYR/ZSYR ABI: complex ALPHA is passed by value. */
#include <assert.h>
#include <complex.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(P,T,R,INT,SUFFIX,TOL) do { \
    typedef INT (*update)(int,char,INT,T,const T *,INT,T *,INT); \
    const char *names[]={"LAPACKE_" #P "syr" SUFFIX,"LAPACKE_" #P "syr_work" SUFFIX}; \
    for(int entry=0;entry<2;entry++) { \
        update call=(update)dlsym(library,names[entry]);assert(call); \
        for(int layout=101;layout<=102;layout++) for(int flag=0;flag<4;flag++) for(int stride_case=0;stride_case<4;stride_case++) { \
            char uplo="LUlu"[flag];int lower=(flag%2)==0; \
            int strides[]={-3,-1,1,2};INT inc=(INT)strides[stride_case];int step=inc<0?-(int)inc:(int)inc; \
            T a[20],expected[20],x[10],logical[3];T alpha=(T)(0.7-0.4*I); \
            for(int q=0;q<20;q++){a[q]=(T)NAN;expected[q]=(T)NAN;} \
            for(int q=0;q<10;q++)x[q]=(T)NAN; \
            for(int k=0;k<3;k++){logical[k]=(T)((1+0.5*k)+(0.3-0.2*k)*I);x[(inc<0?2-k:k)*step]=logical[k];} \
            for(int i=0;i<3;i++)for(int j=0;j<3;j++)if(lower?i>=j:i<=j) { \
                int q=layout==101?i*5+j:i+j*5; \
                a[q]=(T)((2+i+3*j)+(0.2*i-0.3*j)*I); \
                expected[q]=a[q]+alpha*logical[i]*logical[j]; \
            } \
            assert(call(layout,uplo,3,alpha,x,inc,a,5)==0); \
            for(int q=0;q<20;q++) { \
                if(isnan(creal((double complex)expected[q])))assert(isnan(creal((double complex)a[q]))); \
                else assert(cabs((double complex)(a[q]-expected[q]))<(TOL)); \
            } \
        } \
        T alpha=(T)(0.7-0.4*I); \
        assert(call(0,'L',3,alpha,NULL,1,NULL,3)==-1); \
        assert(call(101,'L',3,alpha,NULL,1,NULL,2)==-8); \
        assert(call(101,'L',0,alpha,NULL,1,NULL,0)==0); \
        assert(call(102,'U',3,(T)0,NULL,1,NULL,3)==0); \
        assert(call(101,'L',3,(T)0,NULL,-2,NULL,3)==0); \
        (void)last_error();assert(call(102,'?',3,alpha,NULL,1,NULL,3)==0);assert(last_error()==1); \
        assert(call(102,'L',-1,alpha,NULL,1,NULL,3)==0);assert(last_error()==2); \
        assert(call(102,'L',3,alpha,NULL,0,NULL,3)==0);assert(last_error()==5); \
        assert(call(102,'L',3,alpha,NULL,1,NULL,2)==0);assert(last_error()==7); \
        if(entry==0) { \
            R parts[2]={(R)2,(R)NAN};T poison;memcpy(&poison,parts,sizeof(poison)); \
            assert(creal((double complex)poison)==2 && isnan(cimag((double complex)poison))); \
            T a[4]={poison,1,1,4},x[3]={1,(T)NAN,2}; \
            assert(call(102,'L',2,alpha,x,2,a,2)==-7); \
            a[0]=2;assert(call(102,'L',2,poison,x,2,a,2)==-4); \
            x[2]=poison;assert(call(102,'L',2,alpha,x,2,a,2)==-5); \
            x[2]=2;assert(call(102,'L',2,alpha,x,2,a,2)==0); \
            assert(call(102,'L',0,poison,NULL,1,NULL,1)==-4); \
        } \
    } \
} while(0)

int main(int argc,char **argv) {
    assert(argc==2);void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if(!library){fprintf(stderr,"%s\n",dlerror());return 3;}
    typedef void (*set_nancheck)(int);typedef int (*get_error)(void);
    set_nancheck set=(set_nancheck)dlsym(library,"LAPACKE_set_nancheck");assert(set);set(1);
    get_error last_error=(get_error)dlsym(library,"colmat_blas_last_error");assert(last_error);
    CHECK(c,float complex,float,int32_t,"",1e-5);
    CHECK(z,double complex,double,int32_t,"",1e-12);
    CHECK(c,float complex,float,int64_t,"64_",1e-5);
    CHECK(z,double complex,double,int64_t,"64_",1e-12);
    typedef int64_t (*zupdate)(int,char,int64_t,double complex,const double complex *,int64_t,double complex *,int64_t);
    zupdate call=(zupdate)dlsym(library,"LAPACKE_zsyr64_");assert(call);
    assert(call(102,'L',INT64_C(-4294967293),1,NULL,1,NULL,1)==0);assert(last_error()==2);
    double complex a=2,x=1+I,alpha=0.7-0.4*I,expected=a+alpha*x*x;
    assert(call(102,'L',1,alpha,&x,INT64_C(4294967296),&a,1)==0);assert(cabs(a-expected)<1e-12);
    puts("LAPACKE complex SYR C ABI: by-value coefficient, strided NaN checks, both integer widths passed");
    dlclose(library);return 0;
}
