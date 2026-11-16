/* Independent C ABI caller: Reference-LAPACK v3.12.1 Cholesky declarations. */
#include <assert.h>
#include <complex.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(P,T,INT,SUFFIX,OFF,TOL) do { \
    typedef INT (*single)(int,char,INT,T *,INT); \
    typedef INT (*solve)(int,char,INT,INT,const T *,INT,T *,INT); \
    typedef INT (*driver)(int,char,INT,INT,T *,INT,T *,INT); \
    const char *routines[]={"potrf","potrf2","potri"}; \
    T off=(T)(OFF);double norm=cabs((double complex)off),diag=9+norm*norm; \
    for(int layout=101;layout<=102;layout++) for(int u=0;u<4;u++) { \
        char uplo="LUlu"[u];int lower=(u%2)==0; \
        int selected=lower?(layout==101?3:1):(layout==101?1:3); \
        int ignored=selected==1?3:1;T held=lower?off:(T)conj((double complex)off); \
        for(int routine=0;routine<3;routine++) for(int work=0;work<2;work++) { \
            char name[96];snprintf(name,sizeof(name),"LAPACKE_%s%s%s%s",#P,routines[routine],work?"_work":"",SUFFIX); \
            single call=(single)dlsym(library,name);assert(call); \
            T a[6];for(int q=0;q<6;q++)a[q]=(T)NAN; \
            a[0]=routine==2?2:4;a[4]=routine==2?3:diag;a[selected]=routine==2?held:2*held; \
            assert(call(layout,uplo,2,a,3)==0); \
            T diagonal0=routine==2?(T)(diag/36):(T)2; \
            T diagonal1=routine==2?(T)(4.0/36):(T)3; \
            T off_result=routine==2?-2*held/36:held; \
            assert(cabs((double complex)(a[0]-diagonal0))<(TOL)); \
            assert(cabs((double complex)(a[4]-diagonal1))<(TOL)); \
            assert(cabs((double complex)(a[selected]-off_result))<(TOL)); \
            assert(isnan(creal((double complex)a[ignored])) && isnan(creal((double complex)a[2])) && isnan(creal((double complex)a[5]))); \
            assert(call(layout,uplo,0,NULL,layout==101?0:1)==0); \
            assert(call(101,'?',2,NULL,1)==-5); \
            assert(call(102,'?',2,NULL,2)==-2); \
            a[0]=routine==2?2:-1;a[4]=routine==2?0:diag;a[selected]=held; \
            assert(call(layout,uplo,2,a,3)==(routine==2?2:1)); \
        } \
        for(int operation=0;operation<2;operation++) for(int work=0;work<2;work++) { \
            char name[96];snprintf(name,sizeof(name),"LAPACKE_%s%s%s%s",#P,operation?"posv":"potrs",work?"_work":"",SUFFIX); \
            void *entry=dlsym(library,name);assert(entry); \
            T a[6],b[6];for(int q=0;q<6;q++){a[q]=(T)NAN;b[q]=(T)NAN;} \
            a[0]=operation?4:2;a[4]=operation?diag:3;a[selected]=operation?2*held:held; \
            T saved[6];memcpy(saved,a,sizeof(a)); \
            b[0]=4+4*(T)conj((double complex)off);b[layout==101?3:1]=2*off+2*diag; \
            INT info=operation?((driver)entry)(layout,uplo,2,1,a,3,b,3):((solve)entry)(layout,uplo,2,1,a,3,b,3); \
            assert(info==0);assert(cabs((double complex)(b[0]-1))<(TOL)); \
            assert(cabs((double complex)(b[layout==101?3:1]-2))<(TOL)); \
            if(!operation)assert(memcmp(a,saved,sizeof(a))==0); \
            assert(isnan(creal((double complex)a[ignored])) && isnan(creal((double complex)a[2])) && isnan(creal((double complex)a[5]))); \
            for(int q=0;q<6;q++)if(q!=0 && q!=(layout==101?3:1))assert(isnan(creal((double complex)b[q]))); \
            if(operation) { \
                a[0]=4;a[4]=diag;a[selected]=2*held; \
                assert(((driver)entry)(layout,uplo,2,0,a,3,NULL,layout==101?0:2)==0); \
                assert(cabs((double complex)(a[0]-2))<(TOL) && cabs((double complex)(a[4]-3))<(TOL)); \
                a[0]=-1;a[4]=diag;a[selected]=2*held; \
                b[0]=9;b[layout==101?3:1]=7; \
                assert(((driver)entry)(layout,uplo,2,1,a,3,b,3)==1); \
                assert(b[0]==9 && b[layout==101?3:1]==7); \
                assert(((driver)entry)(101,'?',2,1,NULL,1,NULL,0)==-6); \
                assert(((driver)entry)(101,'?',2,1,NULL,2,NULL,0)==-8); \
            } else { \
                assert(((solve)entry)(layout,uplo,2,0,NULL,3,NULL,layout==101?0:2)==0); \
                assert(((solve)entry)(101,'?',2,1,NULL,1,NULL,0)==-6); \
                assert(((solve)entry)(101,'?',2,1,NULL,2,NULL,0)==-8); \
            } \
        } \
    } \
} while(0)

int main(int argc,char **argv) {
    assert(argc==2);void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if(!library){fprintf(stderr,"%s\n",dlerror());return 3;}
    CHECK(s,float,int32_t,"",1.0f,1e-5);
    CHECK(d,double,int32_t,"",1.0,1e-12);
    CHECK(c,float complex,int32_t,"",1.0f+0.5f*I,1e-5);
    CHECK(z,double complex,int32_t,"",1.0+0.5*I,1e-12);
    CHECK(s,float,int64_t,"64_",1.0f,1e-5);
    CHECK(d,double,int64_t,"64_",1.0,1e-12);
    CHECK(c,float complex,int64_t,"64_",1.0f+0.5f*I,1e-5);
    CHECK(z,double complex,int64_t,"64_",1.0+0.5*I,1e-12);
    typedef int64_t (*zsingle)(int,char,int64_t,double complex *,int64_t);
    zsingle factor=(zsingle)dlsym(library,"LAPACKE_zpotrf64_");assert(factor);
    double parts[2]={2,NAN};double complex poison;memcpy(&poison,parts,sizeof(poison));
    assert(creal(poison)==2 && isnan(cimag(poison)));
    double complex a[4]={poison,1,1,4};assert(factor(102,'L',2,a,2)==-4);
    assert(factor(102,'L',INT64_C(-4294967293),NULL,2)==-3);
    typedef int64_t (*zdriver)(int,char,int64_t,int64_t,double complex *,int64_t,double complex *,int64_t);
    zdriver driver=(zdriver)dlsym(library,"LAPACKE_zposv64_");assert(driver);
    double complex b[2]={1,2};assert(driver(102,'L',2,1,a,2,b,2)==-5);
    a[0]=2;b[0]=poison;assert(driver(102,'L',2,1,a,2,b,2)==-7);
    zsingle factor_work=(zsingle)dlsym(library,"LAPACKE_zpotrf_work64_");assert(factor_work);
    a[0]=poison;a[1]=1;a[2]=1;a[3]=4;
    assert(factor_work(102,'L',2,a,2)==0);
    assert(isfinite(creal(a[0])) && cimag(a[0])==0 && isfinite(creal(a[1])) && isfinite(creal(a[3])));
    zdriver driver_work=(zdriver)dlsym(library,"LAPACKE_zposv_work64_");assert(driver_work);
    a[0]=poison;a[1]=1;a[2]=1;a[3]=4;b[0]=4;b[1]=9;
    assert(driver_work(102,'L',2,1,a,2,b,2)==0);
    assert(cabs(b[0]-1)<1e-12 && cabs(b[1]-2)<1e-12);
    puts("LAPACKE Cholesky C ABI: five families, high/work, four scalars, both integer widths passed");
    dlclose(library);return 0;
}
