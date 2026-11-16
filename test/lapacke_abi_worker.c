#define _POSIX_C_SOURCE 200809L
/* Independent C caller for Reference-LAPACK v3.12.1 LAPACKE GETRF declarations. */
#include <assert.h>
#include <complex.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(P,T,INT,SUFFIX,PHASE,TOL) do { \
    typedef INT (*factor)(int,INT,INT,T *,INT,INT *); \
    const char *names[]={"LAPACKE_" #P "getrf" SUFFIX,"LAPACKE_" #P "getrf_work" SUFFIX}; \
    for (int entry=0;entry<2;entry++) { \
        factor call=(factor)dlsym(library,names[entry]); assert(call); \
        for (int layout=101;layout<=102;layout++) { \
            T a[15],original[3][2]; INT piv[4]={-91,-92,-93,-94}; \
            for (int q=0;q<15;q++) a[q]=(T)777; \
            double values[3][2]={{2,1},{4,3},{1,5}}; \
            for(int i=0;i<3;i++) for(int j=0;j<2;j++) { \
                original[i][j]=(T)(values[i][j]*(PHASE)); \
                a[layout==101?i*5+j:i+j*5]=original[i][j]; \
            } \
            assert(call(layout,3,2,a,5,piv+1)==0); \
            assert(piv[0]==-91 && piv[3]==-94 && piv[1]==2); \
            for(int k=0;k<2;k++) for(int j=0;j<2;j++) { \
                T tmp=original[k][j]; original[k][j]=original[piv[k+1]-1][j]; original[piv[k+1]-1][j]=tmp; \
            } \
            for(int i=0;i<3;i++) for(int j=0;j<2;j++) { \
                T value=0; \
                for(int k=0;k<2;k++) { \
                    T l=i==k?1:(i>k?a[layout==101?i*5+k:i+k*5]:0); \
                    T u=k<=j?a[layout==101?k*5+j:k+j*5]:0; value+=l*u; \
                } \
                assert(cabs((double complex)(value-original[i][j]))<(TOL)); \
            } \
            for(int q=0;q<15;q++) { \
                int used=layout==101?(q/5<3 && q%5<2):(q/5<2 && q%5<3); \
                if(!used) assert(a[q]==(T)777); \
            } \
            T singular[4]={1,2,2,4}; assert(call(layout,2,2,singular,2,piv+1)==2); \
            assert(call(layout,-1,2,NULL,5,NULL)==-2); \
        } \
    } \
} while(0)


#define CHECK_SOLVE(P,T,INT,SUFFIX,TOL) do { \
    typedef INT (*solve)(int,char,INT,INT,const T *,INT,const INT *,T *,INT); \
    const char *names[]={"LAPACKE_" #P "getrs" SUFFIX,"LAPACKE_" #P "getrs_work" SUFFIX}; \
    for(int entry=0;entry<2;entry++) { \
        solve call=(solve)dlsym(library,names[entry]);assert(call); \
        for(int layout=101;layout<=102;layout++) for(int t=0;t<3;t++) { \
            char trans="NTC"[t]; \
            T a[6]={4,0,777,0,-0.5,777}; \
            a[1]=layout==101?3:0.5; a[3]=layout==101?0.5:3; \
            T saved[6];memcpy(saved,a,sizeof(a)); \
            T b[6]={777,777,777,777,777,777}; \
            b[0]=t==0?4:10; b[layout==101?3:1]=t==0?10:7; \
            INT piv[2]={2,2}; \
            assert(call(layout,trans,2,1,a,3,piv,b,3)==0); \
            assert(cabs((double complex)(b[0]-1))<(TOL)); \
            assert(cabs((double complex)(b[layout==101?3:1]-2))<(TOL)); \
            assert(memcmp(a,saved,sizeof(a))==0 && piv[0]==2 && piv[1]==2); \
            for(int q=0;q<6;q++) if(q!=0 && q!=(layout==101?3:1)) assert(b[q]==(T)777); \
            assert(call(layout,'?',0,0,NULL,3,NULL,NULL,3)==-2); \
            assert(call(layout,'N',0,0,NULL,3,NULL,NULL,3)==0); \
            assert(call(101,'?',2,2,NULL,1,NULL,NULL,1)==-6); \
            assert(call(101,'?',2,2,NULL,2,NULL,NULL,1)==-9); \
        } \
    } \
} while(0)

int main(int argc,char **argv) {
    assert(argc==2 || argc==3);
    void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if(!library) {fprintf(stderr,"%s\n",dlerror());return 3;}
    typedef int (*getflag)(void); typedef void (*setflag)(int);
    getflag get=(getflag)dlsym(library,"LAPACKE_get_nancheck");
    setflag set=(setflag)dlsym(library,"LAPACKE_set_nancheck"); assert(get && set);
    if(argc==3) { int expected=atoi(argv[2]); assert(get()==expected); assert(setenv("LAPACKE_NANCHECK", expected?"0":"1",1)==0); assert(get()==expected); puts("LAPACKE NaN environment: passed");return 0; }
    getflag get64=(getflag)dlsym(library,"LAPACKE_get_nancheck64_");
    setflag set64=(setflag)dlsym(library,"LAPACKE_set_nancheck64_");assert(get64 && set64);
    set64(0);assert(get()==0);set(1);assert(get64()==1);
    CHECK(s,float,int32_t,"",1.0f,1e-5);
    CHECK(d,double,int32_t,"",1.0,1e-12);
    CHECK(c,float complex,int32_t,"",0.8f-0.6f*I,1e-5);
    CHECK(z,double complex,int32_t,"",0.8-0.6*I,1e-12);
    CHECK(s,float,int64_t,"64_",1.0f,1e-5);
    CHECK(d,double,int64_t,"64_",1.0,1e-12);
    CHECK(c,float complex,int64_t,"64_",0.8f-0.6f*I,1e-5);
    CHECK(z,double complex,int64_t,"64_",0.8-0.6*I,1e-12);
    CHECK_SOLVE(s,float,int32_t,"",1e-5);
    CHECK_SOLVE(d,double,int32_t,"",1e-12);
    CHECK_SOLVE(c,float complex,int32_t,"",1e-5);
    CHECK_SOLVE(z,double complex,int32_t,"",1e-12);
    CHECK_SOLVE(s,float,int64_t,"64_",1e-5);
    CHECK_SOLVE(d,double,int64_t,"64_",1e-12);
    CHECK_SOLVE(c,float complex,int64_t,"64_",1e-5);
    CHECK_SOLVE(z,double complex,int64_t,"64_",1e-12);
    typedef int64_t (*zsolve)(int,char,int64_t,int64_t,const double complex *,int64_t,const int64_t *,double complex *,int64_t);
    zsolve zs=(zsolve)dlsym(library,"LAPACKE_zgetrs64_");assert(zs);
    double complex fa[4]={NAN,0.5,3,-0.5}, rhs[2]={4,10};int64_t pi[2]={2,2};
    assert(zs(102,'N',2,1,fa,2,pi,rhs,2)==-5);
    fa[0]=4;rhs[0]=NAN*I;assert(zs(102,'N',2,1,fa,2,pi,rhs,2)==-8);
    assert(zs(102,'N',INT64_C(-4294967293),0,NULL,2,NULL,NULL,2)==-3);
    typedef int64_t (*zfactor)(int,int64_t,int64_t,double complex *,int64_t,int64_t *);
    zfactor z=(zfactor)dlsym(library,"LAPACKE_zgetrf64_"); assert(z);
    double complex a[6]={2,NAN,777,1,4,777}; int64_t p[2]={-1,-1};
    assert(z(101,2,2,a,3,p)==-4); assert(p[0]==-1 && p[1]==-1);
    /* NaN check precedes numerical dimension/leading-dimension diagnostics. */
    assert(z(101,2,3,a,2,p)==-4);
    a[1]=1+NAN*I; assert(z(101,2,2,a,3,p)==-4);
    a[1]=1; a[2]=NAN; a[5]=NAN; assert(z(101,2,2,a,3,p)==0);
    set(0);assert(get()==0);
    /* Disabling the scan reaches dimension validation without reading NULL. */
    assert(z(102,INT64_C(-4294967293),2,NULL,2,NULL)==-2);
    set(-9);assert(get()==1);
    puts("LAPACKE C ABI: GETRF/GETRS high/work, four scalars, both integer widths passed");
    dlclose(library);return 0;
}
