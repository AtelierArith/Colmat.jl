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
    const char *names[]={"LAPACKE_" #P "getrf" SUFFIX,"LAPACKE_" #P "getrf_work" SUFFIX,"LAPACKE_" #P "getf2" SUFFIX,"LAPACKE_" #P "getf2_work" SUFFIX,"LAPACKE_" #P "getrf2" SUFFIX,"LAPACKE_" #P "getrf2_work" SUFFIX}; \
    for (int entry=0;entry<6;entry++) { \
        factor call=(factor)dlsym(library,names[entry]); assert(call); \
        for (int layout=101;layout<=102;layout++) for(int shape=0;shape<2;shape++) { \
            int m=shape?2:3,n=shape?3:2; \
            T a[15],original[3][3]; INT piv[4]={-91,-92,-93,-94}; \
            for (int q=0;q<15;q++) a[q]=(T)777; \
            double values[3][3]={{2,1,7},{4,3,-2},{1,5,6}}; \
            for(int i=0;i<m;i++) for(int j=0;j<n;j++) { \
                original[i][j]=(T)(values[i][j]*(PHASE)); \
                a[layout==101?i*5+j:i+j*5]=original[i][j]; \
            } \
            assert(call(layout,m,n,a,5,piv+1)==0); \
            assert(piv[0]==-91 && piv[3]==-94 && piv[1]==2); \
            for(int k=0;k<2;k++) for(int j=0;j<n;j++) { \
                T tmp=original[k][j]; original[k][j]=original[piv[k+1]-1][j]; original[piv[k+1]-1][j]=tmp; \
            } \
            for(int i=0;i<m;i++) for(int j=0;j<n;j++) { \
                T value=0; \
                for(int k=0;k<2;k++) { \
                    T l=i==k?1:(i>k?a[layout==101?i*5+k:i+k*5]:0); \
                    T u=k<=j?a[layout==101?k*5+j:k+j*5]:0; value+=l*u; \
                } \
                assert(cabs((double complex)(value-original[i][j]))<(TOL)); \
            } \
            for(int q=0;q<15;q++) { \
                int used=layout==101?(q/5<m && q%5<n):(q/5<n && q%5<m); \
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


#define CHECK_DRIVER(P,T,INT,SUFFIX,PHASE,TOL) do { \
    typedef INT (*driver)(int,INT,INT,T *,INT,INT *,T *,INT); \
    const char *names[]={"LAPACKE_" #P "gesv" SUFFIX,"LAPACKE_" #P "gesv_work" SUFFIX}; \
    for(int entry=0;entry<2;entry++) { \
        driver call=(driver)dlsym(library,names[entry]);assert(call); \
        for(int layout=101;layout<=102;layout++) { \
            T a[6]={2,0,777,0,3,777}, b[6]={777,777,777,777,777,777}; \
            a[1]=layout==101?1:4;a[3]=layout==101?4:1; \
            for(int q=0;q<6;q++) if(q!=2 && q!=5) a[q]*=(PHASE); \
            b[0]=4*(PHASE);b[layout==101?3:1]=10*(PHASE); \
            INT piv[4]={-91,-92,-93,-94}; \
            assert(call(layout,2,1,a,3,piv+1,b,3)==0); \
            assert(piv[0]==-91 && piv[3]==-94 && piv[1]==2 && piv[2]==2); \
            assert(cabs((double complex)(b[0]-1))<(TOL)); \
            assert(cabs((double complex)(b[layout==101?3:1]-2))<(TOL)); \
            assert(a[2]==(T)777 && a[5]==(T)777); \
            for(int q=0;q<6;q++) if(q!=0 && q!=(layout==101?3:1)) assert(b[q]==(T)777); \
            T z[4]={2,layout==101?1:4,layout==101?4:1,3}; \
            assert(call(layout,2,0,z,2,piv+1,NULL,layout==101?0:2)==0); \
            assert(piv[1]==2 && piv[2]==2); \
            assert(cabs((double complex)(z[0]-4))<(TOL)); \
            assert(cabs((double complex)(z[3]+0.5))<(TOL)); \
            T singular[4]={1,2,2,4}, rhs[2]={9,7}; \
            assert(call(layout,2,1,singular,2,piv+1,rhs,layout==101?1:2)==2); \
            assert(rhs[0]==9 && rhs[1]==7); \
            assert(call(layout,-1,0,NULL,3,NULL,NULL,3)==-2); \
            assert(call(layout,0,0,NULL,3,NULL,NULL,3)==0); \
            assert(call(101,-1,-1,NULL,-2,NULL,NULL,-2)==-5); \
        } \
    } \
} while(0)


#define CHECK_INVERSE(P,T,INT,SUFFIX,PHASE,TOL) do { \
    typedef INT (*inverse)(int,INT,T *,INT,const INT *); \
    typedef INT (*inverse_work)(int,INT,T *,INT,const INT *,T *,INT); \
    inverse high=(inverse)dlsym(library,"LAPACKE_" #P "getri" SUFFIX);assert(high); \
    inverse_work work=(inverse_work)dlsym(library,"LAPACKE_" #P "getri_work" SUFFIX);assert(work); \
    for(int layout=101;layout<=102;layout++) { \
        T w[4]={888,888,888,888}; \
        assert(work(layout,2,NULL,3,NULL,w+1,-1)==0); \
        assert(creal((double complex)w[1])>=2 && cimag((double complex)w[1])==0); \
        assert(w[0]==(T)888 && w[2]==(T)888 && w[3]==(T)888); \
        for(int entry=0;entry<2;entry++) { \
            T a[6]={4*(PHASE),0,777,0,-0.5*(PHASE),777}; \
            a[1]=layout==101?3*(PHASE):0.5;a[3]=layout==101?0.5:3*(PHASE); \
            INT piv[2]={2,2};T saved[6];memcpy(saved,a,sizeof(a)); \
            if(entry==1) { \
                assert(work(layout,2,a,3,piv,w+1,1)==-7);assert(memcmp(a,saved,sizeof(a))==0); \
                assert(work(layout,2,a,3,piv,w+1,2)==0); \
                assert(w[0]==(T)888 && w[3]==(T)888); \
            } else assert(high(layout,2,a,3,piv)==0); \
            double values[2][2]={{1.5,-0.5},{-2,1}}; \
            for(int i=0;i<2;i++) for(int j=0;j<2;j++) \
                assert(cabs((double complex)(a[layout==101?i*3+j:i+j*3]-values[i][j]/(PHASE)))<(TOL)); \
            assert(a[2]==(T)777 && a[5]==(T)777 && piv[0]==2 && piv[1]==2); \
            memcpy(a,saved,sizeof(a));a[4]=0;memcpy(saved,a,sizeof(a)); \
            assert((entry==1?work(layout,2,a,3,piv,w+1,2):high(layout,2,a,3,piv))==2); \
            assert(memcmp(a,saved,sizeof(a))==0); \
        } \
        assert(work(layout,0,NULL,layout==101?0:1,NULL,w+1,1)==0);assert(w[1]==(T)1); \
        assert(high(layout,0,NULL,layout==101?0:1,NULL)==0); \
        assert(work(layout,-1,NULL,3,NULL,NULL,-1)==-2); \
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
    CHECK_DRIVER(s,float,int32_t,"",1.0f,1e-5);
    CHECK_DRIVER(d,double,int32_t,"",1.0,1e-12);
    CHECK_DRIVER(c,float complex,int32_t,"",0.8f-0.6f*I,1e-5);
    CHECK_DRIVER(z,double complex,int32_t,"",0.8-0.6*I,1e-12);
    CHECK_DRIVER(s,float,int64_t,"64_",1.0f,1e-5);
    CHECK_DRIVER(d,double,int64_t,"64_",1.0,1e-12);
    CHECK_DRIVER(c,float complex,int64_t,"64_",0.8f-0.6f*I,1e-5);
    CHECK_DRIVER(z,double complex,int64_t,"64_",0.8-0.6*I,1e-12);
    CHECK_INVERSE(s,float,int32_t,"",1.0f,1e-5);
    CHECK_INVERSE(d,double,int32_t,"",1.0,1e-12);
    CHECK_INVERSE(c,float complex,int32_t,"",0.8f-0.6f*I,1e-5);
    CHECK_INVERSE(z,double complex,int32_t,"",0.8-0.6*I,1e-12);
    CHECK_INVERSE(s,float,int64_t,"64_",1.0f,1e-5);
    CHECK_INVERSE(d,double,int64_t,"64_",1.0,1e-12);
    CHECK_INVERSE(c,float complex,int64_t,"64_",0.8f-0.6f*I,1e-5);
    CHECK_INVERSE(z,double complex,int64_t,"64_",0.8-0.6*I,1e-12);
    typedef int64_t (*dinverse_work)(int,int64_t,double *,int64_t,const int64_t *,double *,int64_t);
    dinverse_work diw=(dinverse_work)dlsym(library,"LAPACKE_dgetri_work64_");assert(diw);
    double large_query=0;int64_t huge_n=INT64_C(4294967297);
    assert(diw(101,huge_n,NULL,huge_n,NULL,&large_query,-1)==0);assert(large_query==(double)huge_n);
    typedef int64_t (*sinverse_work)(int,int64_t,float *,int64_t,const int64_t *,float *,int64_t);
    sinverse_work siw=(sinverse_work)dlsym(library,"LAPACKE_sgetri_work64_");assert(siw);
    float rounded_query=0;int64_t rounded_n=INT64_C(16777217);
    assert(siw(102,rounded_n,NULL,rounded_n,NULL,&rounded_query,-1)==0);assert((double)rounded_query>=(double)rounded_n);
    typedef int64_t (*dinverse)(int,int64_t,double *,int64_t,const int64_t *);
    dinverse di=(dinverse)dlsym(library,"LAPACKE_dgetri64_");assert(di);
    double nan_factor[4]={4,0.5,3,NAN};int64_t nan_piv[2]={2,2};
    assert(di(102,2,nan_factor,2,nan_piv)==-3);assert(isnan(nan_factor[3]));
    typedef int64_t (*ddriver)(int,int64_t,int64_t,double *,int64_t,int64_t *,double *,int64_t);
    ddriver dd=(ddriver)dlsym(library,"LAPACKE_dgesv64_");assert(dd);
    double da[4]={NAN,4,1,3},db[2]={4,10};int64_t dp[2]={-91,-92};
    assert(dd(102,2,1,da,2,dp,db,2)==-4);assert(dp[0]==-91 && dp[1]==-92);
    da[0]=2;db[0]=NAN;assert(dd(102,2,1,da,2,dp,db,2)==-7);assert(dp[0]==-91 && dp[1]==-92);
    assert(dd(102,INT64_C(-4294967293),0,NULL,2,NULL,NULL,2)==-2);
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
    puts("LAPACKE C ABI: GETRF/GETF2/GETRF2/GETRS/GESV/GETRI high/work, four scalars, both integer widths passed");
    dlclose(library);return 0;
}
