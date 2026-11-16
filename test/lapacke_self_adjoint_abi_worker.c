/* Independent LAPACKE v3.12.1 SYEV/HEEV C prototypes. */
#include <assert.h>
#include <complex.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define REAL_DECL
#define REAL_ARG
#define C32_DECL , float *
#define C64_DECL , double *
#define COMPLEX_ARG , rw

static int at(int layout,int i,int j,int ld){return layout==101?i*ld+j:i+j*ld;}

#define CHECK(P,FAMILY,T,R,INT,SUFFIX,RDECL,RARG,TOL) do { \
    typedef INT (*high_fn)(int,char,char,INT,T*,INT,R*); \
    typedef INT (*work_fn)(int,char,char,INT,T*,INT,R*,T*,INT RDECL); \
    high_fn high=(high_fn)dlsym(library,"LAPACKE_" #P #FAMILY SUFFIX); \
    work_fn work=(work_fn)dlsym(library,"LAPACKE_" #P #FAMILY "_work" SUFFIX);assert(high && work); \
    R rw[64]={0};(void)rw;T query=0; \
    for(int layout=101;layout<=102;layout++)for(int flag=0;flag<4;flag++)for(int vectors=0;vectors<2;vectors++)for(int n=0;n<=2;n++){ \
        char uplo="LUlu"[flag],job=vectors?'v':'n';int lower=flag%2==0,ld=n+2; \
        T a[24],original[24],off=(T)(1+2*I);R values[4]={-99,-99,-99,-99}; \
        for(int q=0;q<24;q++){a[q]=(T)NAN;original[q]=(T)NAN;} \
        for(int i=0;i<n;i++)for(int j=0;j<n;j++)if(lower?i>=j:i<=j){int q=at(layout,i,j,ld);a[q]=original[q]=i==j?(T)(i==0?4:7):(i<j?off:(T)conj((double complex)off));} \
        assert(work(layout,job,uplo,n,a,ld,values+1,&query,-1 RARG)==0);assert(memcmp(a,original,sizeof(a))==0); \
        for(int q=0;q<4;q++){assert(values[q]==-99);} \
        int nw=(int)creal((double complex)query);assert(nw>=1 && nw<64);T workspace[64]={0}; \
        assert(work(layout,job,uplo,n,a,ld,values+1,workspace,nw-1 RARG)==-9);assert(memcmp(a,original,sizeof(a))==0); \
        for(int entry=0;entry<2;entry++){ \
            memcpy(a,original,sizeof(a));for(int q=0;q<4;q++){values[q]=-99;} \
            INT info=entry?high(layout,job,uplo,n,a,ld,values+1):work(layout,job,uplo,n,a,ld,values+1,workspace,nw RARG);assert(info==0); \
            assert(values[0]==-99 && values[n+1]==-99); \
            for(int k=0;k<n;k++){double expected=n==1?4:(11+(k==0?-1:1)*sqrt(9+4*pow(cabs((double complex)off),2)))/2;assert(fabs(values[k+1]-expected)<(TOL)*10);} \
            if(vectors){for(int i=0;i<n;i++)for(int k=0;k<n;k++){ \
                double complex product=0;for(int j=0;j<n;j++){double complex x=i==j?(i==0?4:7):(i<j?(double complex)off:conj((double complex)off));product+=x*(double complex)a[at(layout,j,k,ld)];} \
                assert(cabs(product-values[k+1]*(double complex)a[at(layout,i,k,ld)])<(TOL)*30); \
            }for(int i=0;i<n;i++)for(int j=0;j<n;j++){double complex dot=0;for(int k=0;k<n;k++){dot+=conj((double complex)a[at(layout,k,i,ld)])*(double complex)a[at(layout,k,j,ld)];}assert(cabs(dot-(i==j?1:0))<(TOL)*10);}} \
            for(int q=0;q<24;q++){int written=0;for(int i=0;i<n;i++)for(int j=0;j<n;j++){written|=q==at(layout,i,j,ld)&&(vectors||(lower?i>=j:i<=j));}if(!written){assert(isnan(creal((double complex)a[q])));}} \
        } \
    } \
    assert(work(0,'N','L',2,NULL,2,NULL,&query,-1 RARG)==-1); \
    assert(work(102,'?','L',2,NULL,2,NULL,&query,-1 RARG)==-2); \
    assert(work(102,'N','?',2,NULL,2,NULL,&query,-1 RARG)==-3); \
    assert(work(102,'N','L',-1,NULL,1,NULL,&query,-1 RARG)==-4); \
    assert(work(101,'?','?',2,NULL,1,NULL,&query,-1 RARG)==-6); \
    assert(work(102,'N','L',2,NULL,1,NULL,&query,-1 RARG)==-6); \
    T poison=(T)NAN;R values[2]={0};assert(high(102,'N','L',1,&poison,1,values)==-5); \
    assert(high(101,'N','L',0,NULL,0,NULL)==0); \
    if(sizeof(INT)==8){INT negative=(INT)INT64_C(-4294967293),stride=(INT)INT64_C(4294967296); \
        assert(work(102,'N','L',negative,NULL,1,NULL,&query,-1 RARG)==-4); \
        assert(work(102,'N','L',1,NULL,stride,NULL,&query,-1 RARG)==0); \
    } \
} while(0)

int main(int argc,char **argv){
    assert(argc==2);void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if(!library){fprintf(stderr,"%s\n",dlerror());return 3;}
    CHECK(s,syev,float,float,int32_t,"",REAL_DECL,REAL_ARG,1e-5);
    CHECK(d,syev,double,double,int32_t,"",REAL_DECL,REAL_ARG,1e-12);
    CHECK(c,heev,float complex,float,int32_t,"",C32_DECL,COMPLEX_ARG,1e-5);
    CHECK(z,heev,double complex,double,int32_t,"",C64_DECL,COMPLEX_ARG,1e-12);
    CHECK(s,syev,float,float,int64_t,"64_",REAL_DECL,REAL_ARG,1e-5);
    CHECK(d,syev,double,double,int64_t,"64_",REAL_DECL,REAL_ARG,1e-12);
    CHECK(c,heev,float complex,float,int64_t,"64_",C32_DECL,COMPLEX_ARG,1e-5);
    CHECK(z,heev,double complex,double,int64_t,"64_",C64_DECL,COMPLEX_ARG,1e-12);
    puts("LAPACKE self-adjoint C ABI: layouts, triangles, jobs and integer widths passed");
    dlclose(library);return 0;
}
