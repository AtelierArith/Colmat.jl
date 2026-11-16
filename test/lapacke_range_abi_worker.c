/* Independent C11 LAPACKE v3.12.1 SYEVR/HEEVR prototypes. */
#include <assert.h>
#include <complex.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define REAL_DECL(INT)
#define REAL_ARG
#define C32_DECL(INT) ,float *,INT
#define C64_DECL(INT) ,double *,INT
#define COMPLEX_ARG ,rw,lr
static int at(int layout,int i,int j,int ld){return layout==101?i*ld+j:i+j*ld;}

#define CHECK(P,FAMILY,T,R,INT,SUFFIX,RDECL,RARG,TOL) do { \
    typedef INT (*high_fn)(int,char,char,char,INT,T*,INT,R,R,INT,INT,R,INT*,R*,T*,INT,INT*); \
    typedef INT (*work_fn)(int,char,char,char,INT,T*,INT,R,R,INT,INT,R,INT*,R*,T*,INT,INT*,T*,INT RDECL(INT),INT*,INT); \
    high_fn high=(high_fn)dlsym(library,"LAPACKE_" #P #FAMILY SUFFIX); \
    work_fn work=(work_fn)dlsym(library,"LAPACKE_" #P #FAMILY "_work" SUFFIX);assert(high && work); \
    R rw[128]={0};INT iw[66],lr=128,li=64;T workspace[128]={0},query=0;(void)rw;(void)lr; \
    for(int layout=101;layout<=102;layout++)for(int ncase=0;ncase<3;ncase++) \
    for(int flag=0;flag<4;flag++)for(int vectors=0;vectors<2;vectors++)for(int selected=0;selected<3;selected++)for(int entry=0;entry<2;entry++){ \
        int n=ncase==2?3:ncase,ld=n+2,lz=n+2;char uplo="LUlu"[flag],job=vectors?'v':'n',mode="AIV"[selected];int lower=flag%2==0; \
        R vl=sizeof(T)>sizeof(R)?(R)2:(R)3;INT il=n>1?2:1,iu=n,m=-99;T off=(T)(1+I); \
        T a[32],original[32],z[32];R w[5];INT support[10]; \
        for(int q=0;q<32;q++){a[q]=original[q]=(T)NAN;z[q]=(T)777;} \
        for(int q=0;q<5;q++){w[q]=-99;}for(int q=0;q<10;q++){support[q]=-99;}for(int q=0;q<66;q++){iw[q]=-99;} \
        for(int i=0;i<n;i++)for(int j=0;j<n;j++)if(lower?i>=j:i<=j){int q=at(layout,i,j,ld);a[q]=original[q]=i==j?(T)2:(abs(i-j)==1?(i<j?off:(T)conj((double complex)off)):(T)0);} \
        if(n==1){vl=2;} \
        T *zp=vectors?z:NULL; \
        assert(work(layout,job,mode,uplo,n,NULL,ld,vl,10,il,iu,0,&m,NULL,NULL,lz,support,&query,-1 RARG,iw+1,li)==0); \
        assert(m==-99 && creal((double complex)query)>=1); \
        assert(iw[0]==-99 && iw[1]==(n<=1?1:10*n) && iw[2]==-99 && iw[65]==-99); \
        int nw=(int)creal((double complex)query);assert(nw>=1 && nw<=128); \
        assert(work(layout,job,mode,uplo,n,NULL,ld,vl,10,il,iu,0,&m,NULL,NULL,lz,support,workspace,0 RARG,iw+1,li)==-19); \
        INT info=entry?high(layout,job,mode,uplo,n,a,ld,vl,10,il,iu,0,&m,w,zp,lz,support): \
            work(layout,job,mode,uplo,n,a,ld,vl,10,il,iu,0,&m,w,zp,lz,support,workspace,nw RARG,iw+1,li);assert(info==0); \
        int expected=selected==0?n:selected==1?(n>1?n-1:n):(n==3?1:0);assert(m==expected); \
        for(int q=m;q<5;q++){assert(w[q]==-99);}assert(support[8]==-99 && support[9]==-99); \
        if(vectors){ \
            for(int i=0;i<n;i++)for(int k=0;k<m;k++){double complex product=0;for(int j=0;j<n;j++){double complex x=i==j?2:(abs(i-j)==1?(i<j?(double complex)off:conj((double complex)off)):0);product+=x*(double complex)z[at(layout,j,k,lz)];}assert(cabs(product-w[k]*(double complex)z[at(layout,i,k,lz)])<(TOL)*40);} \
            for(int i=0;i<m;i++)for(int j=0;j<m;j++){double complex dot=0;for(int k=0;k<n;k++){dot+=conj((double complex)z[at(layout,k,i,lz)])*(double complex)z[at(layout,k,j,lz)];}assert(cabs(dot-(i==j?1:0))<(TOL)*10);} \
            if(n==1){assert(z[0]==(T)1 && support[0]==1 && support[1]==1);} \
        } \
        for(int q=0;q<32;q++){int used_a=0,used_z=0;for(int i=0;i<n;i++)for(int j=0;j<n;j++){used_a|=q==at(layout,i,j,ld)&&(lower?i>=j:i<=j);used_z|=vectors && q==at(layout,i,j,lz) && j<(n==1?1:m);}if(!used_a){assert(isnan(creal((double complex)a[q])));}if(!used_z){assert(z[q]==(T)777);}} \
    } \
    INT m=-99,support[8]={0};R values[3]={0}; \
    assert(work(0,'N','A','L',3,NULL,3,0,10,1,3,0,&m,NULL,NULL,1,support,&query,-1 RARG,iw,li)==-1); \
    assert(work(101,'?','?','?',3,NULL,2,0,10,1,3,0,&m,NULL,NULL,0,support,&query,-1 RARG,iw,li)==-7); \
    assert(work(101,'V','A','L',3,NULL,3,0,10,1,3,0,&m,NULL,NULL,2,support,&query,-1 RARG,iw,li)==-16); \
    assert(work(102,'?','A','L',3,NULL,3,0,10,1,3,0,&m,NULL,NULL,3,support,&query,-1 RARG,iw,li)==-2); \
    assert(work(102,'N','?','L',3,NULL,3,0,10,1,3,0,&m,NULL,NULL,1,support,&query,-1 RARG,iw,li)==-3); \
    assert(work(102,'N','A','?',3,NULL,3,0,10,1,3,0,&m,NULL,NULL,1,support,&query,-1 RARG,iw,li)==-4); \
    assert(work(102,'N','A','L',-1,NULL,1,0,10,1,0,0,&m,NULL,NULL,1,support,&query,-1 RARG,iw,li)==-5); \
    assert(work(102,'N','V','L',3,NULL,3,10,0,1,3,0,&m,NULL,NULL,1,support,&query,-1 RARG,iw,li)==-9); \
    assert(work(102,'N','I','L',3,NULL,3,0,10,0,3,0,&m,NULL,NULL,1,support,&query,-1 RARG,iw,li)==-10); \
    assert(work(102,'N','I','L',3,NULL,3,0,10,1,4,0,&m,NULL,NULL,1,support,&query,-1 RARG,iw,li)==-11); \
    for(int which=0;which<(sizeof(T)>sizeof(R)?3:2);which++){INT lw=which==0?-1:0;li=which==1?-1:0;lr=which==2?-1:0;m=-99; \
        assert(work(102,'V','A','L',3,NULL,3,0,10,1,3,0,&m,NULL,NULL,3,support,&query,lw RARG,iw,li)==0);assert(m==-99 && iw[0]==30); \
    }lr=128;li=64; \
    T a=(T)2,z=(T)777; \
    assert(high(102,'N','A','L',1,&a,1,NAN,NAN,1,1,0,&m,values,NULL,1,support)==0); \
    assert(high(102,'N','A','L',1,&a,1,0,10,1,1,NAN,&m,values,NULL,1,support)==-12); \
    assert(high(102,'N','V','L',1,&a,1,NAN,10,1,1,0,&m,values,NULL,1,support)==-8); \
    assert(high(102,'N','V','L',1,&a,1,0,NAN,1,1,0,&m,values,NULL,1,support)==-9); \
    a=(T)NAN;assert(high(102,'N','A','L',1,&a,1,0,10,1,1,NAN,&m,values,NULL,1,support)==-6); \
    assert(high(101,'V','V','L',3,(T[9]){2,0,0,0,3,0,0,0,4},3,5,10,1,3,0,&m,values,NULL,3,support)==0 && m==0); \
    assert(high(101,'N','A','L',0,NULL,0,0,10,1,0,0,&m,NULL,NULL,1,support)==0 && m==0); \
    if(sizeof(INT)==8){INT negative=(INT)INT64_C(-4294967293),large=(INT)INT64_C(4294967296); \
        assert(work(102,'N','A','L',negative,NULL,1,0,10,1,0,0,&m,NULL,NULL,1,support,&query,-1 RARG,iw,li)==-5); \
        assert(work(102,'V','A','L',1,NULL,large,0,10,1,1,0,&m,NULL,NULL,large,support,&query,-1 RARG,iw,li)==0); \
    }(void)z; \
}while(0)

int main(int argc,char **argv){
    assert(argc==3);void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!library){fprintf(stderr,"%s\n",dlerror());return 3;}
    void *reference=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);if(!reference){fprintf(stderr,"%s\n",dlerror());return 4;}
    typedef void (*register_fn)(int,void*);register_fn reg=(register_fn)dlsym(library,"colmat_selfadjoint_vr_set_fallback");assert(reg);
    const char *names[]={"ssyevr_64_","dsyevr_64_","cheevr_64_","zheevr_64_"};
    for(int i=0;i<4;i++){void *p=dlsym(reference,names[i]);assert(p && p!=dlsym(library,names[i]));reg(i,p);}
    typedef void (*policy_fn)(int);policy_fn policy=(policy_fn)dlsym(library,"LAPACKE_set_nancheck");assert(policy);policy(1);
    CHECK(s,syevr,float,float,int32_t,"",REAL_DECL,REAL_ARG,5e-5);
    CHECK(d,syevr,double,double,int32_t,"",REAL_DECL,REAL_ARG,2e-12);
    CHECK(c,heevr,float complex,float,int32_t,"",C32_DECL,COMPLEX_ARG,5e-5);
    CHECK(z,heevr,double complex,double,int32_t,"",C64_DECL,COMPLEX_ARG,2e-12);
    CHECK(s,syevr,float,float,int64_t,"64_",REAL_DECL,REAL_ARG,5e-5);
    CHECK(d,syevr,double,double,int64_t,"64_",REAL_DECL,REAL_ARG,2e-12);
    CHECK(c,heevr,float complex,float,int64_t,"64_",C32_DECL,COMPLEX_ARG,5e-5);
    CHECK(z,heevr,double complex,double,int64_t,"64_",C64_DECL,COMPLEX_ARG,2e-12);
    puts("LAPACKE range C ABI: real endpoints, integer outputs, layouts and workspace queries passed");
    dlclose(library);dlclose(reference);return 0;
}
