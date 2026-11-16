/* Independent LAPACKE v3.12.1 GEEV prototypes; no generated Rust declarations. */
#include <assert.h>
#include <complex.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define WI_DECL(R) , R *
#define WI_ARG , wi+1
#define RW_DECL(R)
#define RW_ARG
#define NO_WI_DECL(R)
#define NO_WI_ARG
#define COMPLEX_RW_DECL(R) , R *
#define COMPLEX_RW_ARG , rw
static int at(int layout,int i,int j,int ld){return layout==101?i*ld+j:i+j*ld;}
#define CHECK(P,T,R,J,SUFFIX,WDECL,WARG,RDECL,RARG,TOL) do { \
    typedef J (*high_fn)(int,char,char,J,T*,J,T* WDECL(R),T*,J,T*,J); \
    typedef J (*work_fn)(int,char,char,J,T*,J,T* WDECL(R),T*,J,T*,J,T*,J RDECL(R)); \
    high_fn high=(high_fn)dlsym(library,"LAPACKE_" #P "geev" SUFFIX); \
    work_fn work=(work_fn)dlsym(library,"LAPACKE_" #P "geev_work" SUFFIX);assert(high&&work); \
    R rw[8]={0},wi[5];(void)rw;(void)wi; \
    for(int layout=101;layout<=102;layout++)for(int n=0;n<=3;n++)for(int jl=0;jl<2;jl++)for(int jr=0;jr<2;jr++){ \
        int ld=n+2;char left=jl?'v':'n',right=jr?'v':'n';T a[25],original[25],vl[25],vr[25],w[5],query=0,workspace[16]; \
        for(int q=0;q<25;q++){a[q]=original[q]=(T)777;vl[q]=vr[q]=(T)777;} \
        for(int i=0;i<n;i++)for(int j=0;j<n;j++){a[at(layout,i,j,ld)]=original[at(layout,i,j,ld)]=i==j?(T)(7+i+(sizeof(T)>sizeof(R)?I:0)):0;} \
        for(int q=0;q<5;q++){w[q]=(T)777;wi[q]=777;} \
        assert(work(layout,left,right,n,a,ld,w+1 WARG,vl,ld,vr,ld,&query,-1 RARG)==0); \
        assert(memcmp(a,original,sizeof(a))==0);for(int q=0;q<25;q++){assert(vl[q]==(T)777&&vr[q]==(T)777);} \
        for(int q=0;q<5;q++){assert(w[q]==(T)777&&wi[q]==777);} \
        int minimum=n==0?1:(sizeof(T)>sizeof(R)?2*n:((jl||jr)?4*n:3*n)); \
        assert(creal((double complex)query)>=minimum); \
        assert(work(layout,left,right,n,a,ld,w+1 WARG,vl,ld,vr,ld,workspace,minimum-1 RARG)==(sizeof(T)>sizeof(R)?-13:-14)); \
        assert(memcmp(a,original,sizeof(a))==0); \
        for(int entry=0;entry<2;entry++){ \
            memcpy(a,original,sizeof(a));for(int q=0;q<25;q++){vl[q]=vr[q]=(T)777;} \
            J info=entry?high(layout,left,right,n,a,ld,w+1 WARG,vl,ld,vr,ld):work(layout,left,right,n,a,ld,w+1 WARG,vl,ld,vr,ld,workspace,minimum RARG);assert(info==0); \
            assert(w[0]==(T)777&&w[n+1]==(T)777&&wi[0]==777&&wi[n+1]==777); \
            for(int k=0;k<n;k++){double complex lambda=(double complex)w[k+1];assert(creal(lambda)>=7-(TOL)&&creal(lambda)<=9+(TOL)); \
                assert(fabs(cimag(lambda)-(sizeof(T)>sizeof(R)?1:0))<(TOL)); \
                for(int side=0;side<2;side++){if(!(side?jr:jl)){continue;}T *v=side?vr:vl;double norm=0; \
                    for(int i=0;i<n;i++){double complex eigen=side?lambda:conj(lambda),diag=(double complex)original[at(layout,i,i,ld)];if(!side){diag=conj(diag);} \
                        assert(cabs((diag-eigen)*(double complex)v[at(layout,i,k,ld)])<(TOL)*20);norm+=pow(cabs((double complex)v[at(layout,i,k,ld)]),2);}assert(fabs(norm-1)<(TOL)*10); \
                } \
            } \
            for(int q=0;q<25;q++){int logical=0;for(int i=0;i<n;i++)for(int j=0;j<n;j++){logical|=q==at(layout,i,j,ld);} \
                if(!logical){assert(a[q]==original[q]);}if(!logical||!jl){assert(vl[q]==(T)777);}if(!logical||!jr){assert(vr[q]==(T)777);} \
            } \
        } \
    } \
    T query=0;assert(work(0,'N','N',0,NULL,1,NULL WARG,NULL,1,NULL,1,&query,-1 RARG)==-1); \
    assert(work(102,'?','N',0,NULL,1,NULL WARG,NULL,1,NULL,1,&query,-1 RARG)==-2); \
    assert(work(102,'N','?',0,NULL,1,NULL WARG,NULL,1,NULL,1,&query,-1 RARG)==-3); \
    assert(work(102,'N','N',-1,NULL,1,NULL WARG,NULL,1,NULL,1,&query,-1 RARG)==-4); \
    assert(work(101,'?','?',3,NULL,2,NULL WARG,NULL,3,NULL,3,&query,-1 RARG)==-6); \
    assert(work(102,'N','N',3,NULL,2,NULL WARG,NULL,1,NULL,1,&query,-1 RARG)==-6); \
    assert(work(102,'V','N',3,NULL,3,NULL WARG,NULL,2,NULL,1,&query,-1 RARG)==(sizeof(T)>sizeof(R)?-9:-10)); \
    assert(work(102,'N','V',3,NULL,3,NULL WARG,NULL,1,NULL,2,&query,-1 RARG)==(sizeof(T)>sizeof(R)?-11:-12)); \
    assert(work(101,'N','N',0,NULL,0,NULL WARG,NULL,1,NULL,1,&query,-1 RARG)==0); \
    T poison=(T)NAN,w[2]={0};assert(high(102,'N','N',1,&poison,1,w WARG,NULL,1,NULL,1)==-5); \
    if(sizeof(J)==8){assert(work(102,'N','N',(J)INT64_C(-4294967293),NULL,1,NULL WARG,NULL,1,NULL,1,&query,-1 RARG)==-4);} \
} while(0)
int main(int argc,char **argv){
    assert(argc==2);void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);assert(library);
    CHECK(s,float,float,int32_t,"",WI_DECL,WI_ARG,RW_DECL,RW_ARG,2e-5);
    CHECK(d,double,double,int32_t,"",WI_DECL,WI_ARG,RW_DECL,RW_ARG,1e-12);
    CHECK(c,float complex,float,int32_t,"",NO_WI_DECL,NO_WI_ARG,COMPLEX_RW_DECL,COMPLEX_RW_ARG,2e-5);
    CHECK(z,double complex,double,int32_t,"",NO_WI_DECL,NO_WI_ARG,COMPLEX_RW_DECL,COMPLEX_RW_ARG,1e-12);
    CHECK(s,float,float,int64_t,"64_",WI_DECL,WI_ARG,RW_DECL,RW_ARG,2e-5);
    CHECK(d,double,double,int64_t,"64_",WI_DECL,WI_ARG,RW_DECL,RW_ARG,1e-12);
    CHECK(c,float complex,float,int64_t,"64_",NO_WI_DECL,NO_WI_ARG,COMPLEX_RW_DECL,COMPLEX_RW_ARG,2e-5);
    CHECK(z,double complex,double,int64_t,"64_",NO_WI_DECL,NO_WI_ARG,COMPLEX_RW_DECL,COMPLEX_RW_ARG,1e-12);
    puts("LAPACKE general eigen C ABI: 8 names verified");dlclose(library);return 0;
}
