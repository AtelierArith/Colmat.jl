/* Independent LAPACKE v3.12.1 GEEVX C declarations. */
#include <assert.h>
#include <complex.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define WI_DECL(R) ,R *
#define WI_ARG ,wi+1
#define NO_WI_DECL(R)
#define NO_WI_ARG
static int at(int layout,int i,int j,int ld){return layout==101?i*ld+j:i+j*ld;}
#define CHECK(P,T,R,J,SUFFIX,WDECL,WARG,AUX,TOL) do { \
    typedef J (*high_fn)(int,char,char,char,char,J,T*,J,T* WDECL(R),T*,J,T*,J,J*,J*,R*,R*,R*,R*); \
    typedef J (*work_fn)(int,char,char,char,char,J,T*,J,T* WDECL(R),T*,J,T*,J,J*,J*,R*,R*,R*,R*,T*,J,AUX*); \
    high_fn high=(high_fn)dlsym(library,"LAPACKE_" #P "geevx" SUFFIX);work_fn work=(work_fn)dlsym(library,"LAPACKE_" #P "geevx_work" SUFFIX);assert(high&&work); \
    for(int layout=101;layout<=102;layout++)for(int n=0;n<=2;n++)for(int balance=0;balance<4;balance++)for(int sense=0;sense<4;sense++){ \
        char bal="npsb"[balance],se="nevb"[sense];int ld=n+2;T a[16],saved[16],vl[16],vr[16],w[4],query=0,workspace[64];R wi[4],scale[4],abnrm[3]={777,777,777},ce[4],cv[4];J ilo[3]={-777,-777,-777},ihi[3]={-777,-777,-777};AUX aux[10]; \
        for(int k=0;k<16;k++){a[k]=saved[k]=(T)777;vl[k]=vr[k]=(T)777;}for(int i=0;i<n;i++)for(int j=0;j<n;j++){a[at(layout,i,j,ld)]=saved[at(layout,i,j,ld)]=i==j?(T)(7+i+(sizeof(T)>sizeof(R)?I:0)):0;} \
        for(int k=0;k<4;k++){w[k]=(T)777;wi[k]=scale[k]=ce[k]=cv[k]=777;}for(int k=0;k<10;k++){aux[k]=(AUX)-777;} \
        if(n==2&&balance==0&&sense==0){ \
            assert(work(0,bal,'v','v',se,n,NULL,ld,w+1 WARG,NULL,ld,NULL,ld,ilo+1,ihi+1,scale+1,abnrm+1,ce+1,cv+1,&query,-1,aux+1)==-1); \
            assert(work(102,'?','v','v',se,n,NULL,ld,w+1 WARG,NULL,ld,NULL,ld,ilo+1,ihi+1,scale+1,abnrm+1,ce+1,cv+1,&query,-1,aux+1)==-2); \
            assert(work(102,bal,'?','v',se,n,NULL,ld,w+1 WARG,NULL,ld,NULL,ld,ilo+1,ihi+1,scale+1,abnrm+1,ce+1,cv+1,&query,-1,aux+1)==-3); \
            assert(work(102,bal,'v','?',se,n,NULL,ld,w+1 WARG,NULL,ld,NULL,ld,ilo+1,ihi+1,scale+1,abnrm+1,ce+1,cv+1,&query,-1,aux+1)==-4); \
            assert(work(102,bal,'n','n','e',n,NULL,ld,w+1 WARG,NULL,ld,NULL,ld,ilo+1,ihi+1,scale+1,abnrm+1,ce+1,cv+1,&query,-1,aux+1)==-5); \
            assert(work(102,bal,'v','v',se,-1,NULL,ld,w+1 WARG,NULL,ld,NULL,ld,ilo+1,ihi+1,scale+1,abnrm+1,ce+1,cv+1,&query,-1,aux+1)==-6); \
            assert(work(101,'?','?','?','?',n,NULL,1,w+1 WARG,NULL,ld,NULL,ld,ilo+1,ihi+1,scale+1,abnrm+1,ce+1,cv+1,&query,-1,aux+1)==-8); \
            assert(work(102,bal,'v','v',se,n,NULL,1,w+1 WARG,NULL,ld,NULL,ld,ilo+1,ihi+1,scale+1,abnrm+1,ce+1,cv+1,&query,-1,aux+1)==-8); \
            assert(work(102,bal,'v','v',se,n,NULL,ld,w+1 WARG,NULL,1,NULL,ld,ilo+1,ihi+1,scale+1,abnrm+1,ce+1,cv+1,&query,-1,aux+1)==(sizeof(T)>sizeof(R)?-11:-12)); \
            assert(work(102,bal,'v','v',se,n,NULL,ld,w+1 WARG,NULL,ld,NULL,1,ilo+1,ihi+1,scale+1,abnrm+1,ce+1,cv+1,&query,-1,aux+1)==(sizeof(T)>sizeof(R)?-13:-14)); \
            if(sizeof(J)==8){assert(work(102,bal,'v','v',se,(J)INT64_C(-4294967293),NULL,ld,w+1 WARG,NULL,ld,NULL,ld,ilo+1,ihi+1,scale+1,abnrm+1,ce+1,cv+1,&query,-1,aux+1)==-6);} \
            T poison=(T)NAN;assert(high(102,bal,'n','n',se,1,&poison,1,w+1 WARG,NULL,1,NULL,1,ilo+1,ihi+1,scale+1,abnrm+1,ce+1,cv+1)==-7); \
        } \
        assert(work(layout,bal,'v','v',se,n,a,ld,w+1 WARG,vl,ld,vr,ld,ilo+1,ihi+1,scale+1,abnrm+1,ce+1,cv+1,&query,-1,aux+1)==0); \
        assert(memcmp(a,saved,sizeof(a))==0&&ilo[1]==-777&&ihi[1]==-777&&abnrm[1]==777); \
        int minimum=n==0?1:(sense>=2?n*n+(sizeof(T)>sizeof(R)?2*n:6*n):(sizeof(T)>sizeof(R)?2*n:3*n));assert(creal((double complex)query)>=minimum); \
        assert(work(layout,bal,'v','v',se,n,a,ld,w+1 WARG,vl,ld,vr,ld,ilo+1,ihi+1,scale+1,abnrm+1,ce+1,cv+1,workspace,minimum-1,aux+1)==(sizeof(T)>sizeof(R)?-21:-22)); \
        for(int entry=0;entry<2;entry++){ \
            memcpy(a,saved,sizeof(a));J info=entry?high(layout,bal,'v','v',se,n,a,ld,w+1 WARG,vl,ld,vr,ld,ilo+1,ihi+1,scale+1,abnrm+1,ce+1,cv+1):work(layout,bal,'v','v',se,n,a,ld,w+1 WARG,vl,ld,vr,ld,ilo+1,ihi+1,scale+1,abnrm+1,ce+1,cv+1,workspace,minimum,aux+1);assert(info==0); \
            if(n==0){assert(ilo[1]==-777&&ihi[1]==-777&&abnrm[1]==777);}else {assert(ilo[1]>=1&&ilo[1]<=n&&ihi[1]>=ilo[1]&&ihi[1]<=n&&isfinite(abnrm[1]));} \
            assert(ilo[0]==-777&&ilo[2]==-777&&ihi[0]==-777&&ihi[2]==-777&&abnrm[0]==777&&abnrm[2]==777); \
            assert(w[0]==(T)777&&w[n+1]==(T)777&&wi[0]==777&&wi[n+1]==777&&scale[0]==777&&scale[n+1]==777&&ce[0]==777&&ce[n+1]==777&&cv[0]==777&&cv[n+1]==777); \
            assert(aux[0]==(AUX)-777&&aux[9]==(AUX)-777); \
            for(int j=0;j<n;j++){double complex lambda=(double complex)w[j+1];assert(creal(lambda)>=7-(TOL)&&creal(lambda)<=8+(TOL));assert(fabs(cimag(lambda)-(sizeof(T)>sizeof(R)?1:0))<(TOL)); \
                if(sense==1||sense==3){assert(fabs(ce[j+1]-1)<(TOL)*10);}else {assert(ce[j+1]==777);}if(sense>=2){assert(isfinite(cv[j+1])&&cv[j+1]>0);}else {assert(cv[j+1]==777);} \
                for(int side=0;side<2;side++){T *v=side?vr:vl;double norm=0;for(int i=0;i<n;i++){double complex diagonal=(double complex)saved[at(layout,i,i,ld)],eigen=lambda;if(!side){diagonal=conj(diagonal);eigen=conj(eigen);}assert(cabs((diagonal-eigen)*(double complex)v[at(layout,i,j,ld)])<(TOL)*20);norm+=pow(cabs((double complex)v[at(layout,i,j,ld)]),2);}assert(fabs(norm-1)<(TOL)*10);} \
            } \
            for(int k=0;k<16;k++){int logical=0;for(int i=0;i<n;i++)for(int j=0;j<n;j++){logical|=k==at(layout,i,j,ld);}if(!logical){assert(a[k]==saved[k]&&vl[k]==(T)777&&vr[k]==(T)777);}} \
        } \
    } \
}while(0)
int main(int argc,char **argv){
    assert(argc==3);void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL),*reference=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);assert(library&&reference);
    typedef void (*register_fn)(int,int,void*);register_fn reg=(register_fn)dlsym(library,"colmat_lapack_set_expert_fallback");assert(reg);
    const char *names[]={"sgeevx_64_","dgeevx_64_","cgeevx_64_","zgeevx_64_"};for(int i=0;i<4;i++){void *p=dlsym(reference,names[i]);assert(p&&p!=dlsym(library,names[i]));reg(i,64,p);}
    CHECK(s,float,float,int32_t,"",WI_DECL,WI_ARG,int32_t,2e-5);
    CHECK(d,double,double,int32_t,"",WI_DECL,WI_ARG,int32_t,1e-12);
    CHECK(c,float complex,float,int32_t,"",NO_WI_DECL,NO_WI_ARG,float,2e-5);
    CHECK(z,double complex,double,int32_t,"",NO_WI_DECL,NO_WI_ARG,double,1e-12);
    CHECK(s,float,float,int64_t,"64_",WI_DECL,WI_ARG,int64_t,2e-5);
    CHECK(d,double,double,int64_t,"64_",WI_DECL,WI_ARG,int64_t,1e-12);
    CHECK(c,float complex,float,int64_t,"64_",NO_WI_DECL,NO_WI_ARG,float,2e-5);
    CHECK(z,double complex,double,int64_t,"64_",NO_WI_DECL,NO_WI_ARG,double,1e-12);
    puts("LAPACKE GEEVX C ABI: 8 names verified");dlclose(library);dlclose(reference);return 0;
}
