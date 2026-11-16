/* Independent LAPACKE v3.12.1 GEQRFP with baseline GEQR2 declarations. */
#include <assert.h>
#include <complex.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static int at(int layout,int i,int j,int ld){return layout==101?i*ld+j:i+j*ld;}
#define CHECK(P,T,J,SUFFIX,TOL) do { \
    typedef J (*high_fn)(int,J,J,T*,J,T*);typedef J (*blocked_fn)(int,J,J,T*,J,T*,T*,J);typedef J (*plain_fn)(int,J,J,T*,J,T*,T*); \
    high_fn high[2]={(high_fn)dlsym(library,"LAPACKE_" #P "geqr2" SUFFIX),(high_fn)dlsym(library,"LAPACKE_" #P "geqrfp" SUFFIX)}; \
    blocked_fn blocked=(blocked_fn)dlsym(library,"LAPACKE_" #P "geqrfp_work" SUFFIX);plain_fn plain=(plain_fn)dlsym(library,"LAPACKE_" #P "geqr2_work" SUFFIX);assert(high[0]&&high[1]&&blocked&&plain); \
    for(int layout=101;layout<=102;layout++)for(int shape=0;shape<7;shape++)for(int family=0;family<2;family++)for(int entry=0;entry<2;entry++){ \
        int ms[]={0,0,3,1,2,4,3},ns[]={0,3,0,1,4,2,3};int m=ms[shape],n=ns[shape],k=m<n?m:n,ld=layout==101?n+2:m+2,lw=family&&(m==0||n==0)?1:(n>0?n:1); \
        T a[64],saved[64],tau[8],work[8],query=0;double complex original[16]={0}; \
        for(int q=0;q<64;q++){a[q]=(T)NAN;}for(int q=0;q<8;q++){tau[q]=work[q]=(T)777;} \
        for(int i=0;i<m;i++)for(int j=0;j<n;j++){T value=(T)((i==j?2:0)+.1*(i+j+1)+.2*(i-j)*I);a[at(layout,i,j,ld)]=value;original[i*n+j]=(double complex)value;}memcpy(saved,a,sizeof(a)); \
        if(family&&!entry){assert(blocked(layout,m,n,a,ld,tau+1,&query,-1)==0);assert(memcmp(a,saved,sizeof(a))==0);for(int q=0;q<8;q++){assert(tau[q]==(T)777);}assert(creal((double complex)query)>=lw);if(m==0||n==0){assert(query==(T)1);}assert(blocked(layout,m,n,a,ld,tau+1,work+1,lw-1)==-8);assert(memcmp(a,saved,sizeof(a))==0);} \
        J info=entry?high[family](layout,m,n,a,ld,tau+1):family?blocked(layout,m,n,a,ld,tau+1,work+1,lw):plain(layout,m,n,a,ld,tau+1,work+1);assert(info==0); \
        assert(tau[0]==(T)777&&tau[k+1]==(T)777&&work[0]==(T)777&&work[lw+1]==(T)777);if(family&&!entry&&(m==0||n==0)){assert(work[1]==(T)1);} \
        if(family){for(int j=0;j<k;j++){assert(creal((double complex)a[at(layout,j,j,ld)])>=0&&cimag((double complex)a[at(layout,j,j,ld)])==0);}} \
        double complex qmat[16]={0};for(int i=0;i<m;i++){qmat[i*m+i]=1;} \
        for(int j=0;j<k;j++){double complex v[4]={0},qv[4]={0};v[j]=1;for(int i=j+1;i<m;i++){v[i]=(double complex)a[at(layout,i,j,ld)];}for(int i=0;i<m;i++)for(int l=0;l<m;l++){qv[i]+=qmat[i*m+l]*v[l];}for(int i=0;i<m;i++)for(int l=0;l<m;l++){qmat[i*m+l]-=qv[i]*(double complex)tau[j+1]*conj(v[l]);}} \
        for(int i=0;i<m;i++)for(int j=0;j<n;j++){double complex value=0;for(int l=0;l<m;l++){if(l<=j){value+=qmat[i*m+l]*(double complex)a[at(layout,l,j,ld)];}}assert(cabs(value-original[i*n+j])<(TOL)*30);} \
        for(int i=0;i<m;i++)for(int j=0;j<m;j++){double complex value=0;for(int l=0;l<m;l++){value+=conj(qmat[l*m+i])*qmat[l*m+j];}assert(cabs(value-(i==j?1:0))<(TOL)*20);} \
        for(int pos=0;pos<64;pos++){int i=layout==101?pos/ld:pos%ld,j=layout==101?pos%ld:pos/ld;if(i>=m||j>=n){assert(isnan(creal((double complex)a[pos])));}} \
    } \
    T query=0;assert(blocked(0,0,0,NULL,1,NULL,&query,-1)==-1);assert(blocked(102,-1,0,NULL,1,NULL,&query,-1)==-2);assert(blocked(102,0,-1,NULL,1,NULL,&query,-1)==-3);assert(blocked(101,-1,3,NULL,2,NULL,&query,-1)==-5);assert(blocked(102,3,2,NULL,2,NULL,&query,-1)==-5); \
    assert(plain(0,0,0,NULL,1,NULL,NULL)==-1);assert(plain(102,-1,0,NULL,1,NULL,NULL)==-2);assert(plain(102,0,-1,NULL,1,NULL,NULL)==-3);assert(plain(101,-1,3,NULL,2,NULL,NULL)==-5); \
    T poison=(T)NAN,tau[3]={(T)777,(T)777,(T)777};for(int family=0;family<2;family++){assert(high[family](102,1,1,&poison,1,tau+1)==-4);assert(tau[0]==(T)777&&tau[1]==(T)777&&tau[2]==(T)777);} \
    if(sizeof(J)==8){J negative=(J)INT64_C(-4294967293),large=(J)INT64_C(4294967296);assert(blocked(102,negative,0,NULL,1,NULL,&query,-1)==-2);assert(plain(102,negative,0,NULL,1,NULL,NULL)==-2);assert(blocked(102,0,large,NULL,1,NULL,&query,-1)==0);assert(plain(102,0,large,NULL,1,NULL,NULL)==0);} \
}while(0)
int main(int argc,char **argv){assert(argc==2);void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);assert(library);
    CHECK(s,float,int32_t,"",2e-5);CHECK(d,double,int32_t,"",1e-12);CHECK(c,float complex,int32_t,"",2e-5);CHECK(z,double complex,int32_t,"",1e-12);
    CHECK(s,float,int64_t,"64_",2e-5);CHECK(d,double,int64_t,"64_",1e-12);CHECK(c,float complex,int64_t,"64_",2e-5);CHECK(z,double complex,int64_t,"64_",1e-12);
    puts("LAPACKE positive QR C ABI: 8 positive names verified");dlclose(library);return 0;}
