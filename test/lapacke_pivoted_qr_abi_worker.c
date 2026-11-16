/* Independent C prototypes and explicit Householder-product reconstruction. */
#include <assert.h>
#include <complex.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static int at(int layout,int i,int j,int ld){return layout==101?i*ld+j:i+j*ld;}
#define CHECK(P,T,R,J,SUFFIX,COMPLEX,TOL) do { \
    typedef J (*high_fn)(int,J,J,T*,J,J*,T*); \
    typedef J (*blocked_fn)(int,J,J,T*,J,J*,T*,T*,J); \
    typedef J (*blocked_complex_fn)(int,J,J,T*,J,J*,T*,T*,J,R*); \
    typedef J (*plain_fn)(int,J,J,T*,J,J*,T*,T*); \
    typedef J (*plain_complex_fn)(int,J,J,T*,J,J*,T*,T*,R*); \
    high_fn high[2]={(high_fn)dlsym(library,"LAPACKE_" #P "geqpf" SUFFIX),(high_fn)dlsym(library,"LAPACKE_" #P "geqp3" SUFFIX)}; \
    void *blocked=dlsym(library,"LAPACKE_" #P "geqp3_work" SUFFIX),*plain=dlsym(library,"LAPACKE_" #P "geqpf_work" SUFFIX);assert(high[0]&&high[1]&&blocked&&plain); \
    for(int shape=0;shape<7;shape++)for(int layout=101;layout<=102;layout++)for(int family=0;family<2;family++)for(int entry=0;entry<2;entry++)for(int pattern=0;pattern<3;pattern++){ \
        int ms[]={0,0,3,1,2,4,5},ns[]={0,3,0,1,4,2,3};int m=ms[shape],n=ns[shape],k=m<n?m:n,ld=layout==101?n+2:m+2,lw=3*n+1; \
        T a[96],saved_a[96],tau[8],work[32];R rwork[32];J pivot[8],saved_pivot[8];double complex original[36]={0}; \
        for(int z=0;z<96;z++){a[z]=(T)NAN;}for(int z=0;z<8;z++){tau[z]=(T)778;pivot[z]=(J)-777;}for(int z=0;z<32;z++){work[z]=(T)779;rwork[z]=(R)780;} \
        for(int j=0;j<n;j++){pivot[j+1]=(J)(pattern==0?0:pattern==1?(j%2==0?-2:0):9);} \
        for(int i=0;i<m;i++){for(int j=0;j<n;j++){T value=(T)((i==j?2:0)+.1*(i+j+1)+.07*(i-j)*I);a[at(layout,i,j,ld)]=value;original[i*n+j]=(double complex)value;}}memcpy(saved_a,a,sizeof(a));memcpy(saved_pivot,pivot,sizeof(pivot)); \
        if(family&&!entry){J info=COMPLEX?((blocked_complex_fn)blocked)(layout,m,n,a,ld,pivot+1,tau+1,work+1,-1,rwork+1):((blocked_fn)blocked)(layout,m,n,a,ld,pivot+1,tau+1,work+1,-1);assert(info==0&&creal((double complex)work[1])>=1);assert(memcmp(a,saved_a,sizeof(a))==0&&memcmp(pivot,saved_pivot,sizeof(pivot))==0); \
            info=COMPLEX?((blocked_complex_fn)blocked)(layout,m,n,a,ld,pivot+1,tau+1,work+1,0,rwork+1):((blocked_fn)blocked)(layout,m,n,a,ld,pivot+1,tau+1,work+1,0);assert(info==-9&&creal((double complex)work[1])>=1);assert(memcmp(a,saved_a,sizeof(a))==0&&memcmp(pivot,saved_pivot,sizeof(pivot))==0); \
        } \
        J result=entry?high[family](layout,m,n,a,ld,pivot+1,tau+1):family?(COMPLEX?((blocked_complex_fn)blocked)(layout,m,n,a,ld,pivot+1,tau+1,work+1,lw,rwork+1):((blocked_fn)blocked)(layout,m,n,a,ld,pivot+1,tau+1,work+1,lw)):(COMPLEX?((plain_complex_fn)plain)(layout,m,n,a,ld,pivot+1,tau+1,work+1,rwork+1):((plain_fn)plain)(layout,m,n,a,ld,pivot+1,tau+1,work+1));assert(result==0); \
        assert(pivot[0]==(J)-777&&pivot[n+1]==(J)-777&&tau[0]==(T)778&&tau[k+1]==(T)778&&work[0]==(T)779&&work[lw+1]==(T)779&&rwork[0]==(R)780&&rwork[2*n+1]==(R)780); \
        int seen=0;for(int j=0;j<n;j++){assert(pivot[j+1]>=1&&pivot[j+1]<=n);int bit=1<<((int)pivot[j+1]-1);assert(!(seen&bit));seen|=bit;}int fixed=0;for(int j=0;j<n;j++){if(saved_pivot[j+1]!=0){assert(pivot[fixed+1]==j+1);fixed++;}} \
        double complex q[36]={0};for(int i=0;i<m;i++){q[i*m+i]=1;} \
        for(int j=0;j<k;j++){double complex v[6]={0},qv[6]={0};v[j]=1;for(int i=j+1;i<m;i++){v[i]=(double complex)a[at(layout,i,j,ld)];}for(int i=0;i<m;i++){for(int h=0;h<m;h++){qv[i]+=q[i*m+h]*v[h];}}for(int i=0;i<m;i++){for(int h=0;h<m;h++){q[i*m+h]-=qv[i]*(double complex)tau[j+1]*conj(v[h]);}}assert(cimag((double complex)a[at(layout,j,j,ld)])==0);} \
        for(int i=0;i<m;i++){for(int j=0;j<n;j++){double complex value=0;for(int h=0;h<k;h++){if(h<=j){value+=q[i*m+h]*(double complex)a[at(layout,h,j,ld)];}}double complex expected=original[i*n+(int)pivot[j+1]-1];assert(cabs(value-expected)<(TOL)*100*(1+cabs(expected)));}} \
        for(int i=0;i<m;i++){for(int j=0;j<m;j++){double complex value=0;for(int h=0;h<m;h++){value+=conj(q[h*m+i])*q[h*m+j];}assert(cabs(value-(i==j?1:0))<(TOL)*100);}} \
        for(int z=0;z<96;z++){int i=layout==101?z/ld:z%ld,j=layout==101?z%ld:z/ld;if(i>=m||j>=n){assert(isnan(creal((double complex)a[z])));}} \
    } \
    T a=(T)NAN,tau[3]={(T)778,(T)778,(T)778},query=0;R rwork=0;J pivot[3]={(J)-777,0,(J)-777}; \
    for(int family=0;family<2;family++){assert(high[family](102,1,1,&a,1,pivot+1,tau+1)==-4);assert(pivot[1]==0&&tau[1]==(T)778);} \
    J result=COMPLEX?((blocked_complex_fn)blocked)(101,-1,3,NULL,2,NULL,NULL,&query,-1,&rwork):((blocked_fn)blocked)(101,-1,3,NULL,2,NULL,NULL,&query,-1);assert(result==-5); \
    if(sizeof(J)==8){J large=(J)INT64_C(4294967296),negative=(J)INT64_C(-4294967293);result=COMPLEX?((blocked_complex_fn)blocked)(102,0,large,NULL,1,NULL,NULL,&query,-1,&rwork):((blocked_fn)blocked)(102,0,large,NULL,1,NULL,NULL,&query,-1);assert(result==0&&query==(T)1);result=COMPLEX?((blocked_complex_fn)blocked)(102,negative,0,NULL,1,NULL,NULL,&query,-1,&rwork):((blocked_fn)blocked)(102,negative,0,NULL,1,NULL,NULL,&query,-1);assert(result==-2);} \
}while(0)
int main(int argc,char **argv){assert(argc==2);void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);assert(library);
    CHECK(s,float,float,int32_t,"",0,2e-5);CHECK(d,double,double,int32_t,"",0,1e-12);CHECK(c,float complex,float,int32_t,"",1,2e-5);CHECK(z,double complex,double,int32_t,"",1,1e-12);
    CHECK(s,float,float,int64_t,"64_",0,2e-5);CHECK(d,double,double,int64_t,"64_",0,1e-12);CHECK(c,float complex,float,int64_t,"64_",1,2e-5);CHECK(z,double complex,double,int64_t,"64_",1,1e-12);
    puts("LAPACKE pivoted QR C ABI: 16 names verified");dlclose(library);return 0;}
