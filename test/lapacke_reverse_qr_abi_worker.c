/* Independent LAPACKE declarations and independent OpenBLAS Q consumers. */
#include <assert.h>
#include <complex.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int at(int layout,int i,int j,int ld){return layout==101?i*ld+j:i+j*ld;}
#define CHECK(P,T,J,SUFFIX,QPREFIX,TOL) do { \
    typedef J (*high_fn)(int,J,J,T*,J,T*); \
    typedef J (*work_fn)(int,J,J,T*,J,T*,T*,J); \
    typedef void (*q_fn)(const int64_t*,const int64_t*,const int64_t*,T*,const int64_t*,T*,T*,const int64_t*,int64_t*); \
    for(int rq=0;rq<2;rq++){ \
        char name[96];snprintf(name,sizeof(name),"LAPACKE_%s%s%s",#P,rq?"gerqf":"geqlf",SUFFIX);high_fn high=(high_fn)dlsym(library,name); \
        snprintf(name,sizeof(name),"LAPACKE_%s%s_work%s",#P,rq?"gerqf":"geqlf",SUFFIX);work_fn work=(work_fn)dlsym(library,name); \
        snprintf(name,sizeof(name),"%s%s%s_64_",#P,QPREFIX,rq?"rq":"ql");q_fn generate=(q_fn)dlsym(reference,name);assert(high&&work&&generate); \
        for(int layout=101;layout<=102;layout++)for(int shape=0;shape<7;shape++)for(int entry=0;entry<2;entry++){ \
            int ms[]={0,0,3,1,2,4,3},ns[]={0,3,0,1,4,2,3};int m=ms[shape],n=ns[shape],k=m<n?m:n,order=rq?n:m,ld=layout==101?n+2:m+2; \
            int lw=rq?(n==0?1:(m>0?m:1)):(m==0?1:(n>0?n:1)); \
            T a[64],saved[64],tau[8],scratch[8],query=0;double complex original[16]={0}; \
            for(int z=0;z<64;z++){a[z]=(T)NAN;}for(int z=0;z<8;z++){tau[z]=scratch[z]=(T)777;} \
            for(int i=0;i<m;i++)for(int j=0;j<n;j++){T v=(T)((i==j?2:0)+.1*(i+j+1)+.2*(i-j)*I);a[at(layout,i,j,ld)]=v;original[i*n+j]=(double complex)v;}memcpy(saved,a,sizeof(a)); \
            if(!entry){assert(work(layout,m,n,a,ld,tau+1,&query,-1)==0);assert(creal((double complex)query)>=lw);if(m==0||n==0){assert(query==(T)1);}assert(memcmp(saved,a,sizeof(a))==0);for(int z=0;z<8;z++){assert(tau[z]==(T)777);}assert(work(layout,m,n,a,ld,tau+1,scratch+1,lw-1)==-8);assert(memcmp(saved,a,sizeof(a))==0);} \
            J info=entry?high(layout,m,n,a,ld,tau+1):work(layout,m,n,a,ld,tau+1,scratch+1,lw);assert(info==0); \
            assert(tau[0]==(T)777&&tau[k+1]==(T)777&&scratch[0]==(T)777&&scratch[lw+1]==(T)777); \
            for(int pos=0;pos<64;pos++){int i=layout==101?pos/ld:pos%ld,j=layout==101?pos%ld:pos/ld;if(i>=m||j>=n){assert(isnan(creal((double complex)a[pos])));}} \
            T q[36];for(int z=0;z<36;z++){q[z]=(T)NAN;}int64_t qo=order,qk=k,qld=order+2,qlw=-1,qi=-99; \
            for(int j=0;j<k;j++)for(int i=0;i<order-k+j;i++){if(rq){q[order-k+j+i*qld]=a[at(layout,m-k+j,i,ld)];}else{q[i+(order-k+j)*qld]=a[at(layout,i,n-k+j,ld)];}} \
            generate(&qo,&qo,&qk,q,&qld,tau+1,&query,&qlw,&qi);assert(qi==0);qlw=(int64_t)creal((double complex)query);assert(qlw>=1);T *qw=calloc((size_t)qlw,sizeof(T));assert(qw);generate(&qo,&qo,&qk,q,&qld,tau+1,qw,&qlw,&qi);free(qw);assert(qi==0); \
            for(int i=0;i<m;i++)for(int j=0;j<n;j++){double complex value=0;for(int z=0;z<order;z++){if(rq){if(z-i>=n-m){value+=(double complex)a[at(layout,i,z,ld)]*(double complex)q[z+j*qld];}}else{if(z-j>=m-n){value+=(double complex)q[i+z*qld]*(double complex)a[at(layout,z,j,ld)];}}}assert(cabs(value-original[i*n+j])<(TOL)*40);} \
            for(int i=0;i<order;i++)for(int j=0;j<order;j++){double complex value=0;for(int z=0;z<order;z++){value+=conj((double complex)q[z+i*qld])*(double complex)q[z+j*qld];}assert(cabs(value-(i==j?1:0))<(TOL)*30);} \
        } \
        T query=0;assert(work(0,0,0,NULL,1,NULL,&query,-1)==-1);assert(work(102,-1,0,NULL,1,NULL,&query,-1)==-2);assert(work(102,0,-1,NULL,1,NULL,&query,-1)==-3);assert(work(101,-1,3,NULL,2,NULL,&query,-1)==-5);assert(work(102,3,2,NULL,2,NULL,&query,-1)==-5); \
        T poison=(T)NAN,tau[3]={(T)777,(T)777,(T)777};assert(high(102,1,1,&poison,1,tau+1)==-4);assert(tau[1]==(T)777); \
        if(sizeof(J)==8){J negative=(J)INT64_C(-4294967293),large=(J)INT64_C(4294967296);assert(work(102,negative,0,NULL,1,NULL,&query,-1)==-2);assert(work(102,0,large,NULL,1,NULL,&query,-1)==0);assert(query==(T)1);} \
    } \
}while(0)
int main(int argc,char **argv){assert(argc==3);void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL),*reference=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);assert(library&&reference);
    CHECK(s,float,int32_t,"","org",2e-5);CHECK(d,double,int32_t,"","org",1e-12);CHECK(c,float complex,int32_t,"","ung",2e-5);CHECK(z,double complex,int32_t,"","ung",1e-12);
    CHECK(s,float,int64_t,"64_","org",2e-5);CHECK(d,double,int64_t,"64_","org",1e-12);CHECK(c,float complex,int64_t,"64_","ung",2e-5);CHECK(z,double complex,int64_t,"64_","ung",1e-12);
    puts("LAPACKE QL/RQ C ABI: 16 names verified");dlclose(reference);dlclose(library);return 0;}
