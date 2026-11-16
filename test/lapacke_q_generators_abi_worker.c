/* Independent LAPACKE explicit-Q prototypes; reference Fortran uses ILP64. */
#include <assert.h>
#include <complex.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static int at(int layout,int i,int j,int ld){return layout==101?i*ld+j:i+j*ld;}
#define CHECK(P,T,J,SUFFIX,STEM,TOL) do { \
    typedef J (*high_fn)(int,J,J,J,T*,J,const T*);typedef J (*work_fn)(int,J,J,J,T*,J,const T*,T*,J); \
    typedef void (*ref_fn)(const int64_t*,const int64_t*,const int64_t*,T*,const int64_t*,const T*,T*,const int64_t*,int64_t*); \
    const char *orientations[]={"qr","lq","ql","rq"}; \
    for(int orientation=0;orientation<4;orientation++){int rows=orientation==1||orientation==3;char name[96]; \
        snprintf(name,sizeof(name),"LAPACKE_%s%s%s%s",#P,STEM,orientations[orientation],SUFFIX);high_fn high=(high_fn)dlsym(library,name); \
        snprintf(name,sizeof(name),"LAPACKE_%s%s%s_work%s",#P,STEM,orientations[orientation],SUFFIX);work_fn work=(work_fn)dlsym(library,name); \
        snprintf(name,sizeof(name),"%s%s%s_64_",#P,STEM,orientations[orientation]);ref_fn ref=(ref_fn)dlsym(reference,name);assert(high&&work&&ref); \
        for(int shape=0;shape<5;shape++){int shortdim[]={0,0,1,2,3},longdim[]={0,3,1,4,3};int m=rows?shortdim[shape]:longdim[shape],n=rows?longdim[shape]:shortdim[shape],extent=m<n?m:n; \
            for(int k=0;k<=extent;k++)for(int layout=101;layout<=102;layout++)for(int entry=0;entry<2;entry++){ \
                int ld=layout==101?n+2:m+2,lw=rows?(m>0?m:1):(n>0?n:1);int64_t rm=m,rn=n,rk=k,rld=m+2,rlw=lw,ri[3]={-777,-777,-777}; \
                T a[64],saved[64],expected[32],tau[8],savedtau[8],scratch[12],rscratch[12],query=0; \
                for(int z=0;z<64;z++){a[z]=(T)777;}for(int z=0;z<32;z++){expected[z]=(T)777;}for(int z=0;z<8;z++){tau[z]=(T)779;}for(int z=0;z<12;z++){scratch[z]=rscratch[z]=(T)778;} \
                for(int i=0;i<m;i++){for(int j=0;j<n;j++){T v=(T)(.1*(1+i+j)+.07*(i-j)*I);a[at(layout,i,j,ld)]=expected[i+j*rld]=v;}}for(int h=0;h<k;h++){tau[h+1]=(T)(.5+.1*I);} \
                memcpy(saved,a,sizeof(a));memcpy(savedtau,tau,sizeof(tau));ref(&rm,&rn,&rk,expected,&rld,tau+1,rscratch+1,&rlw,ri+1);assert(ri[0]==-777&&ri[1]==0&&ri[2]==-777); \
                if(!entry){assert(work(layout,m,n,k,a,ld,tau+1,&query,-1)==0);assert(creal((double complex)query)>=lw);assert(memcmp(a,saved,sizeof(a))==0&&memcmp(tau,savedtau,sizeof(tau))==0);assert(work(layout,m,n,k,a,ld,tau+1,scratch+1,lw-1)==-9);assert(memcmp(a,saved,sizeof(a))==0);} \
                J info=entry?high(layout,m,n,k,a,ld,tau+1):work(layout,m,n,k,a,ld,tau+1,scratch+1,lw);assert(info==0);assert(memcmp(tau,savedtau,sizeof(tau))==0);assert(scratch[0]==(T)778&&scratch[lw+1]==(T)778); \
                for(int i=0;i<m;i++)for(int j=0;j<n;j++){double complex value=(double complex)expected[i+j*rld];assert(cabs((double complex)a[at(layout,i,j,ld)]-value)<(TOL)*(1+cabs(value)));} \
                for(int pos=0;pos<64;pos++){int i=layout==101?pos/ld:pos%ld,j=layout==101?pos%ld:pos/ld;if(i>=m||j>=n){assert(a[pos]==(T)777);}} \
            } \
        } \
        T query=0,a[8]={0},tau[3]={(T)1,(T)779,(T)779},saved[8]; \
        assert(work(0,0,0,0,NULL,1,NULL,&query,-1)==-1);assert(work(102,-1,0,0,NULL,1,NULL,&query,-1)==-2);assert(work(102,0,-1,0,NULL,1,NULL,&query,-1)==-3);assert(work(102,2,2,3,NULL,2,NULL,&query,-1)==-4);assert(work(101,-1,3,0,NULL,2,NULL,&query,-1)==-6);assert(work(102,2,2,1,NULL,1,NULL,&query,-1)==-6); \
        a[0]=(T)NAN;memcpy(saved,a,sizeof(a));assert(high(102,2,2,1,a,2,tau)==-5);assert(memcmp(a,saved,sizeof(a))==0);a[0]=0;tau[0]=(T)NAN;memcpy(saved,a,sizeof(a));assert(high(102,2,2,1,a,2,tau)==-7);assert(memcmp(a,saved,sizeof(a))==0); \
        if(sizeof(J)==8){J negative=(J)INT64_C(-4294967293),large=(J)INT64_C(4294967296);assert(work(102,negative,0,0,NULL,1,NULL,&query,-1)==-2);assert(work(102,rows?0:large,rows?large:0,0,NULL,rows?1:large,NULL,&query,-1)==0);assert(query==(T)1);} \
    } \
}while(0)
int main(int argc,char **argv){assert(argc==3);void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL),*reference=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);assert(library&&reference);
    CHECK(s,float,int32_t,"","org",1e-4);CHECK(d,double,int32_t,"","org",1e-12);CHECK(c,float complex,int32_t,"","ung",1e-4);CHECK(z,double complex,int32_t,"","ung",1e-12);
    CHECK(s,float,int64_t,"64_","org",1e-4);CHECK(d,double,int64_t,"64_","org",1e-12);CHECK(c,float complex,int64_t,"64_","ung",1e-4);CHECK(z,double complex,int64_t,"64_","ung",1e-12);
    puts("LAPACKE explicit Q C ABI: 32 names verified");dlclose(reference);dlclose(library);return 0;}
