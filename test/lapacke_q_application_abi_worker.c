/* Independent C prototypes; reference Fortran uses ILP64 and character lengths. */
#include <assert.h>
#include <complex.h>
#include <dlfcn.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static int at(int layout,int i,int j,int ld){return layout==101?i*ld+j:i+j*ld;}
#define CHECK(P,T,J,SUFFIX,STEM,TOL) do { \
    typedef J (*high_fn)(int,char,char,J,J,J,const T*,J,const T*,T*,J); \
    typedef J (*work_fn)(int,char,char,J,J,J,const T*,J,const T*,T*,J,T*,J); \
    typedef void (*ref_fn)(const char*,const char*,const int64_t*,const int64_t*,const int64_t*,const T*,const int64_t*,const T*,T*,const int64_t*,T*,const int64_t*,int64_t*,size_t,size_t); \
    const char *orientations[]={"qr","lq","ql","rq"}; \
    for(int orientation=0;orientation<4;orientation++){int rows=orientation==1||orientation==3;char name[96]; \
        snprintf(name,sizeof(name),"LAPACKE_%s%s%s%s",#P,STEM,orientations[orientation],SUFFIX);high_fn high=(high_fn)dlsym(library,name); \
        snprintf(name,sizeof(name),"LAPACKE_%s%s%s_work%s",#P,STEM,orientations[orientation],SUFFIX);work_fn work=(work_fn)dlsym(library,name); \
        snprintf(name,sizeof(name),"%s%s%s_64_",#P,STEM,orientations[orientation]);ref_fn ref=(ref_fn)dlsym(reference,name);assert(high&&work&&ref); \
        for(int shape=0;shape<6;shape++)for(int right=0;right<2;right++){ \
            int ms[]={0,0,3,1,2,4},ns[]={0,3,0,1,4,2};int m=ms[shape],n=ns[shape],order=right?n:m;char side=right?'R':'L'; \
            for(int k=0;k<=order;k++)for(int adjoint=0;adjoint<2;adjoint++)for(int layout=101;layout<=102;layout++)for(int entry=0;entry<2;entry++){ \
                char trans=adjoint?(STEM[0]=='u'?'C':'T'):'N';int ar=rows?k:order,ac=rows?order:k,la=layout==101?ac+2:ar+2,lc=layout==101?n+2:m+2,lw=right?(m>0?m:1):(n>0?n:1); \
                int64_t rm=m,rn=n,rk=k,rla=ar+2,rlc=m+2,rlw=lw,ri[3]={-777,-777,-777}; \
                T a[96],saved_a[96],ra[48],saved_ra[48],c[96],saved_c[96],rc[48],tau[8],saved_tau[8],scratch[16],rscratch[16],query=0; \
                for(int z=0;z<96;z++){a[z]=c[z]=(T)777;}for(int z=0;z<48;z++){ra[z]=rc[z]=(T)777;}for(int z=0;z<8;z++){tau[z]=(T)779;}for(int z=0;z<16;z++){scratch[z]=rscratch[z]=(T)778;} \
                for(int i=0;i<ar;i++){for(int j=0;j<ac;j++){T v=(T)(.1*(i+j+1)+.07*(i-j)*I);a[at(layout,i,j,la)]=ra[i+j*rla]=v;}} \
                for(int i=0;i<m;i++){for(int j=0;j<n;j++){T v=(T)((i==j?1:0)+.1*(i+j+1)+.2*(i-j)*I);c[at(layout,i,j,lc)]=rc[i+j*rlc]=v;}} \
                for(int h=0;h<k;h++){tau[h+1]=(T)((.7+.2*I)/(h+1));}memcpy(saved_a,a,sizeof(a));memcpy(saved_ra,ra,sizeof(ra));memcpy(saved_c,c,sizeof(c));memcpy(saved_tau,tau,sizeof(tau)); \
                ref(&side,&trans,&rm,&rn,&rk,ra,&rla,tau+1,rc,&rlc,rscratch+1,&rlw,ri+1,1,1);assert(ri[0]==-777&&ri[1]==0&&ri[2]==-777);assert(memcmp(ra,saved_ra,sizeof(ra))==0&&memcmp(tau,saved_tau,sizeof(tau))==0); \
                if(!entry){assert(work(layout,side,trans,m,n,k,a,la,tau+1,c,lc,&query,-1)==0);assert(creal((double complex)query)>=lw);assert(memcmp(a,saved_a,sizeof(a))==0&&memcmp(c,saved_c,sizeof(c))==0);assert(work(layout,side,trans,m,n,k,a,la,tau+1,c,lc,scratch+1,lw-1)==-13);assert(memcmp(c,saved_c,sizeof(c))==0);} \
                J info=entry?high(layout,side,trans,m,n,k,a,la,tau+1,c,lc):work(layout,side,trans,m,n,k,a,la,tau+1,c,lc,scratch+1,lw);assert(info==0); \
                assert(memcmp(a,saved_a,sizeof(a))==0&&memcmp(tau,saved_tau,sizeof(tau))==0);assert(scratch[0]==(T)778&&scratch[lw+1]==(T)778); \
                for(int i=0;i<m;i++){for(int j=0;j<n;j++){double complex expected=(double complex)rc[i+j*rlc];assert(cabs((double complex)c[at(layout,i,j,lc)]-expected)<(TOL)*(1+cabs(expected)));}} \
                for(int pos=0;pos<96;pos++){int i=layout==101?pos/lc:pos%lc,j=layout==101?pos%lc:pos/lc;if(i>=m||j>=n){assert(c[pos]==(T)777);}} \
            } \
        } \
        T query=0,a[16]={0},tau[3]={(T)1,(T)779,(T)779},c[16]={0},saved[16]; \
        assert(work(0,'L','N',0,0,0,NULL,1,NULL,NULL,1,&query,-1)==-1);assert(work(102,'X','N',0,0,0,NULL,1,NULL,NULL,1,&query,-1)==-2);assert(work(102,'L','X',0,0,0,NULL,1,NULL,NULL,1,&query,-1)==-3);assert(work(102,'L','N',-1,0,0,NULL,1,NULL,NULL,1,&query,-1)==-4);assert(work(102,'L','N',0,-1,0,NULL,1,NULL,NULL,1,&query,-1)==-5); \
        assert(work(102,'L','N',2,2,3,NULL,8,NULL,NULL,8,&query,-1)==-6);assert(work(102,'L','N',2,2,1,NULL,0,NULL,NULL,8,&query,-1)==-8);assert(work(102,'L','N',2,2,1,NULL,8,NULL,NULL,0,&query,-1)==-11);assert(work(101,'X','N',2,2,1,NULL,0,NULL,NULL,0,&query,-1)==-8);assert(work(101,'X','N',2,2,1,NULL,8,NULL,NULL,0,&query,-1)==-11); \
        a[0]=(T)NAN;memcpy(saved,c,sizeof(c));assert(high(102,'L','N',2,2,1,a,2,tau,c,2)==-7);assert(memcmp(c,saved,sizeof(c))==0);a[0]=0;c[0]=(T)NAN;tau[0]=(T)NAN;memcpy(saved,c,sizeof(c));assert(high(102,'L','N',2,2,1,a,2,tau,c,2)==-10);assert(memcmp(c,saved,sizeof(c))==0);c[0]=0;assert(high(102,'L','N',2,2,1,a,2,tau,c,2)==-9); \
        if(sizeof(J)==8){J negative=(J)INT64_C(-4294967293),large=(J)INT64_C(4294967296);assert(work(102,'L','N',negative,0,0,NULL,1,NULL,NULL,1,&query,-1)==-4);assert(work(102,'L','N',0,large,0,NULL,1,NULL,NULL,1,&query,-1)==0);assert(creal((double complex)query)>=large);assert(work(102,'L','N',2,2,1,a,2,tau,c,2,&query,negative)==-13);} \
    } \
}while(0)
int main(int argc,char **argv){assert(argc==3);void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL),*reference=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);assert(library&&reference);
    CHECK(s,float,int32_t,"","orm",1e-4);CHECK(d,double,int32_t,"","orm",1e-12);CHECK(c,float complex,int32_t,"","unm",1e-4);CHECK(z,double complex,int32_t,"","unm",1e-12);
    CHECK(s,float,int64_t,"64_","orm",1e-4);CHECK(d,double,int64_t,"64_","orm",1e-12);CHECK(c,float complex,int64_t,"64_","unm",1e-4);CHECK(z,double complex,int64_t,"64_","unm",1e-12);
    puts("LAPACKE Q application C ABI: 32 names verified");dlclose(reference);dlclose(library);return 0;}
