/* Independent C declarations and scalar Householder products; no generated header. */
#include <assert.h>
#include <complex.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define CHECK(P,T,J,SUFFIX,TRANS,TOL) do { \
    typedef J (*factor_high)(int,J,J,J,J,T*,J,T*,J,T*,J); \
    typedef J (*factor_work)(int,J,J,J,J,T*,J,T*,J,T*,J,T*); \
    typedef J (*plain_factor)(int,J,J,J,T*,J,T*,J,T*,J); \
    typedef J (*q_high)(int,char,char,J,J,J,J,J,const T*,J,const T*,J,T*,J,T*,J); \
    typedef J (*q_work)(int,char,char,J,J,J,J,J,const T*,J,const T*,J,T*,J,T*,J,T*); \
    typedef J (*block_high)(int,char,char,char,char,J,J,J,J,const T*,J,const T*,J,T*,J,T*,J); \
    typedef J (*block_work)(int,char,char,char,char,J,J,J,J,const T*,J,const T*,J,T*,J,T*,J,T*,J); \
    factor_high fh=(factor_high)dlsym(library,"LAPACKE_" #P "tpqrt" SUFFIX); \
    factor_work fw=(factor_work)dlsym(library,"LAPACKE_" #P "tpqrt_work" SUFFIX); \
    plain_factor ph=(plain_factor)dlsym(library,"LAPACKE_" #P "tpqrt2" SUFFIX),pw=(plain_factor)dlsym(library,"LAPACKE_" #P "tpqrt2_work" SUFFIX); \
    q_high qh=(q_high)dlsym(library,"LAPACKE_" #P "tpmqrt" SUFFIX);q_work qw=(q_work)dlsym(library,"LAPACKE_" #P "tpmqrt_work" SUFFIX); \
    block_high bh=(block_high)dlsym(library,"LAPACKE_" #P "tprfb" SUFFIX);block_work bw=(block_work)dlsym(library,"LAPACKE_" #P "tprfb_work" SUFFIX); \
    assert(fh&&fw&&ph&&pw&&qh&&qw&&bh&&bw); \
    for(int layout=101;layout<=102;layout++)for(int high=0;high<2;high++)for(int blocked=0;blocked<2;blocked++){ \
        T a[16],b[16],t[16],work[8];for(int i=0;i<16;i++){a[i]=b[i]=t[i]=(T)777;}for(int i=0;i<8;i++){work[i]=(T)778;}a[1]=3;b[1]=4; \
        J info=blocked?(high?fh(layout,1,1,0,1,a+1,3,b+1,4,t+1,5):fw(layout,1,1,0,1,a+1,3,b+1,4,t+1,5,work+1)):(high?ph(layout,1,1,0,a+1,3,b+1,4,t+1,5):pw(layout,1,1,0,a+1,3,b+1,4,t+1,5)); \
        assert(info==0);assert(cabs((double complex)a[1]+5)<TOL);assert(cabs((double complex)b[1]-.5)<TOL);assert(cabs((double complex)t[1]-1.6)<TOL); \
        for(int i=0;i<16;i++){if(i!=1){assert(a[i]==(T)777&&b[i]==(T)777&&t[i]==(T)777);}}assert(work[0]==(T)778&&work[2]==(T)778); \
    } \
    for(int layout=101;layout<=102;layout++)for(int high=0;high<2;high++)for(int right=0;right<2;right++)for(int adj=0;adj<2;adj++){ \
        T v[16],t[16],a[16],b[16],work[16],sv[16],st[16];for(int i=0;i<16;i++){v[i]=t[i]=a[i]=b[i]=(T)777;work[i]=(T)778;}v[1]=(T).5;t[1]=(T)1.6;a[1]=3;b[1]=4;memcpy(sv,v,sizeof(v));memcpy(st,t,sizeof(t)); \
        char side=right?'R':'L',trans=adj?TRANS:'N';J info=high?qh(layout,side,trans,1,1,1,0,1,v+1,3,t+1,4,a+1,5,b+1,6):qw(layout,side,trans,1,1,1,0,1,v+1,3,t+1,4,a+1,5,b+1,6,work+1); \
        assert(info==0);assert(cabs((double complex)a[1]+5)<TOL&&cabs((double complex)b[1])<TOL);assert(memcmp(v,sv,sizeof(v))==0&&memcmp(t,st,sizeof(t))==0); \
        for(int i=0;i<16;i++){if(i!=1){assert(a[i]==(T)777&&b[i]==(T)777);}}assert(work[0]==(T)778&&work[2]==(T)778); \
        for(int backward=0;backward<2;backward++)for(int rowwise=0;rowwise<2;rowwise++){ \
            T value=(T)(.5+.25*I),tau=(T)(.3+.2*I);v[1]=value;t[1]=tau;a[1]=(T)(2+.4*I);b[1]=(T)(3-.2*I); \
            double complex vv=rowwise?conj((double complex)value):(double complex)value,tt=adj?conj((double complex)tau):(double complex)tau,aa=(double complex)a[1],bb=(double complex)b[1]; \
            double complex dot=right?aa+bb*vv:aa+conj(vv)*bb,ea=aa-dot*tt,eb=bb-dot*tt*(right?conj(vv):vv);memcpy(sv,v,sizeof(v));memcpy(st,t,sizeof(t)); \
            info=high?bh(layout,side,trans,backward?'B':'F',rowwise?'R':'C',1,1,1,0,v+1,3,t+1,4,a+1,5,b+1,6):bw(layout,side,trans,backward?'B':'F',rowwise?'R':'C',1,1,1,0,v+1,3,t+1,4,a+1,5,b+1,6,work+1,3); \
            assert(info==0);assert(cabs((double complex)a[1]-ea)<TOL&&cabs((double complex)b[1]-eb)<TOL);assert(memcmp(v,sv,sizeof(v))==0&&memcmp(t,st,sizeof(t))==0); \
            for(int i=0;i<16;i++){if(i!=1){assert(a[i]==(T)777&&b[i]==(T)777);}}assert(work[0]==(T)778&&work[4]==(T)778); \
        } \
    } \
    assert(fw(0,0,0,0,1,NULL,1,NULL,1,NULL,1,NULL)==-1); \
    assert(pw(101,2,3,0,NULL,2,NULL,3,NULL,3)==-5); \
    assert(qw(101,'?', 'N',0,0,0,0,1,NULL,0,NULL,0,NULL,0,NULL,0,NULL)==-2); \
    assert(bw(102,'?', 'N','F','C',0,0,0,0,NULL,1,NULL,1,NULL,1,NULL,1,NULL,1)==0);assert(last_error()==1); \
    if(sizeof(J)==8){J negative=(J)INT64_C(-4294967293);assert(fw(102,negative,0,0,1,NULL,1,NULL,1,NULL,1,NULL)==-2);assert(qw(102,'L','N',negative,0,0,0,1,NULL,1,NULL,1,NULL,1,NULL,1,NULL)==-4);} \
}while(0)
int main(int argc,char **argv){
    assert(argc==2);void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);assert(library);
    int (*last_error)(void)=(int (*)(void))dlsym(library,"colmat_blas_last_error");assert(last_error);
    CHECK(s,float,int32_t,"",'T',2e-5);CHECK(d,double,int32_t,"",'T',1e-12);
    CHECK(c,float complex,int32_t,"",'C',2e-5);CHECK(z,double complex,int32_t,"",'C',1e-12);
    CHECK(s,float,int64_t,"64_",'T',2e-5);CHECK(d,double,int64_t,"64_",'T',1e-12);
    CHECK(c,float complex,int64_t,"64_",'C',2e-5);CHECK(z,double complex,int64_t,"64_",'C',1e-12);
    puts("LAPACKE triangular-pentagonal C ABI: 32 names and scalar Householder products");dlclose(library);return 0;
}
