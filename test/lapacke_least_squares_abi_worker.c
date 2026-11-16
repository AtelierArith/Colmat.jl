/* Independent C declarations and analytic one-column least-squares systems. */
#include <assert.h>
#include <complex.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static int at(int layout,int i,int j,int ld){return layout==101?i*ld+j:i+j*ld;}
#define RANK_CALL(F,H,CX,LW) ((F)==0?((H)?yh(layout,m,n,r,a+1,lda,b+1,ldb,pivot+1,rcond,rank+1):((CX)?yc(layout,m,n,r,a+1,lda,b+1,ldb,pivot+1,rcond,rank+1,work+1,LW,rwork+1):yr(layout,m,n,r,a+1,lda,b+1,ldb,pivot+1,rcond,rank+1,work+1,LW))):((F)==1?((H)?sh(layout,m,n,r,a+1,lda,b+1,ldb,singular+1,rcond,rank+1):((CX)?sc(layout,m,n,r,a+1,lda,b+1,ldb,singular+1,rcond,rank+1,work+1,LW,rwork+1):sr(layout,m,n,r,a+1,lda,b+1,ldb,singular+1,rcond,rank+1,work+1,LW))):((H)?dh(layout,m,n,r,a+1,lda,b+1,ldb,singular+1,rcond,rank+1):((CX)?dc(layout,m,n,r,a+1,lda,b+1,ldb,singular+1,rcond,rank+1,work+1,LW,rwork+1,iwork+1):dr(layout,m,n,r,a+1,lda,b+1,ldb,singular+1,rcond,rank+1,work+1,LW,iwork+1)))))
#define CHECK(P,T,R,J,SUFFIX,CX,TRANS,TOL) do { \
    typedef J (*g_high)(int,char,J,J,J,T*,J,T*,J);typedef J (*g_work)(int,char,J,J,J,T*,J,T*,J,T*,J); \
    typedef J (*y_high)(int,J,J,J,T*,J,T*,J,J*,R,J*); \
    typedef J (*y_real)(int,J,J,J,T*,J,T*,J,J*,R,J*,T*,J);typedef J (*y_complex)(int,J,J,J,T*,J,T*,J,J*,R,J*,T*,J,R*); \
    typedef J (*s_high)(int,J,J,J,T*,J,T*,J,R*,R,J*); \
    typedef J (*s_real)(int,J,J,J,T*,J,T*,J,R*,R,J*,T*,J);typedef J (*s_complex)(int,J,J,J,T*,J,T*,J,R*,R,J*,T*,J,R*); \
    typedef J (*d_real)(int,J,J,J,T*,J,T*,J,R*,R,J*,T*,J,J*);typedef J (*d_complex)(int,J,J,J,T*,J,T*,J,R*,R,J*,T*,J,R*,J*); \
    g_high gh=(g_high)dlsym(library,"LAPACKE_" #P "gels" SUFFIX);g_work gw=(g_work)dlsym(library,"LAPACKE_" #P "gels_work" SUFFIX); \
    y_high yh=(y_high)dlsym(library,"LAPACKE_" #P "gelsy" SUFFIX);void *yp=dlsym(library,"LAPACKE_" #P "gelsy_work" SUFFIX);y_real yr=(y_real)yp;y_complex yc=(y_complex)yp; \
    s_high sh=(s_high)dlsym(library,"LAPACKE_" #P "gelss" SUFFIX),dh=(s_high)dlsym(library,"LAPACKE_" #P "gelsd" SUFFIX); \
    void *sp=dlsym(library,"LAPACKE_" #P "gelss_work" SUFFIX),*dp=dlsym(library,"LAPACKE_" #P "gelsd_work" SUFFIX);s_real sr=(s_real)sp;s_complex sc=(s_complex)sp;d_real dr=(d_real)dp;d_complex dc=(d_complex)dp; \
    assert(gh&&gw&&yh&&yp&&sh&&dh&&sp&&dp); \
    for(int layout=101;layout<=102;layout++)for(int high=0;high<2;high++)for(int adj=0;adj<2;adj++){ \
        T a[64],b[64],work[64],sa[64],sb[64];for(int q=0;q<64;q++){a[q]=b[q]=(T)777;work[q]=(T)779;} \
        int lda=layout==101?3:4,ldb=layout==101?5:6;T av[2]={(T)(3+.2*I),(T)(4-.1*I)},z=(T)(2+.5*I),rhs=(T)(5-.2*I); \
        double norm2=creal((double complex)av[0]*conj((double complex)av[0])+(double complex)av[1]*conj((double complex)av[1])); \
        for(int i=0;i<2;i++){a[1+at(layout,i,0,lda)]=av[i];b[1+at(layout,i,0,ldb)]=adj?(high?(T)0:(T)NAN):av[i]*z;}if(adj){b[1]=rhs;}memcpy(sa,a,sizeof(a));memcpy(sb,b,sizeof(b)); \
        char trans=adj?TRANS:'N';if(!high){assert(gw(layout,trans,2,1,1,a+1,lda,b+1,ldb,work+1,-1)==0);T query=work[1];assert(memcmp(a,sa,sizeof(a))==0&&memcmp(b,sb,sizeof(b))==0);assert(gw(layout,trans,2,1,1,a+1,lda,b+1,ldb,work+1,0)==-11);assert(work[1]==query&&last_error()==10);} \
        assert((high?gh(layout,trans,2,1,1,a+1,lda,b+1,ldb):gw(layout,trans,2,1,1,a+1,lda,b+1,ldb,work+1,62))==0); \
        if(adj){for(int i=0;i<2;i++){double complex expected=(double complex)av[i]*(double complex)rhs/norm2;assert(cabs((double complex)b[1+at(layout,i,0,ldb)]-expected)<TOL);}}else{assert(cabs((double complex)b[1]-(double complex)z)<TOL);assert(cabs((double complex)b[1+at(layout,1,0,ldb)])<TOL);} \
        for(int q=0;q<64;q++){if(q!=1&&q!=1+at(layout,1,0,lda)){assert(a[q]==sa[q]);}if(q!=1&&q!=1+at(layout,1,0,ldb)){assert(b[q]==sb[q]);}}assert(work[0]==(T)779&&work[63]==(T)779); \
    } \
    for(int layout=101;layout<=102;layout++)for(int high=0;high<2;high++)for(int family=0;family<3;family++){ \
        J m=2,n=1,r=1,lda=layout==101?3:4,ldb=layout==101?5:6,pivot[3]={(J)777,(J)-7,(J)777},rank[3]={(J)777,(J)778,(J)777},iwork[256]; \
        T a[64],b[64],work[2048],sa[64],sb[64],av[2]={(T)(3+.2*I),(T)(4-.1*I)},z=(T)(2+.5*I);R singular[3]={(R)782,(R)782,(R)782},rwork[4096],rcond=(R).001; \
        for(int q=0;q<64;q++){a[q]=b[q]=(T)777;}for(int q=0;q<2048;q++){work[q]=(T)779;}for(int q=0;q<4096;q++){rwork[q]=(R)780;}for(int q=0;q<256;q++){iwork[q]=(J)781;} \
        for(int i=0;i<2;i++){a[1+at(layout,i,0,lda)]=av[i];b[1+at(layout,i,0,ldb)]=av[i]*z;}memcpy(sa,a,sizeof(a));memcpy(sb,b,sizeof(b)); \
        if(!high){assert(RANK_CALL(family,0,CX,(J)-1)==0);T query=work[1];assert(rank[1]==(J)778&&singular[1]==(R)782&&pivot[1]==(J)-7);assert(memcmp(a,sa,sizeof(a))==0&&memcmp(b,sb,sizeof(b))==0);if(family==2){assert(iwork[1]>=1);if(CX){assert(rwork[1]>=1);}}assert(RANK_CALL(family,0,CX,(J)0)==-13);assert(work[1]==query&&last_error()==12);assert(rank[1]==(J)778&&singular[1]==(R)782);} \
        assert(RANK_CALL(family,high,CX,(J)2046)==0);assert(rank[1]==1);assert(cabs((double complex)b[1]-(double complex)z)<TOL); \
        if(family==0){assert(pivot[1]==1);}else{double sigma=sqrt(creal((double complex)av[0]*conj((double complex)av[0])+(double complex)av[1]*conj((double complex)av[1])));assert(fabs((double)singular[1]-sigma)<TOL);} \
        for(int q=0;q<64;q++){if(q!=1&&q!=1+at(layout,1,0,lda)){assert(a[q]==sa[q]);}if(q!=1&&q!=1+at(layout,1,0,ldb)){assert(b[q]==sb[q]);}} \
        assert(rank[0]==777&&rank[2]==777&&pivot[0]==777&&pivot[2]==777&&singular[0]==(R)782&&singular[2]==(R)782);assert(work[0]==(T)779&&work[2047]==(T)779&&rwork[0]==(R)780&&rwork[4095]==(R)780&&iwork[0]==(J)781&&iwork[255]==(J)781); \
        if(sizeof(J)==8){m=(J)INT64_C(-4294967293);rank[1]=(J)778;work[1]=(T)779;assert(RANK_CALL(family,0,CX,(J)-1)==-2);assert(rank[1]==(J)778&&work[1]==(T)779);} \
    } \
}while(0)
int main(int argc,char **argv){assert(argc==2);void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);assert(library);int (*last_error)(void)=(int (*)(void))dlsym(library,"colmat_blas_last_error");assert(last_error);
    CHECK(s,float,float,int32_t,"",0,'T',2e-4);CHECK(d,double,double,int32_t,"",0,'T',1e-11);CHECK(c,float complex,float,int32_t,"",1,'C',2e-4);CHECK(z,double complex,double,int32_t,"",1,'C',1e-11);
    CHECK(s,float,float,int64_t,"64_",0,'T',2e-4);CHECK(d,double,double,int64_t,"64_",0,'T',1e-11);CHECK(c,float complex,float,int64_t,"64_",1,'C',2e-4);CHECK(z,double complex,double,int64_t,"64_",1,'C',1e-11);
    puts("LAPACKE least-squares C ABI: 32 names and analytic solutions");dlclose(library);return 0;}
