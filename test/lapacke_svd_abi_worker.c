/* Independent C11 prototypes from LAPACKE v3.12.1. No Rust headers are used. */
#include <assert.h>
#include <complex.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define REAL_DECL
#define REAL_ARG
#define C32_DECL , float *
#define C64_DECL , double *
#define COMPLEX_ARG , rw

static int position(int layout,int i,int j,int ld) { return layout==101?i*ld+j:i+j*ld; }

#define CHECK(P,T,R,INT,SUFFIX,RDECL,RARG,TOL) do { \
    typedef INT (*vd_high)(int,char,char,INT,INT,T*,INT,R*,T*,INT,T*,INT,R*); \
    typedef INT (*vd_work)(int,char,char,INT,INT,T*,INT,R*,T*,INT,T*,INT,T*,INT RDECL); \
    typedef INT (*dd_high)(int,char,INT,INT,T*,INT,R*,T*,INT,T*,INT); \
    typedef INT (*dd_work)(int,char,INT,INT,T*,INT,R*,T*,INT,T*,INT,T*,INT RDECL,INT*); \
    vd_high vh=(vd_high)dlsym(library,"LAPACKE_" #P "gesvd" SUFFIX); \
    vd_work vw=(vd_work)dlsym(library,"LAPACKE_" #P "gesvd_work" SUFFIX); \
    dd_high dh=(dd_high)dlsym(library,"LAPACKE_" #P "gesdd" SUFFIX); \
    dd_work dw=(dd_work)dlsym(library,"LAPACKE_" #P "gesdd_work" SUFFIX); \
    assert(vh && vw && dh && dw); \
    for(int dd=0;dd<2;dd++) for(int layout=101;layout<=102;layout++) \
    for(int shape=0;shape<4;shape++) for(int ju=0;ju<4;ju++) for(int jv=0;jv<(dd?1:4);jv++) { \
        char job="ASNO"[ju],jobv=dd?job:"ASNO"[jv]; \
        if(!dd && job=='O' && jobv=='O')continue; \
        int m=shape==0?3:2,n=shape==0?2:3;if(shape==2)m=0;if(shape==3)n=0; \
        int r=m<n?m:n; \
        int has_u=job=='A'||job=='S'||(!dd && job=='O')||(dd && job=='O'); \
        int has_v=jobv=='A'||jobv=='S'||jobv=='O'; \
        int uc=job=='A'||(dd && job=='O' && m<n)?m:r; \
        int vr=jobv=='A'||(dd && job=='O' && m>=n)?n:r; \
        int use_u=job=='A'||job=='S'||(dd && job=='O' && m<n); \
        int use_v=jobv=='A'||jobv=='S'||(dd && job=='O' && m>=n); \
        int lda=(layout==101?n:m)+2,ldu=(layout==101?uc:m)+2,ldvt=(layout==101?n:vr)+2; \
        T a[64],original[64],u[64],vt[64],query=0;R s[4]={0},superb[4]={0},rw[256]={0};INT iw[32]={0};(void)rw; \
        for(int k=0;k<64;k++)a[k]=original[k]=u[k]=vt[k]=(T)789; \
        for(int i=0;i<m;i++)for(int j=0;j<n;j++) { int q=position(layout,i,j,lda);a[q]=original[q]=(T)((3*(i+1)+j*j)+(0.3*i-0.6*j)*I); } \
        T *up=use_u?u:NULL,*vp=use_v?vt:NULL; \
        INT info=dd?dw(layout,job,m,n,a,lda,s,up,ldu,vp,ldvt,&query,-1 RARG,iw): \
            vw(layout,job,jobv,m,n,a,lda,s,up,ldu,vp,ldvt,&query,-1 RARG); \
        assert(info==0 && creal((double complex)query)>=1);assert(memcmp(a,original,sizeof(a))==0); \
        int nw=(int)creal((double complex)query);assert(nw>0 && nw<1024);T w[1024]={0}; \
        info=dd?dw(layout,job,m,n,a,lda,s,up,ldu,vp,ldvt,w,0 RARG,iw): \
            vw(layout,job,jobv,m,n,a,lda,s,up,ldu,vp,ldvt,w,0 RARG); \
        assert(info==(dd?-13:-14));assert(memcmp(a,original,sizeof(a))==0); \
        for(int high=0;high<2;high++) { \
            memcpy(a,original,sizeof(a));for(int k=0;k<64;k++)u[k]=vt[k]=(T)789; \
            info=dd?(high?dh(layout,job,m,n,a,lda,s,up,ldu,vp,ldvt):dw(layout,job,m,n,a,lda,s,up,ldu,vp,ldvt,w,nw RARG,iw)): \
                (high?vh(layout,job,jobv,m,n,a,lda,s,up,ldu,vp,ldvt,superb):vw(layout,job,jobv,m,n,a,lda,s,up,ldu,vp,ldvt,w,nw RARG)); \
            assert(info==0);double norm=0,snorm=0; \
            for(int i=0;i<m;i++)for(int j=0;j<n;j++){double q=cabs((double complex)original[position(layout,i,j,lda)]);norm+=q*q;} \
            for(int k=0;k<r;k++){assert(s[k]>=0);if(k+1<r)assert(s[k]>=s[k+1]);snorm+=(double)s[k]*s[k];} \
            assert(fabs(norm-snorm)<(TOL)*(1+norm)); \
            T *left=(job=='O' && (!dd || m>=n))?a:u;int ll=left==a?lda:ldu; \
            T *right=(!dd && jobv=='O')||(dd && job=='O' && m<n)?a:vt;int lr=right==a?lda:ldvt; \
            if(has_u && has_v)for(int i=0;i<m;i++)for(int j=0;j<n;j++){ \
                double complex value=0;for(int k=0;k<r;k++)value+=(double complex)left[position(layout,i,k,ll)]*s[k]*(double complex)right[position(layout,k,j,lr)]; \
                assert(cabs(value-(double complex)original[position(layout,i,j,lda)])<(TOL)*(1+sqrt(norm))); \
            } \
            if(has_u)for(int i=0;i<uc;i++)for(int j=0;j<uc;j++){ \
                double complex value=0;for(int k=0;k<m;k++)value+=conj((double complex)left[position(layout,k,i,ll)])*(double complex)left[position(layout,k,j,ll)]; \
                assert(cabs(value-(i==j?1.0:0.0))<(TOL)*10); \
            } \
            if(has_v)for(int i=0;i<vr;i++)for(int j=0;j<vr;j++){ \
                double complex value=0;for(int k=0;k<n;k++)value+=(double complex)right[position(layout,i,k,lr)]*conj((double complex)right[position(layout,j,k,lr)]); \
                assert(cabs(value-(i==j?1.0:0.0))<(TOL)*10); \
            } \
            for(int q=0;q<64;q++){int used_a=0,used_u=0,used_v=0; \
                for(int i=0;i<m;i++)for(int j=0;j<n;j++)used_a|=q==position(layout,i,j,lda); \
                for(int i=0;i<m;i++)for(int j=0;j<uc;j++)used_u|=use_u && q==position(layout,i,j,ldu); \
                for(int i=0;i<vr;i++)for(int j=0;j<n;j++)used_v|=use_v && q==position(layout,i,j,ldvt); \
                if(!used_a){assert(a[q]==(T)789);}if(!used_u){assert(u[q]==(T)789);}if(!used_v){assert(vt[q]==(T)789);} \
            } \
        } \
    } \
    T a[4]={1,0,0,2},query=0;R s[2]={0},superb[2]={0},rw[32]={0};INT iw[16]={0};(void)rw; \
    assert(vw(0,'N','N',1,1,NULL,1,s,NULL,1,NULL,1,&query,-1 RARG)==-1); \
    assert(dw(0,'N',1,1,NULL,1,s,NULL,1,NULL,1,&query,-1 RARG,iw)==-1); \
    assert(vw(102,'?','N',1,1,NULL,1,s,NULL,1,NULL,1,&query,-1 RARG)==-2); \
    assert(vw(102,'N','?',1,1,NULL,1,s,NULL,1,NULL,1,&query,-1 RARG)==-3); \
    assert(vw(102,'O','O',1,1,NULL,1,s,NULL,1,NULL,1,&query,-1 RARG)==-3); \
    assert(dw(102,'?',1,1,NULL,1,s,NULL,1,NULL,1,&query,-1 RARG,iw)==-2); \
    assert(vw(102,'N','N',-1,1,NULL,1,s,NULL,1,NULL,1,&query,-1 RARG)==-4); \
    assert(vw(102,'N','N',1,-1,NULL,1,s,NULL,1,NULL,1,&query,-1 RARG)==-5); \
    assert(dw(102,'N',-1,1,NULL,1,s,NULL,1,NULL,1,&query,-1 RARG,iw)==-3); \
    assert(dw(102,'N',1,-1,NULL,1,s,NULL,1,NULL,1,&query,-1 RARG,iw)==-4); \
    assert(vw(101,'N','N',2,3,NULL,2,s,NULL,1,NULL,1,&query,-1 RARG)==-7); \
    assert(vw(101,'S','S',2,3,NULL,3,s,NULL,1,NULL,3,&query,-1 RARG)==-10); \
    assert(vw(101,'S','S',2,3,NULL,3,s,NULL,2,NULL,2,&query,-1 RARG)==-12); \
    assert(dw(101,'N',2,3,NULL,2,s,NULL,1,NULL,3,&query,-1 RARG,iw)==-6); \
    assert(dw(101,'S',2,3,NULL,3,s,NULL,1,NULL,3,&query,-1 RARG,iw)==-9); \
    assert(dw(101,'N',2,3,NULL,3,s,NULL,1,NULL,2,&query,-1 RARG,iw)==-11); \
    a[0]=(T)NAN;assert(vh(102,'N','N',1,1,a,1,s,NULL,1,NULL,1,superb)==-6); \
    assert(dh(102,'N',1,1,a,1,s,NULL,1,NULL,1)==-5); \
    if(sizeof(INT)==8){ \
        INT negative=(INT)INT64_C(-4294967293),large=(INT)INT64_C(4294967296); \
        assert(vw(102,'N','N',negative,1,NULL,1,s,NULL,1,NULL,1,&query,-1 RARG)==-4); \
        assert(dw(102,'N',negative,1,NULL,1,s,NULL,1,NULL,1,&query,-1 RARG,iw)==-3); \
        assert(vw(102,'N','N',1,1,NULL,large,s,NULL,1,NULL,1,&query,-1 RARG)==0); \
        assert(dw(102,'N',1,1,NULL,large,s,NULL,1,NULL,1,&query,-1 RARG,iw)==0); \
    } \
} while(0)

int main(int argc,char **argv) {
    assert(argc==2);void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if(!library){fprintf(stderr,"%s\n",dlerror());return 3;}
    CHECK(s,float,float,int32_t,"",REAL_DECL,REAL_ARG,3e-5);
    CHECK(d,double,double,int32_t,"",REAL_DECL,REAL_ARG,1e-12);
    CHECK(c,float complex,float,int32_t,"",C32_DECL,COMPLEX_ARG,3e-5);
    CHECK(z,double complex,double,int32_t,"",C64_DECL,COMPLEX_ARG,1e-12);
    CHECK(s,float,float,int64_t,"64_",REAL_DECL,REAL_ARG,3e-5);
    CHECK(d,double,double,int64_t,"64_",REAL_DECL,REAL_ARG,1e-12);
    CHECK(c,float complex,float,int64_t,"64_",C32_DECL,COMPLEX_ARG,3e-5);
    CHECK(z,double complex,double,int64_t,"64_",C64_DECL,COMPLEX_ARG,1e-12);
    puts("LAPACKE SVD C ABI: both integer widths, jobs, layouts and workspaces passed");
    dlclose(library);return 0;
}
