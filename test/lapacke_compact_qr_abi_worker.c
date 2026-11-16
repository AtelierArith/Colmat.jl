/* Independent LAPACKE v3.12.1 declarations and ILP64 Fortran oracles. */
#include <assert.h>
#include <complex.h>
#include <dlfcn.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static int at(int layout,int i,int j,int ld){return layout==101?i*ld+j:i+j*ld;}
#define CHECK(P,T,J,SUFFIX,TRANS,TOL) do { \
    typedef J (*factor_high)(int,J,J,J,T*,J,T*,J); \
    typedef J (*factor_work)(int,J,J,J,T*,J,T*,J,T*); \
    typedef J (*plain_fn)(int,J,J,T*,J,T*,J); \
    typedef void (*ref_factor)(const int64_t*,const int64_t*,const int64_t*,T*,const int64_t*,T*,const int64_t*,T*,int64_t*); \
    typedef void (*ref_plain)(const int64_t*,const int64_t*,T*,const int64_t*,T*,const int64_t*,int64_t*); \
    typedef J (*apply_high)(int,char,char,J,J,J,J,const T*,J,const T*,J,T*,J); \
    typedef J (*apply_work)(int,char,char,J,J,J,J,const T*,J,const T*,J,T*,J,T*); \
    typedef void (*ref_apply)(const char*,const char*,const int64_t*,const int64_t*,const int64_t*,const int64_t*,const T*,const int64_t*,const T*,const int64_t*,T*,const int64_t*,T*,int64_t*,size_t,size_t); \
    factor_high fh=(factor_high)dlsym(library,"LAPACKE_" #P "geqrt" SUFFIX); \
    factor_work fw=(factor_work)dlsym(library,"LAPACKE_" #P "geqrt_work" SUFFIX); \
    plain_fn ph[2][2]={{(plain_fn)dlsym(library,"LAPACKE_" #P "geqrt2" SUFFIX),(plain_fn)dlsym(library,"LAPACKE_" #P "geqrt2_work" SUFFIX)}, \
                      {(plain_fn)dlsym(library,"LAPACKE_" #P "geqrt3" SUFFIX),(plain_fn)dlsym(library,"LAPACKE_" #P "geqrt3_work" SUFFIX)}}; \
    ref_factor rf=(ref_factor)dlsym(reference,#P "geqrt_64_"); \
    ref_plain rp[2]={(ref_plain)dlsym(reference,#P "geqrt2_64_"),(ref_plain)dlsym(reference,#P "geqrt3_64_")}; \
    apply_high ah=(apply_high)dlsym(library,"LAPACKE_" #P "gemqrt" SUFFIX); \
    apply_work aw=(apply_work)dlsym(library,"LAPACKE_" #P "gemqrt_work" SUFFIX); \
    ref_apply ra=(ref_apply)dlsym(reference,#P "gemqrt_64_"); \
    assert(fh&&fw&&ph[0][0]&&ph[0][1]&&ph[1][0]&&ph[1][1]&&rf&&rp[0]&&rp[1]&&ah&&aw&&ra); \
    for(int variant=0;variant<3;variant++)for(int shape=0;shape<5;shape++)for(int layout=101;layout<=102;layout++)for(int high=0;high<2;high++){ \
        int ms[]={0,3,1,5,2},ns[]={0,0,1,3,4};int m=ms[shape],n=ns[shape],k=m<n?m:n;if(variant&&m<n){continue;} \
        int nb=variant?(n>0?n:1):(k>1?2:1),la=layout==101?n+2:m+2,lt=layout==101?k+2:nb+2; \
        int64_t rm=m,rn=n,rnb=nb,rla=m+2,rlt=nb+2,info[3]={-777,-777,-777}; \
        T a[96],t[96],ar[96],tr[96],w[48],rw[48],sa[96],st[96]; \
        for(int z=0;z<96;z++){a[z]=ar[z]=(T)777;t[z]=tr[z]=(T)778;}for(int z=0;z<48;z++){w[z]=rw[z]=(T)779;} \
        for(int i=0;i<m;i++){for(int j=0;j<n;j++){T value=(T)((i==j?2:0)+.1*(i+j+1)+.07*(i-j)*I);a[at(layout,i,j,la)]=ar[i+j*rla]=value;}} \
        for(int i=0;i<nb;i++){for(int j=0;j<k;j++){t[at(layout,i,j,lt)]=tr[i+j*rlt]=0;}}memcpy(sa,a,sizeof(a));memcpy(st,t,sizeof(t)); \
        if(variant){rp[variant==2&&n==0?0:variant-1](&rm,&rn,ar,&rla,tr,&rlt,info+1);}else{rf(&rm,&rn,&rnb,ar,&rla,tr,&rlt,rw+1,info+1);} \
        assert(info[0]==-777&&info[1]==0&&info[2]==-777); \
        J result=variant?ph[variant-1][high](layout,m,n,a,la,t,lt):high?fh(layout,m,n,nb,a,la,t,lt):fw(layout,m,n,nb,a,la,t,lt,w+1);assert(result==0); \
        for(int j=0;j<k;j++){assert(cimag((double complex)a[at(layout,j,j,la)])==0);} \
        for(int direction=0;direction<2;direction++){T fv[96]={0},ft[96]={0},q[96]={0},qw[48];for(int z=0;z<48;z++){qw[z]=(T)779;} \
            for(int i=0;i<m;i++){q[i+i*rla]=1;for(int j=0;j<k;j++){fv[i+j*rla]=direction?ar[i+j*rla]:a[at(layout,i,j,la)];}} \
            for(int i=0;i<nb;i++){for(int j=0;j<k;j++){ft[i+j*rlt]=i<=j%nb?(direction?tr[i+j*rlt]:t[at(layout,i,j,lt)]):0;}} \
            if(direction){assert(ah(102,'L','N',m,m,k,nb,fv,rla,ft,rlt,q,rla)==0);}else{char side='L',trans='N';int64_t rk=k;info[1]=-777;ra(&side,&trans,&rm,&rm,&rk,&rnb,fv,&rla,ft,&rlt,q,&rla,qw+1,info+1,1,1);assert(info[1]==0);} \
            for(int i=0;i<m;i++){for(int j=0;j<n;j++){double complex value=0;for(int h=0;h<k;h++){if(h<=j){value+=(double complex)q[i+h*rla]*(double complex)(direction?ar[h+j*rla]:a[at(layout,h,j,la)]);}}double complex expected=(double complex)sa[at(layout,i,j,la)];assert(cabs(value-expected)<(TOL)*100*(1+cabs(expected)));}} \
            for(int i=0;i<m;i++){for(int j=0;j<m;j++){double complex value=0;for(int h=0;h<m;h++){value+=conj((double complex)q[h+i*rla])*(double complex)q[h+j*rla];}assert(cabs(value-(i==j?1:0))<(TOL)*100);}} \
            assert(qw[0]==(T)779&&qw[nb*m+1]==(T)779); \
        } \
        for(int z=0;z<96;z++){int i=layout==101?z/la:z%la,j=layout==101?z%la:z/la;if(i>=m||j>=n){assert(a[z]==sa[z]);}i=layout==101?z/lt:z%lt;j=layout==101?z%lt:z/lt;if(i>=nb||j>=k){assert(t[z]==st[z]);}} \
        int length=nb*n;if(length<1){length=1;}assert(w[0]==(T)779&&w[length+1]==(T)779); \
    } \
    for(int right=0;right<2;right++)for(int adj=0;adj<2;adj++)for(int shape=0;shape<4;shape++)for(int layout=101;layout<=102;layout++)for(int high=0;high<2;high++)for(int k0=0;k0<=3;k0++){ \
        int ms[]={3,7,0,5},ns[]={7,3,5,0};int m=ms[shape],n=ns[shape],order=right?n:m,k=k0<order?k0:order,nb=k>1?2:1;char side=right?'R':'L',trans=adj?TRANS:'N'; \
        int lv=layout==101?k+2:order+2,lt=layout==101?k+2:nb+2,lc=layout==101?n+2:m+2; \
        int64_t rm=m,rn=n,rk=k,rnb=nb,rlv=order+2,rlt=nb+2,rlc=m+2,info[3]={-777,-777,-777}; \
        T v[128],t[128],c[128],vr[128],tr[128],cr[128],sv[128],st[128],sc[128],w[48],rw[48]; \
        for(int z=0;z<128;z++){v[z]=t[z]=c[z]=vr[z]=tr[z]=cr[z]=(T)777;}for(int z=0;z<48;z++){w[z]=rw[z]=(T)779;} \
        for(int i=0;i<order;i++){for(int j=0;j<k;j++){T value=(T)(.1*(i+j+1)+.03*(i-j)*I);v[at(layout,i,j,lv)]=vr[i+j*rlv]=value;}} \
        for(int i=0;i<nb;i++){for(int j=0;j<k;j++){T value=(T)(.2/(i+j+1)+.02*I);t[at(layout,i,j,lt)]=tr[i+j*rlt]=value;}} \
        for(int i=0;i<m;i++){for(int j=0;j<n;j++){T value=(T)(.1*(i+j+1)+.03*(i-j)*I);c[at(layout,i,j,lc)]=cr[i+j*rlc]=value;}}memcpy(sv,v,sizeof(v));memcpy(st,t,sizeof(t));memcpy(sc,c,sizeof(c)); \
        ra(&side,&trans,&rm,&rn,&rk,&rnb,vr,&rlv,tr,&rlt,cr,&rlc,rw+1,info+1,1,1);assert(info[0]==-777&&info[1]==0&&info[2]==-777); \
        J result=high?ah(layout,side,trans,m,n,k,nb,v,lv,t,lt,c,lc):aw(layout,side,trans,m,n,k,nb,v,lv,t,lt,c,lc,w+1);assert(result==0); \
        assert(memcmp(v,sv,sizeof(v))==0&&memcmp(t,st,sizeof(t))==0); \
        for(int i=0;i<m;i++){for(int j=0;j<n;j++){double complex expected=(double complex)cr[i+j*rlc];assert(cabs((double complex)c[at(layout,i,j,lc)]-expected)<(TOL)*10*(1+cabs(expected)));}} \
        for(int z=0;z<128;z++){int i=layout==101?z/lc:z%lc,j=layout==101?z%lc:z/lc;if(i>=m||j>=n){assert(c[z]==sc[z]);}}int length=nb*(right?m:n);if(length<1){length=1;}assert(w[0]==(T)779&&w[length+1]==(T)779); \
    } \
    T scratch=0;assert(fw(0,0,0,1,NULL,1,NULL,1,&scratch)==-1);assert(ph[0][0](102,-1,-2,NULL,1,NULL,1)==-3);assert(ph[1][1](102,-1,-2,NULL,1,NULL,1)==-3); \
    assert(aw(101,'X','N',3,5,2,1,NULL,0,NULL,0,NULL,0,&scratch)==-13);assert(aw(101,'X','N',3,5,2,1,NULL,0,NULL,0,NULL,5,&scratch)==-11); \
    if(sizeof(J)==8){J negative=(J)INT64_C(-4294967293),large=(J)INT64_C(4294967296);assert(fw(102,negative,0,1,NULL,1,NULL,1,&scratch)==-2);assert(fw(102,0,large,1,NULL,1,NULL,1,&scratch)==0);assert(aw(102,'L','N',0,large,0,1,NULL,1,NULL,1,NULL,1,&scratch)==0);} \
}while(0)
int main(int argc,char **argv){assert(argc==3);void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL),*reference=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);assert(library&&reference);
    CHECK(s,float,int32_t,"",'T',2e-5);CHECK(d,double,int32_t,"",'T',1e-12);CHECK(c,float complex,int32_t,"",'C',2e-5);CHECK(z,double complex,int32_t,"",'C',1e-12);
    CHECK(s,float,int64_t,"64_",'T',2e-5);CHECK(d,double,int64_t,"64_",'T',1e-12);CHECK(c,float complex,int64_t,"64_",'C',2e-5);CHECK(z,double complex,int64_t,"64_",'C',1e-12);
    puts("LAPACKE compact QR C ABI: 32 names verified");dlclose(reference);dlclose(library);return 0;}
