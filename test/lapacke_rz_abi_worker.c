/* Independent LAPACKE declarations, Fortran consumers and explicit reflectors. */
#include <assert.h>
#include <complex.h>
#include <dlfcn.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static int at(int layout,int i,int j,int ld){return layout==101?i*ld+j:i+j*ld;}
#define CHECK(P,T,J,SUFFIX,APPLY,TRANS,TOL) do { \
    typedef J (*factor_high)(int,J,J,T*,J,T*); \
    typedef J (*factor_work)(int,J,J,T*,J,T*,T*,J); \
    typedef J (*apply_high)(int,char,char,J,J,J,J,const T*,J,const T*,T*,J); \
    typedef J (*apply_work)(int,char,char,J,J,J,J,const T*,J,const T*,T*,J,T*,J); \
    typedef void (*ref_factor)(const int64_t*,const int64_t*,T*,const int64_t*,T*,T*,const int64_t*,int64_t*); \
    typedef void (*ref_apply)(const char*,const char*,const int64_t*,const int64_t*,const int64_t*,const int64_t*,const T*,const int64_t*,const T*,T*,const int64_t*,T*,const int64_t*,int64_t*,size_t,size_t); \
    factor_high fh=(factor_high)dlsym(library,"LAPACKE_" #P "tzrzf" SUFFIX);factor_work fw=(factor_work)dlsym(library,"LAPACKE_" #P "tzrzf_work" SUFFIX); \
    apply_high ah=(apply_high)dlsym(library,"LAPACKE_" #P APPLY SUFFIX);apply_work aw=(apply_work)dlsym(library,"LAPACKE_" #P APPLY "_work" SUFFIX); \
    ref_factor rf=(ref_factor)dlsym(reference,#P "tzrzf_64_");ref_apply ra=(ref_apply)dlsym(reference,#P APPLY "_64_");assert(fh&&fw&&ah&&aw&&rf&&ra); \
    for(int shape=0;shape<7;shape++)for(int layout=101;layout<=102;layout++)for(int high=0;high<2;high++){ \
        int ms[]={0,0,1,2,1,2,3},ns[]={0,3,1,2,4,5,7};int m=ms[shape],n=ns[shape],l=n-m,ld=layout==101?n+2:m+2,lw=m==n?1:(m>0?m:1); \
        int64_t rm=m,rn=n,rk=m,rl=l,rla=m+2,rlz=n+2,rlw=n>0?n:1,flw=m>0?m:1,info[3]={-777,-777,-777}; \
        T a[128],saved[128],ar[128],tau[8],rtau[8],w[32],rw[32],af[128]={0},z[128]={0},zw[32];double complex original[32]={0}; \
        for(int q=0;q<128;q++){a[q]=ar[q]=(T)NAN;}for(int q=0;q<8;q++){tau[q]=rtau[q]=(T)778;}for(int q=0;q<32;q++){w[q]=rw[q]=zw[q]=(T)779;} \
        for(int i=0;i<m;i++){for(int j=0;j<n;j++){if(i<=j){T value=(T)((i==j?2:0)+.1*(i+j+1)+.03*(i-j)*I);a[at(layout,i,j,ld)]=ar[i+j*rla]=value;original[i*n+j]=(double complex)value;}else if(high){a[at(layout,i,j,ld)]=0;}}}memcpy(saved,a,sizeof(a)); \
        if(!high){T query=0;assert(fw(layout,m,n,a,ld,tau+1,&query,-1)==0);assert(creal((double complex)query)>=lw);if(m==0||m==n){assert(query==(T)1);}assert(memcmp(a,saved,sizeof(a))==0);assert(fw(layout,m,n,a,ld,tau+1,&query,0)==-8);assert(creal((double complex)query)>=lw);assert(memcmp(a,saved,sizeof(a))==0);} \
        assert((high?fh(layout,m,n,a,ld,tau+1):fw(layout,m,n,a,ld,tau+1,w+1,lw))==0); \
        assert(tau[0]==(T)778&&tau[m+1]==(T)778&&w[0]==(T)779&&w[lw+1]==(T)779);if(!high&&(m==0||m==n)){assert(w[1]==(T)1);} \
        for(int q=0;q<128;q++){int i=layout==101?q/ld:q%ld,j=layout==101?q%ld:q/ld;if(i>=m||j>=n||i>j){assert(memcmp(a+q,saved+q,sizeof(T))==0);}} \
        for(int i=0;i<m;i++){for(int j=m;j<n;j++){af[i+j*rla]=a[at(layout,i,j,ld)];}}for(int i=0;i<n;i++){z[i+i*rlz]=1;} \
        char side='R',trans='N';ra(&side,&trans,&rn,&rn,&rk,&rl,af,&rla,tau+1,z,&rlz,zw+1,&rlw,info+1,1,1);assert(info[0]==-777&&info[1]==0&&info[2]==-777);assert(zw[0]==(T)779&&zw[rlw+1]==(T)779); \
        for(int i=0;i<m;i++){for(int j=0;j<n;j++){double complex value=0;for(int h=i;h<m;h++){value+=(double complex)a[at(layout,i,h,ld)]*(double complex)z[h+j*rlz];}assert(cabs(value-original[i*n+j])<(TOL)*50*(1+cabs(original[i*n+j])));}} \
        for(int i=0;i<n;i++){for(int j=0;j<n;j++){double complex value=0;for(int h=0;h<n;h++){value+=conj((double complex)z[h+i*rlz])*(double complex)z[h+j*rlz];}assert(cabs(value-(i==j?1:0))<(TOL)*50);}} \
        rf(&rm,&rn,ar,&rla,rtau+1,rw+1,&flw,info+1);assert(info[0]==-777&&info[1]==0&&info[2]==-777);assert(rtau[0]==(T)778&&rtau[m+1]==(T)778&&rw[0]==(T)779&&rw[flw+1]==(T)779); \
        T ca[128]={0},cz[128]={0};int cla=layout==101?n+2:m+2,clz=n+2;for(int i=0;i<m;i++){for(int j=m;j<n;j++){ca[at(layout,i,j,cla)]=ar[i+j*rla];}}for(int i=0;i<n;i++){cz[at(layout,i,i,clz)]=1;} \
        assert(ah(layout,'R','N',n,n,m,l,ca,cla,rtau+1,cz,clz)==0); \
        for(int i=0;i<m;i++){for(int j=0;j<n;j++){double complex value=0;for(int h=i;h<m;h++){value+=(double complex)ar[i+h*rla]*(double complex)cz[at(layout,h,j,clz)];}assert(cabs(value-original[i*n+j])<(TOL)*50*(1+cabs(original[i*n+j])));}} \
    } \
    for(int right=0;right<2;right++)for(int adj=0;adj<2;adj++)for(int layout=101;layout<=102;layout++)for(int high=0;high<2;high++)for(int shape=0;shape<4;shape++){ \
        int ms[]={3,7,0,5},ns[]={7,3,5,0};int m=ms[shape],n=ns[shape],order=right?n:m,k=order<2?order:2,lda=layout==101?order+2:k+2,ldc=layout==101?n+2:m+2,lw=right?m:n;if(lw<1){lw=1;} \
        for(int tail=0;tail<4;tail++){int l=tail==0?0:tail==1?(order>0?1:0):tail==2?order-k:order;char side=right?'R':'L',trans=adj?TRANS:'N'; \
            T a[128],ar[128],c[128],cr[128],tau[8],w[32],rw[32],sa[128],sc[128],st[8];double complex qmat[64]={0},original[64]={0}; \
            int64_t rm=m,rn=n,rk=k,rl=l,rla=k+2,rlc=m+2,rlw=lw,info[3]={-777,-777,-777}; \
            for(int z=0;z<128;z++){a[z]=ar[z]=c[z]=cr[z]=(T)NAN;}for(int z=0;z<8;z++){tau[z]=(T)778;}for(int z=0;z<32;z++){w[z]=rw[z]=(T)779;} \
            for(int i=0;i<k;i++){tau[i+1]=(T)((.7+.2*I)/(i+1));for(int j=0;j<order;j++){T value=(T)(j>=order-l?.03*(i+j+1)+.02*(i-j)*I:0);ar[i+j*rla]=value;if(high||j>=order-l){a[at(layout,i,j,lda)]=value;}}} \
            for(int i=0;i<m;i++){for(int j=0;j<n;j++){T value=(T)(.1*(i+j+1)+.03*(i-j)*I);c[at(layout,i,j,ldc)]=cr[i+j*rlc]=value;original[i*n+j]=(double complex)value;}}memcpy(sa,a,sizeof(a));memcpy(sc,c,sizeof(c));memcpy(st,tau,sizeof(tau)); \
            for(int i=0;i<order;i++){qmat[i*order+i]=1;}for(int h=0;h<k;h++){double complex v[8]={0},qv[8]={0};v[h]=1;for(int j=order-l;j<order;j++){v[j]+=(double complex)ar[h+j*rla];}for(int i=0;i<order;i++){for(int j=0;j<order;j++){qv[i]+=qmat[i*order+j]*v[j];}}for(int i=0;i<order;i++){for(int j=0;j<order;j++){qmat[i*order+j]-=qv[i]*(double complex)tau[h+1]*conj(v[j]);}}} \
            if(l<=order-k){ra(&side,&trans,&rm,&rn,&rk,&rl,ar,&rla,tau+1,cr,&rlc,rw+1,&rlw,info+1,1,1);assert(info[0]==-777&&info[1]==0&&info[2]==-777);assert(rw[0]==(T)779&&rw[lw+1]==(T)779);} \
            assert((high?ah(layout,side,trans,m,n,k,l,a,lda,tau+1,c,ldc):aw(layout,side,trans,m,n,k,l,a,lda,tau+1,c,ldc,w+1,lw))==0); \
            assert(memcmp(a,sa,sizeof(a))==0&&memcmp(tau,st,sizeof(tau))==0&&w[0]==(T)779&&w[lw+1]==(T)779); \
            for(int i=0;i<m;i++){for(int j=0;j<n;j++){double complex expected=0;for(int h=0;h<order;h++){double complex op=right?(adj?conj(qmat[j*order+h]):qmat[h*order+j]):(adj?conj(qmat[h*order+i]):qmat[i*order+h]);expected+=right?original[i*n+h]*op:op*original[h*n+j];}assert(cabs((double complex)c[at(layout,i,j,ldc)]-expected)<(TOL)*50*(1+cabs(expected)));if(l<=order-k){assert(cabs((double complex)cr[i+j*rlc]-expected)<(TOL)*50*(1+cabs(expected)));}}} \
            for(int z=0;z<128;z++){int i=layout==101?z/ldc:z%ldc,j=layout==101?z%ldc:z/ldc;if(i>=m||j>=n){assert(memcmp(c+z,sc+z,sizeof(T))==0);}} \
        } \
    } \
    T work[8]={(T)779},tau[3]={0};assert(fw(0,0,0,NULL,1,NULL,work,-1)==-1);assert(fw(102,3,2,NULL,3,NULL,work,-1)==-3); \
    assert(aw(101,'R','N',3,5,1,1,NULL,4,NULL,NULL,7,work,-1)==-9);assert(aw(102,'L','N',2,3,1,0,NULL,1,NULL,NULL,2,work,2)==-14); \
    assert(ah(102,'L','N',1,1,2,0,NULL,2,NULL,NULL,1)==-6);assert(aw(101,'L','N',3,5,1,0,NULL,8,tau,NULL,8,work,5)==0); \
    if(sizeof(J)==8){J large=(J)INT64_C(4294967296),negative=(J)INT64_C(-4294967293);assert(fw(102,0,large,NULL,1,NULL,work,-1)==0&&work[0]==(T)1);assert(fw(102,negative,0,NULL,1,NULL,work,-1)==-2);assert(aw(102,'L','N',0,0,0,large,NULL,1,NULL,NULL,1,work,-1)==-7);} \
}while(0)
int main(int argc,char **argv){assert(argc==3);void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL),*reference=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);assert(library&&reference);
    CHECK(s,float,int32_t,"","ormrz",'T',2e-5);CHECK(d,double,int32_t,"","ormrz",'T',1e-12);CHECK(c,float complex,int32_t,"","unmrz",'C',2e-5);CHECK(z,double complex,int32_t,"","unmrz",'C',1e-12);
    CHECK(s,float,int64_t,"64_","ormrz",'T',2e-5);CHECK(d,double,int64_t,"64_","ormrz",'T',1e-12);CHECK(c,float complex,int64_t,"64_","unmrz",'C',2e-5);CHECK(z,double complex,int64_t,"64_","unmrz",'C',1e-12);
    puts("LAPACKE RZ C ABI: 16 names with explicit reflector reconstruction");dlclose(reference);dlclose(library);return 0;}
