/* Independent LAPACKE v3.12.1 TRTRI/TRTRS declarations. */
#include <assert.h>
#include <complex.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static int at(int layout,int i,int j,int ld){return layout==101?i*ld+j:i+j*ld;}
#define CHECK(P,T,J,SUFFIX,TOL) do { \
    typedef J (*inv_fn)(int,char,char,J,T*,J); \
    typedef J (*solve_fn)(int,char,char,char,J,J,const T*,J,T*,J); \
    for(int entry=0;entry<2;entry++){char inverse_name[64],solve_name[64]; \
        snprintf(inverse_name,sizeof(inverse_name),"LAPACKE_" #P "trtri%s" SUFFIX,entry?"":"_work");snprintf(solve_name,sizeof(solve_name),"LAPACKE_" #P "trtrs%s" SUFFIX,entry?"":"_work"); \
        inv_fn inverse=(inv_fn)dlsym(library,inverse_name);solve_fn solve=(solve_fn)dlsym(library,solve_name);assert(inverse&&solve); \
        for(int layout=101;layout<=102;layout++)for(int n=0;n<=3;n++)for(int upper=0;upper<2;upper++)for(int unit=0;unit<2;unit++){ \
            int ld=n+2;char u=upper?'u':'l',d=unit?'u':'n';T a[25],saved[25];double complex matrix[9]={0}; \
            for(int q=0;q<25;q++){a[q]=(T)NAN;}for(int i=0;i<n;i++)for(int j=0;j<n;j++)if(upper?i<=j:i>=j){T value=i==j?(T)(unit?1:i+2):(T)(.2+.1*I);matrix[i*n+j]=(double complex)value;if(!unit||i!=j){a[at(layout,i,j,ld)]=value;}} \
            memcpy(saved,a,sizeof(a));assert(inverse(layout,u,d,n,a,ld)==0); \
            for(int i=0;i<n;i++)for(int j=0;j<n;j++){double complex sum=0;for(int k=0;k<n;k++){double complex value=k==j&&unit?1:((upper?k<=j:k>=j)?(double complex)a[at(layout,k,j,ld)]:0);sum+=matrix[i*n+k]*value;}assert(cabs(sum-(i==j?1:0))<(TOL)*20);} \
            for(int q=0;q<25;q++){int written=0;for(int i=0;i<n;i++)for(int j=0;j<n;j++){written|=q==at(layout,i,j,ld)&&(upper?i<=j:i>=j)&&(!unit||i!=j);}if(!written){assert(isnan(creal((double complex)a[q])));}} \
            for(int trans=0;trans<3;trans++)for(int nrhs=0;nrhs<=2;nrhs++){char t="ntc"[trans];int lb=layout==101?nrhs+2:n+2;T b[25];for(int q=0;q<25;q++){b[q]=(T)NAN;} \
                for(int i=0;i<n;i++)for(int j=0;j<nrhs;j++){double complex sum=0;for(int k=0;k<n;k++){double complex value=trans?matrix[k*n+i]:matrix[i*n+k];if(trans==2){value=conj(value);}sum+=value*(k+j+1);}b[at(layout,i,j,lb)]=(T)sum;} \
                T factor_before[25];memcpy(factor_before,saved,sizeof(saved)); \
                assert(solve(layout,u,t,d,n,nrhs,saved,ld,b,lb)==0);assert(memcmp(saved,factor_before,sizeof(saved))==0); \
                for(int i=0;i<n;i++)for(int j=0;j<nrhs;j++){assert(cabs((double complex)b[at(layout,i,j,lb)]-(i+j+1))<(TOL)*20);} \
                for(int q=0;q<25;q++){int written=0;for(int i=0;i<n;i++)for(int j=0;j<nrhs;j++){written|=q==at(layout,i,j,lb);}if(!written){assert(isnan(creal((double complex)b[q])));}} \
            } \
        } \
        T singular[9]={2,0,0,0,0,0,0,0,4},before[9],rhs[9];memcpy(before,singular,sizeof(before));for(int i=0;i<9;i++){rhs[i]=(T)777;} \
        assert(inverse(102,'L','N',3,singular,3)==2&&memcmp(singular,before,sizeof(before))==0); \
        assert(solve(102,'U','N','N',3,0,singular,3,rhs,3)==2);for(int i=0;i<9;i++){assert(rhs[i]==(T)777);} \
        assert(inverse(0,'L','N',0,NULL,1)==-1);assert(inverse(102,'?','N',0,NULL,1)==-2);assert(inverse(102,'L','?',0,NULL,1)==-3);assert(inverse(102,'L','N',-1,NULL,1)==-4);assert(inverse(101,'?','?',3,NULL,2)==-6); \
        assert(solve(102,'L','?','N',0,0,NULL,1,NULL,1)==-3);assert(solve(102,'L','N','?',0,0,NULL,1,NULL,1)==-4);assert(solve(102,'L','N','N',-1,0,NULL,1,NULL,1)==-5);assert(solve(102,'L','N','N',0,-1,NULL,1,NULL,1)==-6); \
        assert(solve(101,'?','?','?',3,2,NULL,2,NULL,1)==-8);assert(solve(101,'?','?','?',3,2,NULL,3,NULL,1)==-10); \
        if(sizeof(J)==8){assert(inverse(102,'L','N',(J)INT64_C(-4294967293),NULL,1)==-4);assert(solve(102,'L','N','N',(J)INT64_C(-4294967293),0,NULL,1,NULL,1)==-5);} \
        if(entry){T poison=(T)NAN,b=(T)1;assert(inverse(102,'L','N',1,&poison,1)==-5);assert(inverse(102,'L','U',1,&poison,1)==0&&isnan(creal((double complex)poison)));assert(solve(102,'L','N','N',1,1,&poison,1,&b,1)==-7);assert(solve(102,'L','N','U',1,1,&poison,1,&b,1)==0);T one=1;assert(solve(102,'L','N','N',1,1,&one,1,&poison,1)==-9);} \
    } \
}while(0)
int main(int argc,char **argv){assert(argc==2);void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);assert(library);
    CHECK(s,float,int32_t,"",2e-5);CHECK(d,double,int32_t,"",1e-12);CHECK(c,float complex,int32_t,"",2e-5);CHECK(z,double complex,int32_t,"",1e-12);
    CHECK(s,float,int64_t,"64_",2e-5);CHECK(d,double,int64_t,"64_",1e-12);CHECK(c,float complex,int64_t,"64_",2e-5);CHECK(z,double complex,int64_t,"64_",1e-12);
    puts("LAPACKE triangular C ABI: 16 names verified");dlclose(library);return 0;}
