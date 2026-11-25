/* Independent LAPACKE v3.12.1 LAUUM declarations and direct Fortran reference. */
#include <assert.h>
#include <complex.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int at(int layout,int i,int j,int ld){return layout==101?i*ld+j:i+j*ld;}
#define CHECK(P,T,R,J,SUFFIX,TOL) do { \
    typedef J (*c_fn)(int,char,J,T*,J);typedef void (*f_fn)(const char*,const int64_t*,T*,const int64_t*,int64_t*,size_t); \
    f_fn ref=(f_fn)dlsym(reference,#P "lauum_64_");assert(ref); \
    for(int entry=0;entry<2;entry++){char name[64];snprintf(name,sizeof(name),"LAPACKE_" #P "lauum%s" SUFFIX,entry?"":"_work");c_fn call=(c_fn)dlsym(library,name);assert(call); \
        for(int layout=101;layout<=102;layout++)for(int nc=0;nc<4;nc++)for(int upper=0;upper<2;upper++)for(int mode=0;mode<(sizeof(T)>sizeof(R)?3:1);mode++){ \
            int64_t n=nc==3?70:nc,ld=n+2,lr=layout==101?(n>0?n:1):ld,info=-777;char u=upper?'u':'l';T a[8192],before[8192],expected[8192]; \
            for(int q=0;q<8192;q++){a[q]=before[q]=expected[q]=(T)NAN;} \
            for(int i=0;i<n;i++)for(int j=0;j<n;j++)if(upper?i<=j:i>=j){T value=i==j?(T)3:(T)(.2+.1*I);if(i==j&&sizeof(T)>sizeof(R)){R parts[2]={(R)3,(R)(mode==0?0:mode==1?.75:NAN)};memcpy(&value,parts,sizeof(value));}a[at(layout,i,j,ld)]=before[at(layout,i,j,ld)]=expected[i+j*lr]=value;} \
            if(entry&&mode==2&&n>0){assert(call(layout,u,n,a,ld)==-4);assert(memcmp(a,before,sizeof(a))==0);continue;} \
            ref(&u,&n,expected,&lr,&info,1);assert(info==0);assert(call(layout,u,n,a,ld)==0); \
            for(int i=0;i<n;i++)for(int j=0;j<n;j++)if(upper?i<=j:i>=j){double complex x=(double complex)a[at(layout,i,j,ld)],y=(double complex)expected[i+j*lr]; \
                assert((isnan(creal(x))&&isnan(creal(y)))||fabs(creal(x)-creal(y))<(TOL)*fmax(1,fabs(creal(y)))); \
                assert((isnan(cimag(x))&&isnan(cimag(y)))||fabs(cimag(x)-cimag(y))<(TOL)*fmax(1,fabs(cimag(y)))); \
            } \
            for(int q=0;q<8192;q++){int written=0;int row=layout==101?q/ld:q%ld,col=layout==101?q%ld:q/ld;written=row<n&&col<n&&(upper?row<=col:row>=col);if(!written){assert(isnan(creal((double complex)a[q])));}} \
        } \
        assert(call(0,'L',0,NULL,1)==-1);assert(call(102,'?',0,NULL,1)==-2);assert(call(102,'L',-1,NULL,1)==-3);assert(call(101,'?',3,NULL,2)==-5);assert(call(102,'L',3,NULL,2)==-5); \
        if(sizeof(J)==8){assert(call(102,'L',(J)INT64_C(-4294967293),NULL,1)==-3);} \
    } \
}while(0)
int main(int argc,char **argv){assert(argc==4);void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL),*reference=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);assert(library&&reference);
    typedef void (*register_fn)(int,int,void*);register_fn reg=(register_fn)dlsym(library,"colmat_fallback_set");assert(reg);
    int group=atoi(argv[3]);
    const char *names[]={"clauum_64_","zlauum_64_","clauu2_64_","zlauu2_64_"};for(int i=0;i<4;i++){void *p=dlsym(reference,names[i]);assert(p&&p!=dlsym(library,names[i]));reg(group,i,p);}
    CHECK(s,float,float,int32_t,"",8e-5);CHECK(d,double,double,int32_t,"",8e-13);CHECK(c,float complex,float,int32_t,"",8e-5);CHECK(z,double complex,double,int32_t,"",8e-13);
    CHECK(s,float,float,int64_t,"64_",8e-5);CHECK(d,double,double,int64_t,"64_",8e-13);CHECK(c,float complex,float,int64_t,"64_",8e-5);CHECK(z,double complex,double,int64_t,"64_",8e-13);
    puts("LAPACKE LAUUM C ABI: 8 names verified");dlclose(library);dlclose(reference);return 0;}
