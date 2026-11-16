/* Independent declarations from Reference LAPACK v3.12.1 LAPACKE headers. */
#include <assert.h>
#include <complex.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#define CHECK(P,T,J,SUFFIX,TOL) do { \
 typedef J (*pair_high)(int,J,J,J,T*,J,T*,T*,J,T*); \
 typedef J (*pair_work)(int,J,J,J,T*,J,T*,T*,J,T*,T*,J); \
 typedef J (*driver_high)(int,J,J,J,T*,J,T*,J,T*,T*,T*); \
 typedef J (*driver_work)(int,J,J,J,T*,J,T*,J,T*,T*,T*,T*,J); \
 const char *pairs[]={"LAPACKE_" #P "ggqrf","LAPACKE_" #P "ggrqf"}; \
 const char *drivers[]={"LAPACKE_" #P "gglse","LAPACKE_" #P "ggglm"}; \
 for(int family=0;family<2;family++){ \
  char hn[96],wn[96];snprintf(hn,sizeof(hn),"%s%s",pairs[family],SUFFIX);snprintf(wn,sizeof(wn),"%s_work%s",pairs[family],SUFFIX); \
  pair_high h=(pair_high)dlsym(library,hn);pair_work w=(pair_work)dlsym(library,wn);assert(h&&w); \
  for(int layout=101;layout<=102;layout++){ \
   T a[8]={0},b[8]={0},ta[3]={(T)777,0,(T)777},tb[3]={(T)777,0,(T)777},work[64]; \
   for(int z=0;z<64;z++){work[z]=(T)779;}a[0]=2;b[0]=3; \
   assert(w(layout,1,1,1,a,1,ta+1,b,1,tb+1,work+1,-1)==0);assert(creal((double complex)work[1])>=1);assert(a[0]==2&&b[0]==3&&ta[1]==0&&tb[1]==0); \
   assert(w(layout,1,1,1,a,1,ta+1,b,1,tb+1,work+1,62)==0);assert(cabs((double complex)a[0])>0&&cabs((double complex)b[0])>0); \
   assert(work[0]==(T)779&&work[63]==(T)779&&ta[0]==(T)777&&ta[2]==(T)777&&tb[0]==(T)777&&tb[2]==(T)777); \
   a[0]=2;b[0]=3;assert(h(layout,1,1,1,a,1,ta+1,b,1,tb+1)==0); \
  } \
  assert(w(0,0,0,0,NULL,1,NULL,NULL,1,NULL,NULL,-1)==-1); \
  if(sizeof(J)==8){J bad=(J)INT64_C(-4294967293);T q=0;assert(w(102,bad,0,0,NULL,1,NULL,NULL,1,NULL,&q,-1)==-2);assert(diagnostic()==1);} \
  snprintf(hn,sizeof(hn),"%s%s",drivers[family],SUFFIX);snprintf(wn,sizeof(wn),"%s_work%s",drivers[family],SUFFIX); \
  driver_high dh=(driver_high)dlsym(library,hn);driver_work dw=(driver_work)dlsym(library,wn);assert(dh&&dw); \
  for(int layout=101;layout<=102;layout++)for(int high=0;high<2;high++){ \
   T a[3]={(T)777,(T)(2+.5*I),(T)777},b[3]={(T)777,(T)(3-.2*I),(T)777}; \
   T answer=(T)(1+.25*I),c[3]={(T)777,a[1]*answer,(T)777},d[3]={(T)777,family?0:b[1]*answer,(T)777},x[3]={(T)777,0,(T)777},work[64]; \
   for(int z=0;z<64;z++){work[z]=(T)779;} \
   assert(dw(layout,1,1,1,a+1,1,b+1,1,c+1,d+1,x+1,work+1,-1)==0);assert(creal((double complex)work[1])>=1&&x[1]==0); \
   J info=high?dh(layout,1,1,1,a+1,1,b+1,1,c+1,d+1,x+1):dw(layout,1,1,1,a+1,1,b+1,1,c+1,d+1,x+1,work+1,62);assert(info==0); \
   assert(cabs((double complex)(family?d[1]:x[1])-(double complex)answer)<TOL);if(family){assert(cabs((double complex)x[1])<TOL);} \
   assert(a[0]==(T)777&&a[2]==(T)777&&b[0]==(T)777&&b[2]==(T)777&&c[0]==(T)777&&c[2]==(T)777&&d[0]==(T)777&&d[2]==(T)777&&x[0]==(T)777&&x[2]==(T)777&&work[0]==(T)779&&work[63]==(T)779); \
  } \
  assert(dw(101,1,1,1,NULL,0,NULL,0,NULL,NULL,NULL,NULL,-1)==-6); \
  if(sizeof(J)==8){J bad=(J)INT64_C(-4294967293);T q=0;assert(dw(102,bad,0,0,NULL,1,NULL,1,NULL,NULL,NULL,&q,-1)==-2);assert(diagnostic()==1);} \
 } \
}while(0)
int main(int argc,char **argv){assert(argc==2);void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);assert(library);int (*diagnostic)(void)=(int (*)(void))dlsym(library,"colmat_blas_last_error");assert(diagnostic);
 CHECK(s,float,int32_t,"",2e-5);CHECK(d,double,int32_t,"",1e-12);CHECK(c,float complex,int32_t,"",2e-5);CHECK(z,double complex,int32_t,"",1e-12);
 CHECK(s,float,int64_t,"64_",2e-5);CHECK(d,double,int64_t,"64_",1e-12);CHECK(c,float complex,int64_t,"64_",2e-5);CHECK(z,double complex,int64_t,"64_",1e-12);
 puts("LAPACKE generalized QR C ABI: 32 names verified");dlclose(library);return 0;}
