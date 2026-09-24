#include <algorithm>
#include <cmath>
namespace {
constexpr int D=18;
double dot(const double *x,const double *t){double v=0;for(int k=0;k<D;++k)v+=x[k]*t[k];return v;}
double mse(int n,const double *x,const double *y,const double *t){double v=0;for(int i=0;i<n;++i){double e=dot(x+i*D,t)-y[i];v+=e*e/n;}return v;}
}
extern "C" double evaluate_mse_loss(int n,const double *x,const double *y,const double *t,const int *ids,int count,double unused,double *grad){
  std::fill(grad,grad+D,0.);double loss=0;
  for(int q=0;q<count;++q){int i=ids[q];double e=dot(x+i*D,t)-y[i];loss+=e*e/count;for(int k=0;k<D;++k)grad[k]+=2*e*x[i*D+k]/count;}
  return loss;
}
extern "C" int train_mse_day(int n,const double *x,const double *y,double *t,const int *order,int batch_size,double unused,double *trace,const int *snapids,int ns,double *snaps,double *losses){
  if(n<1||batch_size!=1)return -1;
  losses[0]=mse(n,x,y,t);
  for(int b=0;b<n;++b){
    const int i=order[b];const double *xi=x+i*D;
    const double e=dot(xi,t)-y[i],old=e*e;
    double finance_before[6];for(int k=0;k<6;++k)finance_before[k]=t[12+k];
    double grad[D],norm2=0,inf=0;
    for(int k=0;k<D;++k){grad[k]=2*e*xi[k];norm2+=grad[k]*grad[k];inf=std::max(inf,std::abs(grad[k]));}
    if(!std::isfinite(old)||!std::isfinite(norm2))return -2;
    for(int s=0;s<ns;++s)if(snapids[s]==b){std::copy(t,t+D,snaps+s*2*D);std::copy(grad,grad+D,snaps+s*2*D+D);}
    double rate=0,best=old;int candidates=0,status=1;
    if(inf>1e-10){
      status=2;
      // Every candidate is evaluated at the actual proposed parameter vector,
      // from this batch's frozen pre-update state and gradient.
      auto trial=[&](double eta){
        ++candidates;double proposed[D];for(int k=0;k<D;++k)proposed[k]=t[k]-eta*grad[k];
        double err=dot(xi,proposed)-y[i],loss=err*err;
        if(std::isfinite(loss)&&loss<old&&loss<=old-1e-4*eta*norm2&&loss<best){rate=eta;best=loss;return true;}return false;
      };
      if(trial(.1)){for(int k=2;k<=10;++k)if(!trial(double(k)/10.))break;}
      else {bool ok=false;for(int exponent=-2;exponent>=-6&&!ok;--exponent){double scale=std::pow(10.,exponent);for(int digit=9;digit>=1;--digit)if(trial(digit*scale)){ok=true;break;}}}
      if(rate>0){for(int k=0;k<D;++k)t[k]-=rate*grad[k];status=0;}
    }
    double *r=trace+28*b;
    r[0]=b;r[1]=1;r[2]=old;r[3]=best;r[4]=rate;r[5]=inf;r[6]=norm2;r[7]=candidates;r[8]=status;r[9]=rate==1.;for(int k=0;k<6;++k){r[10+k]=grad[12+k];r[16+k]=xi[12+k];r[22+k]=finance_before[k];}
  }
  losses[1]=mse(n,x,y,t);
  for(int k=0;k<D;++k)if(!std::isfinite(t[k]))return -3;
  return n;
}
