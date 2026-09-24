#include <algorithm>
#include <cmath>
namespace {
constexpr int D=27;
void coefficients(const double*t,double*c){
  std::copy(t,t+12,c);
  for(int m=0;m<3;++m){int j=12+5*m,b=12+4*m;double d=t[24+m];
    c[j]=t[b];c[j+1]=-d*t[b];c[j+2]=d*t[b+1];c[j+3]=d*t[b+2];c[j+4]=d*t[b+3];}
}
double dot(const double*x,const double*c){double s=0;for(int j=0;j<D;++j)s+=x[j]*c[j];return s;}
double loss(int n,const double*x,const double*y,const double*t,const int*ids,int count,double*g){
  double c[D],raw[D]={};coefficients(t,c);double v=0;
  for(int z=0;z<count;++z){int i=ids?ids[z]:z;const double*xi=x+D*i;double e=dot(xi,c)-y[i];v+=e*e/count;
    if(g)for(int j=0;j<D;++j)raw[j]+=2*e*xi[j]/count;}
  if(g){std::copy(raw,raw+12,g);for(int m=0;m<3;++m){int j=12+5*m,b=12+4*m;double d=t[24+m];
    g[b]=raw[j]-d*raw[j+1];g[b+1]=d*raw[j+2];g[b+2]=d*raw[j+3];g[b+3]=d*raw[j+4];
    g[24+m]=-t[b]*raw[j+1]+t[b+1]*raw[j+2]+t[b+2]*raw[j+3]+t[b+3]*raw[j+4];}}
  return v;
}
}
extern "C" double evaluate_model(int n,const double*x,const double*y,const double*t,double*g){return loss(n,x,y,t,nullptr,n,g);}
extern "C" int train_day(int n,const double*x,const double*y,double*t,const int*order,int full,double*trace,double*snap,double*losses,double*checkpoints){
  losses[0]=loss(n,x,y,t,nullptr,n,nullptr);
  for(int z=0;z<n+full;++z){
    if(checkpoints && (z%8==0||z>=n)){int c=z<n?z/8:(n+7)/8+z-n;std::copy(t,t+D,checkpoints+c*D);}
    if(z==n)losses[1]=loss(n,x,y,t,nullptr,n,nullptr);
    const int*ids=z<n?order+z:nullptr;int count=z<n?1:n;
    for(int block=0;block<2;++block){
    double grad[D],old=loss(n,x,y,t,ids,count,grad),norm2=0,inf=0;
    // Alternating blocks avoid changing a coefficient and the discount that
    // multiplies it in the same proposed step. Both remain learned online.
    for(int j=0;j<D;++j)if((block==0&&j>=24)||(block==1&&j<24))grad[j]=0;
    for(int j=0;j<D;++j){norm2+=grad[j]*grad[j];inf=std::max(inf,std::abs(grad[j]));}
    if(!std::isfinite(old)||!std::isfinite(norm2))return -1;
    int si=-1;if(z==0)si=0;else if(z==n/2)si=1;else if(z==n-1)si=2;else if(z>=n)si=3+z-n;
    if(si>=0){double*s=snap+(si*2+block)*2*D;std::copy(t,t+D,s);std::copy(grad,grad+D,s+D);}
    double best=old,rate=0,bestt[D],direction=0;int candidates=0,status=1;
    if(inf>1e-10){
      status=2;
      auto trial=[&](double eta){
        ++candidates;double nt[D],descent=0;for(int j=0;j<D;++j){nt[j]=t[j]-eta*grad[j];if(j>=24)nt[j]=std::clamp(nt[j],0.,1.);descent+=grad[j]*(t[j]-nt[j]);}
        double value=loss(n,x,y,nt,ids,count,nullptr);
        if(std::isfinite(value)&&descent>0&&value<best&&value<old&&value<=old-1e-4*descent){best=value;rate=eta;direction=descent;std::copy(nt,nt+D,bestt);return true;}return false;
      };
      if(trial(.1)){for(int k=2;k<=10;++k)if(!trial(k/10.))break;}
      else {bool ok=false;for(int ex=-2;ex>=-6&&!ok;--ex){double scale=std::pow(10.,ex);for(int k=9;k>=1;--k)if(trial(k*scale)){ok=true;break;}}}
      if(rate>0){std::copy(bestt,bestt+D,t);status=0;}
    }
    double*r=trace+10*(2*z+block);r[0]=z<n?0:1;r[1]=count;r[2]=old;r[3]=best;r[4]=rate;r[5]=inf;r[6]=norm2;r[7]=candidates;r[8]=status;r[9]=direction;
    }
  }
  if(full==0)losses[1]=loss(n,x,y,t,nullptr,n,nullptr);
  losses[2]=loss(n,x,y,t,nullptr,n,nullptr);
  return n+full;
}
