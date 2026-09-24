#include <algorithm>
#include <cmath>
namespace {
double evaluate(int n,const double*x,const double*y,const double*t,int chosen,int block,double*g){
  std::fill(g,g+27,0.);double total=0;int count=chosen<0?n:1;
  for(int z=0;z<count;++z){int i=chosen<0?z:chosen;const double*r=x+27*i;double pred=0,jac[27]={};
    for(int j=0;j<12;++j){pred+=t[j]*r[j];jac[j]=r[j];}
    for(int m=0;m<3;++m){int b=12+4*m,f=12+5*m;double d=t[24+m];
      double actual=r[f]-d*r[f+1];
      pred+=t[b]*actual+d*(t[b+1]*r[f+2]+t[b+2]*r[f+3]+t[b+3]*r[f+4]);
      jac[b]=actual;for(int j=1;j<4;++j)jac[b+j]=d*r[f+j+1];
      jac[24+m]=-t[b]*r[f+1]+t[b+1]*r[f+2]+t[b+2]*r[f+3]+t[b+3]*r[f+4];
    }
    double err=pred-y[i];total+=err*err/count;
    for(int j=0;j<27;++j)if((block==0&&j<24)||(block==1&&j>=24))g[j]+=2*err*jac[j]/count;
  }
  return total;
}
double relative(double a,double b){return std::abs(a-b)/(1+std::abs(b));}
}
extern "C" int replay(int n,const double*x,const double*y,const int*order,int full,const double*trace,double*t,double*stats,double*losses){
  std::fill(stats,stats+5,0.);double g[27];losses[0]=evaluate(n,x,y,t,-1,0,g);
  for(int step=0;step<n+full;++step){
    if(step==n)losses[1]=evaluate(n,x,y,t,-1,0,g);
    for(int block=0;block<2;++block){
      const double*r=trace+10*(2*step+block);int chosen=step<n?order[step]:-1;
      double old=evaluate(n,x,y,t,chosen,block,g),norm=0,inf=0,descent=0;
      for(int j=0;j<27;++j){norm+=g[j]*g[j];inf=std::max(inf,std::abs(g[j]));double next=t[j]-r[4]*g[j];if(j>=24)next=std::clamp(next,0.,1.);descent+=g[j]*(t[j]-next);t[j]=next;}
      double unused[27];double after=evaluate(n,x,y,t,chosen,block,unused);
      stats[0]=std::max(stats[0],relative(old,r[2]));stats[1]=std::max(stats[1],relative(after,r[3]));
      stats[2]=std::max(stats[2],relative(inf,r[5]));stats[3]=std::max(stats[3],relative(norm,r[6]));
      stats[4]=std::max(stats[4],relative(descent,r[9]));
      if(!std::isfinite(after))return -1;
    }
  }
  if(full==0)losses[1]=evaluate(n,x,y,t,-1,0,g);
  losses[2]=evaluate(n,x,y,t,-1,0,g);return n+full;
}
extern "C" int replay_segment(int n,const double*x,const double*y,const int*order,int start,int end,const double*trace,double*t,double*stats){
  std::fill(stats,stats+5,0.);
  for(int step=start;step<end;++step)for(int block=0;block<2;++block){
    const double*r=trace+10*(2*step+block);int chosen=step<n?order[step]:-1;double g[27];
    double old=evaluate(n,x,y,t,chosen,block,g),norm=0,inf=0,descent=0;
    for(int j=0;j<27;++j){norm+=g[j]*g[j];inf=std::max(inf,std::abs(g[j]));double next=t[j]-r[4]*g[j];if(j>=24)next=std::clamp(next,0.,1.);descent+=g[j]*(t[j]-next);t[j]=next;}
    double unused[27];double after=evaluate(n,x,y,t,chosen,block,unused);
    stats[0]=std::max(stats[0],relative(old,r[2]));stats[1]=std::max(stats[1],relative(after,r[3]));
    stats[2]=std::max(stats[2],relative(inf,r[5]));stats[3]=std::max(stats[3],relative(norm,r[6]));stats[4]=std::max(stats[4],relative(descent,r[9]));
    if(!std::isfinite(after))return -1;
  }
  return end-start;
}
