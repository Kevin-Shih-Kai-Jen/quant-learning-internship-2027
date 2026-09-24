#include <algorithm>
#include <cmath>
#include <vector>
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
namespace {
constexpr int D=27;
struct Audit {
 int n;const double*x,*truth;std::vector<double>s,exps,jac,err,weights,gs;bool regular;
 Audit(int n_,const double*x_,const double*y_):n(n_),x(x_),truth(y_),s(n),exps(n),jac(size_t(n)*D),err(n),weights(size_t(n)*n),gs(n){}
 void state(const double*t){for(int i=0;i<n;++i){const double*r=x+i*D;double*j=jac.data()+i*D;std::copy(r,r+12,j);j[0]=0;double score=0;
   for(int k=0;k<12;++k)score+=t[k]*r[k];
   for(int m=0;m<3;++m){int b=12+4*m,f=12+5*m;double d=t[24+m];j[b]=r[f]-d*r[f+1];for(int k=1;k<4;++k)j[b+k]=d*r[f+k+1];
     j[24+m]=-t[b]*r[f+1]+t[b+1]*r[f+2]+t[b+2]*r[f+3]+t[b+3]*r[f+4];
     for(int k=0;k<4;++k)score+=t[b+k]*j[b+k];}s[i]=score;}
   auto mm=std::minmax_element(s.begin(),s.end());regular=std::isfinite(*mm.first)&&std::isfinite(*mm.second)&&*mm.second-*mm.first<600;
   if(regular){double mid=.5**mm.first+.5**mm.second;for(int i=0;i<n;++i)exps[i]=s[i]-mid;vvexp(exps.data(),exps.data(),&n);}}
 double p(int i,int j){if(regular)return exps[j]/(exps[i]+exps[j]);double z=s[j]-s[i];if(z>=0)return 1/(1+std::exp(-z));double e=std::exp(z);return e/(1+e);}
 struct Job{Audit*a;int phase;bool grad;};
 static void worker(void*ctx,size_t id){Job&job=*static_cast<Job*>(ctx);Audit&a=*job.a;int begin=int(id)*a.n/8,end=int(id+1)*a.n/8;
   if(job.phase==0)for(int i=begin;i<end;++i){double rank=1;for(int j=0;j<a.n;++j)if(i!=j){double p=a.p(i,j);rank+=p;if(job.grad)a.weights[size_t(i)*a.n+j]=p*(1-p);}a.err[i]=(rank-a.truth[i])/(a.n-1);}
   else for(int i=begin;i<end;++i){double v=0;for(int j=0;j<a.n;++j)if(j!=i)v+=a.weights[size_t(i)*a.n+j]*(a.err[j]-a.err[i]);a.gs[i]=2*v/(a.n*(a.n-1.));}}
 double value(const double*t,int chosen,int block,double*g){state(t);double loss=0;
   if(chosen<0){Job job{this,0,g!=nullptr};dispatch_apply_f(8,dispatch_get_global_queue(0,0),&job,worker);for(double e:err)loss+=e*e/n;if(g){job.phase=1;dispatch_apply_f(8,dispatch_get_global_queue(0,0),&job,worker);}}
   else{double rank=1,sum=0;for(int j=0;j<n;++j)if(j!=chosen){double q=p(chosen,j);rank+=q;if(g){gs[j]=q*(1-q);sum+=gs[j];}}double e=(rank-truth[chosen])/(n-1.);loss=e*e;if(g){gs[chosen]=-sum;for(double&v:gs)v*=2*e/(n-1.);}}
   if(g){std::fill(g,g+D,0.);for(int i=0;i<n;++i)for(int k=0;k<D;++k)g[k]+=gs[i]*jac[i*D+k];if(block==0)std::fill(g+24,g+D,0.);else std::fill(g,g+24,0.);}return loss;}
};
void check(double got,double expected,double&maximum){maximum=std::max(maximum,std::abs(got-expected)/(1+std::abs(expected)));}
}
extern "C" int replay_day(int n,const double*x,const double*y,const int*order,int full,const double*trace,const double*snaps,double*t,double*stats,double*losses){
 Audit a(n,x,y);losses[0]=a.value(t,-1,0,nullptr);
 for(int step=0;step<n+full;++step){if(step==n)losses[1]=a.value(t,-1,0,nullptr);int chosen=step<n?order[step]:-1;
   for(int block=0;block<2;++block){int si=step==0?0:step==n/2?1:step==n-1?2:step>=n?3+step-n:-1;
     if(si>=0){const double*saved=snaps+(2*si+block)*54;for(int k=0;k<D;++k)check(t[k],saved[k],stats[5]);std::copy(saved,saved+D,t);}
     double g[D],old=a.value(t,chosen,block,g),norm=0,inf=0;for(double v:g){norm+=v*v;inf=std::max(inf,std::abs(v));}
     const double*r=trace+10*(2*step+block);check(old,r[2],stats[0]);check(inf,r[5],stats[2]);check(norm,r[6],stats[3]);
     if(si>=0){const double*saved=snaps+(2*si+block)*54+D;for(int k=0;k<D;++k)check(g[k],saved[k],stats[6]);}
     double nt[D],descent=0;for(int k=0;k<D;++k){nt[k]=t[k]-r[4]*g[k];if(k>=24)nt[k]=std::clamp(nt[k],0.,1.);descent+=g[k]*(t[k]-nt[k]);}
     double after=r[4]>0?a.value(nt,chosen,block,nullptr):old;check(after,r[3],stats[1]);check(descent,r[9],stats[4]);
     if(!std::isfinite(old)||!std::isfinite(after)||!std::isfinite(norm))return -1;
     std::copy(nt,nt+D,t);
   }
 }
 losses[2]=a.value(t,-1,0,nullptr);if(full==0)losses[1]=losses[2];return n+full;
}
