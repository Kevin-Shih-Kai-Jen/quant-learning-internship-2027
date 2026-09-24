#include <algorithm>
#include <cmath>
#include <vector>
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <arm_neon.h>
namespace {
constexpr int D=27;
void coefficients(const double*t,double*c){std::copy(t,t+12,c);for(int m=0;m<3;++m){int f=12+5*m,b=12+4*m;double d=t[24+m];c[f]=t[b];c[f+1]=-d*t[b];for(int j=1;j<4;++j)c[f+j+1]=d*t[b+j];}}
struct Work {
 int n,workers;const double*x,*truth;double tau;
 std::vector<double> scores,proposed,exps,ranks,errors,weights,scoregrad,dir,jd,dpred;
 bool regular;
 Work(int n_,const double*x_,const double*y_,double tau_,int threads):n(n_),workers(threads),x(x_),truth(y_),tau(tau_),scores(n),proposed(n),exps(n),ranks(n),errors(n),weights(size_t(n)*n),scoregrad(n),dir(n),jd(3*n),dpred(n){}
 void score(const double*t){double c[D];coefficients(t,c);for(int i=0;i<n;++i){const double*r=x+i*D;double s=0;for(int j=0;j<D;++j)s+=r[j]*c[j];scores[i]=s;for(int m=0;m<3;++m){int b=12+4*m,f=12+5*m;jd[3*i+m]=-t[b]*r[f+1]+t[b+1]*r[f+2]+t[b+2]*r[f+3]+t[b+3]*r[f+4];}}}
 void prepare(const double*s){auto mm=std::minmax_element(s,s+n);regular=std::isfinite(*mm.first)&&std::isfinite(*mm.second)&&(*mm.second-*mm.first)/tau<600;
   if(regular){double center=(*mm.first+*mm.second)/2;for(int j=0;j<n;++j)exps[j]=(s[j]-center)/tau;vvexp(exps.data(),exps.data(),&n);}}
 double probability(const double*s,int i,int j)const {if(regular)return exps[j]/(exps[i]+exps[j]);double z=(s[j]-s[i])/tau,e=std::exp(-std::abs(z));return z>=0?1/(1+e):e/(1+e);}
 struct Job{Work*w;const double*s;bool grad;int phase;};
 static void worker(void*ctx,size_t worker){Job&job=*static_cast<Job*>(ctx);Work&w=*job.w;
   const int begin=int(worker)*w.n/w.workers,end=int(worker+1)*w.n/w.workers;
   if(job.phase==0)for(int i=begin;i<end;++i){double rank=1;
     if(w.regular){const auto own=vdupq_n_f64(w.exps[i]);int j=0;double ps[2];
       for(;j+1<w.n;j+=2){auto other=vld1q_f64(w.exps.data()+j);auto probability=vdivq_f64(other,vaddq_f64(own,other));vst1q_f64(ps,probability);
         for(int k=0;k<2;++k)if(j+k!=i){double p=ps[k];rank+=p;if(job.grad)w.weights[size_t(i)*w.n+j+k]=p*(1-p)/w.tau;}}
       if(j<w.n&&j!=i){double p=w.probability(job.s,i,j);rank+=p;if(job.grad)w.weights[size_t(i)*w.n+j]=p*(1-p)/w.tau;}
     }else for(int j=0;j<w.n;++j)if(i!=j){double p=w.probability(job.s,i,j);rank+=p;if(job.grad)w.weights[size_t(i)*w.n+j]=p*(1-p)/w.tau;}
     w.ranks[i]=rank;w.errors[i]=rank-w.truth[i];}
   else for(int i=begin;i<end;++i){double sum=0;const double*wi=w.weights.data()+size_t(i)*w.n;for(int j=0;j<w.n;++j)if(j!=i)sum+=wi[j]*(w.errors[j]-w.errors[i]);w.scoregrad[i]=2*sum/(double(w.n)*(w.n-1)*(w.n-1));}
 }
 void parallel(Job&job){if(workers==1)worker(&job,0);else dispatch_apply_f(workers,dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT,0),&job,worker);}
 double evaluate(const double*s,int chosen,double*g,const double*t,int block){
   prepare(s);double result=0;const double denom=double(n-1)*(n-1);
   if(chosen<0){Job a{this,s,g!=nullptr,0};parallel(a);for(int i=0;i<n;++i)result+=errors[i]*errors[i]/(n*denom);if(g){a.phase=1;parallel(a);}}
   else {int i=chosen;double rank=1,own=0;for(int j=0;j<n;++j)if(j!=i){double p=probability(s,i,j);rank+=p;if(g){scoregrad[j]=p*(1-p)/tau;own+=scoregrad[j];}}double err=rank-truth[i];result=err*err/denom;if(g){scoregrad[i]=-own;for(int j=0;j<n;++j)scoregrad[j]*=2*err/denom;}}
   if(g){double raw[D]={};for(int i=0;i<n;++i)for(int j=1;j<D;++j)raw[j]+=scoregrad[i]*x[i*D+j];std::fill(g,g+D,0.);
     if(block!=1){for(int j=1;j<12;++j)g[j]=raw[j];for(int m=0;m<3;++m){int f=12+5*m,b=12+4*m;double d=t[24+m];g[b]=raw[f]-d*raw[f+1];for(int j=1;j<4;++j)g[b+j]=d*raw[f+j+1];}}
     if(block!=0)for(int m=0;m<3;++m){int f=12+5*m,b=12+4*m;g[24+m]=-t[b]*raw[f+1]+t[b+1]*raw[f+2]+t[b+2]*raw[f+3]+t[b+3]*raw[f+4];}
   }
   return result;
 }
};
}
extern "C" double evaluate_rank(int n,const double*x,const double*y,const double*t,int chosen,int block,double tau,double*g,double*ranks,int workers){
 Work w(n,x,y,tau,workers);w.score(t);double loss=w.evaluate(w.scores.data(),chosen,g,t,block);if(ranks&&chosen<0)std::copy(w.ranks.begin(),w.ranks.end(),ranks);return loss;
}
extern "C" int train_day(int n,const double*x,const double*y,double*t,const int*order,int full,double tau,double*trace,double*snaps,double*losses,int workers){
 Work w(n,x,y,tau,workers);w.score(t);losses[0]=w.evaluate(w.scores.data(),-1,nullptr,t,0);
 for(int step=0;step<n+full;++step){
   if(step==n){w.score(t);losses[1]=w.evaluate(w.scores.data(),-1,nullptr,t,0);}
   int chosen=step<n?order[step]:-1;
   for(int block=0;block<2;++block){
     w.score(t);double g[D],old=w.evaluate(w.scores.data(),chosen,g,t,block),norm=0,inf=0;
     for(int j=0;j<D;++j){norm+=g[j]*g[j];inf=std::max(inf,std::abs(g[j]));}
     if(!std::isfinite(old)||!std::isfinite(norm))return -1;
     int si=step==0?0:step==n/2?1:step==n-1?2:step>=n?3+step-n:-1;
     if(si>=0){double*s=snaps+(2*si+block)*54;std::copy(t,t+D,s);std::copy(g,g+D,s+D);}
     double best=old,rate=0,direction=0,nt[D],bestt[D];int count=0,status=1;
     if(inf>1e-10){
       status=2;
       if(block==0){double c[D];coefficients(g,c); // replace coefficients: discounts in g are zero
         for(int m=0;m<3;++m){int f=12+5*m,b=12+4*m;c[f]=g[b];c[f+1]=-t[24+m]*g[b];for(int j=1;j<4;++j)c[f+j+1]=t[24+m]*g[b+j];}
         for(int i=0;i<n;++i){double s=0;for(int j=0;j<D;++j)s+=x[i*D+j]*c[j];w.dir[i]=s;}}
       auto trial=[&](double eta){++count;double descent=0;for(int j=0;j<D;++j){nt[j]=t[j]-eta*g[j];if(j>=24)nt[j]=std::clamp(nt[j],0.,1.);descent+=g[j]*(t[j]-nt[j]);}
         for(int i=0;i<n;++i){double s=w.scores[i];if(block==0)s-=eta*w.dir[i];else for(int m=0;m<3;++m)s+=w.jd[3*i+m]*(nt[24+m]-t[24+m]);w.proposed[i]=s;}
         double value=w.evaluate(w.proposed.data(),chosen,nullptr,t,block);
         if(std::isfinite(value)&&descent>0&&value<best&&value<old&&value<=old-1e-4*descent){best=value;rate=eta;direction=descent;std::copy(nt,nt+D,bestt);return true;}return false;};
       if(trial(.1)){for(int k=2;k<=10;++k)if(!trial(k/10.))break;}
       else {bool ok=false;for(int ex=-2;ex>=-6&&!ok;--ex){double scale=std::pow(10.,ex);for(int k=9;k>=1;--k)if(trial(k*scale)){ok=true;break;}}}
       if(rate>0){std::copy(bestt,bestt+D,t);status=0;}
     }
     double*r=trace+10*(2*step+block);r[0]=step<n?0:1;r[1]=step<n?1:n;r[2]=old;r[3]=best;r[4]=rate;r[5]=inf;r[6]=norm;r[7]=count;r[8]=status;r[9]=direction;
   }
 }
 w.score(t);losses[2]=w.evaluate(w.scores.data(),-1,nullptr,t,0);if(full==0)losses[1]=losses[2];return n+full;
}
