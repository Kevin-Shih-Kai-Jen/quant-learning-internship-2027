#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

namespace {
constexpr int D=13;
struct Workspace {
  int n; double tau;
  const double *x,*truth;
  std::vector<double> scores, direction, z, exponent, prob;
  Workspace(int n_,double tau_,const double *x_,const double *t_):n(n_),tau(tau_),x(x_),truth(t_),scores(n),direction(n),z(n),exponent(n),prob(n){}
  void dot(const double *theta,std::vector<double>& output){
    for(int j=0;j<n;++j){double sum=0;const double *row=x+j*D;for(int k=0;k<D;++k)sum+=row[k]*theta[k];output[j]=sum;}
  }
  void sigmoid(int i,double rate){
    for(int j=0;j<n;++j){z[j]=((scores[j]-scores[i])-rate*(direction[j]-direction[i]))/tau;exponent[j]=-std::abs(z[j]);}
#ifdef __APPLE__
    vvexp(prob.data(),exponent.data(),&n);
#else
    for(int j=0;j<n;++j)prob[j]=std::exp(exponent[j]);
#endif
    for(int j=0;j<n;++j){const double e=prob[j];prob[j]=(z[j]>=0)?1/(1+e):e/(1+e);}
  }
  double evaluate(const int *ids,int count,double rate,double *gradient){
    const double factor=1/(double(count)*double(n-1)*double(n-1));
    double loss=0;
    if(gradient)std::fill(gradient,gradient+D,0.);
    for(int q=0;q<count;++q){
      const int i=ids?ids[q]:q;sigmoid(i,rate);
      double rank=1;for(int j=0;j<n;++j)if(j!=i)rank+=prob[j];
      const double err=rank-truth[i];loss+=err*err*factor;
      if(gradient){
        double dr[D]={0};const double *xi=x+i*D;
        for(int j=0;j<n;++j)if(j!=i){
          const double weight=prob[j]*(1-prob[j])/tau;const double *xj=x+j*D;
          for(int k=1;k<D;++k)dr[k]+=weight*(xj[k]-xi[k]);
        }
        for(int k=1;k<D;++k)gradient[k]+=2*err*factor*dr[k];
      }
    }
    return loss;
  }
};
}

extern "C" double evaluate_rank_loss(int n,const double *x,const double *truth,const double *theta,const int *ids,int count,double tau,double *gradient){
  Workspace ws(n,tau,x,truth);ws.dot(theta,ws.scores);
  return ws.evaluate(ids,count,0.,gradient);
}

// Trace columns: batch index, batch size, before loss, after loss, eta,
// gradient infinity norm, gradient squared norm, candidates, status, hit upper eta.
// Additional columns: beta gradient, mean financial feature in batch, beta before.
// Status 0 accepted, 1 small gradient, 2 no accepted learning rate.
extern "C" int train_rank_day(int n,const double *x,const double *truth,double *theta,
    const int *order,int batch_size,double tau,double *trace,
    const int *snapshot_batches,int snapshot_count,double *snapshots,double *daily_losses){
  if(n<2||batch_size<1||!(tau>0))return -1;
  Workspace ws(n,tau,x,truth);ws.dot(theta,ws.scores);
  daily_losses[0]=ws.evaluate(nullptr,n,0.,nullptr);
  int b=0;
  for(int start=0;start<n;start+=batch_size,++b){
    const int count=std::min(batch_size,n-start);const int *ids=order+start;
    ws.dot(theta,ws.scores);std::fill(ws.direction.begin(),ws.direction.end(),0.);
    const double beta_before=theta[D-1];
    double gradient[D];const double old_loss=ws.evaluate(ids,count,0.,gradient);
    double norm2=0,inf=0;for(int k=0;k<D;++k){norm2+=gradient[k]*gradient[k];inf=std::max(inf,std::abs(gradient[k]));}
    if(!std::isfinite(old_loss)||!std::isfinite(norm2))return -2;
    for(int s=0;s<snapshot_count;++s)if(snapshot_batches[s]==b){
      std::copy(theta,theta+D,snapshots+s*(2*D));std::copy(gradient,gradient+D,snapshots+s*(2*D)+D);
    }
    double best_rate=0,best_loss=old_loss;int candidates=0,status=1;
    if(inf>1e-10){
      status=2;ws.dot(gradient,ws.direction);
      auto trial=[&](double rate){
        ++candidates;const double loss=ws.evaluate(ids,count,rate,nullptr);
        if(std::isfinite(loss)&&loss<old_loss&&loss<=old_loss-1e-4*rate*norm2&&loss<best_loss){
          best_rate=rate;best_loss=loss;return true;
        }
        return false;
      };
      if(trial(.1)){
        for(int k=2;k<=10;++k)if(!trial(double(k)/10.))break;
      }else{
        bool accepted=false;
        for(int exponent=-2;exponent>=-6&&!accepted;--exponent){
          const double scale=std::pow(10.,exponent);
          for(int digit=9;digit>=1;--digit)if(trial(digit*scale)){accepted=true;break;}
        }
      }
      if(best_rate>0){
        for(int k=0;k<D;++k)theta[k]-=best_rate*gradient[k];
        status=0;
      }
    }
    double *row=trace+13*b;
    row[0]=b;row[1]=count;row[2]=old_loss;row[3]=best_loss;row[4]=best_rate;
    row[5]=inf;row[6]=norm2;row[7]=candidates;row[8]=status;row[9]=best_rate==1.;
    row[10]=gradient[D-1];row[11]=0;for(int q=0;q<count;++q)row[11]+=x[ids[q]*D+D-1]/count;row[12]=beta_before;
  }
  ws.dot(theta,ws.scores);std::fill(ws.direction.begin(),ws.direction.end(),0.);
  daily_losses[1]=ws.evaluate(nullptr,n,0.,nullptr);
  for(int k=0;k<D;++k)if(!std::isfinite(theta[k]))return -3;
  return b;
}
