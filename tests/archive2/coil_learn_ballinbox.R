#Fitting of coil directly to Ball-in-Box problem

rm(list=ls())

library(neuralcoil)

#Load in Ball-in-Box data:
data("exampledata_ballinbox")
rawdat<-exampledata_ballinbox
out_goal=as.numeric(rawdat)

#User Selections----
n.s=dim(rawdat)[2]#number of states
sym=F   #Parameter Symmetry
loc=F #Locality
cont=T #Parameter Physicality Controls
sub.num=1 #Number of conserved subgroups
vfara_inert=1 #inertia
vfara_init=1 #initial inertia
Tlen=dim(rawdat)[1] #Steps to run coil

buildcoil(n.s,sym=F)

#Number of needed parameters
L=dim(CoilVals)[1]*2+3+n.s*2

#Learn coil parameters
heteroscedastic_loss=function(y_true,pred_mean,pred_logvar){
  sum(((y_true-pred_mean)^2/(2*exp(pred_logvar))+abs(pred_logvar)))/length(y_true)
}

#Particle Swarm Optimizer
n=100#number of particles

w=0.9 #velocity factor
g_p=0.4#position influence factor
g_g=0.4#swarm influence factor

x.p=matrix(runif(n*L,0,1),nrow=n,ncol=L)
vel=matrix(runif(n*L,-0.1,0.1),nrow=n,ncol=L)
x.p[(x.p)<0]=0
best_p=x.p

pop_coil<-function(aa){
  rdim=dim(CoilVals)[1]
  RandVec=complex(rdim,aa[1:(rdim)],aa[(rdim+1):(rdim*2)])
  rotvals=aa[(rdim*2+1):(rdim*2+3)]*100
  stmat=matrix(aa[(rdim*2+4):length(aa)],nrow=2)
  startvals=complex(n.s,stmat[1,],stmat[2,])
  runcoil(RandVec,rotvals,startvals)[[1]]
}

solvedcoil=apply(x.p,1,pop_coil)

outmat=t(solvedcoil)
outmat[is.na(outmat)]=0
outmat[abs(outmat)==Inf]=0
outvar=apply(outmat,2,var)
outmean=apply(outmat,2,mean)

het.loss=heteroscedastic_loss(out_goal,outmean,log(outvar))
het.save=list(het.loss=het.loss,outmean=outmean,outvar=outvar)

best_p_res=outmat

locality=0.6*n

optf=function(outs){mean(abs(outs[upL]-out_goal[upL]))}


for (iMPSO in 1:1000){
  if (iMPSO<400){
    upL=c(1:15,26:36,51:61,76:86)
  }else{
    upL=1:length(out_goal)
  }

  old_perf=apply(outmat,1,optf)

  print(min(old_perf))


  solvedcoil=apply(x.p,1,pop_coil)

  outmat=t(solvedcoil)
  outmat[is.na(outmat)]=0
  outmat[abs(outmat)==Inf]=0
  outvar=apply(outmat,2,var)
  outmean=apply(outmat,2,mean)



  repulse.factor=0

  if (sum(repulse.factor)>0){

    repulse.factor=repulse.factor/sum(repulse.factor)*(het.loss)*(1+2*w)*1e4/n/dim(x.p)[2]
  }

  n_v=matrix(runif(n*L,-0.01,0.01),nrow=n,ncol=L)
  r_p=matrix(runif(n*L,0,1),nrow=n,ncol=L)
  r_g=matrix(runif(n*L,0,1),nrow=n,ncol=L)

  cmat=t(apply(x.p,1,function(z) order(apply(x.p,1,function(y) sum(abs(y-z))))[-1][1:locality]))

  closest.neighbor=x.p[cmat[,1],]

  best_g_mat=t(apply(cmat,1,function(a) x.p[a,][which.min(apply(outmat[a,],1,optf)),]))
  new.ind=which(apply(outmat,1,optf)<apply(best_p_res,1,optf))

  best_p[new.ind,]=x.p[new.ind,]
  best_p_res[new.ind,]=outmat[new.ind,]

  vel=n_v+w*vel+g_p*r_p*(best_p-x.p)+g_g*r_g*(best_g_mat-x.p)-sweep((x.p-closest.neighbor),2,repulse.factor,"*")

  x.p=x.p+vel
  x.p[(x.p)<0]=0


  matplot(t(outmat),col="grey",lty=1,type="l")
  lines(out_goal,col="blue")

}

matplot(pop_coil(best_p[which.min(apply(best_p_res,1,optf)),]),pch=1,type="l",xlim=c(0,(Tlen)),lwd=2,ylim=c(0,1),xlab="Time",ylab="P(s)")
matlines(rawdat,pch=1,type="l",xlim=c(0,(Tlen)),xlab="Time",ylab="P(s)")

#Demonstrate Sensitivity
rdim=dim(CoilVals)[1]
aa=best_p[which.min(apply(best_p_res,1,optf)),]
aa[(rdim*2+1):(rdim*2+3)]=aa[(rdim*2+1):(rdim*2+3)]+runif(3,-0.01,0.01)
aa[(rdim*2+4):length(aa)]=aa[(rdim*2+4):length(aa)]+runif(aa[(rdim*2+4):length(aa)],-0.4,0.4)
matplot(pop_coil(aa),pch=1,type="l",xlim=c(0,(Tlen)),lwd=2,ylim=c(0,1),xlab="Time",ylab="P(s)")
matlines(rawdat,pch=1,type="l",xlim=c(0,(Tlen)),xlab="Time",ylab="P(s)")
