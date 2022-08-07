#Fitting of coil directly to Ball-in-Box problem

rm(list=ls())

lrad <- function(bb){(Re(bb))}

pop_coil_full<-function(avec){
  rdim=dim(CoilVals)[1]
  RandVec=complex(rdim,avec[1:(rdim)],avec[(rdim+1):(rdim*2)])
  rotvals=avec[(rdim*2+1):(rdim*2+3)]*100
  rotvals=c(200,500,500)
  stmat=matrix(avec[(rdim*2+4):length(avec)],nrow=2)
  startvals=complex(n.s,stmat[1,],stmat[2,])
  runcoil(RandVec,rotvals,startvals)
}
initialize_swarm_full<-function(swarm_size,L,locfac=0.6,setrots=NULL){
  eval_fun<-eval_params
  lowlim<<-(0)

  x.p<<-matrix(runif(swarm_size*L,lowlim,1),nrow=swarm_size,ncol=L)
  vel<<-matrix(runif(swarm_size*L,-0.1,0.1),nrow=swarm_size,ncol=L)
  if (!is.null(setrots)){
    x.p[,c(1,2,3)]<<-t(matrix(setrots,ncol=dim(x.p)[1],nrow=3))
  }

  locality<<-locfac*swarm_size

  solvedcoil=apply(x.p,1,function(x)pop_coil_full(x)[[1]])

  outmat<<-t(solvedcoil)
  best_p_res<<-outmat
  x.p<<-x.p
  best_p<<-x.p
}

step_swarm_full<-function(swarm_size,L,w=0.9,g_p=0.4,g_g=0.4,setrots=NULL){
  old_perf=apply(outmat,1,function(x)optf(x,out_goal,slseq))
  print(min(old_perf))
  solvedcoil=apply(x.p,1,function(x)pop_coil_full(x)[[1]])

  outmat<<-t(solvedcoil)
  outmat[is.na(outmat)]<<-0
  outmat[abs(outmat)==Inf]<<-0

  n_v=matrix(runif(swarm_size*L,-0.01,0.01),nrow=swarm_size,ncol=L)
  r_p=matrix(runif(swarm_size*L,0,1),nrow=swarm_size,ncol=L)
  r_g=matrix(runif(swarm_size*L,0,1),nrow=swarm_size,ncol=L)

  cmat=t(apply(x.p,1,function(z) order(apply(x.p,1,function(y) sum(abs(y-z))))[-1][1:locality]))

  best_g_mat<<-t(apply(cmat,1,function(a) x.p[a,][which.min(apply(outmat[a,],1,function(x)optf(x,out_goal,slseq))),]))

  new.ind=which(apply(outmat,1,function(x)optf(x,out_goal,slseq))<apply(best_p_res,1,function(x)optf(x,out_goal,slseq)))

  best_p[new.ind,]<<-x.p[new.ind,]
  best_p_res[new.ind,]<<-outmat[new.ind,]

  vel<<-n_v+w*vel+g_p*r_p*(best_p-x.p)+g_g*r_g*(best_g_mat-x.p)

  x.p<<-x.p+vel
  x.p[x.p<(lowlim)]<<-lowlim
  x.p<<-x.p
  if (!is.null(setrots)){
    x.p[,c(1,2,3)]<<-t(matrix(setrots,ncol=dim(x.p)[1],nrow=3))
  }

  best_p<<-best_p
  best_p_res<<-best_p_res
}

library(neuralcoil)

#Load in Ball-in-Box data:
data("exampledata_ballinbox")
out_goal=as.numeric(exampledata_ballinbox)

#User Selections----
n.s=dim(exampledata_ballinbox)[2]#number of states
sym=F   #Parameter Symmetry
loc=F #Locality
cont=T #Parameter Physicality Controls
sub.num=1 #Number of conserved subgroups
vfara_inert=1 #inertia
vfara_init=1 #initial inertia
Tlen=dim(exampledata_ballinbox)[1] #Steps to run coil

buildcoil(n.s,sym=F)

#Number of needed parameters
L=dim(CoilVals)[1]*2+3+n.s*2

#Particle Swarm Optimizer
n.part=100#number of particles

w=0.9 #velocity factor
g_p=0.4#position influence factor
g_g=0.4#swarm influence factor

initialize_swarm_full(n.part,L)

for (itt in 1:1000){
  if (itt<400){
    slseq=array(matrix(seq(1:length(out_goal)),ncol=n.s)[1:round(Tlen/2),])
    #slseq=c(1:15,26:36,51:61,76:86)
  }else{
    slseq=1:length(out_goal)
  }

  step_swarm_full(n.part,L)
  matplot(t(outmat),col="grey",lty=1,type="l")
  lines(out_goal,col="blue")
  print(itt)
}

Pmat=pop_coil_full(best_p[which.min(apply(best_p_res,1,function(x)optf(x,out_goal,slseq))),])[[1]]
matplot(pop_coil_full(best_p[which.min(apply(best_p_res,1,function(x)optf(x,out_goal,slseq))),])[[1]],pch=1,type="l",xlim=c(0,(Tlen)),lwd=2,ylim=c(0,1),xlab="Time",ylab="P(s)")
#matplot(pop_coil_full(x.p[which.min(apply(outmat,1,function(x)optf(x,out_goal,slseq))),])[[1]],pch=1,type="l",xlim=c(0,(Tlen)),lwd=2,ylim=c(0,1),xlab="Time",ylab="P(s)")
aa=readRDS("results/ballinbox_weights.RdA")
rdim=dim(CoilVals)[1]
aa[(rdim*2+1):(rdim*2+3)]=c(200,500,500)
matplot(pop_coil_full(aa)[[1]],pch=1,type="l",xlim=c(0,(Tlen)),lwd=2,ylim=c(0,1),xlab="Time",ylab="P(s)")
matlines(exampledata_ballinbox,pch=1,type="l",xlim=c(0,(Tlen)),xlab="Time",ylab="P(s)")
