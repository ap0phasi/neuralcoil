#Fitting of coil directly to Ball-in-Box problem

rm(list=ls())

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

initialize_swarm_full(n.part,L,setrots=c(2,5,5))

for (itt in 1:1000){
  if (itt<400){
    slseq=array(matrix(seq(1:length(out_goal)),ncol=n.s)[1:round(Tlen/2),])
    #slseq=c(1:15,26:36,51:61,76:86)
  }else{
    slseq=1:length(out_goal)
  }
  #slseq=1:length(out_goal)

  step_swarm_full(n.part,L,setrots=c(2,5,5))
  matplot(t(outmat),col="grey",lty=1,type="l")
  lines(out_goal,col="blue")
  print(itt)
}

Pmat=pop_coil_full(best_p[which.min(apply(best_p_res,1,function(x)optf(x,out_goal,slseq))),])[[1]]
matplot(pop_coil_full(best_p[which.min(apply(best_p_res,1,function(x)optf(x,out_goal,slseq))),])[[1]],pch=1,type="l",xlim=c(0,(Tlen)),lwd=2,ylim=c(0,1),xlab="Time",ylab="P(s)")
#matplot(pop_coil_full(x.p[which.min(apply(outmat,1,function(x)optf(x,out_goal,slseq))),])[[1]],pch=1,type="l",xlim=c(0,(Tlen)),lwd=2,ylim=c(0,1),xlab="Time",ylab="P(s)")
matlines(exampledata_ballinbox,pch=1,type="l",xlim=c(0,(Tlen)),xlab="Time",ylab="P(s)")
aa=readRDS("results/ballinbox_weights3.RdA")
rdim=dim(CoilVals)[1]
aa[(rdim*2+1):(rdim*2+3)]=c(2,5,5)
matplot(pop_coil_full(aa)[[1]],pch=1,type="l",xlim=c(0,(Tlen)),lwd=2,ylim=c(0,1),xlab="Time",ylab="P(s)")
matlines(exampledata_ballinbox,pch=1,type="l",xlim=c(0,(Tlen)),xlab="Time",ylab="P(s)")

#Demonstrate Chaos
# bb=aa
# bb[(rdim*2+1):(rdim*2+3)]=c(2,5.00001,5)
# matplot(pop_coil_full(bb)[[1]],pch=1,type="l",xlim=c(0,(Tlen)),lwd=2,ylim=c(0,1),xlab="Time",ylab="P(s)")
# matlines(exampledata_ballinbox,pch=1,type="l",xlim=c(0,(Tlen)),xlab="Time",ylab="P(s)")

par(mfrow = c(2, 2))
for (iP in 1:dim(Pmat)[2]){

  plot(exampledata_ballinbox[,iP],type="l")
  lines(Pmat[,iP],lty=2)
}

par(mfrow = c(1,1))

#saveRDS(file="results/ballinbox_weights4.RdA",best_p[which.min(apply(best_p_res,1,function(x)optf(x,out_goal,slseq))),])
