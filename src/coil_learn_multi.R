#Test rotation value concept

rm(list=ls())

#Source Buoy Data
library(forecastML)

exdat<-function(lookback,lookforward,clean,predvar){
  dfr=c()
  isy=lookback+1
  while(isy<dim(clean)[1]){
    dfr=rbind(dfr,c(unlist(clean[(isy-lookback):(isy-includecurrent),]),clean[isy:(isy+lookforward-1),predvar]))
    isy=isy+1  
  }
  colnames(dfr)=c(as.vector(t(outer(paste0("var",seq(1:(dim(clean)[2]))),paste0("[t-",lookback:includecurrent,"]"),paste0))),paste0("var",predvar,paste0("[t+",0:(lookforward-1),"]")))
  return(dfr)
}

data("data_buoy",package="forecastML")

features<-c("wind_spd","air_temperature","sea_surface_temperature")

#Pull in subset of raw data
raw_dat<-data_buoy[features][1:4000,]
raw_dat=raw_dat[rowSums(is.na(raw_dat))==0,]

#Establish look ranges
lookback=10 #How many previous steps to use?
includecurrent=1
lookforward=30 #How long to predict?
predval=3 #Which column are we using?

scdat=as.data.frame(apply(raw_dat,2,function(x)(x-min(x,na.rm = T))/(max(x,na.rm = T)-min(x,na.rm = T))))
#scdat[,predval]=(scdat[,predval]-mean(scdat[,predval]))/1.5+mean(scdat[,predval])

datex<-exdat(lookback,lookforward,scdat,predval)
#drop missing
datex=datex[rowSums(datex==0)==0,]

Yall<-datex[,(dim(datex)[2]-lookforward+includecurrent):dim(datex)[2]]
Xall<-datex[,-((dim(datex)[2]-lookforward+includecurrent):dim(datex)[2])]

#Select points
selps=sample(1:1000,1,replace = F)
#selps=c(630,629,775,801,30,525)
selps=c(30,237,630,123,41,82)

marketvals=list()
for (iLL in 1:length(selps)){
  xin=t(matrix(Xall[selps[iLL],],ncol=length(features)))
  #xin=t(apply(xin,1,function(x)x/sum(x)))
  yin=as.numeric(Yall[selps[iLL],]) 
  
  #Predict modified percent change
  scale_offset=0.5
  yin=yin/(xin[predval,lookback]+scale_offset)

  marketvals[[iLL]]=list(xin,yin)
}
min.len=min(unlist(lapply(marketvals,function(x)length(x[[2]]))))

#User Selections----
n.s=4#number of states
sym=F   #Parameter Symmetry
loc=F #Locality
cont=T #Parameter Physicality Controls
sub.num=1 #Number of conserved subgroups
vfara_inert=min.len/2 #inertia
vfara_init=1 #initial inertia
Tlen=min.len #Steps to run coil
loadvals=T #Load in previously learned values?

#Generate Coil According to User specifications
source("src/complex_coil_gen.R")

buildcoil(n.s,sym=F)

#Number of needed parameters
L=dim(CoilVals)[1]*2+3*length(selps)+n.s*2*length(selps)
label=c(rep("randvec",dim(CoilVals)[1]*2),
        apply(expand.grid(rep("rots",3),1:length(selps)),1,function(x)paste(x,collapse="")),
        apply(expand.grid(rep("starts",n.s*2),1:length(selps)),1,function(x)paste(x,collapse="")))

#Learn coil parameters
heteroscedastic_loss=function(y_true,pred_mean,pred_logvar){
  sum(((y_true-pred_mean)^2/(2*exp(pred_logvar))+abs(pred_logvar)))/length(y_true)
}

out_goal=matrix(do.call(c,lapply(marketvals,function(x)x[[2]][1:min.len])),ncol=1)

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
  outfull=c()
  for (iS in 1:length(selps)){
    rdim=dim(CoilVals)[1]
    RandVec=complex(rdim,aa[1:(rdim)],aa[(rdim+1):(rdim*2)])
    rotvals=aa[label==paste0("rots",iS)]*10
    stmat=matrix(aa[label==paste0("starts",iS)],nrow=2)
    startvals=complex(n.s,stmat[1,],stmat[2,])
    outfull=c(outfull,(runcoil(RandVec,rotvals,startvals)[[1]][,1]))
  }
  return(outfull)
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
  upL=1:length(out_goal)
  
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

#saveRDS(best_p[which.min(apply(best_p_res,1,optf)),],file="results/buoymulti.RdA")
matplot(pop_coil(best_p[which.min(apply(best_p_res,1,optf)),]),pch=1,type="l",lwd=2,ylim=c(0,1),xlab="Time",ylab="P(s)")
lines(out_goal,col="blue")
