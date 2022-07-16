##Full Machine Learning Coil Stack
rm(list=ls())

#Establish Libraries
library(tensorflow)
library(tfautograph)
library(keras)

scaledata<-function(x,range){
  scaled=((x-min(x))/(max(x)-min(x)))*(range[2]-range[1])+range[1]
  return(list(scaled=scaled,scaler=c(min(x),max(x))))
}

invertscaling<-function(x,scaler,range){
  inverted=((x-range[1])/(range[2]-range[1]))*(scaler[2]-scaler[1])+scaler[1]
  return(inverted)
}

lookwindow<-function(df,lookback,lookforward,predictors,objective){
  lfull<-list()
  listtemp<-list()
  for (ip in c(predictors,objective)){
    ptemp=c()
    for (iL in lookback:1){
      ptemp=cbind(ptemp,df[[ip]][(lookback-iL+1):(length(df[[ip]])-iL)])
    }
    listtemp[[paste(ip)]]=ptemp
  }
  lfull[["predictors"]]=listtemp
  
  listtemp<-list()
  for (io in objective){
    ptemp=c()
    for (iL in lookforward:1){
      ptemp=cbind(ptemp,df[[io]][(lookback+lookforward-iL+1):(length(df[[io]])-iL-lookback)])
    }
    listtemp[[paste(io)]]=ptemp
  }
  lfull[["objective"]]=listtemp
  return(lfull)
}

scalelist<-function(dfw,predrange=c(0,1),objrange=c(0.2,0.6)){
  #Scale list and give dummy data head
  scalesaves=c()
  for (idd in 1:length(dfw$predictors)){
    scaled=scaledata(dfw$predictors[[idd]],predrange)
    dfw$predictors[[idd]]=scaled$scaled
    scalesaves=rbind(scalesaves,scaled$scaler)
  }
  
  for (idd in 1:length(dfw$objective)){
    scaled=scaledata(dfw$objective[[idd]],objrange)
    dfw$objective[[idd]]=scaled$scaled
    scalesaves=rbind(scalesaves,scaled$scaler)
  }
  
  dfw$dummy=matrix(1,nrow=dim(dfw$predictors[[1]])[1],ncol=2)
  return(list(scaledlist=dfw,scalesaves=scalesaves))
}

#Generate Coil According to User specifications
source("src/complex_coil_gen.R")

autogen_cnn<-function(dfs,n.s,params){
  visl<-list()
  cnnl<-list()
  for (ipp in 1:length(dfs$predictors)){
    visl[[ipp]]<-layer_input(shape=c(dim(dfs$predictors[[ipp]])[2],1))
    cnnl[[ipp]]<-visl[[ipp]]%>%
      layer_conv_1d(filters=64,kernel_size=2,input_shape=c(dim(dfs$predictors[[ipp]])[2],1),activation="relu")%>%
      layer_max_pooling_1d(pool_size=2)%>%
      layer_flatten()
  }
  merge<-layer_concatenate(cnnl)%>%
    layer_dense(units=32,activation="relu")%>%
    layer_dense(units=16,activation="relu")
  
  rots<-merge%>%layer_dense(units=3,activation = "linear")
  
  starts<-merge%>%layer_dense(units=n.s*2)%>%layer_activation_leaky_relu(0.01) 
  
  visl[[ipp+1]]<-layer_input(shape=c(dim(dfs$dummy)[2]))
  dumw<-visl[[ipp+1]]%>%layer_dense(units=params*2)%>%layer_activation_leaky_relu(1e-4) 
  
  output <- layer_concatenate(list(rots,starts,dumw))
  
  model<-keras_model(visl,output)
  
  return(model)
}


assign_weights<-function(weights,weightdim,avec){
  weightsnew=weights
  stindx=1
  for (iw in 1:length(weightdim)){
    len=prod(weightdim[[iw]])
    weightsnew[[iw]]=array(avec[stindx:(len+stindx-1)],dim=weightdim[[iw]])
    stindx=len+stindx
  }
  set_weights(model,weightsnew)
}

pop_coil<-function(input,readout=F){
  model(input, training = TRUE)
  rdim<-dim(CoilVals)[1]
  val_out=model(input, training = TRUE)
  cnn_outputs <- as.array(val_out)
  rots<-abs(cnn_outputs[1:3])*10
  stmat<-(matrix(cnn_outputs[(4):(4+n.s*2-1)],nrow=2))
  startvals<-complex(n.s,stmat[1,],stmat[2,])/10
  randmat<-matrix(cnn_outputs[(4+n.s*2):length(cnn_outputs)],nrow=2)
  RandVec<-complex(rdim,randmat[1,],randmat[2,])
  coil_out<-(runcoil(RandVec,rots,startvals))
  if (readout){
    print(rots)
    print(startvals)
  }
  return(coil_out)
}

lossfun<-function(actual,predicted){
  mean((actual-predicted)^2)
}

eval_weights<-function(avec,inputlist,outputs){
  assign_weights(weights,weightdim,avec)
  errors=c()
  for (iii in 1:length(inputlist)){
    inputs=inputlist[[iii]]
    coil_out=pop_coil(inputs)
    errors=c(errors,lossfun(outputs[iii,],coil_out[[1]][,1]))
  }
  errors[is.na(errors)]=1e4
  return(sum(errors))
}

initialize_swarm<-function(swarm_size,L=length(avec),locfac=0.6){
  x.p<<-matrix(runif(swarm_size*L,-1,1),nrow=swarm_size,ncol=L)
  vel<<-matrix(runif(swarm_size*L,-0.1,0.1),nrow=swarm_size,ncol=L)
  locality<<-locfac*swarm_size
  outgs<<-apply(x.p,1,function(aa)eval_weights(aa,inputlist,outputs))
  bestgs<<-outgs
  bestp<<-x.p
}

step_swarm<-function(swarm_size,L=length(avec),w=0.9,g_p=0.4,g_g=0.4){
  
  outgs<<-apply(x.p,1,function(aa)eval_weights(aa,inputlist,outputs))
  
  n_v=matrix(runif(swarm_size*L,-0.01,0.01),nrow=swarm_size,ncol=L)
  r_p=matrix(runif(swarm_size*L,0,1),nrow=swarm_size,ncol=L)
  r_g=matrix(runif(swarm_size*L,0,1),nrow=swarm_size,ncol=L)
  
  cmat=t(apply(x.p,1,function(z) order(apply(x.p,1,function(y) sum(abs(y-z))))[-1][1:locality]))
  closest.neighbor=x.p[cmat[,1],]
  
  best_g_mat<<-t(apply(cmat,1,function(a) x.p[a,][which.min(outgs[a]),]))
  
  new.ind=which(outgs<bestgs)
  bestp[new.ind,]<<-x.p[new.ind,]
  bestgs[new.ind]<<-outgs[new.ind]
  
  repulse.factor=0
  vel<<-n_v+w*vel+g_p*r_p*(bestp-x.p)+g_g*r_g*(best_g_mat-x.p)-sweep((x.p-closest.neighbor),2,repulse.factor,"*")
  
  x.p<<-x.p+vel
  x.p[x.p<-1]=-1
  x.p[x.p>1]=1
  x.p<<-x.p
}


# Running -----------------------------------------------------------------

#Source Buoy Data
library(forecastML)

data("data_buoy",package="forecastML")

clean=data_buoy
clean[is.na(clean)]=0

lookback=20
lookforward=30

df=clean
predictors=c("wind_spd","air_temperature")
objective=c("sea_surface_temperature")

dfw<-lookwindow(df,lookback,lookforward,predictors,objective)

lookback=20
lookforward=30

scaledat<-scalelist(dfw,objrange=c(0.2,0.6))
scalesaves<-scaledat$scalesaves
dfs<-scaledat$scaledlist

#User Selections----
n.s=4#number of states
sym=F   #Parameter Symmetry
loc=F #Locality
cont=T #Parameter Physicality Controls
sub.num=1 #Number of conserved subgroups
vfara_inert=lookforward/2#inertia
vfara_init=1000 #initial inertia
Tlen=lookforward #Steps to run coil
loadvals=T #Load in previously learned values?

buildcoil(n.s,sym=F)

model<-autogen_cnn(dfs,n.s,dim(CoilVals)[1])

weights<-get_weights(model)
weightdim=lapply(weights, dim)
avec<-unlist(weights)


xsamps=sample(1:100,3)
inputlist=list()
for (xsel in xsamps){
  inputs=list()
  for (iaa in 1:length(dfs$predictors)){
    inputs[[iaa]]=t(as.matrix(dfs$predictors[[iaa]][xsel,],ncol=length(xsel)))
  }
  inputs[[iaa+1]]=t(as.matrix(dfs$dummy[xsel,],ncol=length(xsel)))
  inputlist[[paste(xsel)]]=inputs
}

outputs<-dfs$objective[[1]][xsamps,]
outputs<-t(apply(dfs$objective[[1]][xsamps,],1,function(x)scaledata(x,c(0.2,0.6))$scaled))

n.part=10
initialize_swarm(n.part)
Esave=c()
for (itt in 1:10){
  step_swarm(n.part)
  Esave=c(Esave,min(bestgs))
  plot(Esave,type="l")
}

assign_weights(weights,weightdim,bestp[which.min(bestgs),])

par(mfrow=c(length(inputlist),1),mar=c(0,4,0,0))
for (iii in 1:length(inputlist)){
  inputs=inputlist[[iii]]
  coil_out=pop_coil(inputs,readout = T)
  
  plot(outputs[iii,],ylim=c(0,1))
  print(coil_out[[1]][1,1])
  lines(coil_out[[1]][,1])
}
par(mfrow=c(1,1),mar=c(4,4,4,4))
