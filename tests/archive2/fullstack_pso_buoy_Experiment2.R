##Full Machine Learning Coil Stack
rm(list=ls())

#Establish Libraries
library(tensorflow)
library(tfautograph)
library(keras)

Sys.setenv(TF_AVGPOOL_USE_CUDNN=1)

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

layer_activation_seagull<-function(x,alpha,decay=1e3){
  k_minimum(1,k_abs(layer_activation_leaky_relu(x,alpha)/decay))
}

autogen_cnn<-function(dfs,n.s,params){
  visl<-list()
  cnnl<-list()
  for (ipp in 1:length(dfs$predictors)){
    visl[[ipp]]<-layer_input(shape=c(dim(dfs$predictors[[ipp]])[2],1))
    cnnl[[ipp]]<-visl[[ipp]]%>%
      layer_conv_1d(filters=64,kernel_size=2,input_shape=c(dim(dfs$predictors[[ipp]])[2],1),activation="relu",batch_size = 1)%>%
      layer_max_pooling_1d(pool_size=2,batch_size = 1)%>%
      layer_flatten()
  }
  merge<-layer_concatenate(cnnl)%>%
    layer_dense(units=32,activation="relu")%>%
    layer_dense(units=16,activation="relu")
  
  #rots<-merge%>%layer_dense(units=3)%>%layer_activation_seagull(0.1,decay=100)
  rots<-merge%>%layer_dense(units=3,activation="linear")
  
  starts<-merge%>%layer_dense(units=n.s*2)%>%layer_activation_seagull(0.1)
  
  visl[[ipp+1]]<-layer_input(shape=c(dim(dfs$dummy)[2]))
  dumw<-visl[[ipp+1]]%>%layer_dense(units=params*2)%>%layer_activation_seagull(0.1)
  
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

get_params<-function(input){
  rdim<-dim(CoilVals)[1]
  val_out=model(input, training = TRUE)
  cnn_outputs <- as.array(val_out)
  rots<-abs(cnn_outputs[1:3])
  stv<-cnn_outputs[(4):(4+n.s*2-1)]
  rvs<-cnn_outputs[(4+n.s*2):length(cnn_outputs)]
  stmat<-(matrix(stv,nrow=2))
  startvals<-(complex(n.s,stmat[1,],stmat[2,]))
  randmat<-(matrix(rvs,nrow=2))
  RandVec<-complex(rdim,randmat[1,],randmat[2,])
  return(list(RandVec=RandVec,rots=rots,startvals=startvals))
}

pop_coil<-function(input,readout=F){
  model_out<-get_params(input)
  RandVec<-model_out$RandVec
  rots<-model_out$rots
  startvals<-model_out$startvals
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

transform_to_params<-function(avec,inputlist){
  inL<-length(inputlist)
  rot.indx<-1:(inL*3)
  st.indx<-(inL*3+1):length(avec)
  rots<-matrix(avec[rot.indx],ncol=3)*100
  stvals<-matrix(avec[st.indx],ncol=2*n.s)/100
  return(list(rots=rots,stvals=stvals))
}

eval_params<-function(avec,inputlist,outputs,sel=seq(1,lookforward)){
  inL<-length(inputlist)
  paramout<-transform_to_params(avec,inputlist)
  rots=paramout$rots
  stvals=paramout$stvals
  errors=c()
  for (iii in 1:inL){
    stmat<-(matrix(stvals[iii,],nrow=2))
    startvals<-(complex(n.s,stmat[1,],stmat[2,]))
    coil_out<-(runcoil(RandVec,rots[iii,],startvals))
    errors=c(errors,lossfun(outputs[iii,][sel],coil_out[[1]][sel,states]))
  }
  errors[is.na(errors)]=1e4
  return(sum(errors))
}

eval_weights<-function(avec,inputlist,outputs,sel=seq(1,lookforward)){
  assign_weights(weights,weightdim,avec)
  errors=c()
  for (iii in 1:length(inputlist)){
    inputs=inputlist[[iii]]
    coil_out=pop_coil(inputs)
    errors=c(errors,lossfun(outputs[iii,][sel],coil_out[[1]][sel,states]))
  }
  errors[is.na(errors)]=1e4
  return(sum(errors))
}

initialize_swarm<-function(swarm_size,L=length(avec),locfac=0.6,type="neural"){
  if (type=="neural"){
    eval_fun<-eval_weights
    lowlim<<-(-3)
    uplim<<-3
  }else{
    eval_fun<-eval_params
    lowlim<<-(0.01)
    uplim<<-(1)
  }
  x.p<<-matrix(runif(swarm_size*L,lowlim/3,uplim/3),nrow=swarm_size,ncol=L)
  vel<<-matrix(runif(swarm_size*L,-0.01,0.01),nrow=swarm_size,ncol=L)
  if (retrain&type=="neural"){
    x.p[1,]<-savedweights
  }else{
    innew=avec*100
    innew[1:(3*length(xsamps))]=innew[1:(3*length(xsamps))]/100/100
    x.p[1:n.part,]<<-t(matrix(rep(innew,n.part),nrow=length(avec)))
    vel<<-matrix(runif(swarm_size*L,-0.0001,0.0001),nrow=swarm_size,ncol=L)
  }
  locality<<-locfac*swarm_size
  outgs<<-apply(x.p,1,function(aa)eval_fun(aa,inputlist,outputs,sel=slseq))
  bestgs<<-outgs
  bestp<<-x.p
}

step_swarm<-function(swarm_size,L=length(avec),w=0.9,g_p=0.4,g_g=0.4,type="neural"){
  if (type=="neural"){
    eval_fun<-eval_weights
  }else{
    eval_fun<-eval_params
  }
  outgs<<-apply(x.p,1,function(aa)eval_fun(aa,inputlist,outputs,sel=slseq))
  
  n_v=matrix(runif(swarm_size*L,-0.01,0.01),nrow=swarm_size,ncol=L)
  r_p=matrix(runif(swarm_size*L,0,1),nrow=swarm_size,ncol=L)
  r_g=matrix(runif(swarm_size*L,0,1),nrow=swarm_size,ncol=L)
  
  cmat=t(apply(x.p,1,function(z) order(apply(x.p,1,function(y) sum(abs(y-z))))[-1][1:locality]))
  
  best_g_mat<<-t(apply(cmat,1,function(a) x.p[a,][which.min(outgs[a]),]))
  
  new.ind=which(outgs<bestgs)
  bestp[new.ind,]<<-x.p[new.ind,]
  bestgs[new.ind]<<-outgs[new.ind]
  
  vel<<-n_v+w*vel+g_p*r_p*(bestp-x.p)+g_g*r_g*(best_g_mat-x.p)
  
  x.p<<-x.p+vel
  x.p[x.p<(lowlim)]=lowlim
  x.p[x.p>uplim]=uplim
  x.p<<-x.p
}

gen_in_out<-function(dfs,xsamps){
  inputlist=list()
  for (xsel in xsamps){
    inputs=list()
    for (iaa in 1:length(dfs$predictors)){
      inputs[[iaa]]=t(as.matrix(dfs$predictors[[iaa]][xsel,],ncol=length(xsel)))
    }
    inputs[[iaa+1]]=t(as.matrix(dfs$dummy[xsel,],ncol=length(xsel)))
    inputlist[[paste(xsel)]]=inputs
  }
  
  outputs<-matrix(dfs$objective[[1]][xsamps,],nrow=length(xsamps))
  return(list(inputlist,outputs))
}

# Running -----------------------------------------------------------------

#Source Buoy Data
library(forecastML)

data("data_buoy",package="forecastML")

clean=data_buoy
clean[is.na(clean)]=0


clean$group=lubridate::floor_date(data_buoy$date,"10 day")
clean<-as.data.frame(clean%>%group_by(group)%>%summarise_all(mean))
lookback=10

lookforward=30
slen=30 #How many samples in the lookforward to use?
slseq=round(seq(1,lookforward,length.out = slen))
states=1 #How many states to calibrate to?


predictors=c("wind_spd","air_temperature")
objective=c("sea_surface_temperature")

df=clean%>%select(c(predictors,objective))
df=df[df$sea_surface_temperature>0,]

dfw<-lookwindow(df,lookback,lookforward,predictors,objective)

scaleran=c(0.2,0.6)
scaledat<-scalelist(dfw,objrange=scaleran)
scalesaves<-scaledat$scalesaves
dfs<-scaledat$scaledlist

#User Selections----
n.s=4#number of states
sym=F   #Parameter Symmetry
loc=F #Locality
cont=T #Parameter Physicality Controls
sub.num=1 #Number of conserved subgroups
vfara_inert=60#inertia
vfara_init=100 #initial inertia
Tlen=lookforward #Steps to run coil
loadvals=T #Load in previously learned values?
if (loadvals){
  savedweights<-readRDS("results/buoy_single.RdA")
}
retrain=F #Retrain from saved weights?

buildcoil(n.s,sym=sym)

model<-autogen_cnn(dfs,n.s,dim(CoilVals)[1])

weights<-get_weights(model)
weightdim=lapply(weights, dim)
avec<-unlist(weights)


#xsamps=sample(1:dim(dfs$objective[[1]])[1],15)
xsamps=c(3,72,44,100,145)[1]

inout<-gen_in_out(dfs,xsamps)
inputlist<-inout[[1]]
outputs<-inout[[2]]

if (loadvals&!retrain){
  assign_weights(weights,weightdim,savedweights)
}else{
  n.part=20
  initialize_swarm(n.part)
  
  Esave=c()
  for (itt in 1:100){
    step_swarm(n.part)
    Esave=c(Esave,min(bestgs))
    plot(Esave,type="l")
  }
  
  assign_weights(weights,weightdim,bestp[which.min(bestgs),])
}

outsave=c()
par(mfrow=c(length(inputlist),1),mar=c(0,4,0,0))
for (iii in 1:length(inputlist)){
  inputs=inputlist[[iii]]
  coil_out=pop_coil(inputs,readout = T)
  
  plot(outputs[iii,],col="blue",type="l")
  print(coil_out[[1]][1,1])
  matlines(coil_out[[1]],col="grey")
  lines(coil_out[[1]][,1],col="red")
  
  outsave=rbind(outsave,coil_out[[1]][,1])
}
par(mfrow=c(1,1),mar=c(4,4,4,4))

plot(df[[objective]][-(1:lookback)],type="l",lwd=1.5)
for (ix in 1:length(xsamps)){
  invres=invertscaling(outsave[ix,],scalesaves[dim(scalesaves)[1],],scaleran)
  invobs=invertscaling(outputs[ix,],scalesaves[dim(scalesaves)[1],],scaleran)
  lines(xsamps[ix]:(xsamps[ix]+lookforward-1),invres,col="red",lwd=2)
  #=lines(xsamps[ix]:(xsamps[ix]+lookforward-1),invobs,col="blue")
}

calparams<-get_params(inputs)
startvals_opt=calparams$startvals
rotvals_opt=calparams$rots
RandVec=calparams$RandVec

#Do direct particle swarm optimization for other samples to determine ideal rotation and start vals for other samples
xsampsold=xsamps
xsamps=c(72,44,100,145)[1:2]

inout<-gen_in_out(dfs,xsamps)
inputlist<-inout[[1]]
outputs<-inout[[2]]

avec<-c(rep(rotvals_opt,length(xsamps)),rep(c(Re(startvals_opt),Im(startvals_opt)),length(xsamps)))

n.part=20
initialize_swarm(n.part,type="direct")

Esave=c()
for (itt in 1:10){
  step_swarm(n.part,type="direct")
  Esave=c(Esave,min(bestgs))
  plot(Esave,type="l")
  #plot(x.p[,c(1,2)])
}

opt_params<-transform_to_params(bestp[which.min(bestgs),],inputlist)
stvals<-opt_params$stvals
rots<-opt_params$rots

#Append original trained data
rots<-rbind(rotvals_opt,rots)
stvals<-rbind(array(t(matrix(c(Re(startvals_opt),Im(startvals_opt)),ncol=2))),stvals)
xsamps=c(xsampsold,c(72,44,100,145)[1:2])

inout<-gen_in_out(dfs,xsamps)
inputlist<-inout[[1]]
outputs<-inout[[2]]
 
outsave=c()
par(mfrow=c(length(inputlist),1),mar=c(0,4,0,0))
for (iii in 1:length(inputlist)){
  inputs=inputlist[[iii]]
  stmat<-(matrix(stvals[iii,],nrow=2))
  startvals<-(complex(n.s,stmat[1,],stmat[2,]))
  print(rots[iii,])
  print(startvals)
  coil_out<-(runcoil(RandVec,rots[iii,],startvals))
  
  plot(outputs[iii,],col="blue",type="l")
  print(coil_out[[1]][1,1])
  matlines(coil_out[[1]],col="grey")
  lines(coil_out[[1]][,1],col="red")
  
  outsave=rbind(outsave,coil_out[[1]][,1])
}
par(mfrow=c(1,1),mar=c(4,4,4,4))

plot(df[[objective]][-(1:lookback)],type="l",lwd=1.5)
for (ix in 1:length(xsamps)){
  invres=invertscaling(outsave[ix,],scalesaves[dim(scalesaves)[1],],scaleran)
  invobs=invertscaling(outputs[ix,],scalesaves[dim(scalesaves)[1],],scaleran)
  lines(xsamps[ix]:(xsamps[ix]+lookforward-1),invres,col="red",lwd=2)
  #=lines(xsamps[ix]:(xsamps[ix]+lookforward-1),invobs,col="blue")
}


#Train Neural Network to produce these outputs
nn_inputs=sapply(1:length(inputlist[[1]]),function(g) do.call(rbind,lapply(inputlist,function(x)x[[g]])))

RandMat=t(matrix(rep(array(t(matrix(c(Re(RandVec),Im(RandVec)),ncol=2))),length(xsamps)),ncol=length(xsamps)))
nn_obj=cbind(rots,stvals,RandMat)

loss_focus<-function(y_true,y_pred){
  importance_vector=rep(1,(n.s*2+3))
  importance_vector[4:(n.s*2+3)]=100000
  k_mean((y_true[,1:(n.s*2+3)]-y_pred[,1:(n.s*2+3)])^2*importance_vector)
}

print(paste("Difference:",sum((model%>%predict(nn_inputs))[1,]-nn_obj[1,])))
model %>%compile(loss=loss_focus,optimizer="adam")
history<-model%>%fit(
  nn_inputs,
  nn_obj,
  epochs=1000,
  batch_size=1
)

outsave=c()
par(mfrow=c(length(inputlist),1),mar=c(0,4,0,0))
for (iii in 1:length(inputlist)){
  inputs=inputlist[[iii]]
  coil_out=pop_coil(inputs,readout = T)
  
  plot(outputs[iii,],col="blue",type="l")
  print(coil_out[[1]][1,1])
  matlines(coil_out[[1]],col="grey")
  lines(coil_out[[1]][,1],col="red")
  
  outsave=rbind(outsave,coil_out[[1]][,1])
}
par(mfrow=c(1,1),mar=c(4,4,4,4))

plot(df[[objective]][-(1:lookback)],type="l",lwd=1.5)
for (ix in 1:length(xsamps)){
  invres=invertscaling(outsave[ix,],scalesaves[dim(scalesaves)[1],],scaleran)
  invobs=invertscaling(outputs[ix,],scalesaves[dim(scalesaves)[1],],scaleran)
  lines(xsamps[ix]:(xsamps[ix]+lookforward-1),invres,col="red",lwd=2)
  #=lines(xsamps[ix]:(xsamps[ix]+lookforward-1),invobs,col="blue")
}

print(paste("Difference:",sum((model%>%predict(nn_inputs))[1,]-nn_obj[1,])))
