##Full Machine Learning Coil Stack
rm(list=ls())

library(neuralcoil)
library(keras)
library(tensorflow)


autogen_cnn<-function(dfs,n.s,params){
  visl<-list()
  cnnl<-list()
  for (ipp in 1:length(dfs$predictors)){
    visl[[ipp]]<-keras::layer_input(shape=c(dim(dfs$predictors[[ipp]])[2],1))
    cnnl[[ipp]]<-visl[[ipp]]%>%
      keras::layer_conv_1d(filters=64,kernel_size=2,input_shape=c(dim(dfs$predictors[[ipp]])[2],1),activation="relu",batch_size = 1)%>%
      keras::layer_max_pooling_1d(pool_size=2,batch_size = 1)%>%
      keras::layer_flatten()
  }
  merge<-keras::layer_concatenate(cnnl)%>%
    keras::layer_dense(units=32,activation="relu")%>%
    keras::layer_dense(units=16,activation="relu")

  rots<-merge%>%keras::layer_dense(units=3,activation="linear")

  starts<-merge%>%keras::layer_dense(units=n.s*2)%>%layer_activation_seagull(0.1)

  visl[[ipp+1]]<-keras::layer_input(shape=c(dim(dfs$dummy)[2]))
  dumw<-visl[[ipp+1]]%>%keras::layer_dense(units=params*2,trainable = F)%>%layer_activation_seagull(0.1)

  output <- keras::layer_concatenate(list(rots,starts,dumw))

  model<-keras::keras_model(visl,output)

  return(model)
}



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
states=1 #Which states to calibrate to?


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
  savedweights<-readRDS("results/buoy_single4.RdA")
}
retrain=F #Retrain from saved weights?

buildcoil(n.s,sym=sym)

model<-autogen_cnn(dfs,n.s,dim(CoilVals)[1])

weights<-get_weights(model)
weightdim=lapply(weights, dim)
avec<-unlist(weights)


#xsamps=sample(1:dim(dfs$objective[[1]])[1],15)
xsamps=c(3)

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

#saveRDS(file="results/buoy_single4.RdA",unlist(get_weights(model)))

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
for (itt in 1:100){
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

#Couple notes: The dummy data is because we will be directly inserting the coil parameters we learned in step 1 into the weights of the neural network
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
print(paste("Difference 2:",sum((model%>%predict(nn_inputs))[1,1:13]-nn_obj[1,1:13])))

# #Checks
# nn_out=(model%>%predict(nn_inputs))
# stmat=matrix(nn_obj[1,4:11],nrow=2)
# startvals<-(complex(n.s,stmat[1,],stmat[2,]))
# Rmat=matrix(nn_obj[1,12:dim(nn_obj)[2]],nrow=2)
# rvecnew=complex(length(RandVec),Rmat[1,],Rmat[2,])
# plot(runcoil(rvecnew,nn_obj[1,1:3],startvals)[[1]][,1])
# lines(runcoil(RandVec,nn_obj[1,1:3],startvals)[[1]][,1])
# stmat=matrix(nn_out[1,4:11],nrow=2)
# startvals<-(complex(n.s,stmat[1,],stmat[2,]))
# Rmat=matrix(nn_out[1,12:dim(nn_out)[2]],nrow=2)
# rvecnew=complex(length(RandVec),Rmat[1,],Rmat[2,])
# lines(runcoil(rvecnew,nn_out[1,1:3],startvals)[[1]][,1],col="blue")


#saveRDS(unlist(get_weights(model)),file="results/buoyasadasd.RdA")
#assign_weights(weights,weightdim,readRDS("results/buoy_opt5.RdA"))
