rm(list=ls())

library(forecastML)
library(neuralcoil)
library(keras)
library(tensorflow)

data("data_buoy",package="forecastML")

clean=data_buoy
clean[is.na(clean)]=0


clean$group=lubridate::floor_date(data_buoy$date,"10 day")
clean<-as.data.frame(clean%>%group_by(group)%>%summarise_all(mean))
lookback=10

lookforward=30
slen=15 #How many samples in the lookforward to use?
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
  savedweights<-readRDS("results/buoy_opt5.RdA")
}
retrain=F #Retrain from saved weights?

buildcoil(n.s,sym=sym)

model<-autogen_cnn(dfs,n.s,dim(CoilVals)[1])

weights<-get_weights(model)
weightdim=lapply(weights, dim)
avec<-unlist(weights)


#xsamps=sample(1:dim(dfs$objective[[1]])[1],15)
xsamps=c(3,72,44,100,145)[1:5]

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

if (loadvals&!retrain){
  assign_weights(weights,weightdim,savedweights)
}else{
  n.part=200
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
