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

lookback=10
lookforward=5

df=as.data.frame(matrix(runif(5000,0,1),ncol=5))
df$V1=1:1000
predictors=c("V1","V2","V3","V4")
objective=c("V5")

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

dfw<-lookwindow(df,lookback,lookforward,predictors,objective)

scalelist<-function(dfw){
  #Scale list and give dummy data head
  scalesaves=c()
  for (idd in 1:length(dfw$predictors)){
    scaled=scaledata(dfw$predictors[[idd]],c(0,1))
    dfw$predictors[[idd]]=scaled$scaled
    scalesaves=rbind(scalesaves,scaled$scaler)
  }
  
  for (idd in 1:length(dfw$objective)){
    scaled=scaledata(dfw$objective[[idd]],c(0.2,0.6))
    dfw$objective[[idd]]=scaled$scaled
    scalesaves=rbind(scalesaves,scaled$scaler)
  }
  
  dfw$dummy=matrix(1,nrow=dim(dfw$predictors[[1]])[1],ncol=2)
  return(list(scaledlist=dfw,scalesaves=scalesaves))
}

scaledat<-scalelist(dfw)
scalesaves<-scaledat$scalesaves
dfs<-scaledat$scaledlist

#User Selections----
n.s=4#number of states
sym=F   #Parameter Symmetry
loc=F #Locality
cont=T #Parameter Physicality Controls
sub.num=1 #Number of conserved subgroups
vfara_inert=lookforward/2 #inertia
vfara_init=1 #initial inertia
Tlen=lookforward #Steps to run coil
loadvals=T #Load in previously learned values?

#Generate Coil According to User specifications
source("src/complex_coil_gen.R")

buildcoil(n.s,sym=F)

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
  
  rots<-merge%>%layer_dense(units=3,activation="relu")
  
  starts<-merge%>%layer_dense(units=n.s*2,activation="relu")
  
  visl[[ipp+1]]<-layer_input(shape=c(dim(dfs$dummy)[2]))
  dumw<-visl[[ipp+1]]%>%layer_dense(units=params*2)%>%layer_activation_leaky_relu(0.0001) 
  
  output <- layer_concatenate(list(rots,starts,dumw))
  
  model<-keras_model(visl,output)
  
  return(model)
}

model<-autogen_cnn(dfs,n.s,dim(CoilVals)[1])

train_step <- function(input, output) {
  with(tf$GradientTape() %as% tape, {
    rdim<-dim(CoilVals)[1]
    tape$watch(model$trainable_variables)
    val_out=model(input, training = TRUE)
    cnn_outputs <- as.array(val_out)
    rots<-cnn_outputs[1:3]
    stmat<-matrix(cnn_outputs[(4):(4+n.s*2-1)],nrow=2)
    startvals<-complex(n.s,stmat[1,],stmat[2,])
    randmat<-matrix(cnn_outputs[(4+n.s*2):length(cnn_outputs)],nrow=2)
    RandVec<-complex(rdim,randmat[1,],randmat[2,])
    rots=c(200,500,700)+rots
    startvals=1+startvals
    coil_out<-(runcoil(RandVec,rots,startvals))
    result<-coil_out[[1]]
    loss_value <- tf$keras$losses$mean_squared_error(output, as_tensor(result[,1]))
    #loss_value <- tf$keras$losses$mean_squared_error(rep_len(output,691), val_out)

  })
  loss_history <<- append(loss_history, loss_value)
  #grads <- tape$gradient(loss_value, model$trainable_variables)
  #print(grads)
  grads<-lapply(get_weights(model),function(x) as_tensor(-x/100,dtype="float"))
  optimizer$apply_gradients(
    purrr::transpose(list(grads, model$trainable_variables))
  )
}

train <- autograph(function() {
  for (epoch in seq_len(10)) {
    #for (xsel in 1:dim(dfs$predictors[[1]])[1]) {
    for (xsel in 1:5){
      inputs=list()
      for (iaa in 1:length(dfs$predictors)){
        inputs[[iaa]]=t(as.matrix(dfs$predictors[[iaa]][xsel,],ncol=length(xsel)))
      }
      inputs[[iaa+1]]=t(as.matrix(dfs$dummy[xsel,],ncol=length(xsel)))
      train_step(inputs, dfs$objective[[1]][xsel,])
    }
    tf$print("Epoch", epoch, "finished.")
  }
})

optimizer <- optimizer_adam()

loss_history <- list()

train()

history <- loss_history %>% 
  purrr::map(as.numeric) %>% 
  purrr::flatten_dbl()

plot(history,ylim=c(0,0.2))


