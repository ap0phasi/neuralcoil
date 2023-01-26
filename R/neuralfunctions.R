##############################################################################
#' Scale Data
#'
#' This function scaled data to a specified range based on minimum and maximum
#'
#' @param x data array
#' @param range desired scaling range
#' @return scaled data
#' @export
scaledata<-function(x,range){
  scaled=((x-min(x))/(max(x)-min(x)))*(range[2]-range[1])+range[1]
  return(list(scaled=scaled,scaler=c(min(x),max(x))))
}

##############################################################################
#' Invert Scaling
#'
#' This function inverts data scaling provided some scaled data, the values the
#' data was scaled with, and the specified range
#'
#' @param x scaled data
#' @param scaler values with which data was scaled
#' @param range desired scaling range
#' @return unscaled data
#' @export
invertscaling<-function(x,scaler,range){
  inverted=((x-range[1])/(range[2]-range[1]))*(scaler[2]-scaler[1])+scaler[1]
  return(inverted)
}

##############################################################################
#' Rearranges Data for Convolutional Time Series Training
#'
#' This transforms a data frame into arrays of lookbacks and lookforwards
#'
#' @param df data frame with single feature row
#' @param lookback how many timesteps should we look back on
#' @param lookforward how many timesteps in the future do we what to predict
#' @param predictors which features are only used for prediction
#' @param predictors which is the objective feature
#' @return list of transformed data
#' @export
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

##############################################################################
#' Scale Data Frame
#'
#' This applies scaling to an entire data frame with the ability to scale to
#' different ranges for predictor vs objective features, and saves the scaling
#' values.
#'
#' @param dfw data frame with single feature row
#' @param predrange what range should the predictor values be scaled to?
#' @param objrange what range should the objective values be scaled to?
#' @return scaled data frame
#' @return scaling values
#' @export
scalelist<-function(dfw,predrange=c(0,1),objrange=c(0.2,0.6)){
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

##############################################################################
#' Seagull Activation
#'
#' This is a unique activation function for the parameterization of coil values
#' with a neural network
#'
#' @param x what to compose new layer instance with
#' @param alpha float >= 0. Negative slope coefficient.
#' @param decay smoothing value to prevent rapid approach of maximum
#' @export
layer_activation_seagull<-function(x,alpha,decay=1e3){
  keras::k_minimum(1,keras::k_abs(keras::layer_activation_leaky_relu(x,alpha)/decay))
}

##############################################################################
#' Automatically Generate Convolutional Neural Network for Coil Parameterization
#'
#' This is a function for automatically generating a convolutional neural network
#' for a transformed dataset in the unique manner required for coil parameterization
#'
#' @param dfs transformed data
#' @param n.s number of coil presentations
#' @param params number of coil values
#' @export
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
  dumw<-visl[[ipp+1]]%>%keras::layer_dense(units=params*2)%>%layer_activation_seagull(0.1)

  output <- keras::layer_concatenate(list(rots,starts,dumw))

  model<-keras::keras_model(visl,output)

  return(model)
}

##############################################################################
#' Assign Neural Networks Weights from Array
#'
#' This is a function to assign neural network weights from a weight array
#' provided a neural network structure
#'
#' @param weights neural network weights in list structure
#' @param weightdim dimensions of neural network weights
#' @param avec new weight array
#' @param model Model Object
#' @export
assign_weights<-function(weights,weightdim,avec){
  weightsnew=weights
  stindx=1
  for (iw in 1:length(weightdim)){
    len=prod(weightdim[[iw]])
    weightsnew[[iw]]=array(avec[stindx:(len+stindx-1)],dim=weightdim[[iw]])
    stindx=len+stindx
  }
  keras::set_weights(model,weightsnew)
}

##############################################################################
#' Evaluate Neural Network to Get Coil Parameters
#'
#' This is a function to evaluate the neural network for some inputs to get
#' coil parameters
#'
#' @param input input values for neural network
#' @param model Model Object
#' @return coil parameter values
#' @return coil rotation values
#' @return coil start values
#' @export
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

##############################################################################
#' Populate and Run Coil for Provided Patameters
#'
#' This is a function to populate and run a coil by evaluating a neural network
#' for a set of inputs
#'
#' @param input input values for neural network
#' @param readout should the neural network results be displayed?
#' @return coil values
#' @export
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

##############################################################################
#' Generic Mean Squared Loss Function
#'
#' This is a function used to evaluate loss based on actual vs predicted
#'
#' @param actual objective values
#' @param predicted model values
#' @export
lossfun<-function(actual,predicted){
  mean((actual-predicted)^2)
}

##############################################################################
#' Transform Array into Coil Parameters
#'
#' This is a function to transform an array into coil parameters directly.
#' For use in direct particle swarm optimization, unique parameters are generated
#' for each input.
#'
#' @param avec parameter array values
#' @param inputlist input list to track where parameters should be assigned.
#' @return rotations and start values
#' @export
transform_to_params<-function(avec,inputlist){
  inL<-length(inputlist)
  rot.indx<-1:(inL*3)
  st.indx<-(inL*3+1):length(avec)
  rots<-matrix(avec[rot.indx],ncol=3)*100
  stvals<-matrix(avec[st.indx],ncol=2*n.s)/100
  return(list(rots=rots,stvals=stvals))
}

##############################################################################
#' Evaluate Parameters by Populating and Running Coil Directly
#'
#' This is a function to pass parameter values into a coil and evaluate its
#' fitness for a number of inputs and outputs
#'
#' @param avec parameter array values
#' @param inputlist input list to track where parameters should be assigned.
#' @param outputs objective values to evaluate against
#' @param sel which values to use in loss function evaluation
#' @return total error
#' @export
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

##############################################################################
#' Evaluate Parameters by Setting Neural Network Weights and Evaluating Coil
#'
#' This is a function to pass parameter values into a neural network, which
#' takes a set of inputs and produces coil paramters. These parameters are
#' passed into a coil for evaluation against objective outputs.
#'
#' @param avec parameter array values
#' @param inputlist input list to track where parameters should be assigned.
#' @param outputs objective values to evaluate against
#' @param sel which values to use in loss function evaluation
#' @return total error
#' @export
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

##############################################################################
#' Initialize Particle Swarm
#'
#' This is a function to initialize a particle swarm either for direct optimization
#' or neural network optimization
#'
#' @param swarm_size number of particles in swarm
#' @param L number of dimensions needed for swarm
#' @param loc_fac locality factor
#' @param type what type of optimization is the swarm used for?
##' \itemize{
##'  \item{"neural"}{ Optimize weights of convolutional neural network to predict rotation, start values, and coil parameterizations from inputs}
##'  \item{"direct"}{ Optimize unique rotations and start values for multiple outputs using the same coil parameterization}
##' }
#' @export
initialize_swarm<-function(swarm_size,L=length(avec),locfac=0.6,type="neural"){
  if (type=="neural"){
    eval_fun<-eval_weights
    lowlim<<-(-3)
    uplim<<-3
  }else if (type=="direct"){
    eval_fun<-eval_params
    lowlim<<-(0.01)
    uplim<<-(1)
  }
  x.p<<-matrix(runif(swarm_size*L,lowlim/3,uplim/3),nrow=swarm_size,ncol=L)
  vel<<-matrix(runif(swarm_size*L,-0.01,0.01),nrow=swarm_size,ncol=L)
  if (retrain&type=="neural"){
    x.p[1,]<-savedweights
  }else if (type=="direct"){
    innew=avec*100
    innew[1:(3*length(xsamps))]=innew[1:(3*length(xsamps))]/100/100
    x.p[1,]<<-innew
  }
  locality<<-locfac*swarm_size
  outgs<<-apply(x.p,1,function(aa)eval_fun(aa,inputlist,outputs,sel=slseq))
  bestgs<<-outgs
  bestp<<-x.p
}

##############################################################################
#' Step through Particle Swarm
#'
#' This is a function to step through a particle swarm either for direct optimization
#' or neural network optimization
#'
#' @param swarm_size number of particles in swarm
#' @param L number of dimensions needed for swarm
#' @param w momentum factor
#' @param g_p global pull factor
#' @param g_g local pull factor
#' @param loc_fac locality factor
#' @param type what type of optimization is the swarm used for?
##' \itemize{
##'  \item{"neural"}{ Optimize weights of convolutional neural network to predict rotation, start values, and coil parameterizations from inputs}
##'  \item{"direct"}{ Optimize unique rotations and start values for multiple outputs using the same coil parameterization}
##' }
#' @export
step_swarm<-function(swarm_size,L=length(avec),w=0.9,g_p=0.4,g_g=0.4,type="neural"){
  if (type=="neural"){
    eval_fun<-eval_weights
  }else if (type=="direct"){
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
  bestp<<-bestp
  bestgs<<-bestgs
}

##############################################################################
#' Easily generate input and output lists
#'
#' This is a function to generate input and output lists based on some specified
#' sampling index
#'
#' @param dfs transformed data
#' @param xsamps sampled index
#' @return input list and output matrix
#' @export
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

##############################################################################
#' Populate and Run Coil for Rotations, Start Values, and Coil Parameters
#'
#' This is a function to populate and run a coil provided the rotations, start values, and coil parameters
#'
#' @param avec array of rotation, start values, and coil parameters
#' @return coil result
#' @export
pop_coil_full<-function(avec){
  rdim=dim(CoilVals)[1]
  RandVec=complex(rdim,avec[1:(rdim)],avec[(rdim+1):(rdim*2)])
  rotvals=avec[(rdim*2+1):(rdim*2+3)]*100
  stmat=matrix(avec[(rdim*2+4):length(avec)],nrow=2)
  startvals=complex(n.s,stmat[1,],stmat[2,])
  runcoil(RandVec,rotvals,startvals)
}

##############################################################################
#' Initialize Particle Swarm for Full Coil Solution
#'
#' This is a function to initialize a particle swarm to determine coil rotations, starts, and parameters
#'
#' @param swarm_size number of particles in swarm
#' @param L number of dimensions needed for swarm
#' @param loc_fac locality factor
#' @param type what type of optimization is the swarm used for?
#' @param setrots use fixed rotation values?
#' @export
initialize_swarm_full<-function(swarm_size,L,locfac=0.6,setrots=NULL){
  eval_fun<-eval_params
  lowlim<<-(0)

  x.p<<-matrix(runif(swarm_size*L,lowlim,1),nrow=swarm_size,ncol=L)
  vel<<-matrix(runif(swarm_size*L,-0.1,0.1),nrow=swarm_size,ncol=L)
  if (!is.null(setrots)){
    rdim=dim(CoilVals)[1]
    rot.indx<-((rdim*2+1):(rdim*2+3))
    x.p[,rot.indx]<<-t(matrix(setrots,ncol=dim(x.p)[1],nrow=3))
  }

  locality<<-locfac*swarm_size

  solvedcoil=apply(x.p,1,function(x)pop_coil_full(x)[[1]])

  outmat<<-t(solvedcoil)
  best_p_res<<-outmat
  x.p<<-x.p
  best_p<<-x.p
}

##############################################################################
#' Step through Particle Swarm for Full Coil Solution
#'
#' This is a function to step through a particle swarm to determine coil rotations,
#' starts, and parameters
#'
#' @param swarm_size number of particles in swarm
#' @param L number of dimensions needed for swarm
#' @param w momentum factor
#' @param g_p global pull factor
#' @param g_g local pull factor
#' @param setrots use fixed rotation values?
#' @export
step_swarm_full<-function(swarm_size,L,w=0.9,g_p=0.4,g_g=0.4,setrots=NULL){
  old_perf<<-apply(outmat,1,function(x)optf(x,out_goal,slseq))
  print(min(old_perf))
  solvedcoil<<-apply(x.p,1,function(x)pop_coil_full(x)[[1]])

  outmat<<-t(solvedcoil)
  outmat[is.na(outmat)]<<-0
  outmat[abs(outmat)==Inf]<<-0

  n_v<<-matrix(runif(swarm_size*L,-1e-4,1e-4),nrow=swarm_size,ncol=L)
  r_p<<-matrix(runif(swarm_size*L,0,1),nrow=swarm_size,ncol=L)
  r_g<<-matrix(runif(swarm_size*L,0,1),nrow=swarm_size,ncol=L)

  cmat<<-t(apply(x.p,1,function(z) order(apply(x.p,1,function(y) sum(abs(y-z))))[-1][1:locality]))

  best_g_mat<<-t(apply(cmat,1,function(a) x.p[a,][which.min(apply(outmat[a,],1,function(x)optf(x,out_goal,slseq))),]))

  new.ind<<-which(apply(outmat,1,function(x)optf(x,out_goal,slseq))<apply(best_p_res,1,function(x)optf(x,out_goal,slseq)))

  best_p[new.ind,]<<-x.p[new.ind,]
  best_p_res[new.ind,]<<-outmat[new.ind,]

  vel<<-n_v+w*vel+g_p*r_p*(best_p-x.p)+g_g*r_g*(best_g_mat-x.p)

  x.p<<-x.p+vel
  x.p[x.p<(lowlim)]<<-lowlim
  if (!is.null(setrots)){
    rdim=dim(CoilVals)[1]
    rot.indx<-((rdim*2+1):(rdim*2+3))
    x.p[,rot.indx]<<-t(matrix(setrots,ncol=dim(x.p)[1],nrow=3))
  }
  x.p<<-x.p
  best_p<<-best_p
  best_p_res<<-best_p_res
}

##############################################################################
#' Fitness Function for Full Coil Solution
#'
#' This is a function to evaluate the particle swarm for a full coil solution
#' @param outs modeled values
#' @param out_goal objective values
#' @param slseq objective points selection
#' @export
optf<-function(outs,out_goal,slseq){
  mean(abs(outs[slseq]-out_goal[slseq]))
}
