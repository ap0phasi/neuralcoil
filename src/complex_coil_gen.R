#Build a complex-valued coil according to user specifications

#Important Functions
normalize <- function(aa,GoalR,Goali){complex(length(aa),Re(aa)/sum(Re(aa))*GoalR,Im(aa)/sum(Im(aa))*Goali)}
llen <- function(bb){Re(bb*Conj(sum(bb)))}

buildcoil<-function(n.s,sym=FALSE){
  name.states=letters[1:n.s]#Establish State Names
  trans.names=apply(expand.grid(name.states,name.states),1,function(x) paste(x,collapse="-"))#Establish Transition Names
  depS.names=apply(expand.grid(trans.names,name.states),1,function(x) paste(x,collapse="|"))#Transitions Dependent on State
  depT.names=apply(expand.grid(trans.names,trans.names),1,function(x) paste(x,collapse="|"))#Transitions Dependent on Transitions
  
  #Build Coil DF
  CoilVals=data.frame(Name=c(name.states,trans.names,depS.names,depT.names))
  #Establish Initial values (note: must be mostly between 0 and 1)
  CoilVals$Value=0
  
  #Symmetric Groups
  symmat=t(as.data.frame(lapply(strsplit(CoilVals$Name,"[-]|[|]"),function(x) c(x,rep(NA,length(name.states)-length(x))))))
  rownames(symmat)=NULL
  tsymmat=c()
  for (isy in 1:dim(symmat)[2]){tsymmat=cbind(tsymmat,symmat[,isy]==apply(symmat,1,function(bb)name.states[which.max(unlist(lapply(name.states, function(aa) sum(aa==bb,na.rm=T))))]))}
  bincode=gsub("NA","",apply(tsymmat,1,function(x) paste(as.numeric(x),collapse="")))
  
  group.index=list()
  ibC=1
  for (ib in unique(bincode)){
    group.index[[ibC]]=which(bincode==ib)
    ibC=ibC+1
  }
  
  if (sym==FALSE){
    group.index=as.list(1:length(CoilVals$Name))
  }
  
  #Create List of Conserved Groups (Groups that normalize)
  conserved.group=list()
  conserved.group[[1]]=name.states
  iC=2
  for (iNN in name.states){
    conserved.group[[iC]]=CoilVals$Name[grep(paste0("^[a-z]-",iNN,"$"),CoilVals$Name)]
    iC=iC+1
  }
  
  Normsaves=c()
  #Perform Normalization
  for (iG in conserved.group){
    #For each conserved group, make sure states or transitions are normalized
    Normsaves=rbind(Normsaves,match(iG,CoilVals$Name))
    #Within these conserved groups, look at individual components
    for (igg in iG){
      #Find all dependencies on this component
      tempName=CoilVals$Name[grep(paste0("[|]",igg,"$"),CoilVals$Name)]
      #Go through state names, all dependencies representing transitions from this state should normalize
      for (iS in name.states){
        n.indx=match(tempName[grep(paste0("-",iS,"[|]"),tempName)],CoilVals$Name)
        Normsaves=rbind(Normsaves,n.indx)
      }
    }
  }
  
  #Global Assignments
  Normsaves<<-Normsaves
  CoilVals<<-CoilVals
  name.states<<-name.states
  conserved.group<<-conserved.group
  group.index<<-group.index
  trans.names<<-trans.names
}
#For max selection
lrad <- function(bb){abs(Re(bb)+Im(bb))}

runcoil=function(RandVec,rotvals,startvals){

  #build symmetric or asymmetric coils
  for (igc in 1:length(group.index)){
    CoilVals$Value[group.index[[igc]]]=RandVec[igc]
  }
  
  CoilVals$Value[1:n.s]=startvals
  
  #Apply physics-based controls on coil parameterization
  if (cont==T){
    #Force Locality
    ExG=(expand.grid(c(1:length(name.states)),c(1:length(name.states))))
    Exgs=ExG[abs(ExG[[1]]-ExG[[2]])>1,]
    
    #Prevent exchange between subnormals
    
    #Subnormalization
    splits=split((1:length(name.states)), ceiling((1:length(name.states))/(length(name.states)/sub.num)))
    
    subvas=which(diff(ceiling((1:length(name.states))/(length(name.states)/sub.num)))>0)
    sbins=rbind(cbind(subvas,subvas+1),cbind(subvas+1,subvas))
    colnames(sbins)=colnames(Exgs)
    
    if (loc==T){
      Exgs=rbind(Exgs,sbins) #For Locality
    }else{
      Exgs=rbind(as.matrix(expand.grid(splits)),as.matrix(expand.grid(splits)[,sub.num:1])) #No Locality, segmented
    }
    
    Pset=split(t(Exgs), rep(1:nrow(Exgs), each = ncol(Exgs)))
    
    for (ipps in Pset){
      #Cross Influence, conservative, nonentropic, no impossibles
      sw.indx=grepl(paste(name.states[ipps[1]],name.states[ipps[2]],sep="[-]"),CoilVals$Name)
      CoilVals[sw.indx,2]=complex(sum(sw.indx),1e-9,1e-9)
    }
    
    #Inertial Values
    for (iSSa in 1:length(name.states)){
      sw.indx=grepl(paste0(name.states[iSSa],"[-]",name.states[iSSa],"[|]"),CoilVals$Name)
      CoilVals$Value[sw.indx]=CoilVals$Value[sw.indx]*vfara_inert
    }
    
    #Starting Inertial Values
    for (iSSa in 1:length(name.states)){
      sw.indx=grepl(paste0("^",name.states[iSSa],"[-]",name.states[iSSa],"$"),CoilVals$Name)
      CoilVals$Value[sw.indx]=CoilVals$Value[sw.indx]*vfara_init
    }
    
    #Inertial perception
    etemp=expand.grid(c(1:length(name.states)),c(1:length(name.states)))
    etemp=etemp[!(etemp[,1]==etemp[,2]),]
    Exk=split(t(etemp), rep(1:(length(name.states)-1), each = 2))
    mfac=0
    for (iske in Exk){
      Eindx1=grepl(paste0(name.states[iske[1]],"[-]",name.states[iske[2]],"[|]",name.states[iske[2]],"[-]",name.states[iske[1]]),CoilVals$Name)
      CoilVals$Value[Eindx1]=CoilVals$Value[Eindx1]*mfac
      
      Eindx2=grepl(paste0(name.states[iske[2]],"[-]",name.states[iske[1]],"[|]",name.states[iske[1]],"[-]",name.states[iske[2]]),CoilVals$Name)
      CoilVals$Value[Eindx2]=CoilVals$Value[Eindx2]*mfac
    }
  
  }
  #Normalize based off of saved norm groups
  for (iNo in 1:dim(Normsaves)[1]){
    if (max(Normsaves[iNo,])<=n.s){
      GoalR<-Re(exp(rotvals[1]*complex(1,0,1)))
      Goali<-Im(exp(rotvals[1]*complex(1,0,1)))
    }else if ((max(Normsaves[iNo,])<=(n.s^2+n.s))&(min(Normsaves[iNo,])>(n.s))){
      GoalR<-Re(exp(rotvals[2]*complex(1,0,1)))
      Goali<-Im(exp(rotvals[2]*complex(1,0,1)))
    }else{
      GoalR<-Re(exp(rotvals[3]*complex(1,0,1)))
      Goali<-Im(exp(rotvals[3]*complex(1,0,1)))
    }
    CoilVals$Value[Normsaves[iNo,]]=normalize(CoilVals$Value[Normsaves[iNo,]],GoalR,Goali)
  }
  
  #Save initial probability matrix
  Pmat=llen(CoilVals$Value[match(name.states,CoilVals$Name)])
  complex_states=CoilVals$Value[match(name.states,CoilVals$Name)]
  for (iT in 1:(Tlen-1)){
    
    #We need to select our predictive group. Find maximum state:
    max.state=name.states[which.max(lrad(CoilVals$Value[match(name.states,CoilVals$Name)]))]
    min.state=name.states[which.min(lrad(CoilVals$Value[match(name.states,CoilVals$Name)]))]
    
    #Now look at the predictions for transition of max state to min state according to different groups
    preds=c()
    for (iG in conserved.group){
      g.tran=paste(paste0(min.state,"-",max.state),iG,sep="|")
      preds=c(preds,sum(CoilVals$Value[match(g.tran,CoilVals$Name)]*CoilVals$Value[match(iG,CoilVals$Name)]))
    }
    
    #Pick whichever maximizes
    group.sel=which.max(lrad(preds))
    if (length(group.sel)==0){group.sel=1}
    
    sel.v=paste0("[|]",conserved.group[[group.sel]],"$")
    TMat=c()
    for (iS in sel.v){
      TMat=cbind(TMat,CoilVals[grep(iS,CoilVals$Name),]$Value)
    }
    
    #Create transition matrix values based on selected predictive group
    CoilVals$Value[match(trans.names,CoilVals$Name)]=TMat%*%CoilVals$Value[match(conserved.group[[group.sel]],CoilVals$Name)]
    
    SMat=c()
    for (iS in name.states){
      SMat=cbind(SMat,CoilVals[grep(paste0("^[a-z]-",iS,"$"),CoilVals$Name),]$Value)
    }
    SMat=apply(SMat,2,function(aa)aa/sum(llen(aa))) #To handle leaks due to rounding?
    
    #Produce new probabilities
    CoilVals$Value[match(name.states,CoilVals$Name)]=SMat%*%CoilVals$Value[match(name.states,CoilVals$Name)]
    
    #Convert complex values to real
    Pmat=rbind(Pmat,llen(CoilVals$Value[match(name.states,CoilVals$Name)]))
    complex_states=rbind(complex_states,CoilVals$Value[match(name.states,CoilVals$Name)])
  }
  return(list(Pmat,complex_states))
}
