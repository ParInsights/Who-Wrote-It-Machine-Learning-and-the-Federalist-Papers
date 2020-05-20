install.packages("rpart")
install.packages("caret")
install.packages("rpart.plot")
install.packages("RColorBrewer")
install.packages("dplyr")
library("caret")
library("rpart")
library("lattice")
library("ggplot2")
library("rpart.plot")
library("RColorBrewer")
library("dplyr")

##load data 
data_path=file.choose()
fedpapers=read.csv(data_path)
str(fedpapers)



################### PART 1 ###################
################### preprocess ###################
##Reference used: https://rstudio-pubs-static.s3.amazonaws.com/246827_b9a4314244084f00ad5139144a1997d8.html


##clean
## remove author + filename columns
fedpapers_clean = fedpapers[, -c(2)]
##Add Cleaning for HM. Remove all authors = HM. Rerun below code
fedpapers_clean<-subset(fedpapers_clean,author!="HM")
View(fedpapers_clean)

##Prepare  dataframes: training and test data

 
  #nonDisputed Authors 
  #create non_disputed df= excludes disputed files 
nondis_df = subset(fedpapers_clean,author!= 'dispt')

  #split nondis into train and test
  nondis_key<-createDataPartition(y=nondis_df$author,p=0.7,list = FALSE)
  train<-nondis_df[nondis_key,]
  test<-nondis_df[-nondis_key,]
  
  
##Disputed files: create df for disputed = include disputed files  
disputed_df = subset(fedpapers_clean,author == 'dispt')

################### PART 2 ###################
################### Decision tree ###################

##reference :  https://rpubs.com/chengjiun/52658, 
##reference: https://cran.r-project.org/web/packages/rpart/rpart.pdf


#build general decision tree using rpart for only author. 
fed_tree<-rpart(author ~., data = train, method = 'class')
rpart.plot(fed_tree) #viz tree


##inspect general model 1 
summary(fed_tree) #inspect tree
printcp(fed_tree) 
plotcp(fed_tree) #visualize cross-validation results 
##Note: Cross-valuidation  estimates how accurately the
    #model generlizes the unseen data. (how well it performs/perdicts)

    #plotcp = cp values plotted agaisnt geometric mean
      #to depict the deviation until the minimum value is 
      #reached. . 
    ##Look into setting minsplot = 10


##Tune decision  tree: 

##DT Tuned Model 1 
#minsplit = 10, maxdepth=1
  Fed_tuned1<-rpart(author ~., data = train, method = 'class',
                    control = rpart.control( minsplit=10, maxdepth = 1 ))
  rpart.plot(Fed_tuned1) 
  
  ##inspect Tuned Model 1 
  summary(Fed_tuned1) #inspect tree
  printcp(Fed_tuned1) 
  plotcp(Fed_tuned1) #visualize cross-validation results 


##DT Tuned Model 2 
#minsplit = 10, maxdepth=5
  Fed_tuned2<-rpart(author ~., data = train, method = 'class',
                    control = rpart.control( minsplit=10, maxdepth = 5 ))
  rpart.plot(Fed_tuned2) 
  
  ##inspect Tuned Model 2
  summary(Fed_tuned2) #inspect tree
  printcp(Fed_tuned2) 
  plotcp(Fed_tuned2) #visualize cross-validation results 
  
  
  

  ##DT Tuned Model 3
#minsplit = 5, maxdepth=5
  Fed_tuned3<-rpart(author ~., data = train, method = 'class',
                    control = rpart.control( minsplit=4, maxdepth = 2 ))
  rpart.plot(Fed_tuned3) 
  ##inspect Tuned Model 3
  summary(Fed_tuned3) #inspect tree
  printcp(Fed_tuned3) 
  plotcp(Fed_tuned3) #visualize cross-validation results 
  
  
  ##DT Tuned Model 4 - ONLY A CHE
  ##Fed_tuned4<-rpart(author ~., data = train, method = 'class',
                   # control =CK TO MAKE SURE 3 IS BEST 
  #minsplit = 3, maxdepth=4 rpart.control( minsplit=3, maxdepth = 4 ))
  ##rpart.plot(Fed_tuned4) 
  ##inspect Tuned Model 3
  #summary(Fed_tuned4) #inspect tree
  #printcp(Fed_tuned4) 
  #plotcp(Fed_tuned4) #visualize cross-validation results 
  
  
###Testing:#####
  
fed_test<-data.frame(predict(Fed_tuned3, newdata = test))
#fed_test<-data.frame(predict(Fed_tuned2, newdata = test))

fed_test<-fed_test %>% mutate(results = ifelse(Hamilton==1, 'Hamilton', ifelse(Jay==1, 'Jay', 'Madison')))
results<-fed_test %>% mutate(results = ifelse(Hamilton==1, 'Hamilton', ifelse(Jay==1, 'Jay', 'Madison')))

row.names(test)<- NULL
fed_test1<-data.frame(test %>% bind_cols(results))
str(fed_test1)
fed_test1$results<-as.factor(fed_test1$results)


confusionMatrix(fed_test1$results,fed_test1$author)

  ###################PART 3###################
  ################### Prediction ###################

fed_pred<-data.frame(predict(Fed_tuned3, newdata = disputed_df))
fed_pred
plot(fed_pred)

#method 2
fed_pred2<-predict.rpart(Fed_tuned3,newdata = disputed_df, na.action = na.omit, type = "prob")
fed_pred2
plot(fed_pred2)
