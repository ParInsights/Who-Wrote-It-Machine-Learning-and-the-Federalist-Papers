data_path=file.choose()
data=read.csv(data_path)
str(data)
sum(!complete.cases(data))
summary(data)
fp<-data[,c(-1,-2)]

#Scaling fp data 

fp2<-data.frame(scale(fp,center = T,scale = T))



#Plot (elbow method) to decide optimal number of clusters 
set.seed(10)
optim.cluster<-function(k){
  return(kmeans(fp2,k,nstart = 25)$tot.withinss)
}
k_values<-1:14
oc_values<-purrr::map_dbl(k_values,optim.cluster)
plot(x=k_values, y=oc_values,type="b",frame=F,xlab = "Number of clusters K",ylab="Total within-clusters (sum squared)")


#### Option 1: K means_Clustering ####
k_mean<-kmeans(fp2, centers = 4, nstart = 25)
k_mean
k_mean$centers


######Option 2: :K_means using Hartingan-Wong Method####

##HW algorithm to find centroid 
km_output1<-kmeans(fp2,centers=4,nstart = 25,iter.max = 100,algorithm = "Hartigan-Wong")
km_output1
km_output1$centers



#visualize the result
install.packages("cluster")
library(cluster)
install.packages("ggplot2")
library(ggplot2)

clusplot(fp2,km_output1$cluster,color=TRUE,shade=T,labels = 2,lines =0 )

km_df1<-data.frame(author=data$author,cluster1=km_output1$cluster)
ggplot(km_df1)+geom_polygon(aes(x=author,y=cluster1,group=author,fill=as.factor(cluster1)),color="red")+
  coord_fixed(1.3)+
  guides(fill=F)+
  theme_bw()

##### 2nd Algorithm: HAC######
hac_output<-hclust(dist(fp2,method = "euclidean"),method = "average") #average
hac_output2<-hclust(dist(fp2,method = "euclidean"),method = "complete") #complete 


#plot HAC 
plot(hac_output) ##average 
plot(hac_output2) ## complete 


#output desirable number of clusters after modeling

#average linkage
hac_cut<-cutree(hac_output,4)
hac_df1<-data.frame(author=data$author,cluster=hac_cut)
avg_Clus_plot<-clusplot(fp2,hac_cut,color=TRUE,shade=T,labels = 2,lines =0 )

#compelte linkage
hac_cut<-cutree(hac_output2,4)
hac_df1<-data.frame(author=data$author,cluster=hac_cut)
clusplot(fp2,hac_cut,color=TRUE,shade=T,labels = 2,lines =0 )




####3rd Algorithm: EM#####
install.packages("mclust")
library(mclust)

#visualize clusters
clPairs(fp2 [1:30],data$author) ##disputed


fit<-Mclust(fp2)
summary(fit)

#1. BIC (The Bayesian information criterion (BIC) 
## is used my mclust with is a test used to assess the fit of a model)
plot(fit,what= "BIC")
#2. classification
plot(fit, what = "classification")
length(fit$classification)

