library(caret)
library(doParallel)
## read data
setwd("~/GitHub/datasciencecoursera/machine learning")
d <- read.csv("data\\pml-training.csv", na.strings = c("", "#DIV/0!","NA"))
col_nas <- colSums(is.na(d))
td <- d[, col_nas<1]
# get rid of bookkeeping columns
td <- td[,-(1:7)]
rm("d")
## test for nas in data set
sum(is.na(td))
## re varImp(). If you train a initial rf model on 10 to 20% partition
set.seed(1984)
trainPartition <- createDataPartition(y=td$classe, p=0.5, list=FALSE )
train1 <- td[trainPartition,]
test1 <- td[-trainPartition,]
testPartition <- createDataPartition(y=test1$classe, p=0.5, list=FALSE )
test2 <- test1[testPartition,]
# train control


# enable multi-core processing

cl <- makeCluster(detectCores())
registerDoParallel(cl)

# random forest fit
fitControl<-trainControl(method="cv", 3)
rffit<-train(classe~.,data=train1,method="rf",trControl=fitControl)

stopCluster(cl)


rfpredict <- predict(rffit, newdata = test2)
confusionMatrix(data = rfpredict, test2$classe)

# gbm boosting
cl <- makeCluster(detectCores())
registerDoParallel(cl)

# gbm fit
fitControl<-trainControl(method="cv", 3)
gbmfit<-train(classe~.,
             data=train1,
             method="gbm",
             verbose=FALSE,
             trControl=fitControl)

stopCluster(cl)

gbmpredict <- predict(gbmfit, newdata = test2)
confusionMatrix(data = gbmpredict, test2$classe)


### testing

testingRaw <- read.csv("data\\pml-testing.csv", na.strings = c("", "#DIV/0!","NA"))

rfpredict <- predict(rffit, newdata = testingRaw)


###### submission function
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(rfpredict)

