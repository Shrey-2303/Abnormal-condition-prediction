library(ClusterR)
library(cluster)
library(neuralnet)
library(party)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ggplot2)
library(e1071)
library(MASS)
library(caTools)
library(class)

setwd("E:/R/assignment/biomedical")

minmax <- function(x)
{
  (x - min(x))/(max(x) - min(x))
}

# 1) Using logistic regression to find abnormal and normal cases

data <- read.csv("column_2C_weka.csv")
head(data)
summary(data)

response = data[,7]
features <- data[-7]
response_number = as.factor(response)
response_number_1 = ifelse(response == "Abnormal",1,0)
response_number_1

features_norm = apply(features,2,minmax)
data_new = cbind(features_norm,"class"=response_number_1)

data_new <- data.frame(data_new)
split <- sample.split(data_new,SplitRatio = 0.70)
traindata <- subset(data_new,split = TRUE)
testdata <- subset(data_new,split = FALSE)
model <- glm(class~.,data = testdata, family = 'binomial')
pred <- predict(model, traindata, type = 'response')
j<-1; TP=0; TN=0; FP=0; FN=0;
for (i in pred)
{
  if (i>=0.5 && data_new[j,7]==1)
    TP <- TP + 1
  if (i>=0.5 && data_new[j,7]==0)
    FP <- FP + 1
  if (i<=0.5 && data_new[j,7]==0)
    TN <- TN + 1
  if (i<=0.5 && data_new[j,7]==1)
    FN <- FN + 1
  j <- j +1
}
accuracy <- (TP+TN)/(TN+TP+FP+FN)*100
accuracy


# 2) Using decision Trees and random forest for classification

data_more <- read.csv("column_3C_weka.csv")
data_more[,7] <- as.factor(data_more[,7])
head(data_more)

split <- sample.split(data_more$class, SplitRatio = 0.7)
train <- subset(data_more, split == TRUE)
test <- subset(data_more, split == FALSE)


#creating tree using party
tree1 <- ctree(class~., data = train)
tree1

plot(tree1)

#predicting training and testing values and calculating accuracy
train_pd_party <- predict(tree1, train)
train_pd_party
test_pd1_party <- predict(tree1,test)
test_pd1_party

t1 <- table(Acutal = train$class, Predicted = train_pd_party)
t1
train_accuracy_party = sum(diag(t1)/sum(t1)) *100
train_accuracy_party

t1_tested <- table(Acutal = test$class, Predicted = test_pd1_party)
t1_tested
test_accuracy_party = sum(diag(t1_tested)/sum(t1_tested)) *100
test_accuracy_party


### using rpart package
tree2 <- rpart(class~., data = train)
tree2

rpart.plot(tree2)

#pd_rpart <- predict(tree2, train, type="prob")
#pd
pd_rpart_train <- predict(tree2, train,type = "class")
pd_rpart_train
pd_rpart_test <- predict(tree2,test, type = "class")
pd_rpart_test

t_rpart_train <- table(Acutal = train$class, Predicted = pd_rpart_train)
t_rpart_train
accuracy1_train = sum (diag(t_rpart_train)/sum(t_rpart_train)) * 100
accuracy1_train

t_rpart_test <- table(Acutal = test$class, Predicted = pd_rpart_test)
t_rpart_test
accuracy1_test = sum (diag(t_rpart_test)/sum(t_rpart_test))  * 100
accuracy1_test

train_acc_array = c()
test_acc_array = c()

#For 10 fold cross validation
for (i in 1:10)
{
  split <- sample.split(data_more$class, SplitRatio = 0.7)
  train <- subset(data_more, split == TRUE)
  test <- subset(data_more, split == FALSE)
  tree1 <- ctree(class~., data = train)
  train_pd_party <- predict(tree1, train)
  test_pd1_party <- predict(tree1,test)
  t1 <- table(Acutal = train$class, Predicted = train_pd_party)
  train_acc_array[i] = sum(diag(t1)/sum(t1)) *100
  t1_tested <- table(Acutal = test$class, Predicted = test_pd1_party)
  test_acc_array[i] = sum(diag(t1_tested)/sum(t1_tested)) *100
}

x <- c(1:10)
plot(test_acc_array~x,t="l",ylim=c(72,90))
lines(train_acc_array,t='l',col="blue")


# using random forest for better accuracy and importance
# Creating random trees

random_forest_classifier <- randomForest(x = train[-7],
                                         y=train$class,
                                         ntree=500)

# Predicting training data and testing data
random_forest_classifier
train_pred_RF <- predict(random_forest_classifier,train[-7])
test_pred_RF <- predict(random_forest_classifier,test[-7])

# Finding accuracy of training data adn testing data
table_RF_train <- table(Acutal = train$class, Predicted = train_pred_RF)
table_RF_train
train_accuracy_RF = sum(diag(table_RF_train)/sum(table_RF_train)) *100
train_accuracy_RF

table_RF_test <- table(Acutal = test$class, Predicted = test_pred_RF)
table_RF_test
test_accuracy_RF = sum(diag(table_RF_test)/sum(table_RF_test)) *100
test_accuracy_RF

# Plotting the error of random trees and importance of features
plot(random_forest_classifier)
importance(random_forest_classifier)




# 3) Using KNN

data <- read.csv("column_3C_weka.csv")
data[,7] <- as.factor(data[,7])
head(data)


features = data[-7]
response <- data[7]
response
summary(features)

features_norm = apply(features,2,minmax)
data_new = cbind(features_norm,"class"=response)
data_new
accuracy <- c()

split <- sample.split(data_new$class, SplitRatio = 0.7)
train <- subset(data_new, split == TRUE)
test <- subset(data_new, split == FALSE)

predict = knn(train[,-7],test[,-7],train[,7],k=4)
table1 = table(Actual = test[,7],predicted = predict)
table1
accuracy1 = sum (diag(table1)/sum(table1))
accuracy1

for (i in 1:100)
{
  predict = knn(train[,-7],test[,-7],train[,7],k=i)
  table1 = table(actual = test[,7],predicted = predict)
  accuracy1 = sum (diag(table1)/sum(table1))
  accuracy[i]=accuracy1*100
}

plot(accuracy, t="l")
plot(accuracy[1:30],t="l")





# 4) using SVM

data <- read.csv("column_3C_weka.csv")
kernellist <- c("linear","radial","polynomial",
                "sigmoid")
data[,7] <- as.factor(data[,7])
#using different kernels to see which one gives best model
accuracy_kernel <- c()
for( i in 1:4)
{
  model <- svm(class~.,
               data = data,kernel = kernellist[i])
  summary(model)
  pred <- predict(model,data=data)
  t <- table(actual = data$class, Predicted = pred)
  accuracy_kernel[i] = sum(diag(t))/sum(t) *100
  
}


accuracy_kernel

accuracy_cost <- c()

# Checking what is the best cost
# For different kernels using different values

gp <- c(10^seq(0,10))
gp
gp_length <- c(0:10)
gp_length

j=1
for (k in 1:4) 
{
  for( i in gp)
  {
    model <- svm(class~.,
                 data = data,kernel = kernellist[k],cost = i)
    pred <- predict(model,data=data)
    t <- table(actual = data$class, Predicted = pred)
    accuracy_cost[j] = sum(diag(t))/sum(t) *100
    j <- j+1
    #plot(model,data,Petal.Length~Petal.Width)
  }
  
  plot(gp_length,accuracy_cost[(j-11):(j-1)], t="l",
       xlab="powers of 10",ylab="accuracy",main=kernellist[k])
  
}

length(accuracy_cost)
accuracy_cost

