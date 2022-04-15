###Loading Libraries....

library(lattice)  # Used for Data Visualization
require(caret)    # for data pre-processing
require(pROC)     # for ROC Curves
library(ipred)    # for bagging and  k fold cv
library(e1071)    # for SVM
library(corrplot)
library(LogicReg)
###Let us read the data

data <- read.csv("C:/Users/SKY/Desktop/College Stuff/R LAB/HR_comma_sep.csv")
head(data)
summary(data)
colnames(data)
str(data)
###Convert sales and salary to factor

data$sales<-as.factor(data$sales)
data$salary<-as.factor(data$salary)
data$salary<-ordered(data$salary,levels=c("low","medium","high"))



##Data Visualization

### Now let's plot the Correlation Plot



library(corrplot)
cor(data[,1:8])
corrplot(cor(data[,1:8]), method="circle")

### Plotting some box plots

### Class 1 refers to people who left the company

ggplot(data, aes(x =  salary, y = satisfaction_level, fill = factor(left), colour = factor(left))) + 
  geom_boxplot(outlier.colour = "black") + xlab("Salary") + ylab("Satisfacion level") 

ggplot(data, aes(x =  salary, y = time_spend_company, fill = factor(left), colour = factor(left))) + 
  geom_boxplot(outlier.colour = NA) + xlab("Salary") + ylab("time_spend_company") 

ggplot(data, aes(x =  factor(time_spend_company), y = average_montly_hours, fill = factor(left), colour = factor(left))) + 
  geom_boxplot(outlier.colour = NA) + xlab("Time spend Company") + ylab("Average Monthly Hours") 



#Observations :
#### * Boxplot between Time Spent in Company and Salary :  In the dataset we find that the people leaving are more experienced (i.e. higher time spent in company on average) in the low and medium salary class.
#### * Boxplot between Satisfaction levels and Salary : In the dataset we find that the average satisfaction of the employees who left is lower than who haven't left
#### * Boxplot between Average Monthly Hours and Time spent Company: In the dataset we find that the people leaving have spent more hours at work

#Conclusion :
###Common traits of Good people leaving :
#### * Experienced                  
#### * Very low satisfaction levels                 
#### * Spend more time at work

###Possible Reasons for people leaving: 
#### * Experienced people may not be finding any challenges in work. Hence they leave.
#### * Work to Pay ratio may be high (because we find clear correlation only in low and medium salary ranges)


#Now let us predict who is leaving and who is not

###Splitting the data into training and test datasets


library(caret)
set.seed(1234)
splitIndex <- createDataPartition(data$left, p = .80,list = FALSE, times = 1)
trainSplit <- data[ splitIndex,]
testSplit <- data[-splitIndex,]
print(table(trainSplit$left))



####A few important terms :
#### * confusion matrix : In a confusion matrix each column of the matrix represents the instances in a predicted class while each row represents the instances in an actual class (or vice versa)
#### * Sensitivity (also called the true positive rate, the recall, or probability of detection[1] in some fields) measures the proportion of positives that are correctly identified as such (e.g., the percentage of sick people who are correctly identified as having the condition).
#### * Specificity (also called the true negative rate) measures the proportion of negatives that are correctly identified as such (e.g., the percentage of healthy people who are correctly identified as not having the condition).
#### * precision (also called positive predictive value) is the fraction of retrieved instances that are relevant.
#### * recall (also known as sensitivity) is the fraction of relevant instances that are retrieved.



###Now let's try  leftification models

### 1)Logistic regression

library(LogicReg)
ctrl <- trainControl(method = "cv", number = 5)
modelglm <- train(as.factor(left) ~. , data = trainSplit, method = "glm", trControl = ctrl)
summary(modelglm)

### predict
predictors <- names(trainSplit)[names(trainSplit) != 'left']
predglm <- predict(modelglm, testSplit)
summary(predglm)

### score prediction using AUC
confusionMatrix(predglm, as.factor(testSplit$left))
aucglm <- roc(as.numeric(testSplit$left), as.numeric(predglm),  ci=TRUE)
plot(aucglm, ylim=c(0,1), print.thres=TRUE, main=paste('Logistic Regression AUC:',round(aucglm$auc[[1]],3)),col = 'blue')


### 2)Random Forest

library(randomForest)
modelrf <- randomForest(as.factor(left) ~. , data = trainSplit)
importance(modelrf)

### predict
predrf <- predict(modelrf,testSplit)
summary(predrf)

### score prediction using AUC
confusionMatrix(predrf,as.factor(testSplit$left))
library(pROC)
aucrf <- roc(as.numeric(testSplit$left), as.numeric(predrf),  ci=TRUE)
plot(aucrf, ylim=c(0,1), print.thres=TRUE, main=paste('Random Forest AUC:',round(aucrf$auc[[1]],3)),col = 'blue')


##Comparing all the ROC Curves

library(ggplot2)
plot(aucrf, ylim=c(0,1), main=paste('ROC Comparison : RF(blue),C5.0(black),Adaboost(Green),SVM(Yellow),Logistic(Green)'),col = 'blue')
par(new = TRUE)
plot(auc1)
par(new = TRUE)
plot(aucsvm,col = "yellow")
par(new = TRUE)
plot(aucglm,col = "red")

#### Clearly, we find Random Forest is performing well. Therefore, we'll continue ahead with Random Forest

### Feature Engineering


# The importance routine in r for random forest models gives us the mean decrease gini value. Higher the Mean Decrease Gini value, more important the variable.
importance(modelrf)



#### Let's remove them least important variable and build the model with the remaining parameters and check the performance and repeat the process. Stop the process when there is a significant decrease in the performance measures

### Random Forest After Feature Engineering

modelrf2 <- randomForest(as.factor(left) ~.-promotion_last_5years-Work_accident-salary-sales , data = trainSplit)
importance(modelrf2)

### predict
predrf2 <- predict(modelrf2,testSplit)
summary(predrf2)

### score prediction using AUC
confusionMatrix(predrf2,as.factor(testSplit$left))
library(pROC)
aucrf2 <- roc(as.numeric(testSplit$left), as.numeric(predrf2),  ci=TRUE)
plot(aucrf2, ylim=c(0,1), print.thres=TRUE, main=paste('Random Forest AUC:',round(aucrf2$auc[[1]],3)),col = 'blue')



#### * On observing the Mean Decrease Gini for each variable, we find variables 'promotion_last_5years', 'Work_accident', 'salary' and 'sales' are has the least value.
#### * So if we remove 'promotion_last_5years', 'Work_accident', 'salary' and 'sales' one by one and build the model again, there is only a small change  in the AUC and other parameters 
#### * But if we remove any other variable after that and repeat the process, all the performance have a significant decrease

###The Problem Statement is : Why are the best and most experienced employees leaving prematurely ?

####So let's define who are the best and most experienced employees..        (above average)


####Last Evaluation >= 0.75
####time_spend_company >= 4
####number_project >= 5

###Now let us subset them


gp <- data[data$last_evaluation >= 0.75 & data$number_project >= 5 & data$time_spend_company >= 4,]
nrow(gp)



### Correlation Plot with the subsetted data


library(corrplot)
gp1 <- gp[,1:6 ]
gp1 <- cbind(gp1,gp[,8:10])
cor(gp[,1:7])
corrplot(cor(gp1[,1:7]), method="circle")


#Let's now repeat the entire process using Good people data


set.seed(1234)
splitIndex <- createDataPartition(gp$left, p = .80, list = FALSE, times = 1)
trainSplit <- gp[splitIndex,]
testSplit <- gp[-splitIndex,]
print(table(trainSplit$left))

###Now let's try other leftification models

### 1)Logistic regression

library(survival)
library(LogicReg)
library(mcbiopi)
ctrl <- trainControl(method = "cv", number = 5)
modelglm <- train(as.factor(left) ~. , data = trainSplit, method = "glm", trControl = ctrl)
summary(modelglm)

### predict
predictors <- names(trainSplit)[names(trainSplit) != 'left']
predglm <- predict(modelglm, testSplit)
summary(predglm)

### score prediction using AUC
confusionMatrix(predglm, as.factor(testSplit$left))
aucglm <- roc(as.numeric(testSplit$left), as.numeric(predglm),  ci=TRUE)
plot(aucglm, ylim=c(0,1), print.thres=TRUE, main=paste('Logistic Regression AUC:',round(aucglm$auc[[1]],3)),col = 'blue')



### 2)Random Forest

library(randomForest)
modelrf <- randomForest(as.factor(left) ~. , data = trainSplit)
importance(modelrf)

### predict
predrf <- predict(modelrf,testSplit)
summary(predrf)

### score prediction using AUC
confusionMatrix(predrf,as.factor(testSplit$left))
library(pROC)
aucrf <- roc(as.numeric(testSplit$left), as.numeric(predrf),  ci=TRUE)
plot(aucrf, ylim=c(0,1), print.thres=TRUE, main=paste('Random Forest AUC:',round(aucrf$auc[[1]],3)),col = 'blue')



##Comparison of ROC Curves

library(ggplot2)
plot(aucrf, ylim=c(0,1), main=paste('ROC Comparison : RF(blue),Logistic(Red)'),col = 'blue')
par(new = TRUE)

plot(aucglm,col = "red")



#### Clearly, we find Random Forest is performing well. Therefore, we'll continue ahead with Random Forest

### Feature Engineering


importance(modelrf)

### Random Forest After Feature Engineering

modelrf2 <- randomForest(as.factor(left) ~.-promotion_last_5years-Work_accident-salary-sales , data = trainSplit)
importance(modelrf2)

### predict
predrf2 <- predict(modelrf2,testSplit)
summary(predrf2)

### score prediction using AUC
confusionMatrix(predrf2,as.factor(testSplit$left))
library(pROC)
aucrf2 <- roc(as.numeric(testSplit$left), as.numeric(predrf2),  ci=TRUE)
plot(aucrf2, ylim=c(0,1), print.thres=TRUE, main=paste('Random Forest AUC:',round(aucrf2$auc[[1]],3)),col = 'blue')

####On observing the Mean Decrease Gini for each variable, we find variables 'promotion_last_5years', 'Work_accident', 'salary' and 'sales' are has the least value. So if we remove 'promotion_last_5years', 'Work_accident', 'salary' and 'sales' one by one and build the model again, there is only a small change  in the AUC and other parameters . But if we remove any other variable after that and repeat the process, all the performance have a significant decrease
