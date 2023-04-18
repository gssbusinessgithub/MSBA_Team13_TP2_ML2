rm(list = ls())
library(randomForest)
library(e1071)
library(gbm)

#import data set and separate into train/test based on 0/1 or NA in shot_made_flag field
shooting <- read.csv("KobeBryantRIP.csv", header = T, stringsAsFactors = T)

#add feature
shooting$time_remaining <- shooting$minutes_remaining*60 + shooting$seconds_remaining
shooting$post_achilles <- as.factor(ifelse(shooting$shot_id >= 21003, 1, 0))

#modify features
shooting$shot_made_flag <- as.factor(shooting$shot_made_flag)
shooting$opponent[shooting$opponent == "NJN"] <- "BKN"
shooting$opponent[shooting$opponent == "VAN"] <- "MEM"
shooting$opponent[shooting$opponent == "NOH"] <- "CHA"
shooting$opponent[shooting$opponent == "SEA"] <- "OKC"
shooting$opponent <- as.character(shooting$opponent)
shooting$opponent <- as.factor(shooting$opponent)

#remove features
shooting <- subset(shooting, select = -c(game_event_id, game_id, lat, lon, loc_x, loc_y,
                                         minutes_remaining, seconds_remaining, team_id,
                                         team_name, game_date, matchup, shot_id, action_type))

#split into train and test based on shot_made_flag being 1/0 or NA
shooting <- shooting[!is.na(shooting$shot_made_flag),]
trainIndices <- sample(1:nrow(shooting), 0.9*nrow(shooting))
train <- shooting[trainIndices, ]
test <- shooting[-trainIndices, ]

#svm
svm.kobe <- svm(shot_made_flag ~ ., data = train, kernel = "linear", cost = 1)
yhat <- predict(svm.kobe, newdata = train)
confusion <- table(yhat, train$shot_made_flag)
sum(diag(confusion)) / sum(confusion)

#bagging
bag.kobe <- randomForest(shot_made_flag ~ ., data = train, mtry = 12)
yhat <- predict(bag.kobe)
confusion <- table(yhat, train$shot_made_flag)
sum(diag(confusion)) / sum(confusion)

#boosting
train$shot_made_flag <- as.numeric(train$shot_made_flag) - 1
boost.kobe <- gbm(shot_made_flag ~., 
                        data=train, 
                        distribution= 'bernoulli', 
                        n.trees = 5000, 
                        interaction.depth = 5,
                        shrinkage = 0.5,
                        verbose = F)
yhat <- predict(boost.kobe, newdata=train, type = "response")
yhat <- ifelse(yhat > 0.5, 1, 0)
confusion <- table(yhat, train$shot_made_flag)
sum(diag(confusion)) / sum(confusion)

#test svm
svm.test <- predict(svm.kobe, newdata = test)
mean(svm.test == test$shot_made_flag)

#test bagging
bag.test <- predict(bag.kobe, newdata = test)
mean(bag.test == test$shot_made_flag)

#test boosting
test$shot_made_flag <- as.numeric(test$shot_made_flag) - 1
boost.test <- predict(boost.kobe, newdata = test, type = "response")
boost.test <- ifelse(boost.test > 0.5, 1, 0)
mean(boost.test == test$shot_made_flag)

#model voting
svm.num <- as.numeric(svm.test) - 1
bag.num <- as.numeric(bag.test) - 1
boost.num <- as.numeric(boost.test) - 1
votes <- ifelse(svm.num + bag.num + boost.num > 1, 1, 0)
mean(votes == test$shot_made_flag)