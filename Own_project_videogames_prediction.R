install.packages("tidyverse")
install.packages("caret")
install.packages("glmnet")
install.packages("janitor")
install.packages("Hmisc")

library(tidyverse) 
library(caret)     
library(janitor)   
library(Hmisc)
library(glmnet)

#loading the data
df_raw <- read_csv("video_game_sales_with_ratings.csv")
#cleaning up the coloumn names
df_clean <- df_raw %>%
  janitor::clean_names()

#defining the target variable "success" which means that if 1 = hit, if 0 = flop
df_target <- df_clean %>%
  mutate(
    global_sales = as.numeric(global_sales),
    success = ifelse(global_sales >= 1.0, 1, 0), #creating the binary target variable Hit=1,Flop=0
    success = factor(success, levels = c(0, 1), labels = c("Flop", "Hit")) #creating "success" as a factor
  ) %>%
  filter(!is.na(global_sales)) #removing the rows where global_sales cannot be converted or it is NA
df_target %>% #create a tibble to check for distribution of my new factor
  count(success) %>%
  mutate(proportion = n / sum(n))
df_target %>% #checking for missing values in the coloumns
  is.na() %>% 
  colSums()

#Now I need to handle missing values. I chose imputation with the median for this purpose
df_impute <- df_target %>%
  mutate(
    user_score = ifelse(user_score == "tbd", NA, user_score), # the user score often contain tbd, that should be NA first
    user_score = as.numeric(user_score), #checking if values are numeric
    #imputing missing scores with median
    critic_score = impute(critic_score, fun = median), 
    critic_count = impute(critic_count, fun = median),
    user_score = impute(user_score, fun = median),
    user_count = impute(user_count, fun = median),
    year_of_release = impute(year_of_release, fun = median) 
  )
#verifying the missing values are actually gone -> there is still some left
df_impute %>% 
  is.na() %>% 
  colSums()
#for the name and genre only 2 left, so for simplicity i removed them
df_complete <- df_impute %>%
  filter(!is.na(name), !is.na(genre))
#for the rating and developer coloumn, there are over 6600 missing values so I introduce a new category "Unknown"
df_final_clean <- df_complete %>%
  mutate(
    rating = replace_na(rating, "Unknown"),
    developer = replace_na(developer, "Unknown")
  )
#final checking if everything is good
cat("\nFinal Check for NA Counts:\n")
df_final_clean %>% 
  is.na() %>% 
  colSums()
#now I need to rerun the feature encoding 
categorical_features_final <- c("platform", "genre", "publisher", "rating", "developer")
#creating new preprocessing object preP including all relevant predictors
preP_final <- dummyVars(
  formula = success ~ platform + genre + publisher + critic_score + user_score + year_of_release + rating + developer, 
  data = df_final_clean, 
  fullRank = TRUE 
)
#applying the preprocessing recipe
df_processed_final <- predict(preP_final, newdata = df_final_clean) %>%
  as_tibble() %>%
  bind_cols(select(df_final_clean, success)) #adding target variable "success"

#optional code to verify the final data structure - it will fill up the terminal because of the new rows and variables
#print(glimpse(df_processed_final))

#now the data is ready for the splitting into training and test set
set.seed(2025) #reproducibility
#I split the data 80/20 for training and testing
#creating the split index
train_index <- createDataPartition(
  y = df_processed_final$success, # Stratify by the 'success' factor
  p = 0.8, 
  list = FALSE #this ensures that the output is a vector of row numbers
)
train_set <- df_processed_final[train_index, ] #80% for training
test_set <- df_processed_final[-train_index, ] #20% for testing

#now optional checking if the splitting was a success
#verifying for the stratification
#cat("\ntraining set ratio:\n")
#print(prop.table(table(train_set$success)))
#cat("\ntesting set ratio:\n")
#print(prop.table(table(test_set$success)))
#they should be nearly identical and now the data is ready and correctly split

#1.model
#training the logistic regression model
model_logit <- train(
  success ~ ., 
  data = train_set,
  method = "glm",            #specifies the logistic regression
  family = "binomial",       # this is necessary for binary classification
  preProcess = NULL,         #null beacuse the data is preprocessed already
  trControl = trainControl(method = "none") #no cross-validation is needed for the basic model
)
#now I test my model on the unseen testing set
#predicting probabilities on the testing set
predictions_prob <- predict(model_logit, newdata = test_set, type = "prob")

#predicting the class (hit or flop) based on the highest probability
predictions_class <- predict(model_logit, newdata = test_set, type = "raw")

#this is for generating the confusion matrix to evaluate model performance
confusion_matrix <- confusionMatrix(predictions_class, test_set$success)

#displaying the evaluation results
print(confusion_matrix)

#key metrics
cat("\nAccuracy:", confusion_matrix$overall['Accuracy'], "\n")
cat("Sensitivity:", confusion_matrix$byClass['Sensitivity'], "\n")
cat("Specificity:", confusion_matrix$byClass['Specificity'], "\n")

#2.model
#installing the randomForest package and loading
install.packages("randomForest")
library(randomForest)

#using method="rf" for randomforest training on the training set
model_rf <- train(
  success ~ ., 
  data = train_set,
  method = "rf",
  trControl = trainControl(method = "none") #using "none" for simplicity
)
#evaluating the 2. model
#predicting the classes on the unseen test set
predictions_rf_class <- predict(model_rf, newdata = test_set, type = "raw")

#generating the confusion matrix
confusion_matrix_rf <- confusionMatrix(predictions_rf_class, test_set$success)

#displaying the results of confusion matrix
print(confusion_matrix_rf)

#key metrics
cat("accuracy:", confusion_matrix_rf$overall['Accuracy'], "\n")
cat("sensitivity:", confusion_matrix_rf$byClass['Sensitivity'], "\n")
cat("specificity:", confusion_matrix_rf$byClass['Specificity'], "\n")

#3.model - final -> this takes quite a while to run
#training the model
model_rf_balanced <- train(
  success ~ ., 
  data = train_set,
  method = "rf",
  trControl = trainControl(
    method = "none",        #no resampling needed 
    sampling = "smote"      #applying the smote method
  )
)
#predicting on the testing set
predictions_rf_balanced_class <- predict(model_rf_balanced, 
                                         newdata = test_set, 
                                         type = "raw")
#confusion matrix
confusion_matrix_rf_balanced <- confusionMatrix(predictions_rf_balanced_class, 
                                                test_set$success)
#displaying the confusion matrix
print(confusion_matrix_rf_balanced)

#key metrics of the final model
cat("accuracy:", confusion_matrix_rf_balanced$overall['Accuracy'], "\n")
cat("sensitivity:", confusion_matrix_rf_balanced$byClass['Sensitivity'], "\n")
cat("specificity:", confusion_matrix_rf_balanced$byClass['Specificity'], "\n")