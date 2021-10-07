# The goal of this practice session is to use the caret and tidyverse libraries in R. 
# However, a Random Forest Machine learning algorithm is applied on the dataset to predict wine quality. 
library(tidyverse)
library(readr)
library(caret)
# for Plotting
library(gridExtra)
library(grid)
library(ggridges)
library(ggthemes)
theme_set(theme_minimal())

# Loading the datasets
wine_data <- read_csv("winequality-red.csv") %>%
  mutate(quality=as.factor(ifelse(quality <6, "qual_low", "qual_high")))
head(wine_data)

# Using the gsub function to replace or remove spaces in columns
colnames(wine_data) <- gsub(" ", "_", colnames(wine_data))
# Viewing the dataset
glimpse(wine_data)

# Performing exploratory data analysis (EDA)
p1 <- wine_data %>%
  ggplot(aes(x=quality, fill=quality)) +
  geom_bar(alpha=0.8) +
  scale_fill_tableau() +
  guides(fill='none')

# Viewing high and low quality p1.
p1

p2 <- wine_data %>%
  gather(x, y, fixed_acidity:alcohol) %>%
  ggplot(aes(x=y, y=quality, color=quality, fill=quality)) +
  facet_wrap(~ x, scale='free', ncol=3) +
  scale_fill_tableau() +
  scale_color_tableau() +
  geom_density_ridges(alpha=0.8) +
  guides(fill='none', color='none')

# Viewing p2
p2

# Combiniing the p1 and p2
grid.arrange(p1, p2, ncol=2, widths=c(0.3, 0.7))

# To perform a Machine Learing task, let's create training and test sets
set.seed(42)
# First, let's create a data partition
idx <- createDataPartition(wine_data$quality,
                           p=0.8,
                           list=FALSE,
                           times=1)
# Creating the training set
wine_train <- as.vector(wine_data[ idx,])

# Creating  test set
wine_test <- wine_data[-idx,]

# Applying machine learning model. This introduces the cross-validation method. 
fit_control <- trainControl(method='repeatedcv',
                            number=5,
                            repeats=3)
set.seed(42)
# Applying the Random Forest Model
rf <- train(quality ~.,
            data=wine_train,
            method="rf",
            preProcess = c("scale", "center"),
            trControl=fit_control,
            verbose=FALSE)
# calling the Random Forest model
rf
# Let's test how good is the model
test_predict <- predict(rf, wine_test)
confusionMatrix(test_predict, as.factor(wine_test$quality))

# The model has a high accuracy of 81% correctly predicting a red wine as high quality or low quality.
# This model is subject to updating to test for better performance.