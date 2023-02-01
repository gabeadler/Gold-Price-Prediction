library(tidyverse)
library(caret)
library(randomForest)
library(glmnet)
library(reshape2)

gold_df <- read_csv("gld_price_data.csv")

head(gold_df)

## Finding NA
sum(!complete.cases(gold_df))

## Split Data
set.seed(42)
n <- nrow(gold_df)
id <- sample(1:n, size = 0.8*n)
train_df <- gold_df[id, ]
test_df <- gold_df[-id, ]

## Train Model
set.seed(16)
ctrl <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE)

## Linear Regression Model
linear_model <- train(GLD ~ .,
                        data = train_df %>% select(-Date),
                        method = "lm",
                        metric = "RMSE",
                        preProcess = c("center", "scale"),
                        trControl = ctrl)
## K-Nearest Neighbors Model
knn_model <- train(GLD ~ .,
                   data = train_df %>% select(-Date),
                   method = "knn",
                   metric = "RMSE",
                   preProcess = c("center", "scale"),
                   trControl = ctrl)

# Random Forest Model
rf_model <- train(GLD ~ .,
                  data = train_df %>% select(-Date),
                  method = "rf",
                  metric = "RMSE",
                  preProcess = c("center", "scale"),
                  trControl = ctrl)

# Ridge Regression Model
ridge_model <- train(GLD ~ .,
                    data = train_df %>% select(-Date),
                    method = "glmnet",
                    metric = "RMSE",
                    tuneGrid = expand.grid(alpha = 0,
                                          lambda = seq(0.001, 1, length=20)),
                    preProcess = c("center", "scale"),
                    trControl = ctrl)

# Lasso Regression Model
lasso_model <- train(GLD ~ .,
                    data = train_df %>% select(-Date),
                    method = "glmnet",
                    metric = "RMSE",
                    tuneGrid = expand.grid(alpha = 1, lambda = seq(0.001, 1, length=20)),
                    preProcess = c("center", "scale"),
                    trControl = ctrl)

#RMSE Result
result <- resamples(list(
  linearReg = linear_model,
  knn = knn_model,
  randomForest = rf_model,
  ridgeReg = ridge_model,
  lassoReg = lasso_model
))
summary(result)

## Model Performance (RMSE) Visualization
Model <- c("Linear Regression", "K-Nearest Neighbors", "Random Forest", "Ridge Regression", "Lasso Regression")
RMSE <- c(8.08, 2.36, 2.53, 8.30, 8.07)

rmse_df <- data.frame(Model, RMSE)
ggplot(rmse_df , aes(Model, RMSE)) +
  geom_col(fill = "#A7C7E7") + 
  geom_text(aes(label = RMSE), vjust = 1.5, colour = "white") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(title = "Traning Models Performance")

# Feature Importance
varImp(linear_model)

# Correlation
cor(gold_df$GLD, gold_df[ , c(2, 4, 5, 6)], method = "pearson")

## Heat Map Correlation
mydata <- gold_df[, c(3,2,4,5,6)]
cormat <- round(cor(mydata), 2)
melted_cormat <- melt(cormat)

get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}

upper_tri <- get_upper_tri(cormat)

melted_cormat <- melt(upper_tri, na.rm = TRUE)
reorder_cormat <- function(cormat){
  dd <- as.dist((1-cormat)/2)
  hc <- hclust(dd)
  cormat <-cormat[hc$order, hc$order]
}

cormat <- reorder_cormat(cormat)
upper_tri <- get_upper_tri(cormat)

melted_cormat <- melt(upper_tri, na.rm = TRUE)

ggplot(melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "#87CEEB", high = "#ff6961", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed() + 
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.justification = c(1, 0),
    legend.position = c(0.5, 0.7),
    legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                               title.position = "top", title.hjust = 0.5))

## Test Model
predict_gold <- predict(knn_model, newdata=test_df)
RMSE(predict_gold, test_df$GLD)

## Actual Price vs Predicted Price Line Chart
actual_value <- test_df$GLD
predicted_value <- predict_gold
df1 <- data.frame(actual_value, predicted_value)

df <- data.frame(id = seq_along(df1[, 1]),
                 df1)

df <- melt(df, id.vars = "id")

ggplot(df, aes(x = id, y = value, color = variable)) +
  geom_line(size = 0.75) +
  scale_color_manual(values = c("#D43F3A", "#46B8DA"), labels = c("Actual Value", "Predicted Value")) +
  guides(color = guide_legend(title = "")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(title = "Actual Price vs Predicted Price",
       x = "Number of Values",
       y = "Gold Price")
