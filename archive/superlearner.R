
```{r, eval = FALSE}
library(magrittr)
df <- read.csv("dffinal.csv")

too_low_var <-
  c(
    "Condition2",
    "Condition1",
    "Alley",
    "Fence",
    "BsmtCond",
    "Electrical",
    "LandSlope",
    "ExterCond",
    "Functional",
    "Neighborhood2")

# one hot encode
df <- df %>%
  dplyr::select(-too_low_var)%>%
  # one hot encode
  nomvars <- names(df)[sapply(df, class) %in% c('factor',"character")]
df2 <- df %>%
  dplyr::mutate_at(nomvars, as.factor)%>%
  dplyr::select(nomvars) %>%
  dplyr::select(-set)
library(caret)
dummy <- caret::dummyVars(" ~ .", data=df2, fullRank = FALSE)
df2 <- data.frame(predict(dummy, newdata = df2))
dff <- cbind(df2, df %>% dplyr::select(-nomvars))
dff$set <- df$set



train <- dff %>%
  dplyr::filter(set == "train") %>%
  dplyr::select(-set)

test <- dff %>%
  dplyr::filter(set == "test") %>%
  dplyr::select(-set)

library(SuperLearner)
listWrappers()


SuperLearner::listWrappers()
learners <- list("SL.xgboost", "SL.glmnet", "SL.earth")
y <- train %>% dplyr::pull(SalePrice)
x <- train %>% dplyr::select(-SalePrice)
# Stack models
set.seed(840)
sl <- SuperLearner::SuperLearner(
  Y = train$SalePrice,
  X = as.data.frame(x),
  # newX = test %>% dplyr::select(-SalePrice),
  SL.library = learners,
  cvControl=list(V=10))
sl

pred <- predict(sl, test %>% dplyr::select(-SalePrice), onlySL = TRUE)
dftest <- read.csv(here::here("data-raw", "test.csv"))

final <- data.frame(Id = dftest %>% dplyr::pull(Id), SalePrice = exp(pred$pred))
write.csv(final, row.names = FALSE, "predictions.csv")



imp_fun <- function(object, newdata) { # for permutation-based VI scores
  predict(object, newdata = newdata)$pred
}
par_fun <- function(object, newdata) { # for PDPs
  mean(predict(object, newdata = newdata)$pred)
}
library(doParallel) # load the parallel backend
cl <- makeCluster(5) # use 5 workers
registerDoParallel(cl)

var_imp <- vip::vi(sl, method = "permute", train = data, target = outcome, metric = "rmse",
                   pred_wrapper = imp_fun, nsim = 5, parallel = TRUE)
# Add sparkline representation of feature effects (# Figure 19)
vip::add_sparklines(var_imp[1L:15L, ], fit = sl, pred.fun = par_fun, train = data,
                    digits = 2, verbose = TRUE, trim.outliers = TRUE,
                    grid.resolution = 20, parallel = TRUE)

varvec <- var_imp$Variable
varvec <- gsub("\\..*","",varvec)
varvecun <- unique(varvec)
varvecun_short <- varvecun[1:round(length(varvecun)/2)]
dfmodel <- df %>%
  dplyr::select(varvecun_short)
```
