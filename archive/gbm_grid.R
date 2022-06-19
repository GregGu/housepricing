


# hyper_params <- list(ntrees = 300,
#                      learn_rate = c(0.05,0.005),
#                      max_depth  = c(2:5))
# hyper_params <- list(ntrees = 10000,
#                      learn_rate = c(0.005),
#                      max_depth  = c(4:5))
# y <- "SalePrice"
# x <- setdiff(names(train), y)
# # Grid search for selecting the best model
# grid <- h2o.grid(x = x, y = y ,
#                  training_frame = train,
#                  fold_column = "fold",
#                  algorithm = "gbm", grid_id = "id7", hyper_params = hyper_params,seed=1,
#                  search_criteria = list(strategy = "Cartesian"))
# grid
# sortedGrid <- h2o.getGrid("id7", sort_by = "rmse", decreasing = FALSE)
# sortedGrid


y <- "SalePrice"
x <- setdiff(names(train), c(y, "fold"))
gbmfin <- h2o.gbm(x = x, y = y,
                  training_frame = train, seed=1,
                  nfolds = 5,
                  keep_cross_validation_predictions = TRUE,
                  ntrees = 16000,
                  learn_rate = .005,
                  max_depth = 5)
