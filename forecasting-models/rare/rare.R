


library(rjson)
library(xts)
#library(argo)
library(glmnet)

evaluate <- function(true_ts, pred_ts) {
    mae = get_mae(true_ts, pred_ts)
    rmse = get_rmse(true_ts, pred_ts)
    mase = get_mase(true_ts, pred_ts)
    nash_sutcliffe = get_nash_sutcliffe_score(true_ts, pred_ts)
    scores = list()
    scores$mae = mae
    scores$rmse = rmse
    scores$mase = mase
    scores$nash_sutcliffe = nash_sutcliffe
    scores
}

get_nash_sutcliffe_score <- function(true_ts, pred_ts) {
    nash_sutcliffe = 1 - sum((true_ts - pred_ts) ** 2) / sum((true_ts - mean(true_ts)) ** 2)
    nash_sutcliffe
}


get_mae <- function(true_ts, pred_ts) {
    mae = mean(abs(true_ts - pred_ts))
    mae
}

get_rmse <- function(true_ts, pred_ts) {
    mse = mean((true_ts - pred_ts) ** 2)
    rmse = sqrt(mse)
    rmse
}

get_mase <- function(true_ts, pred_ts) {
    # mean absolute error
    mae = get_mae(true_ts, pred_ts)
    # mean absolute scaled error (see wiki)
    i = length(true_ts)
    j = length(true_ts) - 1
    error_with_last_count = get_mae(
        as.numeric(true_ts[2:i]), 
        as.numeric(true_ts[1:j])
        )
    # mase = mae / get_mae(true_ts[2:i], true_ts[1:j])
    mase = mae / error_with_last_count
    
    mase
}


rarx <- function(data, exogen=xts::xts(NULL), N_lag=1:21, N_training=104,
                 alpha=1, use_all_previous=TRUE){
    cat("Running Reguralized ARIMA\n")
    cat("Lenght of data =", length(data), fill = TRUE)
    cat("Rows of exog =", nrow(exogen), fill = TRUE)
    cat("Num of training instances =", N_training, fill = TRUE)
    
    parm <- list(
        N_lag = N_lag,
        N_training = N_training,
        alpha = alpha,
        use_all_previous = use_all_previous
    )
    lasso.pred <- c()
    lasso.coef <- list()
    lasso.fit <- NULL

    if (length(exogen) > 0)
        # exogenous variables must have the same timestamp as y
        if (!all(zoo::index(data) == zoo::index(exogen)))
            stop("error in data and exogen: their time steps must match")
    
    # starttime <- 1 + N_training + max(c(N_lag, 0))
    # starttime <- 1 + N_training
    starttime <- N_training  # start of test indx
    endtime <- nrow(data) - 1  # end of test indx
    # endtime <- nrow(data)
    # endtime <- starttime + 2
    
    print(starttime)
    print(endtime)

    if (use_all_previous) {
        training_idx <- (1 + max(c(N_lag, 0))):starttime
    } else{
        training_idx <- (starttime - N_training + 1):starttime
    }
    
    cat("training instances", length(training_idx), "\n")
       
    lagged_y <- sapply(N_lag, function(l)
        as.numeric(data[(training_idx) - l]))
    
    cat("lagged instances", nrow(lagged_y), "\n")
    
    if (length(lagged_y) == 0) {
        lagged_y <- NULL
    } else{
        colnames(lagged_y) <- paste0("lag_", N_lag)
    }
    
    if (length(exogen) > 0) {
        design_matrix <-
            cbind(lagged_y, data.matrix(exogen[training_idx,]))
    } else{
        design_matrix <- cbind(lagged_y)
    }
    
    y.response <- data[training_idx]
    cat("design matrix =", nrow(design_matrix), "\n")
    cat("response matrix =", length(y.response), "\n")
    
    if (is.finite(alpha)) {
        lasso.fit <- glmnet::cv.glmnet(
            x = design_matrix,
            y = y.response,
            nfolds = 10,
            grouped = FALSE,
            alpha = alpha
        )
    } else{
        lasso.fit <- lm(y.response ~ ., data = data.frame(design_matrix))
    }

    if(is.finite(alpha)){
      lasso.coef[[1]] <- as.matrix(coef(lasso.fit, lambda = lasso.fit$lambda.1se))
    }else{
      lasso.coef[[1]] <- as.matrix(coef(lasso.fit))
    }
    
    cat("Done with estimating parameters\n")
    cat("Estimated parameters:\n")
    print(class(lasso.coef))
    print(typeof(lasso.coef))
    print(class(lasso.coef[[1]]))
    # print(lasso.coef[[1]])
    # print(starttime:endtime)
    # small.lambda.betas <- coef(lasso.fit, s = "lambda.min")
    # print(small.lambda.betas)
    
    for (i in starttime:endtime) {
        # cat("iteration id =", i, "\n")
        lagged_y_next <- matrix(sapply(N_lag, function(l)
            as.numeric(data[i + 1 - l])), nrow = 1)
        # cat("lagged y", nrow(lagged_y_next), "\n")
        # print(lagged_y_next)

        if (length(lagged_y_next) == 0)
            lagged_y_next <- NULL
        if (length(exogen) > 0) {
            newx <- cbind(lagged_y_next, data.matrix(exogen[i + 1, ]))
        } else{
            newx <- lagged_y_next
        }
        # print(newx)
        # invisible(readline(prompt="Press [enter] to continue"))
        # temp = readline(prompt="Press [enter] to continue")

        if (is.finite(alpha)) {
            # cat("alpha is finite\n")
            lasso.pred[i + 1] <-
                predict(lasso.fit, newx = newx, s = "lambda.1se")
        } else{
            colnames(newx) <- c(paste0("lag_", N_lag), names(exogen))
            newx <- as.data.frame(newx)
            lasso.pred[i + 1] <- predict(lasso.fit, newdata = newx)
        }
        # print(lasso.pred)
    }

    # ts_pred = lasso.pred[starttime+1,]
    # print(ts_pred)
    # print(class(lasso.pred))
    # print(typeof(lasso.pred))
    # # print(lasso.pred)
    # print(lasso.pred[(starttime+1):(endtime+1)])
    
    ts_pred = lasso.pred[(starttime+1):(endtime+1)]
    ret = list()
    ret$pred = ts_pred
    ret$model = lasso.coef
    ret$param = parm
    ret$fit = lasso.fit
    ret
    # data$predict <- lasso.pred
    # lasso.coef <- do.call("cbind", lasso.coef)
    # 
    # colnames(lasso.coef) <-
    #     as.character(zoo::index(data))[starttime:endtime]
    # 
    # rarima <- list(pred = data$predict,
    #              coef = lasso.coef,
    #              parm = parm)
    # class(rarima) <- "rarima"
    # rarima
}


main <- function() {
    config_file = commandArgs(trailingOnly = TRUE)
    print(config_file)
    
    cfg <- fromJSON(file = config_file)
    
    outdir = cfg[['output_config']][[1]][['out_dir']]
    cat("Output dir =", outdir, "\n")
    
    cat("Config name =", cfg[['config_name']], "\n")
    
    for (dset in cfg[["data_source"]]) {
        cat(replicate(50, "="), sep = "", fill = TRUE)
        cat("Dataset =", dset[["endoname"]], fill = TRUE)
        cat(replicate(50, "="), sep = "", fill = TRUE)
        
        filepath = dset[["endopath"]]
        # cat("Endo var =", filepath, "\n")
        # print((dset[["name"]]))
        # df = read.csv(filepath)
        endog_z = read.zoo(filepath, sep = ",", index = 1, header = TRUE)
        xts_endog = as.xts(endog_z)
        names(xts_endog)  = "count"
        cat("Num of days =", length(xts_endog), fill = TRUE)
        # print(head(xts_endog, n=2))
        
        # length(x); head(x); tail(x)
        
        filepath = dset[["exogpath"]]
        exog_z = read.zoo(filepath, sep = ",", index = 1, header = TRUE)
        xts_exog = as.xts(exog_z)
        # print(head(xts_exog, n=2))
        
        normalized = dset[['normalized']]
        
        model_class_name = "RARE"
        # for (m in cfg[["model"]]) {
        #     # prediction_model = get_class(m['classname'])()
        #     model_name = m[['name']]
        #     model_param = m[['param']]
        #     
        #     cat(replicate(50, "="), sep = "", fill = TRUE)
        #     cat("Model =", model_name, fill = TRUE)
        #     cat(replicate(50, "="), sep = "", fill = TRUE)
        # }
        for (test_cfg in cfg[["test_config"]]) {
            train_start_date = test_cfg[['train_start_date']]
            train_end_date = test_cfg[['train_end_date']]
            test_start_date = test_cfg[['test_start_date']]
            test_end_date = test_cfg[['test_end_date']]
            # normalized = test_cfg[['normalized']]
            
            if (normalized) {
                xts_endog = (xts_endog - min(xts_endog)) / (max(xts_endog) - min(xts_endog))
                xts_exog = (xts_exog - min(xts_exog)) / (max(xts_exog) - min(xts_exog))
            }
            
            ts_train = xts_endog[paste(
                train_start_date, "/", train_end_date, sep = "")]
            ts_test = xts_endog[paste(
                test_start_date, "/", test_end_date, sep = "")]
            # cat("Train set size =", length(ts_train), fill = TRUE)
            # cat("Test set size =", length(ts_test), fill = TRUE)
            
            # ts_exog_train = xts_exog[paste(
            #     train_start_date, "/", train_end_date, sep = "")]
            # ts_exog_test = xts_exog[paste(
            #     test_start_date, "/", test_end_date, sep = "")]
            
            xts_endog_sub = xts_endog[paste(
                train_start_date, "/", test_end_date, sep = "")]
            xts_exog_sub = xts_exog[paste(
                train_start_date, "/", test_end_date, sep = "")]
            
            # print(length(xts_endog_sub))
            # print(length(ts_train))
            # print(length(ts_test))
            
            # print(nrow(xts_exog_sub))
            # print(head(xts_exog))
            # run argo model
            # rarima(ts_train, ts_exog_train, N_training = length(ts_train))
            # print(head(xts_endog_sub))
            # print(head(xts_exog_sub))
            result = rarx(xts_endog_sub, xts_exog_sub, N_training = length(ts_train))
            # print(result$pred)
            # print(length(result$pred))
            # date = zoo::index(ts_test)
            # print(date)
            df_coef = data.frame(result$model)
            # print(names(df_coef))
            # print(df_coef$X1 > 0)
            df_coef_sub = subset(df_coef, X1 > 0)
            print("Non-zero coefficients:")
            print(df_coef_sub)
            
            # output: save coefficients
            mainpath = file.path(outdir, dset[['name']], cfg[['config_name']], 
                                 "model_results", model_class_name)
            dir.create(mainpath, showWarnings = FALSE, recursive = TRUE)
           
            filename = paste(model_class_name , "_test" ,  test_cfg[['name']] , 
                             sep = "")
            if (normalized) {
                filename = paste(filename , "_normalized", sep = "")
            }
            # filename = paste(filename, "_coef.csv", sep = "")
            filename = paste(filename, "_coef.txt", sep = "")
                
            filepath = file.path(mainpath, filename)
            cat("Saving:", filepath, "\n")
            write.csv(file=filepath, x=df_coef_sub, row.names = TRUE)   
             
            print(class(result$pred))
            ts_test_pred = tail(result$pred, length(ts_test))
            # df = data.frame(zoo::index(ts_test), result$pred)
            df = data.frame(zoo::index(ts_test), ts_test_pred)
            # names(df) = 'count'
            # row.names(df) = 'date'
            # df['date'] = date
            # df['count'] = ts_pred
            # print(head(df))
            
            # xts_pred = xts::xts(x=result$pred, order.by = zoo::index(ts_test))
            xts_pred = xts::xts(x=ts_test_pred, order.by = zoo::index(ts_test))
            names(xts_pred) = 'count'
            row.names(xts_pred) = 'date'
            # print(xts_pred)
            
            # output: save pred series
            filename = paste(model_class_name , "_test" ,  test_cfg[['name']] , 
                             sep = "")
            if (normalized) {
                filename = paste(filename , "_normalized", sep = "")
            }
            filename = paste(filename, "_pred.csv", sep = "")
            filepath = file.path(mainpath, filename)
            cat("Saving:", filepath, "\n")
            write.csv(file=filepath, x=df, row.names = FALSE)
            
            # get various scores
            # print(head(xts_pred))
            # print(head(ts_test))
            
            scores = evaluate(ts_test, xts_pred)
            score_df = data.frame(scores)
            # print(scores)
            
            filename = paste(model_class_name , "_test" , 
                             test_cfg[['name']] , sep = "")
            if (normalized) {
                filename = paste(filename , "_normalized", sep = "")
            }
            filename = paste(filename, "_score.csv", sep = "")
            filepath = file.path(mainpath, filename)
            cat("Saving:", filepath, "\n")
            write.csv(file=filepath, x=score_df, row.names = FALSE)
        }
    }
}

main()




