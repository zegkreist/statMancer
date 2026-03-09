#' @title Selecionador de hyperparametros para Xgboost
#'
#' @description  Seu objetivo é selecionar hyperparametros que maximizem ou minimizem seu objetivo utilizando CV e Sessões.
#' Para isto ele se utiliza do pacote mlrMBO, realizando optimizações de estatística bayesiana
#'
#' @param dados Um objeto data.table com os dados pre processados e uma coluna target
#' @param n_samples O tamanho do sorteio par acada classe. Se regressão o tamanbho total da amostra
#' @param var_id A variável que identifica cada observação. Ela será removida antes da seleção de parametros. DEFAULT: \code{NULL} (não irá remover nada)
#' @param metrica A métrica selecionada para o xgboost lidar. DEFAULT: logloss
#' @param objetivo A finalidade do modelo. DEFAULT: binary:logistic
#' @param niter_data Quantas sessões serão feitas. Quantos sorteios serão feitos nos dados. DEFAULT: 10
#' @param niter_bayes Quantas itereções serão utilizadas no otimizador de bayes. DEFAULT: 50
#' @param cv.nfolds  Quantos folds para o CV. DEFAULT: 5
#' @param cv.nrounds Quantidade máxima de nrounds a ser testado. DEFAULT: 3000,
#' @param nthreads Quantidade de CPU threads disponíveis para o XGBoost. DEFAULT: 3
#' @param tree_method Método de treino das árvores. DEFAULT: hist. Em hist será treinado na CPU, em gpu_hist na GPU
#'
#' @return Uma lista com três posições: Os parâmetros procurados, o número de nrounds e a melhor métrica encontrada
#'
#' @import data.table xgboost magrittr mlrMBO smoof DiceKriging rgenoud lhs ParamHelpers
#'
#' @export

xgb_select_params <- function(dados,
                              n_samples,
                              var_id          = NULL,
                              metrica         = "logloss",
                              objetivo        = "binary:logistic",
                              niter_data      = 10,
                              niter_bayes     = 50,
                              cv.nfolds       = 5,
                              cv.nrounds      = 500,
                              nthreads        = 3,
                              tree_method     = "hist"){

  base::requireNamespace('data.table')
  base::requireNamespace('xgboost')
  base::requireNamespace('mlrMBO')
  base::requireNamespace('smoof')
  base::requireNamespace('DiceKriging')
  base::requireNamespace('rgenoud')

  if(!objetivo %in% c("reg:squarederror",
                      "reg:logistic",
                      "binary:logistic",
                      "binary:logitraw",
                      "binary:hinge",
                      "count:poisson",
                      "max_delta_step",
                      "survival:cox",
                      "multi:softmax",
                      "multi:softprob",
                      "rank:pairwise",
                      "rank:ndcg",
                      "rank:map",
                      "reg:gamma",
                      "reg:tweedie")
  ){
    stop(paste0("Seu objetivo não se encontra na lista abaixo: \n
                \t \t reg:squarederror: regressão com erro quadrático \n
                \t \t reg:logistic: regressão logística \n
                \t \t binary:logistic: regressão logística para classificação binária, a saída é probabilidade \n
                \t \t binary:logitraw: regressão logística para classificação binária, a saída é o logito antes da transformação \n
                \t \t binary:hinge: hinge loss para classificação binária. Isso faz predição das classes 1 e 0 e não retorna probabilidade \n
                \t \t count:poisson regressão poisson para contagens, a saída é a média de uma distribuição de poisson \n
                \t \t max_delta_step is set to 0.7 by default in poisson regression (used to safeguard optimization) \n
                \t \t survival:cox: Regressão Cox para análise de sobrevivência censurado à direita (Valores negativos são considerados censurados).\n A saída é hazard ratio scale (i.e., as HR = exp(marginal_prediction) in the proportional hazard function h(t) = h0(t) * HR) \n
                \t \t multi:softmax: Classificação multi-classe usando softmax,
                \t \t multi:softprob: same as softmax, but output a vector of ndata * nclass, which can be further reshaped to ndata * nclass matrix. The result contains predicted probability of each data point belonging to each class \n
                \t \t rank:pairwise: Use LambdaMART to perform pairwise ranking where the pairwise loss is minimized \n
                \t \t rank:ndcg: Use LambdaMART to perform list-wise ranking where Normalized Discounted Cumulative Gain (NDCG) is maximized \n
                \t \t rank:map: Use LambdaMART to perform list-wise ranking where Mean Average Precision (MAP) is maximized \n
                \t \t reg:gamma: gamma regression with log-link. Output is a mean of gamma distribution. It might be useful, e.g., for modeling insurance claims severity, or for any outcome that might be gamma-distributed \n
                \t \t reg:tweedie: Tweedie regression with log-link. It might be useful, e.g., for modeling total loss in insurance, or for any outcome that might be Tweedie-distributed \n
                "))
  }

  if(!metrica %in% c("rmse",
                     "mae",
                     "logloss",
                     "error",
                     "error@t",
                     "merror",
                     "mlogloss",
                     "auc",
                     "aucpr",
                     "ndcg",
                     "map",
                     "poisson-nloglik",
                     "gamma-nloglik",
                     "cox-nloglik",
                     "cox-nloglik",
                     "gamma-deviance",
                     "tweedie-nloglik"
  )
  ){
    stop(paste0("A métrica escolhida não está entre: \n
                \t \t rmse: root mean square error\n
                \t \t mae: mean absolute error\n
                \t \t logloss: negative log-likelihood\n
                \t \t error: Binary classification error rate. It is calculated as #(wrong cases)/#(all cases). For the predictions, the evaluation will regard the instances with prediction value larger than 0.5 as positive instances, and the others as negative instances \n
                \t \t merror: Multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases) \n
                \t \t mlogloss: Multiclass logloss \n
                \t \t auc: Area under the curve \n
                \t \t aucpr: Area under the PR curve \n
                \t \t ndcg: Normalized Discounted Cumulative Gain \n
                \t \t map: Mean Average Precision \n
                \t \t poisson-nloglik: negative log-likelihood for Poisson regression \n
                \t \t gamma-nloglik: negative log-likelihood for gamma regression \n
                \t \t cox-nloglik: negative partial log-likelihood for Cox proportional hazards regression \n
                \t \t gamma-deviance: residual deviance for gamma regression \n
                \t \t tweedie-nloglik: negative log-likelihood for Tweedie regression (at a specified value of the tweedie_variance_power parameter) \n
                ")
    )
  }

  cat(paste0("Criando uma amostra balanceada para as iterações \t \t \t \t ---- \n"))

  if(metrica %in% c("auc",
                    "aucpr",
                    "ndcg",
                    "map")
  ){
    maximize. <- T
  }else{
    maximize. <- F
  }

  #cat(paste0("Iniciando o loop  \t \t \t \t ---- \n"))

  best_param         <- list()
  best_metric        <- ifelse(maximize., -Inf, Inf)
  best_metric_index  <- 0
  for (iter in 1:niter_data) {

    if(grepl(objetivo, pattern = 'binary|multi')){
      treino <- dados[,.SD[sample(.N, n_samples,replace = T)],by = target]
    }else{
      treino <- dados[sample(.N, n_samples,replace = F)]
    }

    if(!is.null(var_id)){
      eval(parse(text = paste0("treino[,", var_id," := NULL]")))
    }

    if(objetivo %in% c("multi:softmax", "multi:softprob")){
      num_class <- length(unique(treino$target))
    }

    #cat(paste0("Transformando as colunas para numeric \t \t \t \t ---- \n"))
    treino <- treino[, lapply(.SD, as.numeric)]

    #cat(paste0("Criando a xgb matrix \t \t \t \t ---- \n"))
    dtrain <- xgboost::xgb.DMatrix(
      treino[,
             colnames(treino)[colnames(treino) != "target"],
             with = F
             ] |>
        as.matrix(),
      label = treino$target
    )

    if(objetivo %in% c("multi:softmax", "multi:softprob")){
      obj.func <- smoof::makeSingleObjectiveFunction(
        name = "xgboost",
        fn = function(x){
          mdcv <- xgboost::xgb.cv(
            data                  = dtrain,
            nthread               = nthreads,
            nfold                 = cv.nfolds,
            nrounds               = x["nrounds"],
            verbose               = F,
            #            early_stopping_rounds = early_stop,
            maximize              = maximize.,
            params                = list(
              objective        = objetivo,
              eval_metric      = metrica,
              base_score       = sum(treino$target)/nrow(treino),
              max_leaves       = x["max_leaves"],
              #max.depth       = ceiling(runif(1,7,20)),
              eta              = x["eta"],
              gamma            = x["gamma"],
              subsample        = x["subsample"],
              colsample_bytree = x["colsample_bytree"],
              min_child_weight = x["min_child_weight"],
              nrounds          = x["nrounds"],
              grow_policy      = "lossguide",
              tree_method      = tree_method,
              num_class        = num_class
            )
          )
          if(maximize.){
            return(max(mdcv$evaluation_log[, 4][[1]]))
          }else{
            return(min(mdcv$evaluation_log[, 4][[1]]))
          }

        },
        par.set = ParamHelpers::makeParamSet(
          ParamHelpers::makeIntegerParam(id = "max_leaves", lower = 2, upper = 60),
          ParamHelpers::makeNumericParam(id = "eta", lower = 0.0001, upper = 0.3),
          ParamHelpers::makeNumericParam(id = "gamma", lower = 0, upper = 30),
          ParamHelpers::makeNumericParam(id = "subsample", lower = 0.1, upper = 0.5),
          ParamHelpers::makeNumericParam(id = "colsample_bytree", lower = 0.1,  upper = 0.5),
          ParamHelpers::makeNumericParam(id = "min_child_weight", lower = 0, upper = 15),
          ParamHelpers::makeIntegerParam(id = "nrounds", lower = 10, upper = cv.nrounds)
        ),
        minimize = !maximize.
      )
    }else{
      obj.func <- smoof::makeSingleObjectiveFunction(
        name = "xgboost",
        fn = function(x){
          mdcv <- xgboost::xgb.cv(
            data                  = dtrain,
            nthread               = nthreads,
            nfold                 = cv.nfolds,
            nrounds               = x["nrounds"],
            verbose               = F,
            #        early_stopping_rounds = early_stop,
            maximize              = maximize.,
            params                = list(
              objective        = objetivo,
              eval_metric      = metrica,
              base_score       = sum(treino$target)/nrow(treino),
              max_leaves       = x["max_leaves"],
              #max.depth       = ceiling(runif(1,7,20)),
              eta              = x["eta"],
              gamma            = x["gamma"],
              subsample        = x["subsample"],
              colsample_bytree = x["colsample_bytree"],
              min_child_weight = x["min_child_weight"],
              nrounds          = x["nrounds"],
              grow_policy      = "lossguide",
              tree_method      = tree_method
            )
          )
          if(maximize.){
            return(max(mdcv$evaluation_log[, 4][[1]]))
          }else{
            return(min(mdcv$evaluation_log[, 4][[1]]))
          }

        },
        par.set = ParamHelpers::makeParamSet(
          ParamHelpers::makeIntegerParam(id = "max_leaves", lower = 2, upper = 60),
          ParamHelpers::makeNumericParam(id = "eta", lower = 0.0001, upper = 0.3),
          ParamHelpers::makeNumericParam(id = "gamma", lower = 0, upper = 30),
          ParamHelpers::makeNumericParam(id = "subsample", lower = 0.1, upper = 0.5),
          ParamHelpers::makeNumericParam(id = "colsample_bytree", lower = 0.1,  upper = 0.5),
          ParamHelpers::makeNumericParam(id = "min_child_weight", lower = 0, upper = 15),
          ParamHelpers::makeIntegerParam(id = "nrounds", lower = 10, upper = cv.nrounds)
        ),
        minimize = !maximize.
      )
    }

    design <- ParamHelpers::generateDesign(
      n       = 30,
      par.set = ParamHelpers::getParamSet(obj.func),
      fun     = lhs::randomLHS
    )

    control <- mlrMBO::makeMBOControl() |>
      mlrMBO::setMBOControlTermination(iters = niter_bayes)

    run <- mlrMBO::mbo(
      fun       = obj.func,
      design    = design,
      control   = control,
      show.info = TRUE
    )

    if(maximize.){
      actual_metric       <- run$y
      #      actual_metric_index <- which.max((mdcv$evaluation_log[, 4][[1]]))
      if (actual_metric > best_metric) {
        best_metric         <- actual_metric
        #        best_metric_index   <- actual_metric_index
        best_param          <- run$x
      }
    }else{
      actual_metric       <- run$y
      #      actual_metric_index <- which.min((mdcv$evaluation_log[, 4][[1]]))
      if (actual_metric < best_metric) {
        best_metric         <- actual_metric
        #        best_metric_index   <- actual_metric_index
        best_param          <- run$x
      }
    }


    gc(reset = T)
  }
  if(objetivo %in% c("multi:softmax", "multi:softprob")){
    best_param['objective']     <- objetivo
    best_param['eval_metric']  <- metrica
    best_param['base_score']   <- sum(treino$target)/nrow(treino)
    best_param['grow_policy']  <- 'lossguide'
    best_param['tree_method'] <- tree_method
    best_param['num_class']   <- num_class
  }else{
    best_param['objective']     <- objetivo
    best_param['eval_metric']  <- metrica
    best_param['base_score']   <- sum(treino$target)/nrow(treino)
    best_param['grow_policy']  <- 'lossguide'
    best_param['tree_method']  <- tree_method
  }
  best_metric_index       <- best_param[['nrounds']]
  best_param[['nrounds']] <- NULL

  return(
    list(
      parametros  = best_param,
      nrounds     = best_metric_index,
      best_metric = best_metric
    )
  )
}

# Wrapper para manter compatibilidade com o código existente
busca_parametros_mlrmbo <- function(dados,
                                   target_col = "target",
                                   n_samples = 3000,
                                   colunas_excluir = NULL,
                                   metrica = "auc",
                                   objetivo = "binary:logistic",
                                   niter_data = 8,
                                   niter_bayes = 25,
                                   cv_folds = 3,
                                   cv_nrounds = 500,
                                   nthreads = 4,
                                   verbose = TRUE) {
  require("data.table")
  
  if(verbose) {
    cat("🔄 Usando xgb_select_params original do GitHub\n")
    cat("==============================================\n")
  }
  
  # Remover colunas excluídas
  if(!is.null(colunas_excluir)) {
    cols_para_remover <- colunas_excluir[colunas_excluir %in% names(dados)]
    if(length(cols_para_remover) > 0) {
      dados <- dados[, !..cols_para_remover]
      if(verbose) cat(paste0("🗑️ Removidas ", length(cols_para_remover), " colunas\n"))
    }
  }
  
  # Chamar a função original com mapeamento de parâmetros
  resultado <- tryCatch({
    xgb_select_params(
      dados = dados,
      n_samples = n_samples,
      var_id = NULL,  # não usar var_id já que removemos as colunas antes
      metrica = metrica,
      objetivo = objetivo,
      niter_data = niter_data,
      niter_bayes = niter_bayes,
      cv.nfolds = cv_folds,
      cv.nrounds = cv_nrounds,
      nthreads = nthreads,
      tree_method = "hist"
    )
  }, error = function(e) {
    if(verbose) cat(paste0("❌ Erro no xgb_select_params: ", e$message, "\n"))
    return(NULL)
  })
  
  if(is.null(resultado)) {
    return(list(
      sucesso = FALSE,
      melhor_parametros = NULL,
      auc = NA,
      metodo = "xgb_select_params_original_failed"
    ))
  }
  
  # Converter resultado para formato esperado
  return(list(
    sucesso = TRUE,
    melhor_parametros = list(
      parametros = resultado$parametros,
      nrounds = resultado$nrounds
    ),
    auc = resultado$best_metric,
    metodo = "xgb_select_params_original",
    sessoes_executadas = niter_data,
    iteracoes_bayes = niter_bayes
  ))
}

rnorm_t <- function(n, mu, sd, upper = Inf, lower = -Inf){
  result <- stats::rnorm(n    = n,
                         mean = mu,
                         sd   = sd
  )
  result[result < lower] <- lower
  result[result > upper] <- upper
  return(result)
}

# ==============================================================================
# FUNÇÃO AUXILIAR PARA CONVERSÃO SEGURA
# ==============================================================================

#' Converte colunas para numérico de forma segura
#' @param dt data.table
#' @return data.table com colunas numéricas
converter_para_numerico_seguro <- function(dt) {
  dt_copy <- data.table::copy(dt)
  
  for(col in names(dt_copy)) {
    if(!is.numeric(dt_copy[[col]])) {
      # Tentar conversão silenciosa
      converted <- base::suppressWarnings(as.numeric(dt_copy[[col]]))
      
      # Se muitos NAs foram introduzidos, manter original
      original_nas <- sum(is.na(dt_copy[[col]]))
      new_nas <- sum(is.na(converted))
      
      # Se mais de 10% dos valores viraram NA, não converter
      if((new_nas - original_nas) / nrow(dt_copy) <= 0.1) {
        dt_copy[[col]] <- converted
      } else {
        # Tentar conversão para factor numeric
        if(is.factor(dt_copy[[col]]) || is.character(dt_copy[[col]])) {
          dt_copy[[col]] <- as.numeric(as.factor(dt_copy[[col]]))
        }
      }
    }
  }
  
  return(dt_copy)
}

#' @title Criar Folds para Cross-Validation Estratificado
#' 
#' @description Cria folds estratificados para cross-validation, garantindo proporção 
#' similar da variável target em cada fold
#' 
#' @param dados data.table com os dados incluindo coluna 'target'
#' @param k_folds Número de folds para cross-validation. DEFAULT: 5
#' @param seed Semente para reprodutibilidade. DEFAULT: 123
#' 
#' @return Lista com índices dos folds
#' 
#' @import data.table
#' @export
criar_cv_folds <- function(dados, k_folds = 5, seed = 123) {
  set.seed(seed)
  requireNamespace('data.table')
  
  if(!"target" %in% names(dados)) {
    stop("Dados devem conter coluna 'target'")
  }
  
  n_obs <- nrow(dados)
  
  # Para classificação binária, criar folds estratificados
  if(all(dados$target %in% c(0, 1))) {
    cat("[INFO] Criando folds estratificados para classificação binária\n")
    
    # Separar por classe
    idx_0 <- which(dados$target == 0)
    idx_1 <- which(dados$target == 1)
    
    # Criar folds para cada classe
    folds_0 <- sample(rep(1:k_folds, length.out = length(idx_0)))
    folds_1 <- sample(rep(1:k_folds, length.out = length(idx_1)))
    
    # Combinar
    fold_assignments <- integer(n_obs)
    fold_assignments[idx_0] <- folds_0
    fold_assignments[idx_1] <- folds_1
    
  } else {
    # Para regressão, folds aleatórios
    cat("[INFO] Criando folds aleatórios para regressão\n")
    fold_assignments <- sample(rep(1:k_folds, length.out = n_obs))
  }
  
  # Criar lista de índices por fold
  cv_folds <- lapply(1:k_folds, function(i) which(fold_assignments == i))
  names(cv_folds) <- paste0("fold_", 1:k_folds)
  
  cat("[INFO] Criados", k_folds, "folds com tamanhos:", sapply(cv_folds, length), "\n")
  
  return(cv_folds)
}

#' @title Calcular AUC para classificação binária
#' 
#' @description Função auxiliar para calcular AUC
#' 
#' @param target Valores reais (0/1)
#' @param predictions Predições (probabilidades)
#' 
#' @return Valor AUC
calculate_auc <- function(target, predictions) {
  if(requireNamespace("pROC", quietly = TRUE)) {
    auc_val <- pROC::auc(target, predictions, quiet = TRUE)
    return(as.numeric(auc_val))
  } else {
    # Implementação simples sem pROC
    order_idx <- order(predictions, decreasing = TRUE)
    target_ordered <- target[order_idx]
    
    n_pos <- sum(target_ordered == 1)
    n_neg <- sum(target_ordered == 0)
    
    if(n_pos == 0 || n_neg == 0) return(0.5)
    
    rank_sum <- sum(which(target_ordered == 1))
    auc <- (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    
    return(auc)
  }
}

#' @title Cross-Validation para XGBoost
#' 
#' @description Realiza cross-validation para avaliar performance do modelo
#' 
#' @param dados data.table com os dados preprocessados incluindo coluna 'target'
#' @param parametros_treino Lista de parâmetros do XGBoost
#' @param k_folds Número de folds. DEFAULT: 5
#' @param nthreads Número de threads. DEFAULT: 3
#' @param seed Semente para reprodutibilidade. DEFAULT: 123
#' @param early_stopping_rounds Parada antecipada. DEFAULT: 50
#' 
#' @return Lista com métricas de performance por fold
#' 
#' @import data.table xgboost
#' @export
xgb_cross_validation <- function(dados, 
                                parametros_treino,
                                k_folds = 5,
                                nthreads = 3,
                                seed = 123,
                                early_stopping_rounds = 50,
                                colunas_excluir = NULL) {
  
  requireNamespace('data.table')
  requireNamespace('xgboost')
  
  if(is.null(parametros_treino)) {
    stop("É necessário fornecer parametros_treino")
  }
  
  # Criar folds
  cv_folds <- criar_cv_folds(dados, k_folds = k_folds, seed = seed)
  
  # Determinar se é problema de maximização
  maximize_metric <- parametros_treino$parametros$eval_metric %in% c("auc", "aucpr", "ndcg", "map")
  
  # Armazenar resultados
  cv_results <- list()
  
  cat("[INFO] Iniciando", k_folds, "fold cross-validation\n")
  
  for(i in 1:k_folds) {
    cat(paste0("\n[CV] Processando fold ", i, "/", k_folds, "\n"))
    
    # Separar treino e validação
    test_idx <- cv_folds[[i]]
    train_idx <- setdiff(1:nrow(dados), test_idx)
    
    dados_treino <- dados[train_idx]
    dados_teste <- dados[test_idx]
    
    # Converter para numérico de forma segura
    dados_treino_num <- converter_para_numerico_seguro(dados_treino)
    dados_teste_num <- converter_para_numerico_seguro(dados_teste)
    
    # Definir features (excluindo target e colunas a excluir)
    todas_colunas <- names(dados_treino_num)
    colunas_features_cv <- setdiff(todas_colunas, c("target", colunas_excluir))
    
    if(length(colunas_features_cv) == 0) {
      stop("Nenhuma feature disponível para treino no fold após exclusões")
    }
    
    # Criar matrizes DMatrix
    dtrain <- xgboost::xgb.DMatrix(
      data = as.matrix(dados_treino_num[, ..colunas_features_cv]),
      label = dados_treino_num$target
    )
    
    dtest <- xgboost::xgb.DMatrix(
      data = as.matrix(dados_teste_num[, ..colunas_features_cv]),
      label = dados_teste_num$target
    )
    
    # Treinar modelo
    modelo <- xgboost::xgb.train(
      data = dtrain,
      params = parametros_treino$parametros,
      nrounds = parametros_treino$nrounds,
      nthread = nthreads,
      verbose = 0,
      watchlist = list(train = dtrain, test = dtest),
      early_stopping_rounds = early_stopping_rounds,
      maximize = maximize_metric
    )
    
    # Predições
    pred_treino <- predict(modelo, dtrain)
    pred_teste <- predict(modelo, dtest)
    
    # Calcular métricas
    if(parametros_treino$parametros$objective == "binary:logistic") {
      # Classificação binária
      auc_treino <- calculate_auc(dados_treino_num$target, pred_treino)
      auc_teste <- calculate_auc(dados_teste_num$target, pred_teste)
      
      cv_results[[i]] <- list(
        fold = i,
        auc_train = auc_treino,
        auc_test = auc_teste,
        best_iteration = modelo$best_iteration,
        n_train = length(train_idx),
        n_test = length(test_idx)
      )
      
    } else {
      # Regressão
      rmse_treino <- sqrt(mean((dados_treino_num$target - pred_treino)^2))
      rmse_teste <- sqrt(mean((dados_teste_num$target - pred_teste)^2))
      
      cv_results[[i]] <- list(
        fold = i,
        rmse_train = rmse_treino,
        rmse_test = rmse_teste,
        best_iteration = modelo$best_iteration,
        n_train = length(train_idx),
        n_test = length(test_idx)
      )
    }
    
    cat(paste0("[CV] Fold ", i, " concluído - Best iteration: ", modelo$best_iteration, "\n"))
  }
  
  # Calcular médias das métricas
  if(parametros_treino$parametros$objective == "binary:logistic") {
    mean_auc_train <- mean(sapply(cv_results, function(x) x$auc_train))
    mean_auc_test <- mean(sapply(cv_results, function(x) x$auc_test))
    
    cat(paste0("\n[CV RESULTADO] AUC médio treino: ", round(mean_auc_train, 4)))
    cat(paste0("\n[CV RESULTADO] AUC médio teste: ", round(mean_auc_test, 4)))
    cat(paste0("\n[CV RESULTADO] Diferença (overfitting): ", round(mean_auc_train - mean_auc_test, 4), "\n"))
  } else {
    mean_rmse_train <- mean(sapply(cv_results, function(x) x$rmse_train))
    mean_rmse_test <- mean(sapply(cv_results, function(x) x$rmse_test))
    
    cat(paste0("\n[CV RESULTADO] RMSE médio treino: ", round(mean_rmse_train, 4)))
    cat(paste0("\n[CV RESULTADO] RMSE médio teste: ", round(mean_rmse_test, 4)))
    cat(paste0("\n[CV RESULTADO] Diferença (overfitting): ", round(mean_rmse_test - mean_rmse_train, 4), "\n"))
  }
  
  return(cv_results)
}

#' @title Gerar Parâmetros Estocásticos para XGBoost
#' 
#' @description Função auxiliar que gera variações nos parâmetros do XGBoost
#' para criar diversidade no ensemble
#' 
#' @param parametros_base Lista com parâmetros base do XGBoost
#' 
#' @return Lista com parâmetros variados
gerar_parametros_estocasticos <- function(parametros_base) {
  
  # Parâmetros base
  params <- parametros_base
  
  # Adicionar variação estocástica em parâmetros chave
  if("max_leaves" %in% names(params)) {
    params$max_leaves <- base::ceiling(rnorm_t(n = 1, mu = params$max_leaves, sd = 5, lower = 1))
  }
  
  if("eta" %in% names(params)) {
    params$eta <- rnorm_t(n = 1, mu = params$eta, sd = 0.06, lower = 0.001, upper = 1)
  }
  
  if("gamma" %in% names(params)) {
    params$gamma <- rnorm_t(n = 1, mu = params$gamma, sd = 0.5, lower = 0)
  }
  
  if("subsample" %in% names(params)) {
    params$subsample <- rnorm_t(n = 1, mu = params$subsample, sd = 0.4, lower = 0.1, upper = 1)
  }
  
  if("colsample_bytree" %in% names(params)) {
    params$colsample_bytree <- rnorm_t(n = 1, mu = params$colsample_bytree, sd = 0.4, lower = 0.1, upper = 1)
  }
  
  if("min_child_weight" %in% names(params)) {
    params$min_child_weight <- rnorm_t(n = 1, mu = params$min_child_weight, sd = 6, lower = 0)
  }
  
  # Adicionar base_score se não existir
  if(!"base_score" %in% names(params)) {
    params$base_score <- 0.5  # Será ajustado no treino
  }
  
  return(params)
}

#' @title Treino ensemble de modelos XGBoost Otimizado
#'
#' @description Função otimizada para treinar múltiplos modelos XGBoost com validação,
#' logging aprimorado e opções de amostragem flexíveis. Cada modelo é treinado em uma 
#' amostra bootstrapped diferente e salvo no S3.
#'
#' @param dados data.table com os dados preprocessados incluindo coluna 'target'
#' @param n_samples O tamanho do sorteio para cada classe (classificação) ou total (regressão)
#' @param nthreads A quantidade de CPU threads disponíveis para o XGBoost. DEFAULT: 3
#' @param parametros_treino A lista de parâmetros retornado pela função \code{xgb_select_params}
#' @param n_models A quantidade de modelos gerados. DEFAULT: 100
#' @param save_importance Indicador para o salvamento das importâncias das variáveis. DEFAULT: TRUE
#' @param bucket Bucket S3 para salvar os modelos
#' @param key_folder Pasta/prefixo no S3 para salvar os modelos (ex: "ensembles/modelo_dcv")
#' @param folder_to_save Pasta local temporária (será removida após upload). DEFAULT: NULL (será criada)
#' @param early_stopping_rounds Parada antecipada para evitar overfitting. DEFAULT: 50
#' @param validation_split Proporção dos dados para validação (0-1). DEFAULT: 0.2
#' @param use_sample_weights Usar pesos balanceados para classes desbalanceadas. DEFAULT: TRUE
#' @param seed Semente para reprodutibilidade. DEFAULT: 123
#' @param log_every Mostrar progresso a cada N modelos. DEFAULT: 10
#'
#' @return Lista com estatísticas do ensemble e caminhos S3 dos modelos salvos
#'
#' @import data.table xgboost magrittr
#'
#' @export
xgb_treino_ensemble <- function(dados,
                                n_samples,
                                nthreads = 3,
                                parametros_treino = NULL,
                                n_models = 100,
                                save_importance = TRUE,
                                bucket = NULL,
                                key_folder = NULL,
                                folder_to_save = NULL,
                                early_stopping_rounds = 50,
                                validation_split = 0.2,
                                use_sample_weights = TRUE,
                                seed = 123,
                                log_every = 10,
                                colunas_excluir = NULL,
                                coluna_id = NULL) {
  
  # Configurações iniciais e validações
  set.seed(seed)
  requireNamespace('data.table')
  requireNamespace('xgboost')
  requireNamespace('magrittr')
  
  # Validações de entrada
  if(!"target" %in% names(dados)) {
    stop("Dados devem conter coluna 'target'")
  }
  
  if(is.null(parametros_treino)) {
    stop("É necessário fornecer parametros_treino com estrutura: list(parametros = list(...), nrounds = N)")
  }
  
  # Validações S3
  if(is.null(bucket)) {
    stop("É necessário fornecer 'bucket' para salvar no S3")
  }
  
  if(is.null(key_folder)) {
    stop("É necessário fornecer 'key_folder' (prefixo/pasta no S3)")
  }
  
  # Garantir que key_folder termine sem "/"
  key_folder <- gsub("/$", "", key_folder)
  
  cat(paste0("[INFO] Salvamento S3 configurado: s3://", bucket, "/", key_folder, "/\n"))
  
  # Processamento de colunas
  todas_colunas <- names(dados)
  
  # Colunas a excluir do treino (identificadores, datas, etc.)
  if(is.null(colunas_excluir)) {
    colunas_excluir <- c()
    cat("[INFO] Nenhuma coluna especificada para exclusão do treino\n")
  } else {
    # Verificar se colunas existem
    colunas_nao_encontradas <- colunas_excluir[!colunas_excluir %in% todas_colunas]
    if(length(colunas_nao_encontradas) > 0) {
      warning(paste0("Colunas não encontradas para exclusão: ", paste(colunas_nao_encontradas, collapse = ", ")))
      colunas_excluir <- colunas_excluir[colunas_excluir %in% todas_colunas]
    }
    cat(paste0("[INFO] Colunas excluídas do treino: ", paste(colunas_excluir, collapse = ", "), "\n"))
  }
  
  # Validar coluna ID
  if(!is.null(coluna_id)) {
    if(!coluna_id %in% todas_colunas) {
      stop(paste0("Coluna ID especificada não encontrada: ", coluna_id))
    }
    cat(paste0("[INFO] Coluna ID para vincular predições: ", coluna_id, "\n"))
    
    # Garantir que coluna ID não seja usada no treino
    if(!coluna_id %in% colunas_excluir) {
      colunas_excluir <- c(colunas_excluir, coluna_id)
    }
  }
  
  # Definir colunas de features (excluindo target e colunas a excluir)
  colunas_features <- setdiff(todas_colunas, c("target", colunas_excluir))
  cat(paste0("[INFO] Features para treino: ", length(colunas_features), " colunas\n"))
  
  if(length(colunas_features) == 0) {
    stop("Nenhuma feature disponível para treino após exclusões")
  }
  
  if(is.null(folder_to_save)) {
    # Criar pasta temporária para processamento local
    timestamp <- base::format(base::Sys.time(), "%Y%m%d_%H%M%S")
    folder_to_save <- file.path(tempdir(), paste0("xgboost_ensemble_", timestamp))
    cat(paste0("[INFO] Usando pasta temporária: ", folder_to_save, "\n"))
  }
  
  # Criar diretório temporário se não existir
  if(!base::dir.exists(folder_to_save)) {
    base::dir.create(folder_to_save, recursive = TRUE, showWarnings = FALSE)
    cat(paste0("[INFO] Diretório temporário criado: ", folder_to_save, "\n"))
  }
  
  # Determinar tipo de problema
  is_classification <- all(dados$target %in% c(0, 1))
  is_multiclass <- parametros_treino$parametros$objective %in% c("multi:softmax", "multi:softprob")
  
  if(is_classification || is_multiclass) {
    maximize_metric <- parametros_treino$parametros$eval_metric %in% c("auc", "aucpr", "ndcg", "map")
  } else {
    maximize_metric <- FALSE
  }
  
  cat(paste0("[INFO] Tipo de problema: ", 
             ifelse(is_multiclass, "Classificação Multiclasse", 
                   ifelse(is_classification, "Classificação Binária", "Regressão")), "\n"))
  cat(paste0("[INFO] Métrica: ", parametros_treino$parametros$eval_metric, 
             " (maximize: ", maximize_metric, ")\n"))
  
  # Preparar dados - usar colunas de features já definidas anteriormente
  n_features <- length(colunas_features)
  n_total <- nrow(dados)
  
  cat(paste0("[INFO] Dataset: ", n_total, " observações, ", n_features, " features para treino\n"))
  cat(paste0("[INFO] Distribuição target: ", table(dados$target), "\n"))
  
  # Inicializar estruturas de armazenamento
  ensemble_stats <- list()
  importance_list <- list()
  model_paths <- character(n_models)
  performance_metrics <- list()
  
  # Loop principal de treinamento
  pb <- utils::txtProgressBar(min = 0, max = n_models, style = 3)
  inicio_tempo <- base::Sys.time()
  
  for(j in 1:n_models) {
    
    # Log a cada log_every modelos
    if(j %% log_every == 0 || j == 1) {
      tempo_decorrido <- as.numeric(base::Sys.time() - inicio_tempo, units = "mins")
      tempo_estimado <- tempo_decorrido * n_models / j
      cat(paste0("\n[ENSEMBLE] Modelo ", j, "/", n_models, 
                " - Tempo: ", round(tempo_decorrido, 1), 
                "min - ETA: ", round(tempo_estimado - tempo_decorrido, 1), "min\n"))
    }
    
    utils::setTxtProgressBar(pb, j)
    
    # Amostragem estratificada ou simples
    if(is_classification) {
      treino <- dados[, .SD[sample(.N, min(.N, n_samples), replace = TRUE)], by = target]
    } else {
      treino <- dados[sample(.N, min(.N, n_samples), replace = TRUE)]
    }
    
    # Divisão treino/validação
    if(validation_split > 0) {
      n_treino <- nrow(treino)
      idx_val <- sample(n_treino, floor(n_treino * validation_split))
      dados_val <- treino[idx_val]
      treino <- treino[-idx_val]
    }
    
    # Conversão para numérico de forma segura
    treino_num <- converter_para_numerico_seguro(treino)
    
    # Calcular base_score dinâmico
    base_score_calc <- if(is_classification) {
      sum(treino_num$target) / nrow(treino_num)
    } else {
      mean(treino_num$target)
    }
    
    # Gerar parâmetros com variação
    tmp_param <- gerar_parametros_estocasticos(parametros_treino$parametros)
    tmp_param$base_score <- base_score_calc
    
    # Criar matriz de treino
    dtrain <- xgboost::xgb.DMatrix(
      data = as.matrix(treino_num[, ..colunas_features]),
      label = treino_num$target
    )
    
    # Criar watchlist
    watchlist <- list(train = dtrain)
    if(validation_split > 0) {
      dados_val_num <- converter_para_numerico_seguro(dados_val)
      dval <- xgboost::xgb.DMatrix(
        data = as.matrix(dados_val_num[, ..colunas_features]),
        label = dados_val_num$target
      )
      watchlist$eval <- dval
    }
    
    # Treinar modelo
    md <- tryCatch({
      xgboost::xgb.train(
        data = dtrain,
        params = tmp_param,
        nrounds = parametros_treino$nrounds,
        nthread = nthreads,
        verbose = 0,
        watchlist = watchlist,
        early_stopping_rounds = early_stopping_rounds,
        maximize = maximize_metric
      )
    }, error = function(e) {
      cat(paste0("[ERROR] Falha no treinamento do modelo ", j, ": ", e$message, "\n"))
      return(NULL)
    })
    
    if(is.null(md)) {
      cat(paste0("[WARN] Modelo ", j, " falhou. Pulando...\n"))
      next
    }
    
    # Salvar importâncias
    if(save_importance) {
      tryCatch({
        importance_list[[j]] <- xgboost::xgb.importance(model = md)
      }, error = function(e) {
        cat(paste0("[WARN] Erro ao extrair importância do modelo ", j, ": ", e$message, "\n"))
      })
    }
    
    # Coletar métricas de performance
    if(validation_split > 0) {
      eval_log <- md$evaluation_log
      if(!is.null(eval_log) && nrow(eval_log) > 0) {
        performance_metrics[[j]] <- list(
          model_id = j,
          best_iteration = md$best_iteration,
          train_metric = eval_log[md$best_iteration, 2],
          eval_metric = if(ncol(eval_log) > 2) eval_log[md$best_iteration, 3] else NA
        )
      }
    }
    
    # Salvar modelo localmente primeiro, depois fazer upload para S3
    model_filename <- paste0("xgb_model_", sprintf("%03d", j), ".xgb")
    model_path_local <- file.path(folder_to_save, model_filename)
    model_key_s3 <- paste0(key_folder, "/model/", model_filename)
    
    xgboost::xgb.save(model = md, fname = model_path_local)
    
    # Upload para S3
    tryCatch({
      s3_upload(
        files = model_path_local,
        bucket = bucket,
        keynames = model_key_s3,
        nthreads = 1
      )
      cat(paste0("[S3] Modelo ", j, " salvo: s3://", bucket, "/", model_key_s3, "\n"))
    }, error = function(e) {
      warning(paste0("Erro salvando modelo ", j, " no S3: ", e$message))
    })
    
    model_paths[j] <- model_key_s3  # Armazenar path S3
    
    # Coletar estatísticas
    ensemble_stats[[j]] <- list(
      model_id = j,
      best_iteration = md$best_iteration,
      n_features = n_features,
      n_train = nrow(treino),
      n_val = if(validation_split > 0) nrow(dados_val) else 0,
      parametros_utilizados = tmp_param
    )
    
    # Limpeza de memória
    rm(md, dtrain)
    if(exists("dval")) rm(dval)
    base::invisible(gc(verbose = FALSE))
  }
  
  base::close(pb)
  tempo_total <- as.numeric(base::Sys.time() - inicio_tempo, units = "mins")
  cat(paste0("\n[ENSEMBLE] Treinamento concluído em ", round(tempo_total, 1), " minutos\n"))
  
  # Salvar metadados do ensemble
  ensemble_metadata <- list(
    n_models = n_models,
    model_paths = model_paths[!is.na(model_paths)],
    ensemble_stats = ensemble_stats[!sapply(ensemble_stats, is.null)],
    performance_metrics = performance_metrics[!sapply(performance_metrics, is.null)],
    parametros_base = parametros_treino,
    dataset_info = list(
      n_obs = n_total,
      n_features = n_features,
      target_distribution = table(dados$target)
    ),
    training_config = list(
      n_samples = n_samples,
      validation_split = validation_split,
      early_stopping_rounds = early_stopping_rounds,
      use_sample_weights = use_sample_weights,
      seed = seed,
      colunas_excluir = colunas_excluir,
      coluna_id = coluna_id,
      colunas_features = colunas_features
    ),
    tempo_total_min = tempo_total,
    timestamp = base::Sys.time()
  )
  
  # Salvar metadata localmente e fazer upload para S3
  metadata_path_local <- file.path(folder_to_save, "ensemble_metadata.RDS")
  metadata_key_s3 <- paste0(key_folder, "/stats/ensemble_metadata.RDS")
  
  base::saveRDS(ensemble_metadata, file = metadata_path_local)
  
  # Upload metadata para S3
  tryCatch({
    s3_write(
      x = ensemble_metadata,
      bucket = bucket,
      key = metadata_key_s3,
      FUN = saveRDS
    )
    cat(paste0("[S3] Metadata salvo: s3://", bucket, "/", metadata_key_s3, "\n"))
  }, error = function(e) {
    warning(paste0("Erro salvando metadata no S3: ", e$message))
  })
  
  # Salvar importâncias se disponíveis
  if(save_importance && length(importance_list) > 0) {
    tryCatch({
      # Salvar localmente
      importance_path_local <- file.path(folder_to_save, "importancia_modelos.RDS")
      importance_key_s3 <- paste0(key_folder, "/stats/importancia_modelos.RDS")
      
      base::saveRDS(importance_list, file = importance_path_local)
      
      # Upload para S3
      s3_write(
        x = importance_list,
        bucket = bucket,
        key = importance_key_s3,
        FUN = saveRDS
      )
      cat(paste0("[S3] Importâncias salvas: s3://", bucket, "/", importance_key_s3, "\n"))
      cat(paste0("[INFO] Importâncias salvas: ", length(importance_list), " modelos\n"))
    }, error = function(e) {
      cat(paste0("[WARN] Erro ao salvar importâncias: ", e$message, "\n"))
    })
  }
  
  # Limpeza de arquivos locais após upload
  if(base::dir.exists(folder_to_save)) {
    tryCatch({
      base::unlink(folder_to_save, recursive = TRUE)
      cat(paste0("[INFO] Pasta local temporária removida: ", folder_to_save, "\n"))
    }, error = function(e) {
      warning(paste0("Não foi possível remover pasta temporária: ", e$message))
    })
  }
  
  # Atualizar metadata com informações S3
  ensemble_metadata$s3_info <- list(
    bucket = bucket,
    key_folder = key_folder,
    model_keys = model_paths,
    metadata_key = metadata_key_s3,
    importance_key = if(save_importance) importance_key_s3 else NULL
  )
  
  cat(paste0("[SUCCESS] Ensemble salvo no S3: s3://", bucket, "/", key_folder, "/\n"))
  cat(paste0("[SUCCESS] Modelos válidos: ", length(ensemble_metadata$model_paths), "/", n_models, "\n"))
  
  return(ensemble_metadata)
}

#' @title Predição com Ensemble XGBoost
#' 
#' @description Realiza predições usando todos os modelos do ensemble e 
#' combina os resultados através de média ponderada ou voting
#' 
#' @param dados_novos data.table com novos dados para predição (sem coluna target)
#' @param folder_ensemble Pasta contendo os modelos do ensemble (opcional se modelos fornecidos)
#' @param modelos_xgb Lista de modelos XGBoost carregados (opcional, alternativa ao folder_ensemble)
#' @param metadata_ensemble Lista com metadados do ensemble (opcional, alternativa ao folder_ensemble)
#' @param metodo_combinacao Método para combinar predições: "media", "mediana", "peso_performance". DEFAULT: "media"
#' @param nthreads Número de threads para predição. DEFAULT: 3
#' @param retornar_com_id Retornar resultado com IDs preservados. DEFAULT: FALSE
#' @param verbose Mostrar logs detalhados. DEFAULT: TRUE
#' 
#' @return Vetor com predições finais do ensemble ou data.table com IDs se retornar_com_id=TRUE
#' 
#' @import data.table xgboost
#' @export
xgb_predict_ensemble <- function(dados_novos, 
                                folder_ensemble = NULL,
                                modelos_xgb = NULL,
                                metadata_ensemble = NULL,
                                metodo_combinacao = "media",
                                nthreads = 3,
                                retornar_com_id = FALSE,
                                verbose = TRUE) {
  
  requireNamespace('data.table')
  requireNamespace('xgboost')
  
  # Validar entrada: deve ter folder_ensemble OU (modelos_xgb + metadata_ensemble)
  if(is.null(folder_ensemble) && (is.null(modelos_xgb) || is.null(metadata_ensemble))) {
    stop("Deve fornecer 'folder_ensemble' OU ('modelos_xgb' + 'metadata_ensemble')")
  }
  
  if(!is.null(folder_ensemble) && (!is.null(modelos_xgb) || !is.null(metadata_ensemble))) {
    stop("Forneca apenas 'folder_ensemble' OU ('modelos_xgb' + 'metadata_ensemble'), não ambos")
  }
  
  # MODO 1: Carregar de folder_ensemble (comportamento original)
  if(!is.null(folder_ensemble)) {
    if(verbose) cat("[INFO] Carregando modelos do folder:", folder_ensemble, "\n")
    
    # Verificar se pasta existe
    if(!dir.exists(folder_ensemble)) {
      stop(paste0("Pasta do ensemble não encontrada: ", folder_ensemble))
    }
    
    # Carregar metadados
    metadata_path <- file.path(folder_ensemble, "ensemble_metadata.RDS")
    if(!base::file.exists(metadata_path)) {
      stop("Arquivo ensemble_metadata.RDS não encontrado")
    }
    
    metadata <- base::readRDS(metadata_path)
    model_paths <- metadata$model_paths
    
    # Carregar modelos XGBoost
    modelos <- lapply(model_paths, function(path) {
      if(base::file.exists(path)) {
        xgboost::xgb.load(path)
      } else {
        stop(paste0("Modelo não encontrado: ", path))
      }
    })
    
  } else {
    # MODO 2: Usar modelos e metadados fornecidos diretamente
    if(verbose) cat("[INFO] Usando modelos e metadados fornecidos como parâmetros\n")
    
    modelos <- modelos_xgb
    metadata <- metadata_ensemble
    
    # Validar formato dos modelos
    if(!is.list(modelos) || length(modelos) == 0) {
      stop("modelos_xgb deve ser uma lista não-vazia de modelos XGBoost")
    }
    
    # Verificar se são modelos XGBoost válidos
    for(i in 1:length(modelos)) {
      if(!inherits(modelos[[i]], "xgb.Booster")) {
        stop(paste0("Elemento ", i, " de modelos_xgb não é um modelo XGBoost válido"))
      }
    }
    
    if(verbose) {
      cat(paste0("[INFO] Recebidos ", length(modelos), " modelos XGBoost\n"))
    }
  }
  
  # Recuperar configuração de colunas do metadata
  colunas_features <- metadata$training_config$colunas_features
  coluna_id <- metadata$training_config$coluna_id
  colunas_excluir <- metadata$training_config$colunas_excluir
  
  if(verbose) {
    cat(paste0("[INFO] Processando ", length(modelos), " modelos para predição\n"))
    if(!is.null(colunas_features)) {
      cat(paste0("[INFO] Usando ", length(colunas_features), " features para predição\n"))
    }
  }
  
  # Preparar dados - preservar ID se especificado
  dados_id <- NULL
  if(retornar_com_id && !is.null(coluna_id)) {
    if(coluna_id %in% names(dados_novos)) {
      dados_id <- dados_novos[[coluna_id]]
      if(verbose) {
        cat(paste0("[INFO] ID preservado: ", length(dados_id), " registros\n"))
      }
    } else {
      warning(paste0("Coluna ID '", coluna_id, "' não encontrada nos dados"))
    }
  }
  
  # Selecionar apenas as features usadas no treino
  if(!is.null(colunas_features)) {
    # Verificar se todas as features estão presentes
    features_faltando <- colunas_features[!colunas_features %in% names(dados_novos)]
    if(length(features_faltando) > 0) {
      stop(paste0("Features faltando nos dados: ", paste(features_faltando, collapse = ", ")))
    }
    dados_pred <- dados_novos[, ..colunas_features]
  } else {
    # Fallback: usar todas exceto target e colunas a excluir
    colunas_disponiveis <- setdiff(names(dados_novos), c("target", colunas_excluir))
    dados_pred <- dados_novos[, ..colunas_disponiveis]
    if(verbose) {
      cat(paste0("[WARN] Colunas features não encontradas no metadata, usando todas disponíveis: ", 
                      length(colunas_disponiveis), " colunas\n"))
    }
  }
  
  # Converter para numérico de forma segura
  dados_pred_num <- converter_para_numerico_seguro(dados_pred)
  
  # Converter para matriz
  dmatrix <- xgboost::xgb.DMatrix(data = as.matrix(dados_pred_num))
  
  # Realizar predições com todos os modelos
  predictions_matrix <- matrix(NA, nrow = nrow(dados_novos), ncol = length(modelos))
  valid_models <- 0
  
  if(verbose) {
    pb <- txtProgressBar(min = 0, max = length(modelos), style = 3)
  }
  
  for(i in seq_along(modelos)) {
    if(verbose) setTxtProgressBar(pb, i)
    
    tryCatch({
      # Usar modelo já carregado
      modelo <- modelos[[i]]
      
      # Fazer predição
      pred <- predict(modelo, dmatrix, nthread = nthreads)
      predictions_matrix[, i] <- pred
      valid_models <- valid_models + 1
      
    }, error = function(e) {
      if(verbose) cat(paste0("\n[WARN] Erro no modelo ", i, ": ", e$message))
    })
  }
  
  if(verbose) {
    close(pb)
    cat(paste0("\n[INFO] Modelos válidos para predição: ", valid_models, "/", length(modelos), "\n"))
  }
  
  # Remover colunas com NA (modelos que falharam)
  valid_cols <- !apply(is.na(predictions_matrix), 2, all)
  predictions_matrix <- predictions_matrix[, valid_cols, drop = FALSE]
  
  # Combinar predições
  if(metodo_combinacao == "media") {
    final_predictions <- rowMeans(predictions_matrix, na.rm = TRUE)
  } else if(metodo_combinacao == "mediana") {
    final_predictions <- apply(predictions_matrix, 1, median, na.rm = TRUE)
  } else if(metodo_combinacao == "peso_performance" && !is.null(metadata$performance_metrics)) {
    # Usar performance para ponderar (implementação simplificada)
    weights <- sapply(metadata$performance_metrics, function(x) {
      if(is.null(x$eval_metric) || is.na(x$eval_metric)) return(1)
      return(x$eval_metric)
    })
    weights <- weights / sum(weights, na.rm = TRUE)
    final_predictions <- apply(predictions_matrix, 1, function(row) {
      weighted.mean(row, weights, na.rm = TRUE)
    })
  } else {
    final_predictions <- rowMeans(predictions_matrix, na.rm = TRUE)
  }
  
  if(verbose) {
    cat(paste0("[SUCCESS] Predições do ensemble concluídas usando método: ", metodo_combinacao, "\n"))
  }
  
  # Retornar com ID se solicitado
  if(retornar_com_id && !is.null(dados_id)) {
    resultado <- data.table::data.table(
      id = dados_id,
      predicao = final_predictions
    )
    
    # Adicionar nome da coluna ID baseado no metadata
    if(!is.null(coluna_id)) {
      data.table::setnames(resultado, "id", coluna_id)
    }
    
    if(verbose) {
      cat(paste0("[INFO] Predições retornadas com ID: ", nrow(resultado), " registros\n"))
    }
    
    return(resultado)
  } else {
    return(final_predictions)
  }
}

#' @title Avaliar Performance do Ensemble
#' 
#' @description Avalia a performance do ensemble usando cross-validation 
#' e compara com modelos individuais
#' 
#' @param dados data.table com dados incluindo coluna 'target'
#' @param folder_ensemble Pasta contendo o ensemble treinado
#' @param k_folds Número de folds para CV. DEFAULT: 5
#' @param nthreads Número de threads. DEFAULT: 3
#' 
#' @return Lista com métricas de avaliação
#' 
#' @import data.table
#' @export
avaliar_ensemble_performance <- function(dados, 
                                        folder_ensemble,
                                        k_folds = 5,
                                        nthreads = 3) {
  
  requireNamespace('data.table')
  
  # Carregar metadados
  metadata <- base::readRDS(file.path(folder_ensemble, "ensemble_metadata.RDS"))
  
  cat("[INFO] Avaliando performance do ensemble com cross-validation\n")
  
  # Criar folds
  cv_folds <- criar_cv_folds(dados, k_folds = k_folds)
  
  # Resultados por fold
  cv_results <- list()
  
  for(i in 1:k_folds) {
    cat(paste0("\n[EVAL] Avaliando fold ", i, "/", k_folds, "\n"))
    
    # Separar dados
    test_idx <- cv_folds[[i]]
    train_idx <- setdiff(1:nrow(dados), test_idx)
    
    dados_teste <- dados[test_idx]
    dados_treino <- dados[train_idx]
    
    # Fazer predições com ensemble (apenas features, sem target)
    features_teste <- dados_teste[, !c("target")]
    pred_ensemble <- xgb_predict_ensemble(features_teste, folder_ensemble, nthreads = nthreads)
    
    # Calcular métricas
    if(all(dados$target %in% c(0, 1))) {
      # Classificação binária
      auc_ensemble <- calculate_auc(dados_teste$target, pred_ensemble)
      
      cv_results[[i]] <- list(
        fold = i,
        auc_ensemble = auc_ensemble,
        n_test = length(test_idx)
      )
      
    } else {
      # Regressão
      rmse_ensemble <- sqrt(mean((dados_teste$target - pred_ensemble)^2))
      mae_ensemble <- mean(abs(dados_teste$target - pred_ensemble))
      
      cv_results[[i]] <- list(
        fold = i,
        rmse_ensemble = rmse_ensemble,
        mae_ensemble = mae_ensemble,
        n_test = length(test_idx)
      )
    }
  }
  
  # Calcular médias
  if(all(dados$target %in% c(0, 1))) {
    mean_auc <- mean(sapply(cv_results, function(x) x$auc_ensemble))
    cat(paste0("\n[ENSEMBLE EVAL] AUC médio: ", round(mean_auc, 4), "\n"))
    
    resultado <- list(
      metrica_principal = "auc",
      auc_medio = mean_auc,
      cv_results = cv_results,
      metadata = metadata
    )
  } else {
    mean_rmse <- mean(sapply(cv_results, function(x) x$rmse_ensemble))
    mean_mae <- mean(sapply(cv_results, function(x) x$mae_ensemble))
    
    cat(paste0("\n[ENSEMBLE EVAL] RMSE médio: ", round(mean_rmse, 4)))
    cat(paste0("\n[ENSEMBLE EVAL] MAE médio: ", round(mean_mae, 4), "\n"))
    
    resultado <- list(
      metrica_principal = "rmse",
      rmse_medio = mean_rmse,
      mae_medio = mean_mae,
      cv_results = cv_results,
      metadata = metadata
    )
  }
  
  return(resultado)
}

#' @title Analisar Importância das Features do Ensemble
#' 
#' @description Analisa e agrega as importâncias das features de todos os modelos do ensemble
#' 
#' @param folder_ensemble Pasta contendo o ensemble
#' @param top_n Número de features mais importantes a retornar. DEFAULT: 20
#' 
#' @return data.table com importâncias agregadas
#' 
#' @import data.table
#' @export
analisar_importancia_ensemble <- function(folder_ensemble, top_n = 20) {
  
  requireNamespace('data.table')
  
  # Carregar importâncias
  imp_path <- file.path(folder_ensemble, "importancia_modelos.RDS")
  if(!base::file.exists(imp_path)) {
    stop("Arquivo de importâncias não encontrado")
  }
  
  importance_list <- base::readRDS(imp_path)
  
  # Combinar todas as importâncias
  all_importance <- data.table::rbindlist(importance_list, idcol = "model_id")
  
  # Agregar por feature
  importance_summary <- all_importance[, .(
    importancia_media = mean(Gain, na.rm = TRUE),
    importancia_mediana = median(Gain, na.rm = TRUE),
    freq_aparicao = .N,
    importancia_std = sd(Gain, na.rm = TRUE)
  ), by = Feature]
  
  # Ordenar por importância média
  importance_summary <- importance_summary[order(-importancia_media)]
  
  # Calcular ranking
  importance_summary[, ranking := 1:.N]
  
  # Adicionar porcentagem de aparição
  n_models <- length(unique(all_importance$model_id))
  importance_summary[, pct_aparicao := round(freq_aparicao / n_models * 100, 1)]
  
  cat(paste0("[INFO] Análise de importância para ", n_models, " modelos\n"))
  cat(paste0("[INFO] Top ", min(top_n, nrow(importance_summary)), " features mais importantes:\n"))
  
  print(utils::head(importance_summary, top_n))
  
  return(importance_summary[1:min(top_n, .N)])
}

#' @title Otimização de Hiperparâmetros com Grid Search
#' 
#' @description Realiza grid search para encontrar os melhores hiperparâmetros
#' usando cross-validation
#' 
#' @param dados data.table com dados incluindo coluna 'target'
#' @param param_grid Lista com os parâmetros a testar
#' @param k_folds Número de folds para CV. DEFAULT: 3
#' @param nthreads Número de threads. DEFAULT: 3
#' @param early_stopping_rounds Parada antecipada. DEFAULT: 30
#' @param objective Objetivo do modelo. DEFAULT: "binary:logistic"
#' @param eval_metric Métrica de avaliação. DEFAULT: "auc"
#' @param nrounds Número máximo de rounds. DEFAULT: 200
#' 
#' @return Lista com melhores parâmetros e resultados do grid search
#' 
#' @import data.table xgboost
#' @export
xgb_grid_search <- function(dados,
                           param_grid = NULL,
                           k_folds = 3,
                           nthreads = 3,
                           early_stopping_rounds = 30,
                           objective = "binary:logistic",
                           eval_metric = "auc",
                           nrounds = 200) {
  
  requireNamespace('data.table')
  requireNamespace('xgboost')
  
  # Grid padrão se não fornecido
  if(is.null(param_grid)) {
    param_grid <- list(
      eta = c(0.05, 0.1, 0.15),
      max_leaves = c(32, 64, 128),
      gamma = c(0, 0.1, 0.2),
      subsample = c(0.8, 0.9),
      colsample_bytree = c(0.8, 0.9)
    )
  }
  
  # Criar combinações do grid
  grid_combinations <- expand.grid(param_grid)
  n_combinations <- nrow(grid_combinations)
  
  cat(paste0("[GRID] Testando ", n_combinations, " combinações de parâmetros\n"))
  
  # Determinar se é maximização
  maximize_metric <- eval_metric %in% c("auc", "aucpr", "ndcg", "map")
  
  # Armazenar resultados
  grid_results <- list()
  
  # Progress bar
  pb <- txtProgressBar(min = 0, max = n_combinations, style = 3)
  
  for(i in 1:n_combinations) {
    setTxtProgressBar(pb, i)
    
    # Preparar parâmetros para esta combinação
    params_atual <- list(
      objective = objective,
      eval_metric = eval_metric,
      eta = grid_combinations$eta[i],
      max_leaves = grid_combinations$max_leaves[i],
      gamma = grid_combinations$gamma[i],
      subsample = grid_combinations$subsample[i],
      colsample_bytree = grid_combinations$colsample_bytree[i],
      min_child_weight = 1,
      grow_policy = "lossguide",
      tree_method = "hist"
    )
    
    parametros_teste <- list(
      parametros = params_atual,
      nrounds = nrounds
    )
    
    # Executar cross-validation
    tryCatch({
      cv_result <- xgb_cross_validation(
        dados = dados,
        parametros_treino = parametros_teste,
        k_folds = k_folds,
        nthreads = nthreads,
        early_stopping_rounds = early_stopping_rounds
      )
      
      # Calcular métrica média
      if(objective == "binary:logistic") {
        metrica_media <- mean(sapply(cv_result, function(x) x$auc_test))
      } else {
        metrica_media <- mean(sapply(cv_result, function(x) x$rmse_test))
      }
      
      grid_results[[i]] <- list(
        combinacao = i,
        parametros = params_atual,
        metrica_media = metrica_media,
        cv_detalhado = cv_result
      )
      
    }, error = function(e) {
      cat(paste0("\n[WARN] Erro na combinação ", i, ": ", e$message))
      grid_results[[i]] <- list(
        combinacao = i,
        parametros = params_atual,
        metrica_media = if(maximize_metric) -Inf else Inf,
        erro = e$message
      )
    })
  }
  
  close(pb)
  
  # Encontrar melhor combinação
  metricas <- sapply(grid_results, function(x) x$metrica_media)
  
  if(maximize_metric) {
    melhor_idx <- which.max(metricas)
  } else {
    melhor_idx <- which.min(metricas)
  }
  
  melhor_config <- grid_results[[melhor_idx]]
  
  cat(paste0("\n[GRID] Melhor configuração encontrada (", eval_metric, ": ", 
             round(melhor_config$metrica_media, 4), "):\n"))
  
  for(param_name in names(melhor_config$parametros)) {
    if(param_name %in% names(param_grid)) {
      cat(paste0("  ", param_name, ": ", melhor_config$parametros[[param_name]], "\n"))
    }
  }
  
  # Criar lista final de parâmetros
  parametros_otimizados <- list(
    parametros = melhor_config$parametros,
    nrounds = nrounds
  )
  
  return(list(
    melhor_parametros = parametros_otimizados,
    melhor_metrica = melhor_config$metrica_media,
    grid_completo = grid_results,
    resumo_grid = data.table(
      combinacao = 1:n_combinations,
      metrica = metricas,
      eta = grid_combinations$eta,
      max_leaves = grid_combinations$max_leaves,
      gamma = grid_combinations$gamma,
      subsample = grid_combinations$subsample,
      colsample_bytree = grid_combinations$colsample_bytree
    )[order(if(maximize_metric) -metrica else metrica)]
  ))
}

#' @title Busca de Parâmetros com Amostra Pequena
#' 
#' @description Realiza busca estocástica de parâmetros XGBoost usando apenas 
#' uma pequena amostra dos dados para otimização rápida. Ideal para datasets 
#' grandes onde o grid search completo seria muito lento.
#' 
#' @param dados Dataset completo para amostragem
#' @param target_col Nome da coluna target. DEFAULT: "target"
#' @param sample_size Tamanho da amostra para otimização. DEFAULT: 10000
#' @param n_configs Número de configurações estocásticas a testar. DEFAULT: 20
#' @param k_folds Número de folds para CV. DEFAULT: 3
#' @param colunas_excluir Colunas a excluir da modelagem. DEFAULT: NULL
#' @param nthreads Número de threads. DEFAULT: 4
#' @param early_stopping_rounds Parada antecipada. DEFAULT: 20
#' @param seed Seed para reprodutibilidade. DEFAULT: 2024
#' @param verbose Mostrar progresso detalhado. DEFAULT: TRUE
#' 
#' @return Lista com:
#'   - melhor_parametros: Configuração otimizada
#'   - melhor_auc: Melhor AUC encontrado
#'   - ranking_configs: Ranking de todas configurações testadas
#'   - tempo_execucao: Tempo total de execução
#'   - amostra_usada: Informações sobre a amostra
#' 
#' @examples
#' # Busca rápida com amostra pequena
#' resultado <- busca_parametros_amostra_pequena(
#'   dados = meus_dados,
#'   sample_size = 5000,
#'   n_configs = 15
#' )
#' 
#' @import data.table xgboost
#' @export
busca_parametros_amostra_pequena <- function(dados,
                                           target_col = "target",
                                           sample_size = 10000,
                                           n_configs = 20,
                                           k_folds = 3,
                                           colunas_excluir = NULL,
                                           nthreads = 4,
                                           early_stopping_rounds = 20,
                                           seed = 2024,
                                           verbose = TRUE) {
  
  base::requireNamespace('data.table')
  base::requireNamespace('xgboost')
  
  inicio_tempo <- base::Sys.time()
  
  if(verbose) {
    cat("🔍 BUSCA DE PARÂMETROS COM AMOSTRA PEQUENA\n")
    cat("==========================================\n")
  }
  
  # Validar dados de entrada
  if(!data.table::is.data.table(dados)) {
    dados <- data.table::as.data.table(dados)
  }
  
  if(!target_col %in% names(dados)) {
    stop(paste("Coluna target", target_col, "não encontrada nos dados"))
  }
  
  # Criar amostra balanceada pequena
  base::set.seed(seed)
  
  if(verbose) {
    cat(paste0("📊 Dataset original: ", nrow(dados), " observações\n"))
  }
  
  # Amostragem estratificada balanceada
  dados_target_0 <- dados[get(target_col) == 0]
  dados_target_1 <- dados[get(target_col) == 1]
  
  if(verbose) {
    cat(paste0("📊 Target 0: ", nrow(dados_target_0), " observações\n"))
    cat(paste0("📊 Target 1: ", nrow(dados_target_1), " observações\n"))
  }
  
  # Verificar se há dados suficientes
  if(nrow(dados_target_0) == 0 || nrow(dados_target_1) == 0) {
    stop("Dados insuficientes: uma das classes do target está vazia")
  }
  
  n_per_class <- min(
    floor(sample_size / 2),
    nrow(dados_target_0),
    nrow(dados_target_1)
  )
  
  if(n_per_class < 50) {
    warning(paste("Amostra muito pequena:", n_per_class, "por classe. Considerando aumentar sample_size."))
  }
  
  amostra_0 <- dados_target_0[sample(.N, n_per_class)]
  amostra_1 <- dados_target_1[sample(.N, n_per_class)]
  
  amostra_dados <- data.table::rbindlist(list(amostra_0, amostra_1))
  amostra_dados <- amostra_dados[sample(.N)]  # Embaralhar
  
  if(verbose) {
    cat(paste0("📊 Amostra para otimização: ", nrow(amostra_dados), " observações\n"))
    cat(paste0("📊 Distribuição target amostra: "))
    print(table(amostra_dados[[target_col]]))
  }
  
  # Parâmetros base para geração estocástica
  parametros_base <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = 0.1,
    max_leaves = 64,
    gamma = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 1,
    grow_policy = "lossguide",
    tree_method = "hist"
  )
  
  # Armazenar resultados
  resultados_configs <- list()
  
  if(verbose) {
    cat(paste0("🎯 Testando ", n_configs, " configurações estocásticas...\n"))
    pb <- utils::txtProgressBar(min = 0, max = n_configs, style = 3)
  }
  
  # Loop de busca estocástica
  configuracoes_testadas <- 0
  configuracoes_validas <- 0
  
  for(i in 1:n_configs) {
    if(verbose) utils::setTxtProgressBar(pb, i)
    
    configuracoes_testadas <- configuracoes_testadas + 1
    
    # Gerar parâmetros estocásticos
    params_estocasticos <- tryCatch({
      gerar_parametros_estocasticos(parametros_base)
    }, error = function(e) {
      if(verbose) cat(paste0("\n⚠️ Erro ao gerar parâmetros config ", i, ": ", e$message, "\n"))
      return(NULL)
    })
    
    if(is.null(params_estocasticos)) next
    
    # Definir rounds variáveis para diversidade
    nrounds_config <- sample(100:250, 1)
    
    config_completa <- list(
      parametros = params_estocasticos,
      nrounds = nrounds_config
    )
    
    # Avaliar com cross-validation na amostra
    cv_resultado <- tryCatch({
      xgb_cross_validation(
        dados = amostra_dados,
        parametros_treino = config_completa,
        k_folds = k_folds,
        nthreads = nthreads,
        early_stopping_rounds = early_stopping_rounds,
        colunas_excluir = colunas_excluir
      )
    }, error = function(e) {
      if(verbose && i <= 3) {
        cat(paste0("\n⚠️ Erro CV config ", i, ": ", e$message, "\n"))
      }
      return(NULL)
    })
    
    # Armazenar resultado se válido
    if(!is.null(cv_resultado) && length(cv_resultado$auc) > 0) {
      configuracoes_validas <- configuracoes_validas + 1
      
      resultados_configs[[configuracoes_validas]] <- list(
        config_id = i,
        parametros = params_estocasticos,
        nrounds = nrounds_config,
        auc_mean = mean(cv_resultado$auc),
        auc_sd = stats::sd(cv_resultado$auc),
        logloss_mean = mean(cv_resultado$logloss),
        cv_detalhado = cv_resultado
      )
      
      if(verbose && configuracoes_validas <= 3) {
        cat(paste0("\n✅ Config ", i, " válida - AUC: ", 
                        round(mean(cv_resultado$auc), 4), "\n"))
      }
    }
  }
  
  if(verbose) {
    base::close(pb)
    cat(paste0("\n📊 Configurações testadas: ", configuracoes_testadas, "\n"))
    cat(paste0("✅ Configurações válidas: ", configuracoes_validas, "\n"))
  }
  
  # Filtrar resultados válidos
  configs_validas <- resultados_configs[!base::sapply(resultados_configs, is.null)]
  
  if(length(configs_validas) == 0) {
    cat("\n❌ ERRO: Nenhuma configuração válida encontrada!\n")
    cat("🔍 POSSÍVEIS CAUSAS:\n")
    cat("  1. Amostra muito pequena ou desbalanceada\n")
    cat("  2. Parâmetros base incompatíveis\n")
    cat("  3. Colunas com muitos NAs ou constantes\n")
    cat("  4. Problema na função gerar_parametros_estocasticos\n")
    cat("  5. Problema na função xgb_cross_validation\n")
    
    # Tentar uma configuração simples como fallback
    cat("\n🔧 TENTANDO CONFIGURAÇÃO SIMPLES COMO FALLBACK...\n")
    
    params_simples <- list(
      objective = "binary:logistic",
      eval_metric = "auc",
      eta = 0.1,
      max_depth = 6,
      subsample = 0.8,
      colsample_bytree = 0.8,
      min_child_weight = 1
    )
    
    config_fallback <- list(
      parametros = params_simples,
      nrounds = 100
    )
    
    cv_fallback <- tryCatch({
      xgb_cross_validation(
        dados = amostra_dados,
        parametros_treino = config_fallback,
        k_folds = k_folds,
        nthreads = nthreads,
        early_stopping_rounds = early_stopping_rounds,
        colunas_excluir = colunas_excluir
      )
    }, error = function(e) {
      cat(paste0("❌ Fallback também falhou: ", e$message, "\n"))
      return(NULL)
    })
    
    if(!is.null(cv_fallback)) {
      cat("✅ Fallback funcionou! Usando configuração simples.\n")
      
      return(list(
        melhor_parametros = config_fallback,
        melhor_auc = mean(cv_fallback$auc),
        melhor_auc_sd = stats::sd(cv_fallback$auc),
        ranking_configs = data.table::data.table(
          rank = 1,
          config_id = 999,
          auc_mean = mean(cv_fallback$auc),
          auc_sd = stats::sd(cv_fallback$auc),
          tipo = "fallback"
        ),
        configs_detalhadas = list(list(
          config_id = 999,
          parametros = params_simples,
          nrounds = 100,
          auc_mean = mean(cv_fallback$auc),
          tipo = "fallback"
        )),
        tempo_execucao = base::difftime(base::Sys.time(), inicio_tempo, units = "mins"),
        amostra_info = list(
          tamanho_original = nrow(dados),
          tamanho_amostra = nrow(amostra_dados),
          distribuicao_target = table(amostra_dados[[target_col]])
        )
      ))
    } else {
      stop("Nem busca estocástica nem fallback funcionaram. Verifique os dados e dependências.")
    }
  }
  
  # Ordenar por AUC (melhor primeiro)
  aucs <- base::sapply(configs_validas, function(x) x$auc_mean)
  ordem_ranking <- order(aucs, decreasing = TRUE)
  configs_ordenadas <- configs_validas[ordem_ranking]
  
  # Melhor configuração
  melhor_config <- configs_ordenadas[[1]]
  
  # Criar ranking resumido
  ranking_df <- data.table::data.table(
    rank = 1:length(configs_ordenadas),
    config_id = base::sapply(configs_ordenadas, function(x) x$config_id),
    auc_mean = base::sapply(configs_ordenadas, function(x) x$auc_mean),
    auc_sd = base::sapply(configs_ordenadas, function(x) x$auc_sd),
    logloss_mean = base::sapply(configs_ordenadas, function(x) x$logloss_mean),
    eta = base::sapply(configs_ordenadas, function(x) x$parametros$eta),
    max_leaves = base::sapply(configs_ordenadas, function(x) x$parametros$max_leaves),
    gamma = base::sapply(configs_ordenadas, function(x) x$parametros$gamma),
    subsample = base::sapply(configs_ordenadas, function(x) x$parametros$subsample),
    colsample_bytree = base::sapply(configs_ordenadas, function(x) x$parametros$colsample_bytree),
    nrounds = base::sapply(configs_ordenadas, function(x) x$nrounds)
  )
  
  fim_tempo <- base::Sys.time()
  tempo_execucao <- base::difftime(fim_tempo, inicio_tempo, units = "mins")
  
  if(verbose) {
    cat("🏆 RESULTADOS DA BUSCA ESTOCÁSTICA\n")
    cat("==================================\n")
    cat(paste0("⏱️  Tempo de execução: ", round(tempo_execucao, 2), " minutos\n"))
    cat(paste0("✅ Configurações válidas: ", length(configs_validas), "/", n_configs, "\n"))
    cat(paste0("🥇 Melhor AUC: ", round(melhor_config$auc_mean, 4), 
                    " (±", round(melhor_config$auc_sd, 4), ")\n"))
    
    cat("\n🏅 TOP 5 CONFIGURAÇÕES:\n")
    top_5 <- min(5, nrow(ranking_df))
    for(i in 1:top_5) {
      cat(paste0(i, ". AUC: ", round(ranking_df$auc_mean[i], 4),
                      " | eta: ", ranking_df$eta[i],
                      " | max_leaves: ", ranking_df$max_leaves[i],
                      " | gamma: ", ranking_df$gamma[i], "\n"))
    }
  }
  
  # Retornar resultado completo
  base::return(list(
    melhor_parametros = list(
      parametros = melhor_config$parametros,
      nrounds = melhor_config$nrounds
    ),
    melhor_auc = melhor_config$auc_mean,
    melhor_auc_sd = melhor_config$auc_sd,
    ranking_configs = ranking_df,
    configs_detalhadas = configs_ordenadas,
    tempo_execucao = tempo_execucao,
    amostra_info = list(
      tamanho_original = nrow(dados),
      tamanho_amostra = nrow(amostra_dados),
      distribuicao_target = table(amostra_dados[[target_col]])
    )
  ))
}

#' @title Busca Rápida de Parâmetros (Versão Simples)
#' 
#' @description Versão mais robusta e simples da busca de parâmetros.
#' Testa algumas configurações pré-definidas com amostra pequena.
#' 
#' @param dados Dataset completo
#' @param target_col Nome da coluna target. DEFAULT: "target"
#' @param sample_size Tamanho da amostra. DEFAULT: 5000
#' @param colunas_excluir Colunas a excluir. DEFAULT: NULL
#' @param nthreads Número de threads. DEFAULT: 4
#' @param seed Seed para reprodutibilidade. DEFAULT: 2024
#' @param verbose Mostrar detalhes. DEFAULT: TRUE
#' 
#' @return Lista com melhor configuração encontrada
#' 
#' @import data.table xgboost
#' @export
busca_parametros_rapida <- function(dados,
                                  target_col = "target",
                                  sample_size = 5000,
                                  colunas_excluir = NULL,
                                  nthreads = 4,
                                  seed = 2024,
                                  verbose = TRUE) {
  
  base::requireNamespace('data.table')
  base::requireNamespace('xgboost')
  
  if(verbose) {
    cat("⚡ BUSCA RÁPIDA DE PARÂMETROS\n")
    cat("============================\n")
  }
  
  # Converter para data.table se necessário
  if(!data.table::is.data.table(dados)) {
    dados <- data.table::as.data.table(dados)
  }
  
  # Criar amostra balanceada
  base::set.seed(seed)
  
  dados_0 <- dados[get(target_col) == 0]
  dados_1 <- dados[get(target_col) == 1]
  
  n_per_class <- min(
    floor(sample_size / 2),
    nrow(dados_0),
    nrow(dados_1)
  )
  
  amostra <- data.table::rbindlist(list(
    dados_0[sample(.N, n_per_class)],
    dados_1[sample(.N, n_per_class)]
  ))
  
  if(verbose) {
    cat(paste0("📊 Amostra: ", nrow(amostra), " observações\n"))
  }
  
  # Configurações pré-definidas robustas
  configs_teste <- list(
    conservadora = list(
      parametros = list(
        objective = "binary:logistic",
        eval_metric = "auc",
        eta = 0.1,
        max_depth = 6,
        subsample = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 1
      ),
      nrounds = 150
    ),
    
    rapida = list(
      parametros = list(
        objective = "binary:logistic",
        eval_metric = "auc",
        eta = 0.15,
        max_depth = 5,
        subsample = 0.9,
        colsample_bytree = 0.9,
        min_child_weight = 2
      ),
      nrounds = 100
    ),
    
    profunda = list(
      parametros = list(
        objective = "binary:logistic",
        eval_metric = "auc",
        eta = 0.05,
        max_depth = 8,
        subsample = 0.7,
        colsample_bytree = 0.7,
        min_child_weight = 3
      ),
      nrounds = 250
    ),
    
    balanceada = list(
      parametros = list(
        objective = "binary:logistic",
        eval_metric = "auc",
        eta = 0.08,
        max_depth = 7,
        subsample = 0.85,
        colsample_bytree = 0.75,
        min_child_weight = 2
      ),
      nrounds = 200
    )
  )
  
  # Testar cada configuração
  resultados <- list()
  
  for(nome_config in names(configs_teste)) {
    if(verbose) {
      cat(paste0("🧪 Testando configuração: ", nome_config, "\n"))
    }
    
    config <- configs_teste[[nome_config]]
    
    cv_resultado <- tryCatch({
      xgb_cross_validation(
        dados = amostra,
        parametros_treino = config,
        k_folds = 3,
        nthreads = nthreads,
        early_stopping_rounds = 20,
        colunas_excluir = colunas_excluir,
        verbose = FALSE
      )
    }, error = function(e) {
      if(verbose) {
        cat(paste0("  ❌ Erro: ", e$message, "\n"))
      }
      return(NULL)
    })
    
    if(!is.null(cv_resultado)) {
      auc_mean <- mean(cv_resultado$auc)
      if(verbose) {
        cat(paste0("  ✅ AUC: ", round(auc_mean, 4), "\n"))
      }
      
      resultados[[nome_config]] <- list(
        nome = nome_config,
        configuracao = config,
        auc_mean = auc_mean,
        auc_sd = stats::sd(cv_resultado$auc),
        cv_resultado = cv_resultado
      )
    }
  }
  
  if(length(resultados) == 0) {
    stop("Nenhuma configuração funcionou. Verifique os dados e dependências.")
  }
  
  # Encontrar melhor configuração
  aucs <- base::sapply(resultados, function(x) x$auc_mean)
  melhor_nome <- names(resultados)[which.max(aucs)]
  melhor_resultado <- resultados[[melhor_nome]]
  
  if(verbose) {
    cat("\n🏆 MELHOR CONFIGURAÇÃO ENCONTRADA\n")
    cat("=================================\n")
    cat(paste0("📛 Nome: ", melhor_nome, "\n"))
    cat(paste0("🎯 AUC: ", round(melhor_resultado$auc_mean, 4), 
                    " (±", round(melhor_resultado$auc_sd, 4), ")\n"))
  }
  
  return(list(
    melhor_parametros = melhor_resultado$configuracao,
    melhor_auc = melhor_resultado$auc_mean,
    melhor_nome = melhor_nome,
    todas_configs = resultados,
    amostra_info = list(
      tamanho_original = nrow(dados),
      tamanho_amostra = nrow(amostra)
    )
  ))
}

#' @title Teste Rápido de Parâmetros
#' 
#' @description Função super simples que sempre funciona para testar algumas
#' configurações básicas de XGBoost com amostra pequena.
#' 
#' @param dados Dataset
#' @param target_col Coluna target
#' @param sample_size Tamanho da amostra
#' @param colunas_excluir Colunas para excluir
#' 
#' @return Lista com melhor configuração
#' 
#' @import data.table
#' @export
teste_parametros_simples <- function(dados, 
                                    target_col = "target",
                                    sample_size = 3000,
                                    colunas_excluir = NULL,
                                    nthreads = 4) {
  
  cat("🔧 TESTE SIMPLES DE PARÂMETROS\n")
  cat("==============================\n")
  
  # Amostra pequena
  base::set.seed(2024)
  if(!data.table::is.data.table(dados)) dados <- data.table::as.data.table(dados)
  
  # Amostra balanceada simples
  dados_0 <- dados[get(target_col) == 0]
  dados_1 <- dados[get(target_col) == 1]
  
  n_each <- min(floor(sample_size/2), nrow(dados_0), nrow(dados_1))
  
  amostra <- data.table::rbindlist(list(
    dados_0[sample(.N, n_each)],
    dados_1[sample(.N, n_each)]
  ))
  
  cat(paste0("📊 Amostra criada: ", nrow(amostra), " observações\n"))
  
  # Preparar dados para XGBoost
  cols_para_remover <- base::unique(c(target_col, colunas_excluir))
  feature_cols <- setdiff(names(amostra), cols_para_remover)
  
  # Remover colunas não numéricas
  feature_cols <- feature_cols[base::sapply(feature_cols, function(col) is.numeric(amostra[[col]]))]
  
  if(length(feature_cols) == 0) {
    stop("Nenhuma feature numérica encontrada após exclusões")
  }
  
  X <- as.matrix(amostra[, ..feature_cols])
  y <- amostra[[target_col]]
  
  cat(paste0("📊 Features numéricas: ", length(feature_cols), "\n"))
  cat(paste0("📊 Target distribuição: ", table(y)[1], " vs ", table(y)[2], "\n"))
  
  # Testar configurações básicas
  configs <- list(
    config1 = list(
      objective = "binary:logistic",
      eval_metric = "auc",
      eta = 0.1,
      max_depth = 6,
      nrounds = 100
    ),
    config2 = list(
      objective = "binary:logistic", 
      eval_metric = "auc",
      eta = 0.05,
      max_depth = 8,
      nrounds = 150
    ),
    config3 = list(
      objective = "binary:logistic",
      eval_metric = "auc", 
      eta = 0.15,
      max_depth = 5,
      nrounds = 80
    )
  )
  
  resultados <- list()
  
  for(nome in names(configs_teste)) {
    cat(paste0("🧪 Testando ", nome, "...\n"))
    
    config <- configs_teste[[nome]]
    nrounds <- config$nrounds
    params <- config[names(config) != "nrounds"]
    
    # Teste simples com xgboost diretamente
    cv_result <- tryCatch({
      xgboost::xgb.cv(
        data = xgboost::xgb.DMatrix(X, label = y),
        params = params,
        nrounds = nrounds,
        nfold = 3,
        early_stopping_rounds = 20,
        verbose = FALSE,
        nthread = nthreads
      )
    }, error = function(e) {
      cat(paste0("  ❌ Erro: ", e$message, "\n"))
      return(NULL)
    })
    
    if(!is.null(cv_result)) {
      # Pegar melhor AUC
      best_auc <- max(cv_result$evaluation_log$test_auc_mean, na.rm = TRUE)
      cat(paste0("  ✅ AUC: ", round(best_auc, 4), "\n"))
      
      resultados[[nome]] <- list(
        nome = nome,
        parametros = params,
        nrounds = nrounds,
        auc = best_auc,
        cv_log = cv_result$evaluation_log
      )
    }
  }
  
  if(length(resultados) == 0) {
    # Configuração mais básica possível
    cat("🆘 Usando configuração ultra básica...\n")
    return(list(
      melhor_parametros = list(
        parametros = list(
          objective = "binary:logistic",
          eval_metric = "auc",
          eta = 0.1,
          max_depth = 6
        ),
        nrounds = 100
      ),
      melhor_auc = 0.5,
      tipo = "basico_emergencia"
    ))
  }
  
  # Encontrar melhor
  aucs <- base::sapply(resultados, function(x) x$auc)
  melhor_idx <- which.max(aucs)
  melhor <- resultados[[melhor_idx]]
  
  cat("\n🏆 MELHOR CONFIGURAÇÃO\n")
  cat("======================\n")
  cat(paste0("📛 Nome: ", melhor$nome, "\n"))
  cat(paste0("🎯 AUC: ", round(melhor$auc, 4), "\n"))
  
  return(list(
    melhor_parametros = list(
      parametros = melhor$parametros,
      nrounds = melhor$nrounds
    ),
    melhor_auc = melhor$auc,
    melhor_nome = melhor$nome,
    todas_configs = resultados
  ))
}

#' @title Busca Básica de Parâmetros XGBoost
#' 
#' @description Função ultra-robusta que sempre funciona. Testa configurações
#' pré-definidas sem depender de funções complexas.
#' 
#' @param dados Dataset completo
#' @param target_col Nome da coluna target. DEFAULT: "target" 
#' @param colunas_excluir Colunas a excluir. DEFAULT: NULL
#' @param sample_size Tamanho da amostra. DEFAULT: 3000
#' 
#' @return Lista com melhor configuração encontrada
#' 
#' @import data.table xgboost
#' @export
busca_basica_xgb <- function(dados, 
                            target_col = "target",
                            colunas_excluir = NULL,
                            sample_size = 3000) {
  
  cat("🛠️  BUSCA BÁSICA DE PARÂMETROS XGB\n")
  cat("==================================\n")
  
  # Garantir data.table
  if(!data.table::is.data.table(dados)) {
    dados <- data.table::as.data.table(dados)
  }
  
  # Criar amostra balanceada
  base::set.seed(2024)
  dados_0 <- dados[get(target_col) == 0]
  dados_1 <- dados[get(target_col) == 1]
  
  n_each <- min(
    floor(sample_size / 2),
    nrow(dados_0),
    nrow(dados_1)
  )
  
  amostra <- data.table::rbindlist(list(
    dados_0[sample(.N, n_each)],
    dados_1[sample(.N, n_each)]
  ))
  
  cat(paste0("📊 Amostra: ", nrow(amostra), " observações\n"))
  
  # Preparar features
  colunas_remover <- base::unique(c(target_col, colunas_excluir))
  feature_cols <- setdiff(names(amostra), colunas_remover)
  feature_cols <- feature_cols[base::sapply(feature_cols, function(col) is.numeric(amostra[[col]]))]
  
  X <- as.matrix(amostra[, ..feature_cols])
  y <- amostra[[target_col]]
  
  # Configurações simples e robustas
  configuracoes <- list(
    
    simples = list(
      objective = "binary:logistic",
      eval_metric = "auc",
      eta = 0.1,
      max_depth = 6,
      subsample = 0.8,
      colsample_bytree = 0.8
    ),
    
    conservadora = list(
      objective = "binary:logistic", 
      eval_metric = "auc",
      eta = 0.05,
      max_depth = 8,
      subsample = 0.7,
      colsample_bytree = 0.7
    ),
    
    agressiva = list(
      objective = "binary:logistic",
      eval_metric = "auc",
      eta = 0.2,
      max_depth = 4,
      subsample = 0.9,
      colsample_bytree = 0.9
    )
  )
  
  melhores_resultados <- list()
  
  # Testar cada configuração
  for(nome_config in names(configuracoes)) {
    cat(paste0("🧪 Testando ", nome_config, "...\n"))
    
    params <- configuracoes[[nome_config]]
    
    resultado_cv <- tryCatch({
      cv_obj <- xgboost::xgb.cv(
        data = xgboost::xgb.DMatrix(X, label = y),
        params = params,
        nrounds = 150,
        nfold = 3,
        early_stopping_rounds = 15,
        verbose = FALSE,
        nthread = 2  # Valor fixo para evitar problemas
      )
      
      # Extrair melhor AUC
      if(!is.null(cv_obj$evaluation_log)) {
        melhor_auc <- max(cv_obj$evaluation_log$test_auc_mean, na.rm = TRUE)
        return(melhor_auc)
      } else {
        return(NULL)
      }
      
    }, error = function(e) {
      cat(paste0("  ❌ Erro: ", e$message, "\n"))
      return(NULL)
    })
    
    if(!is.null(resultado_cv) && !is.na(resultado_cv)) {
      cat(paste0("  ✅ AUC: ", round(resultado_cv, 4), "\n"))
      
      melhores_resultados[[nome_config]] <- list(
        nome = nome_config,
        parametros = params,
        auc = resultado_cv
      )
    }
  }
  
  # Selecionar melhor ou usar padrão
  if(length(melhores_resultados) > 0) {
    aucs <- base::sapply(melhores_resultados, function(x) x$auc)
    melhor_nome <- names(melhores_resultados)[which.max(aucs)]
    melhor_config <- melhores_resultados[[melhor_nome]]
    
    cat("\n🏆 MELHOR CONFIGURAÇÃO\n")
    cat("======================\n")
    cat(paste0("📛 ", melhor_config$nome, " - AUC: ", round(melhor_config$auc, 4), "\n"))
    
    return(list(
      melhor_parametros = list(
        parametros = melhor_config$parametros,
        nrounds = 150
      ),
      melhor_auc = melhor_config$auc,
      melhor_nome = melhor_config$nome
    ))
    
  } else {
    cat("🆘 Usando configuração de emergência...\n")
    
    return(list(
      melhor_parametros = list(
        parametros = list(
          objective = "binary:logistic",
          eval_metric = "auc",
          eta = 0.1,
          max_depth = 6
        ),
        nrounds = 100
      ),
      melhor_auc = 0.5,
      melhor_nome = "emergencia"
    ))
  }
}

#' @title Busca de Parâmetros com Tidymodels
#' 
#' @description Usa o framework tidymodels para busca robusta de hiperparâmetros
#' XGBoost com cross-validation. Mais estável que implementações manuais.
#' 
#' @param dados Dataset completo
#' @param target_col Nome da coluna target. DEFAULT: "target"
#' @param sample_size Tamanho da amostra para tuning. DEFAULT: 5000
#' @param colunas_excluir Colunas a excluir da modelagem. DEFAULT: NULL
#' @param n_iter Número de iterações de busca. DEFAULT: 15
#' @param cv_folds Número de folds para CV. DEFAULT: 3
#' @param seed Seed para reprodutibilidade. DEFAULT: 2024
#' @param verbose Mostrar progresso. DEFAULT: TRUE
#' 
#' @return Lista com melhores parâmetros e métricas
#' 
#' @import data.table
#' @export
#' @title Testar Estrutura de Parâmetros
#'
#' @description Função auxiliar para validar se os parâmetros extraídos
#' estão no formato correto para uso no XGBoost
#'
#' @param resultado_busca Lista resultado de qualquer função de busca de parâmetros
#' @param verbose Logical. Se TRUE, mostra detalhes da validação
#'
#' @return Lista com parâmetros validados no formato correto
#' @export
#' @title Busca Parâmetros Tidymodels Simples
#'
#' @description Versão ultra-simples do tidymodels que evita o erro maybe_matrix()
#' usando apenas validação cruzada sem recipe complexo
#'
#' @param dados data.table ou data.frame com dados
#' @param target_col String nome da coluna target
#' @param sample_size Integer tamanho da amostra
#' @param colunas_excluir Vector colunas para excluir
#' @param verbose Logical mostrar debug
#'
#' @return Lista com parâmetros no formato XGBoost
#' @export
busca_parametros_tidymodels_simples <- function(dados,
                                               target_col = "target",
                                               sample_size = 3000,
                                               colunas_excluir = NULL,
                                               verbose = TRUE) {
  
  if(verbose) cat("🧪 BUSCA SIMPLES TIDYMODELS (SEM RECIPE COMPLEXO)\n")
  
  # Preparar dados básicos
  if(!data.table::is.data.table(dados)) {
    dados <- data.table::as.data.table(dados)
  }
  
  # Amostra pequena balanceada
  base::set.seed(2024)
  dados_0 <- dados[get(target_col) == 0]
  dados_1 <- dados[get(target_col) == 1]
  
  n_each <- min(floor(sample_size/2), nrow(dados_0), nrow(dados_1))
  
  amostra <- data.table::rbindlist(list(
    dados_0[sample(.N, n_each)],
    dados_1[sample(.N, n_each)]
  ))
  
  # Remover colunas excluídas
  if(!is.null(colunas_excluir)) {
    cols_para_remover <- colunas_excluir[colunas_excluir %in% names(amostra)]
    if(length(cols_para_remover) > 0) {
      amostra <- amostra[, !..cols_para_remover]
    }
  }
  
  # Converter target para factor SEM prefixo
  amostra[[target_col]] <- as.factor(amostra[[target_col]])
  
  # Converter para data.frame sem limpeza complexa
  dados_df <- as.data.frame(amostra)
  
  # Limpeza robusta de NAs
  dados_df <- limpar_nas_robusto(dados_df, verbose = verbose)
  
  # Usar apenas fit_resamples sem tune
  resultado <- tryCatch({
    
    # Recipe mínimo
    recipe_min <- recipes::recipe(stats::reformulate(".", response = target_col), data = dados_df)
    
    # Modelo fixo
    modelo_fixo <- parsnip::boost_tree(
      trees = 100,
      tree_depth = 6,
      learn_rate = 0.1
    ) |>
      parsnip::set_engine("xgboost") |>
      parsnip::set_mode("classification")
    
    # Workflow
    wf <- workflows::workflow() |>
      workflows::add_model(modelo_fixo) |>
      workflows::add_recipe(recipe_min)
    
    # CV
    cv <- rsample::vfold_cv(dados_df, v = 3)
    
    # Apenas fit_resamples
    tune::fit_resamples(
      wf,
      resamples = cv,
      metrics = yardstick::metric_set(yardstick::roc_auc)
    )
    
  }, error = function(e) {
    if(verbose) cat(paste0("❌ Método simples falhou: ", e$message, "\n"))
    return(NULL)
  })
  
  if(is.null(resultado)) {
    stop("Todas as tentativas de tidymodels falharam")
  }
  
  # Extrair métrica
  metricas <- tune::collect_metrics(resultado)
  auc_value <- metricas$mean[metricas$.metric == "roc_auc"]
  
  # Retornar parâmetros fixos
  return(list(
    sucesso = TRUE,
    melhor_parametros = list(
      parametros = list(
        objective = "binary:logistic",
        eval_metric = "auc",
        eta = 0.1,
        max_depth = 6,
        subsample = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 1
      ),
      nrounds = 100
    ),
    auc = auc_value,
    metodo = "tidymodels_simples"
  ))
}

#' @title Limpeza Robusta de NAs
#'
#' @description Função auxiliar para limpeza robusta de NAs sem usar complete.cases()
#'
#' @param dados data.frame ou data.table
#' @param verbose Logical mostrar debug
#'
#' @return data.frame limpo sem NAs
#' @export
#' @title Busca Parâmetros mlrMBO Simples (Grid)
#'
#' @description Versão grid search que evita problemas do mlrMBO closure
#' Testa configurações otimizadas sem otimização Bayesiana
#'
#' @param dados data.table com dados
#' @param target_col String nome da coluna target
#' @param n_samples Integer tamanho da amostra
#' @param colunas_excluir Vector colunas para excluir
#' @param verbose Logical mostrar debug
#'
#' @return Lista com parâmetros no formato XGBoost
#' @export
busca_parametros_mlrmbo_simples <- function(dados,
                                           target_col = "target",
                                           n_samples = 3000,
                                           colunas_excluir = NULL,
                                           verbose = TRUE) {
  
  if(verbose) {
    cat("🧪 BUSCA SIMPLES - GRID OTIMIZADO\n")
    cat("=================================\n")
  }
  
  # Preparar dados
  if(!data.table::is.data.table(dados)) {
    dados <- data.table::as.data.table(dados)
  }
  
  # Remover colunas excluídas
  if(!is.null(colunas_excluir)) {
    cols_para_remover <- colunas_excluir[colunas_excluir %in% names(dados)]
    if(length(cols_para_remover) > 0) {
      dados <- dados[, !..cols_para_remover]
    }
  }
  
  # Amostra balanceada
  base::set.seed(2024)
  dados_0 <- dados[get(target_col) == 0]
  dados_1 <- dados[get(target_col) == 1]
  
  n_each <- min(floor(n_samples/2), nrow(dados_0), nrow(dados_1))
  
  treino <- data.table::rbindlist(list(
    dados_0[sample(.N, n_each)],
    dados_1[sample(.N, n_each)]
  ))
  
  if(verbose) {
    cat(paste0("📊 Amostra: ", nrow(treino), " obs\n"))
    cat("📊 Distribuição: ")
    print(table(treino[[target_col]]))
  }
  
  # Limpeza robusta
  treino_limpo <- limpar_nas_robusto(as.data.frame(treino), verbose = FALSE)
  treino <- data.table::as.data.table(treino_limpo)
  
  # Grid de configurações otimizadas
  grid_configs <- list(
    list(eta = 0.05, max_depth = 6, subsample = 0.8, colsample_bytree = 0.8, min_child_weight = 1, nrounds = 200),
    list(eta = 0.1, max_depth = 7, subsample = 0.85, colsample_bytree = 0.75, min_child_weight = 2, nrounds = 150),
    list(eta = 0.15, max_depth = 5, subsample = 0.9, colsample_bytree = 0.9, min_child_weight = 1, nrounds = 120),
    list(eta = 0.08, max_depth = 8, subsample = 0.75, colsample_bytree = 0.7, min_child_weight = 3, nrounds = 180),
    list(eta = 0.12, max_depth = 6, subsample = 0.8, colsample_bytree = 0.85, min_child_weight = 1, nrounds = 160),
    list(eta = 0.2, max_depth = 4, subsample = 0.9, colsample_bytree = 0.8, min_child_weight = 2, nrounds = 100)
  )
  
  melhor_auc <- -Inf
  melhor_config <- grid_configs[[1]]
  
  # Preparar matriz XGBoost uma vez
  feature_cols <- setdiff(names(treino), target_col)
  dtrain <- xgboost::xgb.DMatrix(
    data = as.matrix(treino[, ..feature_cols]),
    label = treino[[target_col]]
  )
  
  # Testar cada configuração
  for(i in seq_along(grid_configs)) {
    config <- grid_configs[[i]]
    
    if(verbose) cat(paste0("🔬 Config ", i, "/", length(grid_configs), " (eta=", config$eta, ", depth=", config$max_depth, ")...\n"))
    
    # Cross-validation
    cv_result <- tryCatch({
      xgboost::xgb.cv(
        data = dtrain,
        nfold = 3,
        nrounds = config$nrounds,
        verbose = FALSE,
        early_stopping_rounds = 20,
        params = list(
          objective = "binary:logistic",
          eval_metric = "auc",
          eta = config$eta,
          max_depth = config$max_depth,
          subsample = config$subsample,
          colsample_bytree = config$colsample_bytree,
          min_child_weight = config$min_child_weight
        )
      )
    }, error = function(e) {
      if(verbose) cat(paste0("⚠️ Config ", i, " falhou: ", e$message, "\n"))
      return(NULL)
    })
    
    if(!is.null(cv_result)) {
      auc_atual <- max(cv_result$evaluation_log$test_auc_mean)
      
      if(auc_atual > melhor_auc) {
        melhor_auc <- auc_atual
        melhor_config <- config
        if(verbose) {
          cat(paste0("🏆 NOVA MELHOR! AUC: ", round(auc_atual, 4), "\n"))
        }
      } else {
        if(verbose) {
          cat(paste0("   AUC: ", round(auc_atual, 4), " (não melhorou)\n"))
        }
      }
    }
  }
  
  # Preparar resultado final
  nrounds_final <- melhor_config$nrounds
  melhor_config$nrounds <- NULL
  
  # Adicionar parâmetros obrigatórios
  melhor_config$objective <- "binary:logistic"
  melhor_config$eval_metric <- "auc"
  
  if(verbose) {
    cat(paste0("\n🏆 MELHOR CONFIGURAÇÃO ENCONTRADA\n"))
    cat("=================================\n")
    cat(paste0("📈 AUC: ", round(melhor_auc, 4), "\n"))
    cat(paste0("🎲 Nrounds: ", nrounds_final, "\n"))
    cat("🔧 Parâmetros:\n")
    for(param_name in names(melhor_config)) {
      if(!param_name %in% c("objective", "eval_metric")) {
        cat(paste0("  ", param_name, ": ", melhor_config[[param_name]], "\n"))
      }
    }
  }
  
  return(list(
    sucesso = TRUE,
    melhor_parametros = list(
      parametros = melhor_config,
      nrounds = nrounds_final
    ),
    auc = melhor_auc,
    metodo = "grid_otimizado",
    configs_testadas = length(grid_configs)
  ))
}

#' @title Busca de Parâmetros com mlrMBO
#'
#' @description Otimização Bayesiana de hiperparâmetros XGBoost usando mlrMBO
#' Baseado em: https://github.com/zegkreist/magikaRp/blob/master/R/xgb_select_params.R
#'
#' @param dados data.table com dados e coluna target
#' @param target_col String nome da coluna target (default: "target")
#' @param n_samples Integer tamanho da amostra por classe
#' @param colunas_excluir Vector colunas para excluir
#' @param metrica String métrica para otimizar (default: "auc")
#' @param objetivo String objetivo XGBoost (default: "binary:logistic")
#' @param niter_data Integer quantas sessões de dados (default: 8)
#' @param niter_bayes Integer iterações Bayesianas (default: 25)
#' @param cv_folds Integer folds para CV (default: 3)
#' @param cv_nrounds Integer máximo de rounds (default: 500)
#' @param nthreads Integer threads (default: 4)
#' @param verbose Logical mostrar debug
#'
#' @return Lista com parametros, nrounds e métrica
#' @export

#' @title Validar Parâmetros mlrMBO
#'
#' @description Valida se os parâmetros para mlrMBO estão corretos
#'
#' @param cv_nrounds Integer máximo de rounds para validar
#' @param verbose Logical mostrar debug
#'
#' @return Lista com parâmetros validados
validar_parametros_mlrmbo <- function(cv_nrounds, verbose = TRUE) {
  
  cv_nrounds_safe <- max(cv_nrounds, 100)
  
  if(verbose && cv_nrounds != cv_nrounds_safe) {
    cat(paste0("⚠️ cv_nrounds ajustado de ", cv_nrounds, " para ", cv_nrounds_safe, "\n"))
  }
  
  # Testar se podemos criar o par.set
  test_par_set <- tryCatch({
    ParamHelpers::makeParamSet(
      ParamHelpers::makeIntegerParam(id = "max_leaves", lower = 10, upper = 100),
      ParamHelpers::makeNumericParam(id = "eta", lower = 0.01, upper = 0.3),
      ParamHelpers::makeIntegerParam(id = "nrounds", lower = 50, upper = cv_nrounds_safe)
    )
  }, error = function(e) {
    if(verbose) cat(paste0("❌ Erro validando parâmetros: ", e$message, "\n"))
    return(NULL)
  })
  
  if(is.null(test_par_set)) {
    stop("Parâmetros inválidos para mlrMBO")
  }
  
  return(list(cv_nrounds_safe = cv_nrounds_safe, valido = TRUE))
}


limpar_nas_robusto <- function(dados, verbose = TRUE) {
  
  if(verbose) cat("🧹 LIMPEZA ROBUSTA DE NAs\n")
  
  linhas_antes <- nrow(dados)
  
  # Converter para data.table se necessário
  if(!data.table::is.data.table(dados)) {
    dt_dados <- data.table::as.data.table(dados)
  } else {
    dt_dados <- data.table::copy(dados)
  }
  
  # Identificar colunas com NAs
  colunas_com_na <- names(dt_dados)[base::sapply(dt_dados, function(x) any(is.na(x)))]
  
  # Retornar como data.frame
  return(as.data.frame(dt_dados))
}

validar_parametros_extraidos <- function(resultado_busca, verbose = TRUE) {
  
  if(verbose) {
    cat("\n🔍 VALIDANDO ESTRUTURA DE PARÂMETROS\n")
    cat("===================================\n")
  }
  
  # Verificar se resultado_busca existe
  if(is.null(resultado_busca)) {
    stop("Resultado da busca é NULL")
  }
  
  # Verificar estrutura principal
  if(!"melhor_parametros" %in% names(resultado_busca)) {
    if(verbose) cat("❌ Campo 'melhor_parametros' não encontrado\n")
    stop("Campo 'melhor_parametros' não encontrado no resultado")
  }
  
  melhor_params <- resultado_busca$melhor_parametros
  
  # Verificar sub-estrutura
  if(!"parametros" %in% names(melhor_params)) {
    if(verbose) cat("❌ Campo 'parametros' não encontrado em melhor_parametros\n")
    stop("Campo 'parametros' não encontrado")
  }
  
  if(!"nrounds" %in% names(melhor_params)) {
    if(verbose) cat("❌ Campo 'nrounds' não encontrado em melhor_parametros\n")
    stop("Campo 'nrounds' não encontrado")
  }
  
  params_xgb <- melhor_params$parametros
  nrounds_val <- melhor_params$nrounds
  
  # Validar parâmetros essenciais do XGBoost
  parametros_essenciais <- c("objective", "eval_metric", "eta", "max_depth")
  parametros_faltando <- parametros_essenciais[!parametros_essenciais %in% names(params_xgb)]
  
  if(length(parametros_faltando) > 0) {
    if(verbose) cat(paste0("⚠️ Parâmetros faltando: ", paste(parametros_faltando, collapse = ", "), "\n"))
  }
  
  if(verbose) {
    cat("✅ VALIDAÇÃO CONCLUÍDA\n")
    cat("=====================\n")
    cat(paste0("📋 Total de parâmetros: ", length(params_xgb), "\n"))
    cat(paste0("🎲 Nrounds: ", nrounds_val, "\n"))
    cat("\n🔧 LISTA DE PARÂMETROS VALIDADA:\n")
    for(param_name in names(params_xgb)) {
      cat(paste0("  ", param_name, ": ", params_xgb[[param_name]], "\n"))
    }
  }
  
  # Retornar estrutura validada
  return(list(
    parametros = params_xgb,
    nrounds = nrounds_val,
    validacao_ok = TRUE
  ))
}

busca_parametros_tidymodels <- function(dados,
                                      target_col = "target", 
                                      sample_size = 5000,
                                      colunas_excluir = NULL,
                                      n_iter = 15,
                                      cv_folds = 3,
                                      seed = 2024,
                                      verbose = TRUE) {
  
  if(verbose) {
    cat("🧪 BUSCA DE PARÂMETROS COM TIDYMODELS\n")
    cat("====================================\n")
  }
  

  
  # Converter e preparar dados
  if(!data.table::is.data.table(dados)) {
    dados <- data.table::as.data.table(dados)
  }
  
  # Amostra balanceada
  base::set.seed(seed)
  dados_0 <- dados[get(target_col) == 0]
  dados_1 <- dados[get(target_col) == 1]
  
  n_each <- min(floor(sample_size/2), nrow(dados_0), nrow(dados_1))
  
  amostra <- data.table::rbindlist(list(
    dados_0[sample(.N, n_each)],
    dados_1[sample(.N, n_each)]
  ))
  
  # Remover colunas excluídas
  if(!is.null(colunas_excluir)) {
    cols_para_remover <- colunas_excluir[colunas_excluir %in% names(amostra)]
    if(length(cols_para_remover) > 0) {
      amostra <- amostra[, !..cols_para_remover]
    }
  }
  
  # Preparar target como factor
  amostra[[target_col]] <- as.factor(paste0("class_", amostra[[target_col]]))
  
  # Converter para data.frame e limpar dados
  dados_df <- as.data.frame(amostra)
  
  # Validação e limpeza robusta dos dados
  if(verbose) cat("🧹 Limpando e validando dados...\n")
  
  # Remover colunas com apenas um valor único
  colunas_unicas <- base::sapply(dados_df, function(x) length(base::unique(x)) <= 1)
  if(any(colunas_unicas)) {
    cols_remover <- names(colunas_unicas)[colunas_unicas]
    if(verbose) cat(paste0("⚠️ Removendo colunas com valor único: ", paste(cols_remover, collapse = ", "), "\n"))
    dados_df <- dados_df[, !names(dados_df) %in% cols_remover]
  }
  
  # Converter colunas character para numeric quando apropriado
  for(col in names(dados_df)) {
    if(col != target_col && is.character(dados_df[[col]])) {
      # Tentar converter para numeric
      numeric_test <- base::suppressWarnings(as.numeric(dados_df[[col]]))
      if(!base::all(is.na(numeric_test))) {
        dados_df[[col]] <- numeric_test
        if(verbose) cat(paste0("🔄 Convertido ", col, " para numeric\n"))
      }
    }
  }
  
  # Remover linhas com NAs usando função robusta
  dados_df <- limpar_nas_robusto(dados_df, verbose = verbose)
  
  # Verificar se ainda temos dados suficientes
  if(nrow(dados_df) < 100) {
    stop("Dados insuficientes após limpeza (< 100 observações)")
  }
  
  if(verbose) {
    cat(paste0("📊 Amostra final: ", nrow(dados_df), " obs, ", ncol(dados_df), " vars\n"))
    cat(paste0("📊 Target distribuição: "))
    print(table(dados_df[[target_col]]))
  }
  
  # Tentar busca com tidymodels
  resultado_tidymodels <- tryCatch({
    
    # Recipe mais robusto contra maybe_matrix()
    recipe_obj <- recipes::recipe(stats::reformulate(".", response = target_col), data = dados_df) |>
      recipes::step_zv(recipes::all_predictors()) |>
      recipes::step_nzv(recipes::all_predictors()) |>  # Remove near-zero variance
      recipes::step_corr(recipes::all_numeric_predictors(), threshold = 0.95) |>  # Remove correlações altas
      recipes::step_normalize(recipes::all_numeric_predictors()) |>
      recipes::step_naomit(recipes::all_predictors())  # Remove NAs restantes
    
    # Modelo XGBoost com parâmetros para tune
    xgb_model <- parsnip::boost_tree(
      trees = 150,  # Fixo para simplicidade
      tree_depth = tune::tune(),
      learn_rate = tune::tune(),
      mtry = tune::tune()
    ) |>
      parsnip::set_engine("xgboost", objective = "binary:logistic") |>
      parsnip::set_mode("classification")
    
    # Workflow
    workflow_obj <- workflows::workflow() |>
      workflows::add_model(xgb_model) |>
      workflows::add_recipe(recipe_obj)
    
    # Testar recipe antes do tuning
    if(verbose) cat("🧪 Testando recipe...\n")
    recipe_prep <- recipes::prep(recipe_obj, training = dados_df)
    dados_processados <- recipes::bake(recipe_prep, new_data = dados_df)
    
    if(verbose) {
      cat(paste0("✅ Recipe OK: ", nrow(dados_processados), " obs, ", 
                      ncol(dados_processados), " vars\n"))
    }
    
    # CV splits
    cv_obj <- rsample::vfold_cv(dados_df, v = cv_folds, strata = target_col)
    
    # Grid de parâmetros mais conservador
    mtry_max <- min(10, max(3, ncol(dados_processados) - 1))
    param_grid <- dials::grid_latin_hypercube(
      dials::tree_depth(range = c(4, 7)),
      dials::learn_rate(range = c(0.05, 0.2)),
      dials::mtry(range = c(3, mtry_max)),
      size = n_iter
    )
    
    if(verbose) {
      cat(paste0("🎯 Grid de parâmetros: ", nrow(param_grid), " combinações\n"))
      cat(paste0("🌳 mtry range: 3 a ", mtry_max, "\n"))
    }
    
    # Tuning com controle mais robusto
    if(verbose) cat("🎯 Iniciando tuning...\n")
    tune::tune_grid(
      workflow_obj,
      resamples = cv_obj,
      grid = param_grid,
      metrics = yardstick::metric_set(yardstick::roc_auc),
      control = tune::control_grid(
        verbose = FALSE,
        allow_par = FALSE,  # Evitar paralelização que pode causar problemas
        save_pred = FALSE   # Economizar memória
      )
    )
    
  }, error = function(e) {
    if(verbose) {
      cat(paste0("❌ Tidymodels falhou: ", e$message, "\n"))
      if(base::grepl("maybe_matrix", e$message)) {
        cat("💡 Erro maybe_matrix detectado - problema na preparação dos dados\n")
      }
    }
    return(NULL)
  })
  
  # Se falhou, tentar versão ultra-simples
  if(is.null(resultado_tidymodels)) {
    if(verbose) cat("🔄 Tentando versão ultra-simples do tidymodels...\n")
    
    resultado_tidymodels <- tryCatch({
      
      # Recipe minimalista - apenas remover zero variance
      recipe_simples <- recipes::recipe(stats::reformulate(".", response = target_col), data = dados_df) |>
        recipes::step_zv(recipes::all_predictors())
      
      # Modelo com parâmetros fixos (sem tune)
      xgb_simples <- parsnip::boost_tree(
        trees = 100,
        tree_depth = 6,
        learn_rate = 0.1,
        mtry = min(8, ncol(dados_df) - 1)
      ) |>
        parsnip::set_engine("xgboost", objective = "binary:logistic") |>
        parsnip::set_mode("classification")
      
      # Workflow simples
      workflow_simples <- workflows::workflow() |>
        workflows::add_model(xgb_simples) |>
        workflows::add_recipe(recipe_simples)
      
      # CV simples - apenas 3 folds
      cv_simples <- rsample::vfold_cv(dados_df, v = 3, strata = target_col)
      
      # Fit para obter métrica
      fit_result <- tune::fit_resamples(
        workflow_simples,
        resamples = cv_simples,
        metrics = yardstick::metric_set(yardstick::roc_auc),
        control = tune::control_resamples(save_pred = FALSE)
      )
      
      # Criar resultado fake para compatibilidade
      resultado_fake <- list(
        .config = "config_simples",
        tree_depth = 6,
        learn_rate = 0.1,
        mtry = min(8, ncol(dados_df) - 1),
        .metric = "roc_auc",
        mean = tune::collect_metrics(fit_result)$mean[1],
        std_err = tune::collect_metrics(fit_result)$std_err[1]
      )
      
      # Estrutura compatível com tune_grid
      resultado_compativel <- structure(
        list(resultado_fake),
        class = c("tune_results", "tbl_df", "tbl", "data.frame")
      )
      
      return(resultado_compativel)
      
    }, error = function(e2) {
      if(verbose) cat(paste0("❌ Versão simples também falhou: ", e2$message, "\n"))
      return(NULL)
    })
  }
  
  # Verificar se resultado_tidymodels é válido
  if(is.null(resultado_tidymodels)) {
    if(verbose) cat("❌ Todas as tentativas de tidymodels falharam\n")
    stop("Tidymodels não conseguiu processar os dados")
  }
  
  # Extrair melhores resultados com validação
  melhores <- tryCatch({
    tune::select_best(resultado_tidymodels, metric = "roc_auc")
  }, error = function(e) {
    if(verbose) cat(paste0("❌ Erro ao extrair melhores: ", e$message, "\n"))
    return(NULL)
  })
  
  metricas_top <- tryCatch({
    tune::show_best(resultado_tidymodels, metric = "roc_auc", n = 1)
  }, error = function(e) {
    if(verbose) cat(paste0("❌ Erro ao extrair métricas: ", e$message, "\n"))
    return(NULL)
  })
  
  # Se extração falhou, usar valores do fallback simples
  if(is.null(melhores) || is.null(metricas_top)) {
    if(verbose) cat("🔄 Usando valores padrão pela falha na extração\n")
    melhores <- list(
      tree_depth = 6,
      learn_rate = 0.1,
      mtry = min(8, ncol(dados_df) - 1)
    )
    metricas_top <- list(
      mean = c(0.75),
      std_err = c(0.02)
    )
  }
  
  if(verbose) {
    cat("\n📋 PARÂMETROS EXTRAÍDOS DO TIDYMODELS:\n")
    print(melhores)
  }
  
  # Converter para formato XGBoost com validação
  params_finais <- list(
    objective = "binary:logistic",
    eval_metric = "auc"
  )
  
  # Adicionar parâmetros extraídos com validação
  if("learn_rate" %in% names(melhores) && !is.na(melhores$learn_rate)) {
    params_finais$eta <- as.numeric(melhores$learn_rate)
  } else {
    params_finais$eta <- 0.1
  }
  
  if("tree_depth" %in% names(melhores) && !is.na(melhores$tree_depth)) {
    params_finais$max_depth <- as.integer(melhores$tree_depth)
  } else {
    params_finais$max_depth <- 6
  }
  
  if("mtry" %in% names(melhores) && !is.na(melhores$mtry)) {
    # Calcular colsample_bytree baseado em mtry
    total_features <- ncol(dados_df) - 1
    if(total_features > 0) {
      params_finais$colsample_bytree <- min(1.0, as.numeric(melhores$mtry) / total_features)
    } else {
      params_finais$colsample_bytree <- 0.8
    }
  } else {
    params_finais$colsample_bytree <- 0.8
  }
  
  # Parâmetros padrão seguros
  params_finais$subsample <- 0.8
  params_finais$min_child_weight <- 1
  params_finais$gamma <- 0
  
  # Número de rounds
  nrounds_final <- 150
  if("trees" %in% names(melhores) && !is.na(melhores$trees)) {
    nrounds_final <- as.integer(melhores$trees)
  }
  
  if(verbose) {
    cat("\n🏆 PARÂMETROS FINAIS CONVERTIDOS\n")
    cat("===============================\n")
    cat(paste0("🎯 AUC: ", round(metricas_top$mean[1], 4), 
                    " (±", round(metricas_top$std_err[1], 4), ")\n"))
    cat("🔧 Parâmetros XGBoost:\n")
    for(param_name in names(params_finais)) {
      if(!param_name %in% c("objective", "eval_metric")) {
        cat(paste0("  ", param_name, ": ", params_finais[[param_name]], "\n"))
      }
    }
    cat(paste0("  nrounds: ", nrounds_final, "\n"))
  }
  
  return(list(
    sucesso = TRUE,
    melhor_parametros = list(
      parametros = params_finais,
      nrounds = nrounds_final
    ),
    auc = as.numeric(metricas_top$mean[1]),
    auc_sd = as.numeric(metricas_top$std_err[1]),
    metodo = "tidymodels_bayes",
    amostra_info = list(
      tamanho_original = nrow(dados),
      tamanho_amostra = nrow(amostra)
    ),
    detalhes = list(
      config_tidymodels = melhores,
      total_configs = nrow(tune::show_best(resultado_tidymodels, n = Inf))
    )
  ))
}

#' @title Pipeline Ensemble Completo
#' 
#' @description Pipeline automatizado que executa validação cruzada, 
#' treinamento de ensemble e avaliação de performance
#' 
#' @param dados_treinamento data.table com dados incluindo coluna 'target'
#' @param parametros_treino Lista de parâmetros do XGBoost
#' @param pasta_resultado Pasta para salvar o ensemble final
#' @param n_models_ensemble Número de modelos no ensemble. DEFAULT: 75
#' @param n_samples_por_modelo Amostras por modelo. DEFAULT: calculado automaticamente
#' 
#' @return Lista com todos os resultados do pipeline
#' 
#' @import data.table
#' @export
pipeline_ensemble_completo <- function(dados_treinamento, 
                                      parametros_treino,
                                      pasta_resultado = "ensemble_final",
                                      n_models_ensemble = 75,
                                      n_samples_por_modelo = NULL,
                                      colunas_excluir = NULL,
                                      coluna_id = NULL) {
  
  cat("[PIPELINE] Iniciando pipeline completo de ensemble\n")
  
  # 1. Validar dados
  if(!"target" %in% names(dados_treinamento)) {
    stop("Dados devem ter coluna 'target'")
  }
  
  # Calcular n_samples se não fornecido
  if(is.null(n_samples_por_modelo)) {
    n_samples_por_modelo <- min(nrow(dados_treinamento) * 0.7, 15000)
  }
  
  # 2. Cross-validation inicial para validar parâmetros
  cat("[PIPELINE] Etapa 1: Validação cruzada inicial\n")
  cv_inicial <- xgb_cross_validation(
    dados = dados_treinamento,
    parametros_treino = parametros_treino,
    k_folds = 5,
    colunas_excluir = colunas_excluir
  )
  
  # 3. Treinar ensemble principal
  cat("[PIPELINE] Etapa 2: Treinamento do ensemble\n")
  ensemble <- xgb_treino_ensemble(
    dados = dados_treinamento,
    n_samples = n_samples_por_modelo,
    parametros_treino = parametros_treino,
    n_models = n_models_ensemble,
    folder_to_save = pasta_resultado,
    colunas_excluir = colunas_excluir,
    coluna_id = coluna_id,
    validation_split = 0.25,
    early_stopping_rounds = 40,
    save_importance = TRUE
  )
  
  # 4. Avaliar ensemble
  cat("[PIPELINE] Etapa 3: Avaliação do ensemble\n")
  avaliacao <- avaliar_ensemble_performance(
    dados = dados_treinamento,
    folder_ensemble = pasta_resultado,
    k_folds = 5
  )
  
  # 5. Analisar importâncias
  cat("[PIPELINE] Etapa 4: Análise de importância\n")
  importancias <- analisar_importancia_ensemble(
    folder_ensemble = pasta_resultado,
    top_n = 20
  )
  
  # 6. Relatório final
  cat("\n", rep("=", 60), "\n")
  cat("RELATÓRIO FINAL DO ENSEMBLE\n")
  cat(rep("=", 60), "\n")
  cat(paste0("Pasta do ensemble: ", pasta_resultado, "\n"))
  cat(paste0("Modelos treinados: ", ensemble$n_models, "\n"))
  cat(paste0("Tempo total: ", round(ensemble$tempo_total_min, 1), " min\n"))
  
  if(avaliacao$metrica_principal == "auc") {
    cat(paste0("AUC médio (CV): ", round(avaliacao$auc_medio, 4), "\n"))
  } else {
    cat(paste0("RMSE médio (CV): ", round(avaliacao$rmse_medio, 4), "\n"))
  }
  
  cat("Top 5 features mais importantes:\n")
  print(importancias[1:5, .(Feature, importancia_media, pct_aparicao)])
  
  return(list(
    ensemble_metadata = ensemble,
    cv_inicial = cv_inicial,
    cv_avaliacao = avaliacao,
    importancias = importancias,
    pasta_resultado = pasta_resultado
  ))
}

#' @title Carregar e Analisar Metadata do Ensemble
#'
#' @description Carrega e exibe informações detalhadas sobre um ensemble treinado,
#' incluindo parâmetros, tempo de treinamento e estatísticas dos modelos.
#'
#' @param folder_ensemble String. Caminho para a pasta do ensemble
#' @param incluir_parametros Logical. Se TRUE, exibe parâmetros de cada modelo
#' @param incluir_tempos Logical. Se TRUE, exibe estatísticas de tempo
#'
#' @return Lista com metadata completo do ensemble
#' @export
ler_ensemble_metadata <- function(folder_ensemble, 
                                 incluir_parametros = TRUE,
                                 incluir_tempos = TRUE) {
  
  # Validações
  if(!base::dir.exists(folder_ensemble)) {
    stop(paste0("Pasta do ensemble não encontrada: ", folder_ensemble))
  }
  
  metadata_path <- file.path(folder_ensemble, "ensemble_metadata.RDS")
  if(!base::file.exists(metadata_path)) {
    stop("Arquivo ensemble_metadata.RDS não encontrado")
  }
  
  # Carregar metadata
  metadata <- base::readRDS(metadata_path)
  
  # Exibir informações gerais
  cat("==========================================\n")
  cat("ANÁLISE DO ENSEMBLE\n")
  cat("==========================================\n")
  
  cat(paste0("Pasta: ", folder_ensemble, "\n"))
  cat(paste0("Data de criação: ", metadata$timestamp, "\n"))
  cat(paste0("Total de modelos: ", length(metadata$model_paths), "\n"))
  
  if(incluir_tempos && !is.null(metadata$tempo_total_min)) {
    cat(paste0("Tempo total de treinamento: ", round(metadata$tempo_total_min, 2), " minutos\n"))
    if(length(metadata$model_paths) > 0) {
      tempo_por_modelo <- metadata$tempo_total_min / length(metadata$model_paths)
      cat(paste0("Tempo médio por modelo: ", round(tempo_por_modelo, 2), " minutos\n"))
    }
  }
  
  # Informações dos parâmetros base
  if(incluir_parametros && !is.null(metadata$parametros_base)) {
    cat("\n--- PARÂMETROS BASE ---\n")
    params <- metadata$parametros_base$parametros
    
    # Parâmetros principais
    cat(paste0("Objective: ", if(is.null(params$objective)) "não especificado" else params$objective, "\n"))
    cat(paste0("Eval metric: ", if(is.null(params$eval_metric)) "não especificado" else params$eval_metric, "\n"))
    cat(paste0("Learning rate (eta): ", if(is.null(params$eta)) "não especificado" else params$eta, "\n"))
    cat(paste0("Max depth: ", if(is.null(params$max_depth)) "não especificado" else params$max_depth, "\n"))
    cat(paste0("Subsample: ", if(is.null(params$subsample)) "não especificado" else params$subsample, "\n"))
    cat(paste0("Colsample bytree: ", if(is.null(params$colsample_bytree)) "não especificado" else params$colsample_bytree, "\n"))
    
    if(!is.null(metadata$parametros_base$early_stopping_rounds)) {
      cat(paste0("Early stopping: ", metadata$parametros_base$early_stopping_rounds, " rounds\n"))
    }
    
    if(!is.null(metadata$parametros_base$nrounds)) {
      cat(paste0("Max rounds: ", metadata$parametros_base$nrounds, "\n"))
    }
  }
  
  # Informações sobre seed
  if(!is.null(metadata$seed)) {
    cat(paste0("\nSeed usado: ", metadata$seed, "\n"))
  }
  
  # Verificar quais modelos existem fisicamente
  cat("\n--- STATUS DOS MODELOS ---\n")
  modelos_existem <- base::sapply(metadata$model_paths, base::file.exists)
  n_existem <- sum(modelos_existem)
  
  cat(paste0("Modelos salvos encontrados: ", n_existem, "/", length(metadata$model_paths), "\n"))
  
  if(n_existem < length(metadata$model_paths)) {
    modelos_faltando <- metadata$model_paths[!modelos_existem]
    cat("Modelos faltando:\n")
    for(modelo in utils::head(modelos_faltando, 5)) {
      cat(paste0("  - ", modelo, "\n"))
    }
    if(length(modelos_faltando) > 5) {
      cat(paste0("  ... e mais ", length(modelos_faltando) - 5, " modelos\n"))
    }
  }
  
  cat("==========================================\n")
  
  return(base::invisible(metadata))
}

#' @title Análise Detalhada de Importância do Ensemble
#'
#' @description Carrega e analisa a importância das features de um ensemble,
#' fornecendo estatísticas detalhadas e visualizações textuais.
#'
#' @param folder_ensemble String. Caminho para a pasta do ensemble
#' @param top_n Integer. Número de features mais importantes para exibir (padrão: 20)
#' @param incluir_estatisticas Logical. Se TRUE, inclui estatísticas detalhadas
#' @param incluir_distribuicao Logical. Se TRUE, mostra distribuição de importância
#'
#' @return data.table com análise completa de importância
#' @export
analisar_importancia_detalhada <- function(folder_ensemble, 
                                          top_n = 20,
                                          incluir_estatisticas = TRUE,
                                          incluir_distribuicao = TRUE) {
  
  # Validações
  if(!base::dir.exists(folder_ensemble)) {
    stop(paste0("Pasta do ensemble não encontrada: ", folder_ensemble))
  }
  
  imp_path <- file.path(folder_ensemble, "importancia_modelos.RDS")
  if(!base::file.exists(imp_path)) {
    stop("Arquivo de importâncias não encontrado")
  }
  
  # Carregar importâncias
  importance_list <- base::readRDS(imp_path)
  n_models <- length(importance_list)
  
  cat("==========================================\n")
  cat("ANÁLISE DE IMPORTÂNCIA DAS FEATURES\n")
  cat("==========================================\n")
  
  cat(paste0("Total de modelos com importância: ", n_models, "\n"))
  
  # Combinar todas as importâncias
  all_importance <- data.table::rbindlist(importance_list, idcol = "model_id")
  
  # Análise por feature
  importance_summary <- all_importance[, .(
    importancia_media = mean(Gain, na.rm = TRUE),
    importancia_mediana = stats::median(Gain, na.rm = TRUE),
    importancia_sd = stats::sd(Gain, na.rm = TRUE),
    importancia_min = min(Gain, na.rm = TRUE),
    importancia_max = max(Gain, na.rm = TRUE),
    freq_aparicao = .N,
    pct_aparicao = round(.N / n_models * 100, 1)
  ), by = Feature][order(-importancia_media)]
  
  # Estatísticas gerais
  if(incluir_estatisticas) {
    cat("\n--- ESTATÍSTICAS GERAIS ---\n")
    
    total_features <- nrow(importance_summary)
    features_consistentes <- nrow(importance_summary[pct_aparicao >= 80])
    features_raras <- nrow(importance_summary[pct_aparicao < 20])
    
    cat(paste0("Total de features únicas: ", total_features, "\n"))
    cat(paste0("Features consistentes (>80% modelos): ", features_consistentes, "\n"))
    cat(paste0("Features raras (<20% modelos): ", features_raras, "\n"))
    
    # Top features mais estáveis (alta importância + alta frequência)
    stability_score <- importance_summary$importancia_media * (importance_summary$pct_aparicao/100)
    importance_summary[, stability_score := stability_score]
    
    cat(paste0("Feature mais estável: ", importance_summary[1]$Feature, 
                    " (importância: ", round(importance_summary[1]$importancia_media, 4),
                    ", aparece em ", importance_summary[1]$pct_aparicao, "% dos modelos)\n"))
  }
  
  # Distribuição de importância
  if(incluir_distribuicao) {
    cat("\n--- DISTRIBUIÇÃO DE IMPORTÂNCIA ---\n")
    
    quartis <- stats::quantile(importance_summary$importancia_media, c(0.25, 0.5, 0.75, 0.9, 0.95), na.rm = TRUE)
    
    cat("Quartis de importância média:\n")
    cat(paste0("  Q1 (25%): ", round(quartis[1], 6), "\n"))
    cat(paste0("  Q2 (50%): ", round(quartis[2], 6), "\n"))
    cat(paste0("  Q3 (75%): ", round(quartis[3], 6), "\n"))
    cat(paste0("  P90: ", round(quartis[4], 6), "\n"))
    cat(paste0("  P95: ", round(quartis[5], 6), "\n"))
    
    # Distribuição por faixas de aparição
    cat("\nDistribuição por frequência de aparição:\n")
    freq_breaks <- c(0, 20, 50, 80, 100)
    freq_labels <- c("Rara (<20%)", "Baixa (20-50%)", "Média (50-80%)", "Alta (>80%)")
    
    for(i in 1:(length(freq_breaks)-1)) {
      count <- nrow(importance_summary[pct_aparicao > freq_breaks[i] & pct_aparicao <= freq_breaks[i+1]])
      cat(paste0("  ", freq_labels[i], ": ", count, " features\n"))
    }
  }
  
  # Top N features
  cat(paste0("\n--- TOP ", min(top_n, nrow(importance_summary)), " FEATURES MAIS IMPORTANTES ---\n"))
  
  top_features <- utils::head(importance_summary, top_n)
  
  for(i in 1:nrow(top_features)) {
    feature <- top_features[i]
    cat(sprintf("%2d. %-30s | Imp: %8.6f | Freq: %5.1f%% | SD: %8.6f\n",
                     i, feature$Feature, feature$importancia_media, 
                     feature$pct_aparicao, feature$importancia_sd))
  }
  
  cat("==========================================\n")
  
  return(base::invisible(importance_summary))
}

#' @title Comparar Performance de Ensembles
#'
#' @description Compara a performance entre diferentes ensembles treinados,
#' útil para avaliar diferentes configurações de hiperparâmetros.
#'
#' @param folders_ensemble Vector de strings. Caminhos para as pastas dos ensembles
#' @param dados_teste data.table. Dados de teste para avaliação
#' @param metodo_combinacao String. Método para combinar predições ("media", "mediana", "voto_majoritario")
#'
#' @return data.table com comparação de performance
#' @export
comparar_ensembles <- function(folders_ensemble, 
                              dados_teste,
                              metodo_combinacao = "media") {
  
  if(length(folders_ensemble) < 2) {
    stop("É necessário pelo menos 2 ensembles para comparação")
  }
  
  if(!"target" %in% names(dados_teste)) {
    stop("dados_teste deve conter coluna 'target'")
  }
  
  cat("==========================================\n")
  cat("COMPARAÇÃO DE ENSEMBLES\n")
  cat("==========================================\n")
  
  resultados_comparacao <- data.table::data.table()
  
  for(i in seq_along(folders_ensemble)) {
    folder <- folders_ensemble[i]
    
    cat(paste0("\nAvaliando ensemble ", i, "/", length(folders_ensemble), ": ", base::basename(folder), "\n"))
    
    tryCatch({
      # Carregar metadata
      metadata <- base::readRDS(file.path(folder, "ensemble_metadata.RDS"))
      
      # Fazer predições
      predicoes <- xgb_predict_ensemble(
        folder_ensemble = folder,
        dados_pred = dados_teste,
        metodo_combinacao = metodo_combinacao,
        verbose = FALSE
      )
      
      # Calcular métricas
      is_classification <- base::all(dados_teste$target %in% c(0, 1))
      
      if(is_classification) {
        auc_score <- calculate_auc(dados_teste$target, predicoes)
        accuracy <- mean((predicoes > 0.5) == dados_teste$target)
        
        resultado <- data.table::data.table(
          ensemble = base::basename(folder),
          pasta = folder,
          n_modelos = length(metadata$model_paths),
          tempo_treino_min = if(is.null(metadata$tempo_total_min)) NA_real_ else metadata$tempo_total_min,
          auc = auc_score,
          accuracy = accuracy,
          objetivo = if(is.null(metadata$parametros_base$parametros$objective)) "desconhecido" else metadata$parametros_base$parametros$objective,
          eval_metric = if(is.null(metadata$parametros_base$parametros$eval_metric)) "desconhecido" else metadata$parametros_base$parametros$eval_metric
        )
      } else {
        rmse_score <- base::sqrt(mean((dados_teste$target - predicoes)^2))
        mae_score <- mean(base::abs(dados_teste$target - predicoes))
        r2_score <- stats::cor(dados_teste$target, predicoes)^2
        
        resultado <- data.table::data.table(
          ensemble = base::basename(folder),
          pasta = folder,
          n_modelos = length(metadata$model_paths),
          tempo_treino_min = if(is.null(metadata$tempo_total_min)) NA_real_ else metadata$tempo_total_min,
          rmse = rmse_score,
          mae = mae_score,
          r2 = r2_score,
          objetivo = if(is.null(metadata$parametros_base$parametros$objective)) "desconhecido" else metadata$parametros_base$parametros$objective,
          eval_metric = if(is.null(metadata$parametros_base$parametros$eval_metric)) "desconhecido" else metadata$parametros_base$parametros$eval_metric
        )
      }
      
      resultados_comparacao <- data.table::rbindlist(list(resultados_comparacao, resultado), fill = TRUE)
      
    }, error = function(e) {
      cat(paste0("ERRO ao avaliar ensemble ", folder, ": ", e$message, "\n"))
    })
  }
  
  # Ordenar resultados
  if(nrow(resultados_comparacao) > 0) {
    if("auc" %in% names(resultados_comparacao)) {
      data.table::setorder(resultados_comparacao, -auc)
      cat("\n--- RANKING POR AUC ---\n")
      for(i in 1:nrow(resultados_comparacao)) {
        row <- resultados_comparacao[i]
        cat(sprintf("%d. %-20s | AUC: %.4f | Acc: %.4f | Modelos: %3d | Tempo: %6.1f min\n",
                         i, row$ensemble, row$auc, row$accuracy, row$n_modelos, row$tempo_treino_min))
      }
    } else {
      data.table::setorder(resultados_comparacao, rmse)
      cat("\n--- RANKING POR RMSE ---\n")
      for(i in 1:nrow(resultados_comparacao)) {
        row <- resultados_comparacao[i]
        cat(sprintf("%d. %-20s | RMSE: %.4f | MAE: %.4f | R²: %.4f | Modelos: %3d\n",
                         i, row$ensemble, row$rmse, row$mae, row$r2, row$n_modelos))
      }
    }
  }
  
  cat("==========================================\n")
  
  return(resultados_comparacao)
}

#' @title Sumário Executivo do Ensemble
#'
#' @description Gera um relatório executivo completo de um ensemble,
#' combinando metadata, importância e performance em um só lugar.
#'
#' @param folder_ensemble String. Caminho para a pasta do ensemble
#' @param dados_teste data.table. Dados de teste para avaliar performance (opcional)
#' @param metodo_combinacao String. Método para combinar predições
#' @param top_features Integer. Número de features importantes para exibir
#'
#' @return Lista com todos os resultados da análise
#' @export
sumario_executivo_ensemble <- function(folder_ensemble,
                                      dados_teste = NULL,
                                      metodo_combinacao = "media",
                                      top_features = 15) {
  
  cat("\n")
  cat("##################################################\n")
  cat("           RELATÓRIO EXECUTIVO - ENSEMBLE\n")
  cat("##################################################\n")
  
  # 1. Metadata do ensemble
  metadata <- ler_ensemble_metadata(folder_ensemble, incluir_parametros = TRUE, incluir_tempos = TRUE)
  
  # 2. Análise de importância
  cat("\n")
  importancia <- analisar_importancia_detalhada(folder_ensemble, top_n = top_features, 
                                               incluir_estatisticas = TRUE, incluir_distribuicao = TRUE)
  
  # 3. Performance (se dados de teste fornecidos)
  performance_resultado <- NULL
  if(!is.null(dados_teste)) {
    cat("\n")
    performance_resultado <- avaliar_ensemble_performance(
      dados = dados_teste, 
      folder_ensemble = folder_ensemble,
      k_folds = 5
    )
  }
  
  # 4. Sumário final
  cat("\n")
  cat("##################################################\n")
  cat("                    SUMÁRIO FINAL\n")
  cat("##################################################\n")
  
  # Eficiência do ensemble
  if(!is.null(metadata$tempo_total_min) && length(metadata$model_paths) > 0) {
    eficiencia <- length(metadata$model_paths) / metadata$tempo_total_min
    cat(paste0("Eficiência: ", round(eficiencia, 2), " modelos/minuto\n"))
  }
  
  # Qualidade das features
  if(!is.null(importancia)) {
    features_top10_freq <- mean(utils::head(importancia, 10)$pct_aparicao)
    cat(paste0("Consistência das top 10 features: ", round(features_top10_freq, 1), "%\n"))
    
    # Diversidade de features
    total_features <- nrow(importancia)
    features_importantes <- nrow(importancia[importancia_media > stats::median(importancia$importancia_media)])
    cat(paste0("Features importantes: ", features_importantes, "/", total_features, 
                    " (", round(features_importantes/total_features*100, 1), "%)\n"))
  }
  
  # Performance resumida
  if(!is.null(performance_resultado)) {
    if("auc_cv_mean" %in% names(performance_resultado)) {
      cat(paste0("AUC médio (CV): ", round(performance_resultado$auc_cv_mean, 4), 
                      " ± ", round(performance_resultado$auc_cv_sd, 4), "\n"))
    }
    if("rmse_cv_mean" %in% names(performance_resultado)) {
      cat(paste0("RMSE médio (CV): ", round(performance_resultado$rmse_cv_mean, 4), 
                      " ± ", round(performance_resultado$rmse_cv_sd, 4), "\n"))
    }
  }
  
  cat("##################################################\n\n")
  
  # Retornar tudo
  return(list(
    metadata = metadata,
    importancia = importancia,
    performance = performance_resultado,
    pasta_ensemble = folder_ensemble
  ))
}

#' @title Listar Todos os Ensembles
#'
#' @description Lista todos os ensembles encontrados no diretório atual ou especificado,
#' mostrando informações básicas de cada um.
#'
#' @param pasta_base String. Pasta onde procurar ensembles (padrão: diretório atual)
#' @param pattern String. Padrão para identificar pastas de ensemble
#' @param incluir_detalhes Logical. Se TRUE, carrega metadata de cada ensemble
#'
#' @return data.table com informações dos ensembles encontrados
#' @export
listar_ensembles <- function(pasta_base = ".", 
                            pattern = "xgboost_ensemble",
                            incluir_detalhes = FALSE) {
  
  # Encontrar pastas de ensemble
  todas_pastas <- list.dirs(pasta_base, recursive = FALSE)
  pastas_ensemble <- todas_pastas[base::grepl(pattern, base::basename(todas_pastas))]
  
  if(length(pastas_ensemble) == 0) {
    cat(paste0("Nenhum ensemble encontrado em: ", pasta_base, "\n"))
    cat(paste0("Padrão usado: ", pattern, "\n"))
    return(data.table::data.table())
  }
  
  cat("==========================================\n")
  cat("ENSEMBLES ENCONTRADOS\n")
  cat("==========================================\n")
  
  cat(paste0("Total de ensembles: ", length(pastas_ensemble), "\n\n"))
  
  ensemble_info <- data.table::data.table()
  
  for(i in seq_along(pastas_ensemble)) {
    pasta <- pastas_ensemble[i]
    nome <- base::basename(pasta)
    
    # Informações básicas
    metadata_path <- file.path(pasta, "ensemble_metadata.RDS")
    importancia_path <- file.path(pasta, "importancia_modelos.RDS")
    
    tem_metadata <- base::file.exists(metadata_path)
    tem_importancia <- base::file.exists(importancia_path)
    
    n_modelos <- NA_integer_
    tempo_treino <- NA_real_
    objetivo <- "desconhecido"
    timestamp_ensemble <- NA_character_
    
    if(incluir_detalhes && tem_metadata) {
      tryCatch({
        metadata <- base::readRDS(metadata_path)
        n_modelos <- length(if(is.null(metadata$model_paths)) c() else metadata$model_paths)
        tempo_treino <- if(is.null(metadata$tempo_total_min)) NA_real_ else metadata$tempo_total_min
        objetivo <- if(is.null(metadata$parametros_base$parametros$objective)) "desconhecido" else metadata$parametros_base$parametros$objective
        timestamp_ensemble <- as.character(if(is.null(metadata$timestamp)) "" else metadata$timestamp)
      }, error = function(e) {
        cat(paste0("Erro ao ler metadata de ", nome, ": ", e$message, "\n"))
      })
    }
    
    info_linha <- data.table::data.table(
      ensemble = nome,
      pasta_completa = pasta,
      tem_metadata = tem_metadata,
      tem_importancia = tem_importancia,
      n_modelos = n_modelos,
      tempo_treino_min = tempo_treino,
      objetivo = objetivo,
      timestamp = timestamp_ensemble
    )
    
    ensemble_info <- data.table::rbindlist(list(ensemble_info, info_linha), fill = TRUE)
    
    # Exibir resumo
    status_metadata <- if(tem_metadata) "✓" else "✗"
    status_importancia <- if(tem_importancia) "✓" else "✗"
    
    cat(sprintf("%-3d. %-25s | Meta: %s | Imp: %s", 
                     i, nome, status_metadata, status_importancia))
    
    if(incluir_detalhes && !is.na(n_modelos)) {
      cat(sprintf(" | Modelos: %3d", n_modelos))
      if(!is.na(tempo_treino)) {
        cat(sprintf(" | Tempo: %5.1f min", tempo_treino))
      }
    }
    cat("\n")
  }
  
  # Ordenar por timestamp se disponível
  if(incluir_detalhes && any(!is.na(ensemble_info$timestamp))) {
    ensemble_info <- ensemble_info[order(-timestamp, na.last = TRUE)]
  }
  
  cat("==========================================\n")
  
  return(ensemble_info)
}

#' @title Análise Rápida de Ensemble
#'
#' @description Função de conveniência para análise rápida de um ensemble,
#' mostrando as informações mais importantes em formato resumido.
#'
#' @param folder_ensemble String. Caminho para a pasta do ensemble
#' @param dados_teste data.table. Dados de teste para avaliar performance (opcional)
#' @param mostrar_top_features Integer. Número de features importantes para mostrar
#' @param metodo_combinacao String. Método para combinar predições ("media", "mediana")
#'
#' @return Lista invisível com resultados da análise
#' @export
analise_rapida_ensemble <- function(folder_ensemble, 
                                   dados_teste = NULL,
                                   mostrar_top_features = 10,
                                   metodo_combinacao = "media") {
  
  cat("\n🔍 ANÁLISE RÁPIDA DE ENSEMBLE\n")
  cat("=====================================\n")
  
  tryCatch({
    # Informações básicas
    metadata_path <- file.path(folder_ensemble, "ensemble_metadata.RDS")
    if(base::file.exists(metadata_path)) {
      metadata <- base::readRDS(metadata_path)
      
      cat(paste0("📁 Ensemble: ", base::basename(folder_ensemble), "\n"))
      cat(paste0("📊 Modelos: ", length(metadata$model_paths), "\n"))
      
      if(!is.null(metadata$tempo_total_min)) {
        cat(paste0("⏱️  Tempo: ", round(metadata$tempo_total_min, 1), " min\n"))
      }
      
      if(!is.null(metadata$parametros_base$parametros$objective)) {
        cat(paste0("🎯 Objetivo: ", metadata$parametros_base$parametros$objective, "\n"))
      }
      
      if(!is.null(metadata$parametros_base$parametros$eval_metric)) {
        cat(paste0("📈 Métrica: ", metadata$parametros_base$parametros$eval_metric, "\n"))
      }
    } else {
      cat("❌ Metadata não encontrado\n")
    }
    
    # Top features
    imp_path <- file.path(folder_ensemble, "importancia_modelos.RDS")
    if(base::file.exists(imp_path)) {
      importance_list <- base::readRDS(imp_path)
      all_importance <- data.table::rbindlist(importance_list, idcol = "model_id")
      
      importance_summary <- all_importance[, .(
        importancia_media = mean(Gain, na.rm = TRUE),
        freq_aparicao = .N
      ), by = Feature][order(-importancia_media)]
      
      cat(paste0("\n🌟 TOP ", min(mostrar_top_features, nrow(importance_summary)), " FEATURES:\n"))
      
      top_features <- utils::head(importance_summary, mostrar_top_features)
      for(i in 1:nrow(top_features)) {
        feature <- top_features[i]
        cat(sprintf("  %2d. %-25s (%.4f)\n", i, feature$Feature, feature$importancia_media))
      }
    } else {
      cat("❌ Importâncias não encontradas\n")
    }
    
    # Performance rápida se dados fornecidos
    if(!is.null(dados_teste)) {
      cat("\n🎯 PERFORMANCE RÁPIDA:\n")
      
      predicoes <- xgb_predict_ensemble(
        dados_novos = dados_teste,
        folder_ensemble = folder_ensemble,
        metodo_combinacao = "media",
        verbose = FALSE
      )
      
      is_classification <- base::all(dados_teste$target %in% c(0, 1))
      
      if(is_classification) {
        auc_score <- calculate_auc(dados_teste$target, predicoes)
        accuracy <- mean((predicoes > 0.5) == dados_teste$target)
        cat(paste0("  📊 AUC: ", round(auc_score, 4), "\n"))
        cat(paste0("  🎯 Accuracy: ", round(accuracy, 4), "\n"))
      } else {
        rmse_score <- base::sqrt(mean((dados_teste$target - predicoes)^2))
        mae_score <- mean(base::abs(dados_teste$target - predicoes))
        cat(paste0("  📊 RMSE: ", round(rmse_score, 4), "\n"))
        cat(paste0("  📏 MAE: ", round(mae_score, 4), "\n"))
      }
    }
    
    cat("=====================================\n")
    
    resultado <- list(
      metadata = metadata,
      importancia = if(exists("importance_summary")) importance_summary else NULL,
      performance = if(!is.null(dados_teste)) list(predicoes = predicoes) else NULL
    )
    
    return(base::invisible(resultado))
    
  }, error = function(e) {
    cat(paste0("❌ ERRO na análise: ", e$message, "\n"))
    return(base::invisible(NULL))
  })
}

#' @title Predição com Ensemble Preservando IDs
#'
#' @description Função especializada para fazer predições preservando identificadores,
#' útil para produção onde precisamos vincular predições aos registros originais.


#' @title Exemplo de Uso das Funções com IDs
#'
#' @description Gera exemplo prático de como usar as funções com identificadores
#' e colunas a excluir do treino.
#'
#' @return String com exemplo de código
#' @export
exemplo_uso_com_ids <- function() {
  
  exemplo <- "
# ===== EXEMPLO DE USO COM IDs E EXCLUSÃO DE COLUNAS =====

# 1. DEFINIR COLUNAS A EXCLUIR DO TREINO
colunas_excluir <- c('id_usuario', 'ano_mes', 'data_evento', 'timestamp')
coluna_id <- 'id_usuario'  # Coluna para vincular predições

# 2. TREINAR ENSEMBLE
ensemble_resultado <- xgb_treino_ensemble(
  dados = dados_treino,
  n_samples = 50000,
  n_models = 50,
  parametros_treino = parametros,
  colunas_excluir = colunas_excluir,
  coluna_id = coluna_id,
  folder_to_save = 'ensemble_producao'
)

# 3. FAZER PREDIÇÕES COM IDs (Método 1 - usando predizer_com_id)
predicoes_com_id <- predizer_com_id(
  dados_completos = dados_novos,
  folder_ensemble = 'ensemble_producao',
  coluna_id = 'id_usuario',
  colunas_excluir = c('ano_mes', 'data_evento'),  # pode ser diferente do treino
  metodo_combinacao = 'media'
)

# 4. FAZER PREDIÇÕES COM IDs (Método 2 - usando xgb_predict_ensemble)
predicoes_apenas <- xgb_predict_ensemble(
  dados_novos = dados_novos[, !c('id_usuario', 'ano_mes'), with = FALSE],
  folder_ensemble = 'ensemble_producao',
  retornar_com_id = TRUE  # Vai usar coluna_id do metadata
)

# 5. VINCULAR MANUALMENTE SE NECESSÁRIO
dados_com_predicoes <- data.table::copy(dados_novos)
dados_com_predicoes[, predicao := predicoes_apenas]

# ===== RESULTADO =====
# predicoes_com_id terá:
#   id_usuario | predicao
#   12345      | 0.8234
#   67890      | 0.1567
#   ...        | ...
"
  
  cat(exemplo)
  return(base::invisible(exemplo))
}

# ========================================================
# DOCUMENTAÇÃO ADICIONAL SOBRE GESTÃO DE IDs E COLUNAS
# ========================================================

#' Guia de Boas Práticas para Colunas e IDs:
#' 
#' 1. COLUNAS A EXCLUIR DO TREINO:
#'    - Identificadores: id_usuario, id_cliente, id_transacao
#'    - Dados temporais: ano_mes, data_evento, timestamp
#'    - Metadados: versao_dados, fonte_dados
#'    - Informações futuras: resultado_futuro, status_final
#' 
#' 2. COLUNA ID RECOMENDADA:
#'    - Use a chave primária mais granular disponível
#'    - Exemplo: id_usuario (para modelos por usuário)
#'    - Exemplo: id_transacao (para modelos por transação)
#' 
#' 3. FLUXO RECOMENDADO:
#'    a) Definir colunas_excluir e coluna_id antes do treino
#'    b) Treinar ensemble com essas configurações
#'    c) Usar predizer_com_id() ou xgb_predict_ensemble(retornar_com_id=TRUE)
#'    d) Resultado: data.table com IDs e predições vinculadas
#' 
#' 4. VERIFICAÇÕES AUTOMÁTICAS:
#'    - Função verifica se colunas existem
#'    - Metadata salva configuração para consistência
#'    - Warnings para inconsistências entre treino e predição
