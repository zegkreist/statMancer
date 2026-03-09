#' @title Treino de modelo XGBoost
#'
#' @description Treina um modelo XGBoost a partir de um \code{data.table}
#' de treino, retornando um objeto padronizado compatível com \code{xgb_predict}.
#'
#' Os parâmetros do modelo podem vir diretamente de \code{xgb_select_params}
#' (campo \code{$parametros}) ou ser definidos manualmente.
#'
#' @param dt_treino    Um \code{data.table} com os dados de treino.
#' @param var_target   Nome da coluna target (character).
#' @param vars_excluir Colunas a excluir do treino — ex: var_id, datas.
#'                     DEFAULT: \code{NULL}.
#' @param params       Lista de parâmetros XGBoost. Aceita a saída de
#'                     \code{xgb_select_params()$parametros}.
#'                     DEFAULT: parâmetros razoáveis para classificação binária.
#' @param nrounds      Número de rounds. Aceita \code{xgb_select_params()$nrounds}.
#'                     DEFAULT: \code{100}.
#' @param nthreads     Número de threads CPU. DEFAULT: \code{parallel::detectCores() - 1}.
#' @param verbose      Mostrar log do XGBoost. DEFAULT: \code{FALSE}.
#'
#' @return Uma lista com:
#'   \item{\code{$modelo}}{Objeto \code{xgb.Booster} treinado.}
#'   \item{\code{$features}}{Nomes das features usadas (na ordem do treino).}
#'   \item{\code{$params}}{Parâmetros utilizados.}
#'   \item{\code{$nrounds}}{Número de rounds utilizado.}
#'   \item{\code{$var_target}}{Nome da coluna target.}
#'
#' @import data.table xgboost
#' @export
xgb_train <- function(dt_treino,
                      var_target,
                      vars_excluir = NULL,
                      params       = list(
                        objective        = "binary:logistic",
                        eval_metric      = "auc",
                        eta              = 0.05,
                        max_leaves       = 31L,
                        subsample        = 0.8,
                        colsample_bytree = 0.8,
                        min_child_weight = 5,
                        gamma            = 1,
                        grow_policy      = "lossguide",
                        tree_method      = "hist"
                      ),
                      nrounds  = 100L,
                      nthreads = max(1L, parallel::detectCores() - 1L),
                      verbose  = FALSE) {

  data.table::setDT(dt_treino)
  dt_treino <- data.table::copy(dt_treino)

  if (!var_target %in% names(dt_treino))
    stop(paste0("Coluna target '", var_target, "' não encontrada nos dados."))

  features <- setdiff(names(dt_treino), unique(c(var_target, vars_excluir)))

  if (length(features) == 0)
    stop("Nenhuma feature disponível para treino após exclusões.")

  X <- dt_treino[, features, with = FALSE]
  X <- X[, lapply(.SD, as.numeric)]
  y <- as.numeric(dt_treino[[var_target]])

  dtrain <- xgboost::xgb.DMatrix(as.matrix(X), label = y)

  params$nthread <- nthreads

  modelo <- xgboost::xgb.train(
    params  = params,
    data    = dtrain,
    nrounds = as.integer(nrounds),
    verbose = as.integer(verbose)
  )

  return(list(
    modelo     = modelo,
    features   = features,
    params     = params,
    nrounds    = nrounds,
    var_target = var_target
  ))
}
