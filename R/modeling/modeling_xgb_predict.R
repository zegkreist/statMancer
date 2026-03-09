#' @title Predição com modelo XGBoost
#'
#' @description Gera predições a partir de um objeto retornado por
#' \code{xgb_train}, garantindo que as features estejam na mesma ordem
#' usada no treino.
#'
#' @param modelo_obj  Objeto retornado por \code{xgb_train}.
#' @param dt_novo     \code{data.table} com os dados para predição.
#'                    Deve conter todas as features usadas no treino.
#' @param var_id      Nome da coluna de ID a incluir no resultado.
#'                    DEFAULT: \code{NULL}.
#'
#' @return Um \code{data.table} com as colunas:
#'   \item{\code{<var_id>}}{ID da observação (se fornecido).}
#'   \item{\code{predito}}{Score/probabilidade predito pelo modelo.}
#'
#' @import data.table xgboost
#' @export
xgb_predict <- function(modelo_obj, dt_novo, var_id = NULL) {

  data.table::setDT(dt_novo)
  dt_novo <- data.table::copy(dt_novo)

  features         <- modelo_obj$features
  features_faltando <- setdiff(features, names(dt_novo))

  if (length(features_faltando) > 0)
    stop(paste0(
      "Features ausentes em dt_novo: ", paste(features_faltando, collapse = ", ")
    ))

  # Aplica o mapeamento de níveis do treino para codificação consistente
  enc   <- codificar_categoricas(dt_novo, features,
                                 factor_map = modelo_obj$factor_map)
  X     <- enc$X[, lapply(.SD, as.numeric)]
  dtest <- xgboost::xgb.DMatrix(as.matrix(X))
  preds <- predict(modelo_obj$modelo, dtest)

  if (!is.null(var_id) && var_id %in% names(dt_novo)) {
    resultado <- data.table::data.table(id = dt_novo[[var_id]], predito = preds)
    data.table::setnames(resultado, "id", var_id)
  } else {
    resultado <- data.table::data.table(predito = preds)
  }

  return(resultado)
}


# ─── importância das features ────────────────────────────────────────────────

#' @title Importância das features do modelo XGBoost
#'
#' @description Extrai e organiza a importância das features de um modelo
#' criado com \code{xgb_train}.
#'
#' @param modelo_obj  Objeto retornado por \code{xgb_train}.
#' @param top_n       Limitar a N features mais importantes.
#'                    DEFAULT: \code{NULL} (retorna todas).
#'
#' @return Um \code{data.table} com as colunas \code{Feature}, \code{Gain},
#'   \code{Cover} e \code{Frequency}, ordenado por \code{Gain} decrescente.
#'
#' @import data.table xgboost
#' @export
xgb_importancia <- function(modelo_obj, top_n = NULL) {

  imp <- xgboost::xgb.importance(
    model        = modelo_obj$modelo,
    feature_names = modelo_obj$features
  )

  data.table::setDT(imp)
  imp <- imp[order(-Gain)]

  if (!is.null(top_n))
    imp <- imp[seq_len(min(top_n, .N))]

  return(imp)
}
