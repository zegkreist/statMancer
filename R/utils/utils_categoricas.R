#' @title Utilitários para variáveis categóricas
#'
#' @description Funções para codificar variáveis categóricas (character/factor)
#' em inteiros de forma consistente entre treino e predição, preservando o
#' mapeamento de níveis para aplicação futura.
#'
#' A abordagem usada é a **codificação ordinal** (integer encoding), em que cada
#' nível único recebe um inteiro de 1 a K. Modelos de árvore — como XGBoost com
#' \code{tree_method = "hist"} — lidam bem com esta codificação, pois cada split
#' pode separar qualquer subconjunto de inteiros.
#'
#' @section Funções disponíveis:
#' \describe{
#'   \item{\code{codificar_categoricas}}{Codifica colunas character/factor para inteiros.}
#'   \item{\code{detectar_cols_categoricas}}{Retorna nomes das colunas categóricas em um vetor de features.}
#' }


# ─── codificar_categoricas ────────────────────────────────────────────────────

#' @title Codificação ordinal de variáveis categóricas
#'
#' @description Converte colunas \code{character} ou \code{factor} para inteiros
#' (1, 2, 3, …) de forma consistente. Em modo de treino (\code{factor_map = NULL}),
#' constrói o mapeamento de níveis a partir dos dados. Em modo de predição
#' (\code{factor_map} fornecido), aplica o mapeamento do treino — garantindo que
#' o mesmo nível receba sempre o mesmo código.
#'
#' Níveis presentes na predição mas ausentes no treino são codificados como
#' \code{NA} (comportamento padrão de \code{factor()}).
#'
#' @param dt         \code{data.table} (ou \code{data.frame}) com os dados.
#' @param features   Vetor de nomes de colunas a processar.
#' @param factor_map Lista nomeada de vetores de níveis, gerada por uma chamada
#'                   anterior em modo de treino. Se \code{NULL} (padrão), o mapa
#'                   é construído a partir de \code{dt}.
#'
#' @return Lista com dois campos:
#'   \itemize{
#'     \item \code{X}:          \code{data.table} com as colunas categóricas
#'                              convertidas para inteiros e as demais inalteradas.
#'     \item \code{factor_map}: Lista nomeada de vetores de níveis (character).
#'                              Em modo de predição retorna o \code{factor_map}
#'                              recebido.
#'   }
#'
#' @examples
#' \dontrun{
#' library(data.table)
#' source("R/utils/utils_categoricas.R")
#'
#' dt_treino <- data.table(regiao = c("norte","sul","leste","norte"), score = c(1,2,3,4))
#' enc_treino <- codificar_categoricas(dt_treino, c("regiao","score"))
#' # enc_treino$X$regiao : 2, 3, 1, 2  (alphabetical: leste=1, norte=2, sul=3)
#' # enc_treino$factor_map$regiao : c("leste","norte","sul")
#'
#' dt_teste <- data.table(regiao = c("sul","norte","leste"), score = c(5,6,7))
#' enc_teste <- codificar_categoricas(dt_teste, c("regiao","score"),
#'                                    factor_map = enc_treino$factor_map)
#' # enc_teste$X$regiao : 3, 2, 1  (same mapping as training)
#' }
#'
#' @import data.table
#' @export
codificar_categoricas <- function(dt, features, factor_map = NULL) {
  if (!data.table::is.data.table(dt))
    dt <- data.table::as.data.table(dt)

  # Subset and copy only the needed features
  features_presentes <- intersect(features, names(dt))
  X <- data.table::copy(dt[, features_presentes, with = FALSE])

  modo_treino <- is.null(factor_map)
  if (modo_treino) factor_map <- list()

  for (col in features_presentes) {
    vals <- X[[col]]

    if (is.character(vals) || is.factor(vals)) {
      vals_char <- as.character(vals)

      if (modo_treino) {
        # Training: build sorted levels from non-NA values
        lvls <- sort(unique(vals_char[!is.na(vals_char)]))
        factor_map[[col]] <- lvls
      } else {
        lvls <- if (!is.null(factor_map[[col]])) factor_map[[col]] else
          sort(unique(vals_char[!is.na(vals_char)]))
      }

      X[[col]] <- as.integer(factor(vals_char, levels = lvls))
    }
  }

  list(X = X, factor_map = factor_map)
}


# ─── detectar_cols_categoricas ────────────────────────────────────────────────

#' @title Detecta colunas categóricas em um vetor de features
#'
#' @description Retorna o subconjunto de \code{features} cujas colunas em
#' \code{dt} são do tipo \code{character} ou \code{factor}.
#'
#' @param dt       \code{data.table} com os dados.
#' @param features Vetor de nomes de colunas a inspecionar.
#'
#' @return Vetor (character) com os nomes das colunas categóricas.
#'
#' @export
detectar_cols_categoricas <- function(dt, features) {
  features[vapply(features, function(col) {
    col %in% names(dt) &&
      (is.character(dt[[col]]) || is.factor(dt[[col]]))
  }, logical(1))]
}
