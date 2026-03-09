#' @title Métricas de avaliação de modelos
#'
#' @description Funções para calcular e organizar métricas de avaliação de
#' modelos preditivos, suportando classificação binária. Não possui dependências
#' externas além de \code{data.table} e funções de \code{stats} do R base.
#'
#' @section Funções disponíveis:
#' \describe{
#'   \item{\code{metricas_binario}}{AUC, KS, Gini, Precision, Recall, F1, Accuracy.}
#'   \item{\code{tabela_decis}}{Tabela de lift por decil.}
#'   \item{\code{curva_roc}}{Pontos da curva ROC para plotagem.}
#' }


# ─── metricas_binario ────────────────────────────────────────────────────────

#' @title Métricas de classificação binária
#'
#' @description Calcula as principais métricas de avaliação para modelos
#' de classificação binária: AUC, KS, Gini, e métricas de corte em 0.5.
#'
#' @param dt         \code{data.table} com predições e target.
#' @param var_pred   Nome da coluna de scores/probabilidades preditos (0–1).
#' @param var_target Nome da coluna target binário (0 = negativo, 1 = positivo).
#'
#' @return Lista nomeada com:
#'   \itemize{
#'     \item \code{n_total}, \code{n_pos}, \code{n_neg}, \code{taxa_evento}
#'     \item \code{auc}: Area Under the ROC Curve
#'     \item \code{ks}: Estatística KS (Kolmogorov-Smirnov)
#'     \item \code{gini}: Coeficiente de Gini (= 2 * AUC - 1)
#'     \item \code{precision}, \code{recall}, \code{f1}, \code{accuracy} (threshold 0.5)
#'     \item \code{tp}, \code{fp}, \code{tn}, \code{fn}
#'   }
#'
#' @import data.table
#' @export
metricas_binario <- function(dt, var_pred, var_target) {

  data.table::setDT(dt)

  if (!var_pred %in% names(dt))
    stop(paste0("Coluna de predição '", var_pred, "' não encontrada."))
  if (!var_target %in% names(dt))
    stop(paste0("Coluna target '", var_target, "' não encontrada."))

  scores <- as.numeric(dt[[var_pred]])
  labels <- as.integer(dt[[var_target]])

  n_total <- length(labels)
  n_pos   <- sum(labels, na.rm = TRUE)
  n_neg   <- n_total - n_pos

  if (n_pos == 0 || n_neg == 0)
    stop("Target deve ter pelo menos uma observação positiva e uma negativa.")

  auc  <- .calc_auc(scores, labels)
  ks   <- .calc_ks(scores, labels)
  gini <- 2 * auc - 1

  cm <- .confusion_at_threshold(scores, labels, threshold = 0.5)

  list(
    n_total     = n_total,
    n_pos       = n_pos,
    n_neg       = n_neg,
    taxa_evento = round(n_pos / n_total, 4),
    auc         = round(auc,  4),
    ks          = round(ks,   4),
    gini        = round(gini, 4),
    precision   = round(cm$precision, 4),
    recall      = round(cm$recall,    4),
    f1          = round(cm$f1,        4),
    accuracy    = round(cm$accuracy,  4),
    tp          = cm$tp,
    fp          = cm$fp,
    tn          = cm$tn,
    fn          = cm$fn
  )
}


# ─── tabela_decis ────────────────────────────────────────────────────────────

#' @title Tabela de lift por decil
#'
#' @description Ordena as observações por score decrescente, divide em
#' \code{n_decis} grupos iguais e calcula taxa de evento, lift e captura
#' acumulada em cada grupo.
#'
#' @param dt         \code{data.table} com predições e target.
#' @param var_pred   Nome da coluna de scores preditos.
#' @param var_target Nome da coluna target binário (0/1).
#' @param n_decis    Número de faixas. DEFAULT: \code{10}.
#'
#' @return \code{data.table} com: \code{decil}, \code{n}, \code{n_pos},
#'   \code{score_min}, \code{score_max}, \code{score_medio},
#'   \code{taxa_evento}, \code{lift}, \code{captura}, \code{captura_acum}.
#'
#' @import data.table
#' @export
tabela_decis <- function(dt, var_pred, var_target, n_decis = 10L) {

  data.table::setDT(dt)

  dt_tmp <- data.table::copy(
    dt[, .(score  = as.numeric(get(var_pred)),
           target = as.integer(get(var_target)))]
  )
  dt_tmp <- dt_tmp[!is.na(score) & !is.na(target)]
  dt_tmp <- dt_tmp[order(-score)]

  n            <- nrow(dt_tmp)
  taxa_global  <- dt_tmp[, mean(target)]
  n_total_pos  <- dt_tmp[, sum(target)]

  dt_tmp[, decil := ceiling(seq_len(n) / n * n_decis)]
  dt_tmp[decil == 0L, decil := 1L]

  result <- dt_tmp[, .(
    n           = .N,
    n_pos       = sum(target),
    score_min   = round(min(score),  4),
    score_max   = round(max(score),  4),
    score_medio = round(mean(score), 4)
  ), by = decil]

  result <- result[order(decil)]

  result[, taxa_evento  := round(n_pos / n, 4)]
  result[, lift         := round(taxa_evento / taxa_global, 3)]
  result[, captura      := round(n_pos / n_total_pos, 4)]
  result[, captura_acum := round(cumsum(n_pos) / n_total_pos, 4)]

  return(result)
}


# ─── curva_roc ───────────────────────────────────────────────────────────────

#' @title Pontos para a Curva ROC
#'
#' @description Gera uma tabela de pontos (FPR, TPR) para plotagem da curva
#' ROC, ordenando as observações por score decrescente e acumulando as taxas.
#'
#' @param dt         \code{data.table} com predições e target.
#' @param var_pred   Nome da coluna de scores preditos.
#' @param var_target Nome da coluna target binário (0/1).
#'
#' @return \code{data.table} com colunas \code{fpr} e \code{tpr}.
#'
#' @import data.table
#' @export
curva_roc <- function(dt, var_pred, var_target) {

  data.table::setDT(dt)

  scores <- as.numeric(dt[[var_pred]])
  labels <- as.integer(dt[[var_target]])

  n_pos <- sum(labels)
  n_neg <- length(labels) - n_pos

  if (n_pos == 0 || n_neg == 0)
    stop("Target deve ter observações positivas e negativas.")

  ord     <- order(scores, decreasing = TRUE)
  lab_ord <- labels[ord]

  tpr <- cumsum(lab_ord)       / n_pos
  fpr <- cumsum(1L - lab_ord)  / n_neg

  data.table::data.table(fpr = c(0, fpr), tpr = c(0, tpr))
}


# ─── helpers internos ────────────────────────────────────────────────────────

#' AUC via estatística de Wilcoxon (equivalência exata)
#' @noRd
.calc_auc <- function(scores, labels) {
  pos_scores <- scores[labels == 1]
  neg_scores <- scores[labels == 0]
  n_pos      <- length(pos_scores)
  n_neg      <- length(neg_scores)
  if (n_pos == 0L || n_neg == 0L) return(NA_real_)
  w <- stats::wilcox.test(pos_scores, neg_scores, exact = FALSE)$statistic
  as.numeric(w) / (n_pos * n_neg)
}

#' KS statistic entre distribuições de scores positivos e negativos
#' @noRd
.calc_ks <- function(scores, labels) {
  pos_scores <- scores[labels == 1]
  neg_scores <- scores[labels == 0]
  if (length(pos_scores) == 0L || length(neg_scores) == 0L) return(NA_real_)
  res <- stats::ks.test(pos_scores, neg_scores)
  unname(res$statistic)
}

#' Métricas de matriz de confusão a um determinado threshold
#' @noRd
.confusion_at_threshold <- function(scores, labels, threshold = 0.5) {
  pred_class <- as.integer(scores >= threshold)
  tp <- sum(pred_class == 1L & labels == 1L)
  fp <- sum(pred_class == 1L & labels == 0L)
  tn <- sum(pred_class == 0L & labels == 0L)
  fn <- sum(pred_class == 0L & labels == 1L)

  precision <- if ((tp + fp) > 0L) tp / (tp + fp) else 0
  recall    <- if ((tp + fn) > 0L) tp / (tp + fn) else 0
  f1        <- if ((precision + recall) > 0) 2 * precision * recall / (precision + recall) else 0
  accuracy  <- (tp + tn) / length(labels)

  list(tp = tp, fp = fp, tn = tn, fn = fn,
       precision = precision, recall = recall,
       f1 = f1, accuracy = accuracy)
}


# ─── metricas_por_cortes ─────────────────────────────────────────────────────

#' @title Métricas de classificação por múltiplos cortes
#'
#' @description Calcula métricas de classificação (precision, recall, F1,
#' accuracy, specificity, matriz de confusão) para cada threshold em
#' \code{cortes}. Útil para análise de sensibilidade do modelo.
#'
#' @param dt         \code{data.table} com predições e target.
#' @param var_pred   Nome da coluna de scores preditos (0–1).
#' @param var_target Nome da coluna target binário (0/1).
#' @param cortes     Vetor numérico de thresholds. DEFAULT:
#'                   \code{c(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)}.
#'
#' @return \code{data.table} com uma linha por corte e colunas:
#'   \code{corte}, \code{precision}, \code{recall}, \code{f1},
#'   \code{accuracy}, \code{specificity}, \code{tp}, \code{tn},
#'   \code{fp}, \code{fn}.
#'
#' @import data.table
#' @export
metricas_por_cortes <- function(dt, var_pred, var_target,
                                cortes = c(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) {

  data.table::setDT(dt)

  if (!var_pred %in% names(dt))
    stop(paste0("Coluna de predição '", var_pred, "' não encontrada."))
  if (!var_target %in% names(dt))
    stop(paste0("Coluna target '", var_target, "' não encontrada."))

  scores <- as.numeric(dt[[var_pred]])
  labels <- as.integer(dt[[var_target]])

  rows <- lapply(cortes, function(c) {
    cm <- .confusion_at_threshold(scores, labels, threshold = c)
    specificity <- if ((cm$tn + cm$fp) > 0L) cm$tn / (cm$tn + cm$fp) else 0
    data.table::data.table(
      corte       = c,
      precision   = round(cm$precision,   4),
      recall      = round(cm$recall,      4),
      f1          = round(cm$f1,          4),
      accuracy    = round(cm$accuracy,    4),
      specificity = round(specificity,    4),
      tp          = cm$tp,
      tn          = cm$tn,
      fp          = cm$fp,
      fn          = cm$fn
    )
  })

  data.table::rbindlist(rows)
}
