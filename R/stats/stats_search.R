#' @title Busca estatística de variáveis relacionadas ao target
#'
#' @description Executa automaticamente testes estatísticos para ranquear
#' variáveis candidatas por relevância em relação ao target.
#'
#' A estratégia de teste é determinada pela combinação do tipo da variável
#' candidata e do tipo do target:
#'
#' \tabular{lll}{
#'   \strong{Variável}   \tab \strong{Target}      \tab \strong{Teste}              \cr
#'   Numérica            \tab Classificação         \tab KS (binário) / KS par-a-par \cr
#'   Categórica          \tab Classificação         \tab Chi-quadrado + V de Cramér  \cr
#'   Numérica            \tab Regressão             \tab Pearson ou Spearman         \cr
#'   Categórica          \tab Regressão             \tab ANOVA + Eta²                \cr
#' }
#'
#' @section Funções disponíveis:
#' \describe{
#'   \item{\code{stats_search}}{Busca completa de variáveis relevantes.}
#' }


# ─── stats_search ────────────────────────────────────────────────────────────

#' @title Busca estatística de variáveis relacionadas ao target
#'
#' @description Testa cada variável candidata contra o target e retorna um
#' \code{data.table} ranqueado por relevância, com p-valor e tamanho de efeito.
#'
#' @param dt              Um \code{data.table} com os dados.
#' @param var_target      Nome da coluna target (character).
#' @param vars_excluir    Vetor de nomes de colunas a excluir da busca
#'                        (ex: var_id, datas). DEFAULT: \code{NULL}.
#' @param tipo_target     \code{"auto"}, \code{"classificacao"} ou
#'                        \code{"regressao"}. Com \code{"auto"}, detecta
#'                        pelo número de valores únicos. DEFAULT: \code{"auto"}.
#' @param alpha           Nível de significância para marcar significativas.
#'                        DEFAULT: \code{0.05}.
#' @param max_categorias  Limite de valores únicos para tratar variável como
#'                        categórica. DEFAULT: \code{20}.
#' @param min_obs         Mínimo de observações válidas para testar uma variável.
#'                        DEFAULT: \code{30}.
#'
#' @return Um \code{data.table} com as colunas:
#'   \itemize{
#'     \item \code{variavel}: nome da coluna testada.
#'     \item \code{tipo_variavel}: \code{"numerica"} ou \code{"categorica"}.
#'     \item \code{teste}: nome do teste aplicado.
#'     \item \code{estatistica}: valor da estatística (D de KS, V de Cramér, r, F, ηˆ2).
#'     \item \code{p_valor}: p-valor do teste.
#'     \item \code{relevancia}: tamanho de efeito padronizado (0–1).
#'     \item \code{significativa}: \code{TRUE} se p_valor < alpha.
#'   }
#'   Ordenado por \code{p_valor} crescente (mais relevantes primeiro).
#'
#' @import data.table
#' @export
stats_search <- function(dt,
                         var_target,
                         vars_excluir   = NULL,
                         tipo_target    = "auto",
                         alpha          = 0.05,
                         max_categorias = 20,
                         min_obs        = 30) {

  data.table::setDT(dt)
  dt <- data.table::copy(dt)

  if (!var_target %in% names(dt))
    stop(paste0("Coluna target '", var_target, "' não encontrada nos dados."))

  # ── Detectar tipo do target ──────────────────────────────────────────────
  target_vals <- dt[[var_target]]

  if (tipo_target == "auto") {
    n_unicos    <- length(unique(target_vals[!is.na(target_vals)]))
    tipo_target <- if (n_unicos <= max_categorias ||
                       is.character(target_vals)   ||
                       is.factor(target_vals)) {
      "classificacao"
    } else {
      "regressao"
    }
    message(paste0(
      "[stats_search] Target detectado como: ", tipo_target,
      " (", n_unicos, " valores únicos)."
    ))
  }

  # ── Variáveis candidatas ─────────────────────────────────────────────────
  vars_candidatas <- setdiff(names(dt), c(var_target, vars_excluir))

  if (length(vars_candidatas) == 0)
    stop("Nenhuma variável candidata disponível após exclusões.")

  # ── Testar cada variável ─────────────────────────────────────────────────
  resultados <- data.table::rbindlist(
    lapply(vars_candidatas, function(var) {
      tryCatch(
        .testar_variavel(dt, var, var_target, tipo_target, max_categorias, min_obs),
        error = function(e) {
          data.table::data.table(
            variavel      = var,
            tipo_variavel = NA_character_,
            teste         = paste0("erro: ", conditionMessage(e)),
            estatistica   = NA_real_,
            p_valor       = NA_real_,
            relevancia    = NA_real_
          )
        }
      )
    }),
    use.names = TRUE, fill = TRUE
  )

  # ── Pós-processamento ────────────────────────────────────────────────────
  resultados <- resultados[!is.na(p_valor)]
  resultados <- resultados[order(p_valor)]
  resultados[, significativa := p_valor < alpha]

  return(resultados)
}


# ─── helper: testa uma variável ──────────────────────────────────────────────

#' @noRd
.testar_variavel <- function(dt, var, var_target, tipo_target,
                              max_categorias, min_obs) {

  x <- dt[[var]]
  y <- dt[[var_target]]

  # Pares válidos
  idx_ok <- !is.na(x) & !is.na(y)
  x <- x[idx_ok]
  y <- y[idx_ok]

  if (length(x) < min_obs)
    return(data.table::data.table(
      variavel = var, tipo_variavel = NA_character_,
      teste = "obs_insuficientes", estatistica = NA_real_,
      p_valor = NA_real_, relevancia = NA_real_
    ))

  n_unicos  <- length(unique(x))
  tipo_var  <- if (is.numeric(x) && n_unicos > max_categorias) "numerica" else "categorica"

  if (tipo_target == "classificacao") {
    if (tipo_var == "numerica") {
      res <- .ks_vs_classes(x, y)
    } else {
      res <- .chisq_cramer(x, y)
    }
  } else {  # regressao
    if (tipo_var == "numerica") {
      res <- .correlacao(x, suppressWarnings(as.numeric(y)))
    } else {
      res <- .anova_eta(x, suppressWarnings(as.numeric(y)))
    }
  }

  data.table::data.table(
    variavel      = var,
    tipo_variavel = tipo_var,
    teste         = res$teste,
    estatistica   = as.numeric(res$estatistica),
    p_valor       = as.numeric(res$p_valor),
    relevancia    = as.numeric(res$relevancia)
  )
}


# ─── helpers de teste ────────────────────────────────────────────────────────

#' KS test: variável numérica vs target de classificação
#' @noRd
.ks_vs_classes <- function(x, y) {
  classes <- unique(y)

  if (length(classes) == 2) {
    res  <- stats::ks.test(x[y == classes[1]], x[y == classes[2]])
    return(list(
      teste       = "KS",
      estatistica = unname(res$statistic),
      p_valor     = res$p.value,
      relevancia  = unname(res$statistic)   # D de KS: 0–1
    ))
  }

  # Multiclasse: menor p-valor entre todos os pares
  pares   <- utils::combn(classes, 2, simplify = FALSE)
  ks_list <- lapply(pares, function(par) {
    stats::ks.test(x[y == par[[1]]], x[y == par[[2]]])
  })
  d_max  <- max(sapply(ks_list, function(r) unname(r$statistic)))
  p_min  <- min(sapply(ks_list, function(r) r$p.value))

  list(
    teste       = "KS_multiclasse",
    estatistica = d_max,
    p_valor     = p_min,
    relevancia  = d_max
  )
}

#' Chi-quadrado + V de Cramér: variável categórica vs target de classificação
#' @noRd
.chisq_cramer <- function(x, y) {
  tabela  <- table(x, y)
  chi_res <- suppressWarnings(stats::chisq.test(tabela))
  n       <- sum(tabela)
  k       <- min(nrow(tabela), ncol(tabela))
  v       <- sqrt(pmax(0, chi_res$statistic - (k - 1)) / (n * (k - 1)))

  list(
    teste       = "ChiSq_CramerV",
    estatistica = unname(v),
    p_valor     = chi_res$p.value,
    relevancia  = unname(v)
  )
}

#' Pearson / Spearman: variável numérica vs target numérico
#' @noRd
.correlacao <- function(x, y) {
  cor_p  <- tryCatch(stats::cor.test(x, y, method = "pearson"),  error = function(e) NULL)
  cor_s  <- tryCatch(
    stats::cor.test(x, y, method = "spearman", exact = FALSE),
    error = function(e) NULL
  )

  p_p <- if (!is.null(cor_p)) cor_p$p.value else 1
  p_s <- if (!is.null(cor_s)) cor_s$p.value else 1

  if (p_p <= p_s && !is.null(cor_p)) {
    list(
      teste       = "Pearson",
      estatistica = unname(cor_p$estimate),
      p_valor     = p_p,
      relevancia  = abs(unname(cor_p$estimate))
    )
  } else {
    list(
      teste       = "Spearman",
      estatistica = unname(cor_s$estimate),
      p_valor     = p_s,
      relevancia  = abs(unname(cor_s$estimate))
    )
  }
}

#' ANOVA + Eta²: variável categórica vs target numérico
#' @noRd
.anova_eta <- function(x, y) {
  df_tmp   <- data.frame(x = as.factor(x), y = y)
  aov_res  <- stats::aov(y ~ x, data = df_tmp)
  summ     <- summary(aov_res)[[1]]
  f_stat   <- summ$`F value`[1]
  p_val    <- summ$`Pr(>F)`[1]
  ss       <- summ$`Sum Sq`
  eta_sq   <- ss[1] / sum(ss)

  list(
    teste       = "ANOVA_Eta2",
    estatistica = f_stat,
    p_valor     = p_val,
    relevancia  = eta_sq
  )
}
