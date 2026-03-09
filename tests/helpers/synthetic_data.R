#' @title Dados sintéticos para testes
#'
#' @description Funções auxiliares que geram dados sintéticos reproduzíveis
#' para os testes do statMancer. Cada função cobre um cenário diferente.
#'
#' @section Funções disponíveis:
#' \describe{
#'   \item{\code{sintetico_binario}}{Classificação binária desbalanceada, múltiplas features.}
#'   \item{\code{sintetico_multiclasse}}{Classificação multiclasse balanceada.}
#'   \item{\code{sintetico_regressao}}{Target numérico contínuo.}
#'   \item{\code{sintetico_multiplos_registros}}{Múltiplos registros por ID (ex: sinistros).}
#' }


# ─── binário ─────────────────────────────────────────────────────────────────

#' Dados sintéticos para classificação binária desbalanceada
#'
#' @param n_pos  Tamanho da classe positiva. DEFAULT: 200.
#' @param n_neg  Tamanho da classe negativa. DEFAULT: 800.
#' @param seed   Semente. DEFAULT: 42.
#'
#' @return data.table com colunas: id, target, regiao, score_a, score_b,
#'   score_ruido, cat_ruido, idade.
#' @noRd
sintetico_binario <- function(n_pos = 200, n_neg = 800, seed = 42) {
  set.seed(seed)
  n <- n_pos + n_neg

  data.table::data.table(
    id         = seq_len(n),
    target     = c(rep(1L, n_pos), rep(0L, n_neg)),
    regiao     = sample(c("norte", "sul", "leste", "oeste"), n, replace = TRUE),
    score_a    = c(stats::rnorm(n_pos, mean = 75, sd = 10),
                   stats::rnorm(n_neg, mean = 50, sd = 15)),
    score_b    = c(stats::rnorm(n_pos, mean = 60, sd = 8),
                   stats::rnorm(n_neg, mean = 55, sd = 12)),
    score_ruido = stats::rnorm(n, mean = 50, sd = 20),
    cat_ruido  = sample(letters[1:5], n, replace = TRUE),
    idade      = sample(18L:80L, n, replace = TRUE)
  )
}


# ─── multiclasse ─────────────────────────────────────────────────────────────

#' Dados sintéticos para classificação multiclasse
#'
#' @param n_por_classe  Amostras por classe. DEFAULT: 300.
#' @param n_classes     Número de classes. DEFAULT: 3.
#' @param seed          Semente. DEFAULT: 42.
#'
#' @return data.table com colunas: id, target (0..n_classes-1), score_a,
#'   score_b, regiao.
#' @noRd
sintetico_multiclasse <- function(n_por_classe = 300, n_classes = 3, seed = 42) {
  set.seed(seed)

  blocos <- lapply(seq_len(n_classes) - 1L, function(cls) {
    data.table::data.table(
      target  = cls,
      score_a = stats::rnorm(n_por_classe, mean = cls * 20 + 40, sd = 10),
      score_b = stats::rnorm(n_por_classe, mean = 50 - cls * 5, sd = 12)
    )
  })

  dt <- data.table::rbindlist(blocos)
  dt[, id     := seq_len(.N)]
  dt[, regiao := sample(c("norte", "sul", "leste", "oeste"), .N, replace = TRUE)]

  data.table::setcolorder(dt, c("id", "target"))
  return(dt)
}


# ─── regressão ───────────────────────────────────────────────────────────────

#' Dados sintéticos para regressão
#'
#' @param n     Número de observações. DEFAULT: 1000.
#' @param seed  Semente. DEFAULT: 42.
#'
#' @return data.table com colunas: id, target (numérica), x1..x4, ruido.
#' @noRd
sintetico_regressao <- function(n = 1000, seed = 42) {
  set.seed(seed)

  x1 <- stats::rnorm(n)
  x2 <- stats::rnorm(n)
  x3 <- stats::rnorm(n)
  x4 <- stats::rnorm(n)

  data.table::data.table(
    id      = seq_len(n),
    target  = 3 * x1 - 2 * x2 + 0.5 * x3 + stats::rnorm(n, sd = 0.5),
    x1      = x1,
    x2      = x2,
    x3      = x3,
    x4      = x4,  # fraca relação
    ruido   = stats::rnorm(n)  # sem relação
  )
}


# ─── múltiplos registros por ID ──────────────────────────────────────────────

#' Dados sintéticos com múltiplos registros por ID
#'
#' Simula um cenário onde cada beneficiário (ID) tem vários eventos/sinistros,
#' mas um único target (ex: diagnóstico de doença).
#'
#' @param n_ids          Número de indivíduos únicos. DEFAULT: 300.
#' @param max_registros  Máximo de registros por indivíduo. DEFAULT: 5.
#' @param prop_positivos Proporção de IDs com target = 1. DEFAULT: 0.25.
#' @param seed           Semente. DEFAULT: 42.
#'
#' @return data.table com colunas: id, target, evento, valor, data_evento.
#' @noRd
sintetico_multiplos_registros <- function(n_ids          = 300,
                                          max_registros  = 5,
                                          prop_positivos = 0.25,
                                          seed           = 42) {
  set.seed(seed)

  n_pos <- round(n_ids * prop_positivos)
  n_neg <- n_ids - n_pos

  targets <- c(rep(1L, n_pos), rep(0L, n_neg))

  lista_ids <- lapply(seq_len(n_ids), function(i) {
    n_reg <- sample(1L:max_registros, 1L)
    data.table::data.table(
      id          = i,
      target      = targets[i],
      evento      = sample(c("consulta", "exame", "internacao"), n_reg, replace = TRUE),
      valor       = stats::runif(n_reg, min = 50, max = 5000),
      data_evento = Sys.Date() - sample(1L:365L, n_reg, replace = TRUE)
    )
  })

  data.table::rbindlist(lista_ids)
}


# ─── binário com categóricas ──────────────────────────────────────────────────

#' Dados sintéticos para classificação binária com variáveis categóricas
#'
#' Inclui variáveis \code{factor} e \code{character} com sinal real em relação
#' ao target, além das variáveis numéricas usuais.
#'
#' @param n_pos  Tamanho da classe positiva. DEFAULT: 200.
#' @param n_neg  Tamanho da classe negativa. DEFAULT: 800.
#' @param seed   Semente. DEFAULT: 42.
#'
#' @return data.table com colunas:
#'   \code{id}, \code{target},
#'   \code{sexo} (factor, COM sinal),
#'   \code{canal} (factor, COM sinal),
#'   \code{faixa_risco} (character, COM sinal),
#'   \code{regiao} (factor, sem sinal),
#'   \code{score_a} (numérica, COM sinal),
#'   \code{score_b} (numérica, sinal fraco),
#'   \code{score_ruido} (numérica, sem sinal),
#'   \code{idade} (inteira, sinal leve).
#' @noRd
sintetico_binario_categorico <- function(n_pos = 200, n_neg = 800, seed = 42) {
  set.seed(seed)
  n <- n_pos + n_neg

  data.table::data.table(
    id     = seq_len(n),
    target = c(rep(1L, n_pos), rep(0L, n_neg)),

    # ── Categóricas COM sinal (factor) ────────────────────────────────────
    # Positivos: maioria "M"; negativos: maioria "F"
    sexo = factor(c(
      sample(c("M", "F"), n_pos, replace = TRUE, prob = c(0.70, 0.30)),
      sample(c("M", "F"), n_neg, replace = TRUE, prob = c(0.35, 0.65))
    )),

    # Positivos: preferem "app"/"online"; negativos: preferem "loja"/"telefone"
    canal = factor(c(
      sample(c("app", "online", "loja", "telefone"), n_pos,
             replace = TRUE, prob = c(0.50, 0.30, 0.10, 0.10)),
      sample(c("app", "online", "loja", "telefone"), n_neg,
             replace = TRUE, prob = c(0.15, 0.20, 0.35, 0.30))
    )),

    # Character (não factor), COM sinal
    # Positivos: maioria "alto" e "medio"; negativos: maioria "baixo"
    faixa_risco = c(
      sample(c("alto", "medio", "baixo"), n_pos,
             replace = TRUE, prob = c(0.60, 0.30, 0.10)),
      sample(c("alto", "medio", "baixo"), n_neg,
             replace = TRUE, prob = c(0.15, 0.35, 0.50))
    ),

    # ── Categórica SEM sinal (factor) ─────────────────────────────────────
    regiao = factor(
      sample(c("norte", "sul", "leste", "oeste"), n, replace = TRUE)
    ),

    # ── Numéricas ─────────────────────────────────────────────────────────
    score_a     = c(stats::rnorm(n_pos, mean = 75, sd = 10),
                    stats::rnorm(n_neg, mean = 50, sd = 15)),
    score_b     = c(stats::rnorm(n_pos, mean = 60, sd = 8),
                    stats::rnorm(n_neg, mean = 55, sd = 12)),
    score_ruido = stats::rnorm(n, mean = 50, sd = 20),
    idade       = sample(18L:80L, n, replace = TRUE)
  )
}

