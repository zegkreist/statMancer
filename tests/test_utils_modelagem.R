# ============================================================================
# test_utils_modelagem.R
# Testes TDD para as funções utilitárias de R/utils/utils_modelagem.R
#
# Cobre:
#   - rnorm_t
#   - converter_para_numerico_seguro
#   - criar_cv_folds
#   - calculate_auc
#   - gerar_parametros_estocasticos
#   - xgb_cross_validation
# ============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(xgboost)
})

# ── Path resolution (RStudio interativo ou Rscript) ──────────────────────────
.proj_root <- tryCatch(
  dirname(dirname(rstudioapi::getSourceEditorContext()$path)),
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    f    <- args[grepl("--file=", args)]
    if (length(f) > 0) dirname(dirname(sub("--file=", "", f)))
    else getwd()
  }
)

source(file.path(.proj_root, "R", "utils", "utils_modelagem.R"))
source(file.path(.proj_root, "tests", "helpers", "synthetic_data.R"))

.assert <- function(condicao, msg) {
  if (!condicao) stop(paste0("[FAIL] ", msg))
  cat(paste0("  [OK] ", msg, "\n"))
}

cat("\n======= TESTES: utils_modelagem =======\n\n")


# ── rnorm_t ──────────────────────────────────────────────────────────────────
cat("--- rnorm_t() ---\n")

# T1: retorna vetor de tamanho n
test_rnorm_t_comprimento <- function() {
  r <- rnorm_t(n = 100, mu = 0, sd = 1)
  .assert(length(r) == 100, "Deve retornar n = 100 valores")
  cat("PASS: test_rnorm_t_comprimento\n\n")
}

# T2: respeita limite superior
test_rnorm_t_limite_superior <- function() {
  r <- rnorm_t(n = 2000, mu = 0, sd = 10, upper = 2)
  .assert(all(r <= 2), "Todos os valores devem ser <= upper = 2")
  cat("PASS: test_rnorm_t_limite_superior\n\n")
}

# T3: respeita limite inferior
test_rnorm_t_limite_inferior <- function() {
  r <- rnorm_t(n = 2000, mu = 0, sd = 10, lower = -1)
  .assert(all(r >= -1), "Todos os valores devem ser >= lower = -1")
  cat("PASS: test_rnorm_t_limite_inferior\n\n")
}

# T4: ambos os limites ao mesmo tempo
test_rnorm_t_ambos_limites <- function() {
  r <- rnorm_t(n = 2000, mu = 0.5, sd = 5, lower = 0.1, upper = 0.9)
  .assert(all(r >= 0.1 & r <= 0.9), "Valores devem estar no intervalo [0.1, 0.9]")
  cat("PASS: test_rnorm_t_ambos_limites\n\n")
}

test_rnorm_t_comprimento()
test_rnorm_t_limite_superior()
test_rnorm_t_limite_inferior()
test_rnorm_t_ambos_limites()


# ── converter_para_numerico_seguro ────────────────────────────────────────────
cat("--- converter_para_numerico_seguro() ---\n")

# T5: colunas já numéricas passam sem alteração de valor
test_converter_numerico_passthrough <- function() {
  dt  <- data.table(a = 1:5, b = c(1.1, 2.2, 3.3, 4.4, 5.5))
  res <- converter_para_numerico_seguro(dt)
  .assert(is.numeric(res$a),             "Coluna inteira deve permanecer numérica")
  .assert(is.numeric(res$b),             "Coluna double deve permanecer numérica")
  .assert(all(res$a == dt$a),            "Valores de 'a' devem ser inalterados")
  cat("PASS: test_converter_numerico_passthrough\n\n")
}

# T6: character numérico é convertido
test_converter_character_numerico <- function() {
  dt  <- data.table(x = c("1", "2", "3", "4"))
  res <- converter_para_numerico_seguro(dt)
  .assert(is.numeric(res$x),  "Character numérico deve ser convertido para numeric")
  .assert(res$x[1] == 1,      "Valor '1' deve virar 1")
  cat("PASS: test_converter_character_numerico\n\n")
}

# T7: factor é convertido para inteiro numérico
test_converter_factor_numerico <- function() {
  dt  <- data.table(cat = factor(c("A", "B", "A", "C", "B")))
  res <- converter_para_numerico_seguro(dt)
  .assert(is.numeric(res$cat),            "Factor deve ser convertido para numérico")
  .assert(length(unique(res$cat)) == 3L,  "Deve preservar 3 categorias distintas")
  cat("PASS: test_converter_factor_numerico\n\n")
}

# T8: não altera o data.table original (cópia)
test_converter_nao_altera_original <- function() {
  dt  <- data.table(x = c("1", "2", "3"))
  res <- converter_para_numerico_seguro(dt)
  .assert(is.character(dt$x),  "Original não deve ser alterado (deve fazer copy)")
  cat("PASS: test_converter_nao_altera_original\n\n")
}

test_converter_numerico_passthrough()
test_converter_character_numerico()
test_converter_factor_numerico()
test_converter_nao_altera_original()


# ── criar_cv_folds ────────────────────────────────────────────────────────────
cat("--- criar_cv_folds() ---\n")

.dt_folds <- sintetico_binario(n_pos = 200, n_neg = 800, seed = 1)

# T9: retorna exatamente k folds
test_cv_folds_retorna_k <- function() {
  folds <- criar_cv_folds(.dt_folds, k_folds = 5, seed = 42)
  .assert(length(folds) == 5L, "Deve retornar exatamente 5 folds")
  cat("PASS: test_cv_folds_retorna_k\n\n")
}

# T10: união dos folds cobre todas as linhas
test_cv_folds_cobertura_total <- function() {
  folds   <- criar_cv_folds(.dt_folds, k_folds = 5, seed = 42)
  all_idx <- sort(unlist(folds))
  .assert(identical(all_idx, 1:nrow(.dt_folds)), "União dos folds deve cobrir todas as linhas")
  cat("PASS: test_cv_folds_cobertura_total\n\n")
}

# T11: nenhuma linha aparece em mais de um fold
test_cv_folds_sem_sobreposicao <- function() {
  folds   <- criar_cv_folds(.dt_folds, k_folds = 5, seed = 42)
  all_idx <- unlist(folds)
  .assert(length(all_idx) == length(unique(all_idx)), "Nenhuma linha deve estar em mais de um fold")
  cat("PASS: test_cv_folds_sem_sobreposicao\n\n")
}

# T12: estratificação binária — cada fold contém ambas as classes
test_cv_folds_estratificacao_binaria <- function() {
  folds <- criar_cv_folds(.dt_folds, k_folds = 5, seed = 42)
  for (i in seq_along(folds)) {
    classes_fold <- unique(.dt_folds[folds[[i]], target])
    .assert(length(classes_fold) == 2L, paste0("Fold ", i, " deve conter ambas as classes"))
  }
  cat("PASS: test_cv_folds_estratificacao_binaria\n\n")
}

# T13: seed garante reprodutibilidade
test_cv_folds_reproducibilidade <- function() {
  f1 <- criar_cv_folds(.dt_folds, k_folds = 5, seed = 99)
  f2 <- criar_cv_folds(.dt_folds, k_folds = 5, seed = 99)
  .assert(identical(f1, f2), "Mesma seed deve gerar folds idênticos")
  cat("PASS: test_cv_folds_reproducibilidade\n\n")
}

test_cv_folds_retorna_k()
test_cv_folds_cobertura_total()
test_cv_folds_sem_sobreposicao()
test_cv_folds_estratificacao_binaria()
test_cv_folds_reproducibilidade()


# ── calculate_auc ─────────────────────────────────────────────────────────────
cat("--- calculate_auc() ---\n")

# T14: classificador perfeito → AUC = 1
test_auc_classificador_perfeito <- function() {
  target <- c(rep(1L, 100), rep(0L, 100))
  pred   <- c(rep(0.99, 100), rep(0.01, 100))
  auc    <- calculate_auc(target, pred)
  .assert(abs(auc - 1.0) < 0.01, "Classificador perfeito deve ter AUC ≈ 1.0")
  cat("PASS: test_auc_classificador_perfeito\n\n")
}

# T15: classificador invertido → AUC ≈ 0
test_auc_classificador_invertido <- function() {
  target <- c(rep(1L, 100), rep(0L, 100))
  pred   <- c(rep(0.01, 100), rep(0.99, 100))
  auc    <- calculate_auc(target, pred)
  .assert(auc < 0.1, "Classificador invertido deve ter AUC próximo de 0")
  cat("PASS: test_auc_classificador_invertido\n\n")
}

# T16: classificador aleatório → AUC ≈ 0.5
test_auc_aleatorio <- function() {
  set.seed(7)
  target <- rbinom(1000L, 1L, 0.5)
  pred   <- runif(1000L)
  auc    <- calculate_auc(target, pred)
  .assert(auc > 0.35 && auc < 0.65, "Classificador aleatório deve ter AUC ≈ 0.5")
  cat("PASS: test_auc_aleatorio\n\n")
}

# T17: retorna escalar numérico
test_auc_retorna_escalar <- function() {
  auc <- calculate_auc(c(1L, 0L, 1L, 0L), c(0.9, 0.1, 0.8, 0.2))
  .assert(length(auc) == 1L && is.numeric(auc), "AUC deve ser um escalar numérico")
  cat("PASS: test_auc_retorna_escalar\n\n")
}

test_auc_classificador_perfeito()
test_auc_classificador_invertido()
test_auc_aleatorio()
test_auc_retorna_escalar()


# ── gerar_parametros_estocasticos ─────────────────────────────────────────────
cat("--- gerar_parametros_estocasticos() ---\n")

.params_base <- list(
  objective        = "binary:logistic",
  eval_metric      = "auc",
  eta              = 0.1,
  max_leaves       = 64L,
  gamma            = 0.5,
  subsample        = 0.8,
  colsample_bytree = 0.7,
  min_child_weight = 2,
  grow_policy      = "lossguide",
  tree_method      = "hist"
)

# T18: retorna uma lista
test_gerar_params_retorna_lista <- function() {
  res <- gerar_parametros_estocasticos(.params_base)
  .assert(is.list(res), "Deve retornar uma lista")
  cat("PASS: test_gerar_params_retorna_lista\n\n")
}

# T19: campos fixos (objective, eval_metric) são preservados intactos
test_gerar_params_campos_fixos_preservados <- function() {
  res <- gerar_parametros_estocasticos(.params_base)
  .assert(res$objective   == "binary:logistic", "objective deve ser preservado")
  .assert(res$eval_metric == "auc",             "eval_metric deve ser preservado")
  cat("PASS: test_gerar_params_campos_fixos_preservados\n\n")
}

# T20: eta sempre no intervalo válido do XGBoost [0.001, 1]
test_gerar_params_eta_em_range <- function() {
  set.seed(42)
  etas <- replicate(100L, gerar_parametros_estocasticos(.params_base)$eta)
  .assert(all(etas >= 0.001 & etas <= 1.0), "eta deve estar sempre no intervalo [0.001, 1.0]")
  cat("PASS: test_gerar_params_eta_em_range\n\n")
}

# T21: subsample sempre no intervalo válido [0.1, 1]
test_gerar_params_subsample_em_range <- function() {
  set.seed(42)
  subs <- replicate(100L, gerar_parametros_estocasticos(.params_base)$subsample)
  .assert(all(subs >= 0.1 & subs <= 1.0), "subsample deve estar sempre no intervalo [0.1, 1.0]")
  cat("PASS: test_gerar_params_subsample_em_range\n\n")
}

# T22: gamma nunca negativo  
test_gerar_params_gamma_nao_negativo <- function() {
  set.seed(42)
  gammas <- replicate(100L, gerar_parametros_estocasticos(.params_base)$gamma)
  .assert(all(gammas >= 0), "gamma nunca deve ser negativo")
  cat("PASS: test_gerar_params_gamma_nao_negativo\n\n")
}

test_gerar_params_retorna_lista()
test_gerar_params_campos_fixos_preservados()
test_gerar_params_eta_em_range()
test_gerar_params_subsample_em_range()
test_gerar_params_gamma_nao_negativo()


# ── xgb_cross_validation ──────────────────────────────────────────────────────
cat("--- xgb_cross_validation() ---\n")

# Dados pequenos com sinal real para CV rápido
# NOTA: xgb_cross_validation usa "target" como nome da coluna alvo (hardcoded)
.dt_cv <- {
  set.seed(13)
  n      <- 300L
  x1     <- rnorm(n)
  x2     <- rnorm(n)
  target <- as.integer(x1 - x2 + rnorm(n, sd = 0.4) > 0)
  data.table(target = target, x1 = x1, x2 = x2, ruido = rnorm(n))
}

.params_cv <- list(
  parametros = list(
    objective        = "binary:logistic",
    eval_metric      = "auc",
    eta              = 0.15,
    max_depth        = 4L,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    grow_policy      = "lossguide",
    tree_method      = "hist"
  ),
  nrounds = 20L
)

# T23: retorna k resultados (um por fold)
test_xgb_cv_retorna_k_folds <- function() {
  res <- xgb_cross_validation(.dt_cv, .params_cv,
                               k_folds = 3L, nthreads = 1L,
                               early_stopping_rounds = 5L)
  .assert(length(res) == 3L, "Deve retornar 3 resultados de fold")
  cat("PASS: test_xgb_cv_retorna_k_folds\n\n")
}

# T24: estrutura de cada fold contém auc_train, auc_test e n_train
test_xgb_cv_estrutura_resultado <- function() {
  res  <- xgb_cross_validation(.dt_cv, .params_cv,
                                k_folds = 3L, nthreads = 1L,
                                early_stopping_rounds = 5L)
  fold <- res[[1]]
  .assert("auc_train" %in% names(fold), "Fold deve conter 'auc_train'")
  .assert("auc_test"  %in% names(fold), "Fold deve conter 'auc_test'")
  .assert("n_train"   %in% names(fold), "Fold deve conter 'n_train'")
  cat("PASS: test_xgb_cv_estrutura_resultado\n\n")
}

# T25: AUC de teste sempre entre 0 e 1
test_xgb_cv_auc_em_range <- function() {
  res  <- xgb_cross_validation(.dt_cv, .params_cv,
                                k_folds = 3L, nthreads = 1L,
                                early_stopping_rounds = 5L)
  aucs <- sapply(res, function(x) x$auc_test)
  .assert(all(aucs >= 0 & aucs <= 1), "AUC test de cada fold deve estar entre 0 e 1")
  cat("PASS: test_xgb_cv_auc_em_range\n\n")
}

# T26: AUC médio > 0.6 em dados com sinal real
test_xgb_cv_treina_melhor_que_chance <- function() {
  res      <- xgb_cross_validation(.dt_cv, .params_cv,
                                   k_folds = 3L, nthreads = 1L,
                                   early_stopping_rounds = 5L)
  auc_med  <- mean(sapply(res, function(x) x$auc_test))
  .assert(auc_med > 0.6, paste0("AUC médio deve superar 0.60 em dados com sinal (obtido: ",
                                round(auc_med, 3), ")"))
  cat("PASS: test_xgb_cv_treina_melhor_que_chance\n\n")
}

test_xgb_cv_retorna_k_folds()
test_xgb_cv_estrutura_resultado()
test_xgb_cv_auc_em_range()
test_xgb_cv_treina_melhor_que_chance()


cat("\n======= TODOS OS TESTES PASSARAM: utils_modelagem =======\n")
