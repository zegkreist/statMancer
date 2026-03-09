# ============================================================================
# exemplo_completo.R
# Pipeline completo: Dados Sintéticos → Busca → Split → Modelo → Relatório
#
# Como executar:
#   No RStudio: abra este arquivo e pressione Source (Ctrl+Shift+S)
#   No terminal: Rscript exemplo_completo.R  (dentro da pasta statMancer/)
# ============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(xgboost)
})

# ── 0. Caminhos ──────────────────────────────────────────────────────────────

# Detecta a raiz do projeto (funciona via RStudio Interactive e Rscript)
root_dir <- tryCatch(
  dirname(rstudioapi::getSourceEditorContext()$path),
  error = function(e) {
    args     <- commandArgs(trailingOnly = FALSE)
    file_arg <- args[grepl("--file=", args)]
    if (length(file_arg) > 0) dirname(normalizePath(sub("--file=", "", file_arg)))
    else getwd()
  }
)

src <- function(...) file.path(root_dir, ...)

source(src("R/sampling/sampling_balance.R"))
source(src("R/sampling/sampling_split.R"))
source(src("R/stats/stats_search.R"))
source(src("R/modeling/modeling_xgb_train.R"))
source(src("R/modeling/modeling_xgb_predict.R"))
source(src("R/reporting/reporting_metrics.R"))
source(src("R/reporting/reporting_render.R"))

# Diretório de saída
output_dir <- src("output")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

cat("\n")
cat("╔══════════════════════════════════════════════════════╗\n")
cat("║       statMancer — Pipeline de Exemplo Completo     ║\n")
cat("╚══════════════════════════════════════════════════════╝\n\n")


# ── 1. Dados Sintéticos ───────────────────────────────────────────────────────
cat("── 1. Gerando dados sintéticos ────────────────────────\n")
set.seed(2025)

N_TOTAL  <- 2000L
N_POS    <- 400L   # 20% de positivos (desbalanceado)
N_NEG    <- N_TOTAL - N_POS

dados <- data.table(
  id       = seq_len(N_TOTAL),
  target   = c(rep(1L, N_POS), rep(0L, N_NEG)),
  regiao   = sample(c("norte", "sul", "leste", "oeste"), N_TOTAL, replace = TRUE),
  # Features com sinal real (médias distintas entre classes)
  score_a  = c(rnorm(N_POS, mean = 72, sd = 10), rnorm(N_NEG, mean = 48, sd = 15)),
  score_b  = c(rnorm(N_POS, mean = 62, sd = 8),  rnorm(N_NEG, mean = 55, sd = 12)),
  score_c  = c(rnorm(N_POS, mean = 58, sd = 9),  rnorm(N_NEG, mean = 52, sd = 11)),
  # Features sem sinal (ruído)
  ruido_1  = rnorm(N_TOTAL),
  ruido_2  = rnorm(N_TOTAL),
  # Feature de idade (sinal leve)
  idade    = c(sample(40L:70L, N_POS, replace = TRUE),
               sample(20L:65L, N_NEG, replace = TRUE))
)

cat(sprintf("   Total: %s obs | Positivos: %s (%.0f%%) | Negativos: %s\n\n",
            format(N_TOTAL, big.mark = "."),
            format(N_POS,   big.mark = "."), 100 * N_POS / N_TOTAL,
            format(N_NEG,   big.mark = ".")))


# ── 2. Busca Estatística de Variáveis ────────────────────────────────────────
cat("── 2. Busca estatística de variáveis ──────────────────\n")

busca <- suppressMessages(
  stats_search(dados,
               var_target   = "target",
               vars_excluir = c("id", "regiao"),   # excluir ID; regiao é categórica
               tipo_target  = "classificacao")
)

cat(sprintf("   %d variáveis testadas\n", nrow(busca)))
cat(sprintf("   Top 3 mais relevantes: %s\n",
            paste(busca[1:3, variavel], collapse = ", ")))
cat(sprintf("   Significativas (p < 0.05): %d/%d\n\n",
            busca[, sum(significativa)], nrow(busca)))


# ── 3. Divisão Treino / Teste ────────────────────────────────────────────────
cat("── 3. Dividindo em treino (80%) e teste (20%) ─────────\n")

split <- train_test_split(
  dados,
  var_id             = "id",
  prop_treino        = 0.80,
  var_estratificacao = "target",
  seed               = 42L
)

cat(sprintf("   Treino: %s registros | Taxa evento: %.1f%%\n",
            format(nrow(split$treino), big.mark = "."),
            100 * split$treino[, mean(target)]))
cat(sprintf("   Teste:  %s registros | Taxa evento: %.1f%%\n\n",
            format(nrow(split$teste), big.mark = "."),
            100 * split$teste[, mean(target)]))

# ── 4. Balanceamento do Treino ───────────────────────────────────────────────
cat("── 4. Upsample da classe minoritária no treino ────────\n")

treino_bal <- upsample(
  split$treino,
  var_id       = "id",
  var_target   = "target",
  n_por_classe = 600L,
  seed         = 42L
)

cat(sprintf("   Treino balanceado: %s obs | Positivos: %.0f%% | Negativos: %.0f%%\n\n",
            format(nrow(treino_bal), big.mark = "."),
            100 * treino_bal[, mean(target == 1)],
            100 * treino_bal[, mean(target == 0)]))


# ── 5. Treino do Modelo XGBoost ──────────────────────────────────────────────
cat("── 5. Treinando modelo XGBoost ────────────────────────\n")

params_xgb <- list(
  objective        = "binary:logistic",
  eval_metric      = "auc",
  eta              = 0.05,
  max_leaves       = 16L,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 5,
  gamma            = 1,
  grow_policy      = "lossguide",
  tree_method      = "hist"
)

modelo <- xgb_train(
  dt_treino    = treino_bal,
  var_target   = "target",
  vars_excluir = c("id", "regiao"),    # regiao precisa de encoding antes de entrar
  params       = params_xgb,
  nrounds      = 150L,
  verbose      = FALSE
)

cat(sprintf("   Features: %d | Rounds: %d\n",
            length(modelo$features), modelo$nrounds))
cat(sprintf("   Features utilizadas: %s\n\n",
            paste(modelo$features, collapse = ", ")))


# ── 6. Predição no Teste ─────────────────────────────────────────────────────
cat("── 6. Predizendo no conjunto de teste ─────────────────\n")

predicoes_raw <- xgb_predict(modelo, split$teste, var_id = "id")

# Juntar predições com o target real
dt_avaliacao <- merge(
  split$teste[, .(id, target)],
  predicoes_raw,
  by = "id"
)

cat(sprintf("   Score médio geral: %.3f\n",   dt_avaliacao[, mean(predito)]))
cat(sprintf("   Score médio positivos: %.3f\n", dt_avaliacao[target == 1, mean(predito)]))
cat(sprintf("   Score médio negativos: %.3f\n\n", dt_avaliacao[target == 0, mean(predito)]))


# ── 7. Métricas de Avaliação ─────────────────────────────────────────────────
cat("── 7. Métricas de avaliação ───────────────────────────\n")

metricas <- metricas_binario(dt_avaliacao,
                             var_pred   = "predito",
                             var_target = "target")

cat(sprintf("   AUC       : %.4f\n",  metricas$auc))
cat(sprintf("   KS        : %.4f\n",  metricas$ks))
cat(sprintf("   Gini      : %.4f\n",  metricas$gini))
cat(sprintf("   Precision : %.4f\n",  metricas$precision))
cat(sprintf("   Recall    : %.4f\n",  metricas$recall))
cat(sprintf("   F1        : %.4f\n",  metricas$f1))
cat(sprintf("   Accuracy  : %.4f\n\n", metricas$accuracy))

# Tabela de decis (preview)
decis <- tabela_decis(dt_avaliacao, var_pred = "predito", var_target = "target")
cat("   Lift por decil: ")
cat(paste(sprintf("D%d=%.2f", decis$decil, decis$lift), collapse = " | "))
cat("\n\n")


# ── 8. K-Fold Cross-Validation (demonstração) ────────────────────────────────
cat("── 8. K-Fold (5 folds) — AUC por fold ────────────────\n")

folds <- kfold_split(dados, var_id = "id", k = 5L,
                     var_estratificacao = "target", seed = 42L)

aucs_cv <- sapply(folds, function(fold) {
  # Treinar no treino do fold (sem balanceamento para ser rápido)
  m_fold <- xgb_train(
    dt_treino    = fold$treino,
    var_target   = "target",
    vars_excluir = c("id", "regiao"),
    params       = params_xgb,
    nrounds      = 50L,
    verbose      = FALSE
  )
  # Prever no teste do fold
  preds_fold <- xgb_predict(m_fold, fold$teste)
  dt_fold    <- cbind(fold$teste[, .(target)], preds_fold)
  metricas_binario(dt_fold, "predito", "target")$auc
})

cat(sprintf("   AUC por fold:  %s\n",
            paste(sprintf("F%d=%.4f", seq_along(aucs_cv), aucs_cv), collapse = " | ")))
cat(sprintf("   AUC médio: %.4f (±%.4f)\n\n", mean(aucs_cv), sd(aucs_cv)))


# ── 9. Relatório Quarto ──────────────────────────────────────────────────────
cat("── 9. Gerando relatório HTML ──────────────────────────\n")

dados_rds <- file.path(output_dir, "dados_relatorio.rds")

preparar_dados_relatorio(
  titulo            = "Modelo Preditivo — Classificação Binária (Exemplo statMancer)",
  descricao         = paste0(
    "Pipeline completo com dados sintéticos. ",
    "AUC = ", sprintf("%.4f", metricas$auc),
    " | KS = ", sprintf("%.4f", metricas$ks),
    " | Gini = ", sprintf("%.4f", metricas$gini)
  ),
  dt_treino         = split$treino,
  dt_teste          = split$teste,
  var_target        = "target",
  var_id            = "id",
  busca_estatistica = busca,
  modelo_obj        = modelo,
  predicoes_teste   = dt_avaliacao,
  caminho_saida     = dados_rds
)

renderizar_relatorio(
  template_qmd = src("report/template_modelo.qmd"),
  dados_rds    = dados_rds,
  output_file  = "relatorio_modelo.html",
  output_dir   = output_dir
)


cat("\n╔══════════════════════════════════════════════════════╗\n")
cat("║           Pipeline concluído com sucesso!           ║\n")
cat("╚══════════════════════════════════════════════════════╝\n")
cat(sprintf("   Relatório: %s\n\n",
            file.path(output_dir, "relatorio_modelo.html")))
