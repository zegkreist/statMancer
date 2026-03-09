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
source(src("R/utils/utils_categoricas.R"))
source(src("R/modeling/modeling_xgb_train.R"))
source(src("R/modeling/modeling_xgb_predict.R"))
source(src("R/utils/utils_modelagem.R"))
source(src("R/modeling/modeling_xgb_ensemble.R"))
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

N_TOTAL  <- 20000L
N_POS    <- 400L   # 20% de positivos (desbalanceado)
N_NEG    <- N_TOTAL - N_POS

dados <- data.table(
  id       = seq_len(N_TOTAL),
  target   = c(rep(1L, N_POS), rep(0L, N_NEG)),

  # ── Variáveis categóricas COM sinal (factor) ─────────────────────────────
  # Positivos: maioria "M"; negativos: maioria "F"
  sexo = factor(c(
    sample(c("M", "F"), N_POS, replace = TRUE, prob = c(0.68, 0.32)),
    sample(c("M", "F"), N_NEG, replace = TRUE, prob = c(0.38, 0.62))
  )),

  # Positivos: preferem "app"/"online"; negativos: preferem "loja"/"telefone"
  canal = factor(c(
    sample(c("app", "online", "loja", "telefone"), N_POS,
           replace = TRUE, prob = c(0.50, 0.30, 0.10, 0.10)),
    sample(c("app", "online", "loja", "telefone"), N_NEG,
           replace = TRUE, prob = c(0.15, 0.20, 0.35, 0.30))
  )),

  # Character (não factor) COM sinal — também suportado automaticamente
  faixa_risco = c(
    sample(c("alto", "medio", "baixo"), N_POS,
           replace = TRUE, prob = c(0.60, 0.30, 0.10)),
    sample(c("alto", "medio", "baixo"), N_NEG,
           replace = TRUE, prob = c(0.15, 0.35, 0.50))
  ),

  # ── Variável categórica SEM sinal (factor) ────────────────────────────────
  regiao = factor(
    sample(c("norte", "sul", "leste", "oeste"), N_TOTAL, replace = TRUE)
  ),

  # ── Variáveis numéricas ───────────────────────────────────────────────────
  score_a  = c(rnorm(N_POS, mean = 72, sd = 10), rnorm(N_NEG, mean = 48, sd = 15)),
  score_b  = c(rnorm(N_POS, mean = 62, sd = 8),  rnorm(N_NEG, mean = 55, sd = 12)),
  score_c  = c(rnorm(N_POS, mean = 58, sd = 9),  rnorm(N_NEG, mean = 52, sd = 11)),
  ruido_1  = rnorm(N_TOTAL),
  ruido_2  = rnorm(N_TOTAL),
  idade    = c(sample(40L:70L, N_POS, replace = TRUE),
               sample(20L:65L, N_NEG, replace = TRUE))
)

cat(sprintf("   Total: %s obs | Positivos: %s (%.0f%%) | Negativos: %s\n",
            format(N_TOTAL, big.mark = "."),
            format(N_POS,   big.mark = "."), 100 * N_POS / N_TOTAL,
            format(N_NEG,   big.mark = ".")))
cat("   Categ. COM sinal: sexo (factor), canal (factor), faixa_risco (character)\n")
cat("   Categ. sem sinal: regiao (factor)\n\n")


# ── 2. Busca Estatística de Variáveis ────────────────────────────────────────
cat("── 2. Busca estatística de variáveis (numéricas + categóricas) ─\n")

busca <- suppressMessages(
  stats_search(dados,
               var_target   = "target",
               vars_excluir = c("id"),   # inclui categóricas na busca
               tipo_target  = "classificacao")
)

cat(sprintf("   %d variáveis testadas\n", nrow(busca)))
cat(sprintf("   Significativas (p < 0.05): %d/%d\n", busca[, sum(significativa)], nrow(busca)))
cat(sprintf("   Top 5 mais relevantes:\n"))
for (i in seq_len(min(5L, nrow(busca)))) {
  cat(sprintf("     %d. %-15s [%s] %-18s p=%.2e  rel=%.3f\n",
              i, busca[i, variavel], busca[i, tipo_variavel],
              paste0("(", busca[i, teste], ")"),
              busca[i, p_valor], busca[i, relevancia]))
}
cat("\n")


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


# ── 5. Busca de Hiperparâmetros (MBO Bayesiano + CV) ─────────────────────────
cat("── 5. Busca de hiperparâmetros via MBO Bayesiano (+ CV interno) ──\n")

# Parâmetros de partida — usados como fallback se MBO falhar
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
nrounds_xgb <- 150L

# MBO exige features numéricas: codifica categóricas, remove id
dt_mbo     <- data.table::copy(treino_bal)
dt_mbo[, id := NULL]
enc_mbo    <- codificar_categoricas(dt_mbo, setdiff(names(dt_mbo), "target"))
dt_mbo_enc <- enc_mbo$X
dt_mbo_enc[, target := dt_mbo$target]

resultado_mbo <- tryCatch(
  busca_parametros_mlrmbo(
    dados           = dt_mbo_enc,
    target_col      = "target",
    n_samples       = 600L,
    colunas_excluir = NULL,
    metrica         = "auc",
    objetivo        = "binary:logistic",
    niter_data      = 3L,   # sessões de reamostragem
    niter_bayes     = 10L,  # iterações bayesianas por sessão
    cv_folds        = 3L,
    cv_nrounds      = 200L,
    nthreads        = 1L,
    verbose         = FALSE
  ),
  error = function(e) list(sucesso = FALSE, erro = conditionMessage(e))
)

if (isTRUE(resultado_mbo$sucesso)) {
  p_mbo <- resultado_mbo$melhor_parametros
  # p_mbo$parametros já inclui objective / eval_metric / grow_policy / tree_method
  params_xgb  <- p_mbo$parametros
  nrounds_xgb <- p_mbo$nrounds
  cat(sprintf("   AUC-CV best : %.4f\n", resultado_mbo$auc))
  cat(sprintf("   eta=%.4f  max_leaves=%d  nrounds=%d\n",
              params_xgb$eta, params_xgb$max_leaves, nrounds_xgb))
} else {
  cat("   \u26a0  mlrMBO indisponível — usando parâmetros padrão\n")
}
cat("\n")


# ── 6. Treino do Ensemble XGBoost ────────────────────────────────────────────
cat("── 6. Treinando ensemble XGBoost (10 modelos) com params otimizados ─\n")

ensemble <- xgb_treino_ensemble(
  dt                    = treino_bal,
  var_id                = "id",
  var_target            = "target",
  vars_excluir          = NULL,   # todas as categóricas entram como features
  parametros_treino     = list(parametros = params_xgb, nrounds = nrounds_xgb),
  n_models              = 10L,
  metodo_reamostragem   = "upsample",
  folder_saida          = file.path(output_dir, "ensemble"),
  early_stopping_rounds = 20L,
  validation_split      = 0.1,
  seed                  = 42L,
  nthreads              = 1L
)

cat(sprintf("   Modelos treinados: %d\n",     length(ensemble$modelos)))
cat(sprintf("   Features: %d | Pasta: %s\n\n",
            length(ensemble$metadata$training_config$colunas_features),
            ensemble$folder_saida))


# ── 7. Predição com o Ensemble no Teste ──────────────────────────────────────
cat("── 7. Predizendo com ensemble no conjunto de teste ─────\n")

# xgb_prever_ensemble usa o factor_map do treino para codificação consistente
preds_ensemble <- xgb_prever_ensemble(
  dados_novos     = split$teste,
  ensemble_obj    = ensemble,
  metodo_combinacao = "media",
  retornar_com_id = TRUE,
  nthreads        = 1L
)

# preds_ensemble retorna data.table(id, predicao) — renomeamos para 'predito'
# para compatibilidade com metricas_binario()
dt_avaliacao <- merge(
  split$teste[, .(id, target)],
  preds_ensemble[, .(id, predito = predicao)],
  by = "id"
)

cat(sprintf("   Score médio geral: %.3f\n",   dt_avaliacao[, mean(predito)]))
cat(sprintf("   Score médio positivos: %.3f\n", dt_avaliacao[target == 1, mean(predito)]))
cat(sprintf("   Score médio negativos: %.3f\n\n", dt_avaliacao[target == 0, mean(predito)]))


# ── 8. Métricas de Avaliação ─────────────────────────────────────────────────
cat("── 8. Métricas de avaliação ───────────────────────────\n")

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


# ── 9. K-Fold Cross-Validation (demonstração) ────────────────────────────────
cat("── 9. K-Fold (5 folds) — AUC por fold com params ótimos ─\n")

folds <- kfold_split(dados, var_id = "id", k = 5L,
                     var_estratificacao = "target", seed = 42L)

aucs_cv <- sapply(folds, function(fold) {
  # Treinar no treino do fold com os params otimizados pelo MBO
  m_fold <- xgb_train(
    dt_treino    = fold$treino,
    var_target   = "target",
    vars_excluir = c("id"),
    params       = params_xgb,
    nrounds      = min(nrounds_xgb, 80L),
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


# ── 10. Relatório Quarto ─────────────────────────────────────────────────────
cat("── 10. Gerando relatório HTML ─────────────────────────\n")

dados_rds <- file.path(output_dir, "dados_relatorio.rds")

# Constrói um objeto modelo_obj compatível com preparar_dados_relatorio(),
# usando o primeiro modelo do ensemble como representante
modelo_report <- list(
  modelo     = ensemble$modelos[[1]],
  features   = ensemble$metadata$training_config$colunas_features,
  params     = ensemble$metadata$parametros_treino$parametros,
  nrounds    = ensemble$metadata$parametros_treino$nrounds,
  var_target = "target"
)

preparar_dados_relatorio(
  titulo            = "Ensemble XGBoost — Classificação Binária com Categóricas (statMancer)",
  descricao         = paste0(
    "Pipeline com ensemble de 10 modelos (upsample), inclui variáveis factor e character. ",
    "AUC = ", sprintf("%.4f", metricas$auc),
    " | KS = ", sprintf("%.4f", metricas$ks),
    " | Gini = ", sprintf("%.4f", metricas$gini)
  ),
  dt_treino         = split$treino,
  dt_teste          = split$teste,
  var_target        = "target",
  var_id            = "id",
  busca_estatistica = busca,
  modelo_obj        = modelo_report,
  predicoes_teste   = dt_avaliacao,
  ensemble_obj      = ensemble,
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

