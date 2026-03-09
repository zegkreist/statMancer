# ============================================================================
# test_reporting_metrics.R
# Testes TDD para metricas_binario(), tabela_decis() e curva_roc()
# de R/reporting/reporting_metrics.R
# ============================================================================

suppressPackageStartupMessages(library(data.table))

source(file.path(dirname(dirname(rstudioapi::getSourceEditorContext()$path)),
                 "R", "reporting", "reporting_metrics.R"))

.assert <- function(condicao, msg) {
  if (!condicao) stop(paste0("[FAIL] ", msg))
  cat(paste0("  [OK] ", msg, "\n"))
}

cat("\n======= TESTES: reporting_metrics =======\n\n")

# ── Dados de teste com resultados conhecidos ──────────────────────────────────

# Classificador perfeito: todos os positivos têm score > todos os negativos
dt_perfeito <- data.table(
  predito = c(0.95, 0.90, 0.85, 0.80,  0.20, 0.15, 0.10, 0.05),
  target  = c(1L,   1L,   1L,   1L,    0L,   0L,   0L,   0L)
)

# Classificador invertido: todos os positivos têm score < todos os negativos
dt_invertido <- data.table(
  predito = c(0.05, 0.10, 0.15, 0.20,  0.80, 0.85, 0.90, 0.95),
  target  = c(1L,   1L,   1L,   1L,    0L,   0L,   0L,   0L)
)

# Classificador parcialmente bom
set.seed(42)
dt_parcial <- data.table(
  predito = c(runif(100, 0.55, 0.95), runif(100, 0.15, 0.65)),
  target  = c(rep(1L, 100), rep(0L, 100))
)

# Dataset maior para decis
set.seed(99)
n_grande <- 1000L
dt_grande <- data.table(
  predito = c(runif(200, 0.6, 1.0), runif(800, 0.0, 0.5)),
  target  = c(rep(1L, 200), rep(0L, 800))
)


# ── METRICAS_BINARIO ─────────────────────────────────────────────────────────
cat("--- metricas_binario() ---\n")

# T1: AUC = 1 para classificador perfeito
test_metricas_auc_perfeito <- function() {
  res <- metricas_binario(dt_perfeito, var_pred = "predito", var_target = "target")
  .assert(res$auc == 1.0, "AUC deve ser 1.0 para classificador perfeito")
  .assert(res$ks  == 1.0, "KS deve ser 1.0 para classificador perfeito")
  .assert(res$gini == 1.0, "Gini deve ser 1.0 para classificador perfeito")
  cat("PASS: test_metricas_auc_perfeito\n\n")
}

# T2: AUC = 0 para classificador invertido (pior que aleatório)
test_metricas_auc_invertido <- function() {
  res <- metricas_binario(dt_invertido, var_pred = "predito", var_target = "target")
  .assert(res$auc == 0.0, "AUC deve ser 0.0 para classificador completamente invertido")
  .assert(res$gini == -1.0, "Gini deve ser -1.0 para classificador invertido")
  cat("PASS: test_metricas_auc_invertido\n\n")
}

# T3: AUC > 0.5 para classificador parcialmente bom
test_metricas_auc_parcial <- function() {
  res <- metricas_binario(dt_parcial, var_pred = "predito", var_target = "target")
  .assert(res$auc > 0.5, "AUC deve ser > 0.5 para classificador com sinal")
  .assert(res$auc < 1.0, "AUC deve ser < 1.0 (não é perfeito)")
  cat("PASS: test_metricas_auc_parcial\n\n")
}

# T4: estrutura correta do retorno
test_metricas_estrutura <- function() {
  res <- metricas_binario(dt_grande, var_pred = "predito", var_target = "target")

  campos_esperados <- c("n_total", "n_pos", "n_neg", "taxa_evento",
                        "auc", "ks", "gini",
                        "precision", "recall", "f1", "accuracy",
                        "tp", "fp", "tn", "fn")

  .assert(is.list(res),                                   "Resultado deve ser lista")
  .assert(all(campos_esperados %in% names(res)),          "Deve ter todos os campos")
  .assert(res$n_total == res$n_pos + res$n_neg,           "n_total = n_pos + n_neg")
  .assert(res$auc  >= 0 && res$auc  <= 1,                 "AUC em [0,1]")
  .assert(res$ks   >= 0 && res$ks   <= 1,                 "KS em [0,1]")
  .assert(res$gini >= -1 && res$gini <= 1,                "Gini em [-1,1]")
  .assert(res$precision >= 0 && res$precision <= 1,       "Precision em [0,1]")
  .assert(res$recall    >= 0 && res$recall    <= 1,       "Recall em [0,1]")
  .assert(res$f1        >= 0 && res$f1        <= 1,       "F1 em [0,1]")
  .assert(res$accuracy  >= 0 && res$accuracy  <= 1,       "Accuracy em [0,1]")
  .assert(res$tp + res$fp + res$tn + res$fn == res$n_total, "Soma da matriz de confusão = n_total")
  cat("PASS: test_metricas_estrutura\n\n")
}

# T5: taxas corretas no dataset grande
test_metricas_taxas <- function() {
  res <- metricas_binario(dt_grande, var_pred = "predito", var_target = "target")
  .assert(res$n_total == 1000L,                               "n_total deve ser 1000")
  .assert(res$n_pos   == 200L,                                "n_pos deve ser 200")
  .assert(res$n_neg   == 800L,                                "n_neg deve ser 800")
  .assert(abs(res$taxa_evento - 0.2) < 0.001,                 "taxa_evento deve ser 0.2")
  cat("PASS: test_metricas_taxas\n\n")
}

# T6: erro se target é unário
test_metricas_erro_target_unario <- function() {
  dt_uni <- data.table(predito = runif(10), target = rep(1L, 10))
  err <- tryCatch(
    metricas_binario(dt_uni, "predito", "target"),
    error = function(e) e$message
  )
  .assert(is.character(err), "Deve lançar erro se target tem apenas uma classe")
  cat("PASS: test_metricas_erro_target_unario\n\n")
}

test_metricas_auc_perfeito()
test_metricas_auc_invertido()
test_metricas_auc_parcial()
test_metricas_estrutura()
test_metricas_taxas()
test_metricas_erro_target_unario()


# ── TABELA_DECIS ──────────────────────────────────────────────────────────────
cat("--- tabela_decis() ---\n")

# T7: retorna exatamente n_decis linhas
test_decis_n_linhas <- function() {
  res <- tabela_decis(dt_grande, var_pred = "predito", var_target = "target")
  .assert(nrow(res) == 10L, "Deve retornar 10 decis por default")

  res5 <- tabela_decis(dt_grande, var_pred = "predito", var_target = "target", n_decis = 5L)
  .assert(nrow(res5) == 5L, "Deve retornar 5 decis quando n_decis = 5")
  cat("PASS: test_decis_n_linhas\n\n")
}

# T8: colunas obrigatórias presentes
test_decis_colunas <- function() {
  res  <- tabela_decis(dt_grande, var_pred = "predito", var_target = "target")
  cols <- c("decil", "n", "n_pos", "score_min", "score_max",
            "score_medio", "taxa_evento", "lift", "captura", "captura_acum")
  .assert(all(cols %in% names(res)), "Deve ter todas as colunas esperadas")
  cat("PASS: test_decis_colunas\n\n")
}

# T9: decil 1 (maior score) deve ter lift > 1 para modelo com sinal
test_decis_lift_decil1 <- function() {
  res <- tabela_decis(dt_grande, var_pred = "predito", var_target = "target")
  .assert(res[decil == 1L, lift] > 1, "Decil 1 deve ter lift > 1 para modelo com sinal")
  cat("PASS: test_decis_lift_decil1\n\n")
}

# T10: captura acumulada no último decil = 100%
test_decis_captura_total <- function() {
  res <- tabela_decis(dt_grande, var_pred = "predito", var_target = "target")
  .assert(abs(res[decil == max(decil), captura_acum] - 1.0) < 0.001,
          "Captura acumulada no último decil deve ser ~1.0 (100%)")
  cat("PASS: test_decis_captura_total\n\n")
}

# T11: soma de n_pos de todos os decis = n_pos total
test_decis_n_pos_total <- function() {
  res    <- tabela_decis(dt_grande, var_pred = "predito", var_target = "target")
  n_pos  <- dt_grande[, sum(target)]
  .assert(sum(res$n_pos) == n_pos, "Soma de n_pos nos decis deve igualar total de positivos")
  cat("PASS: test_decis_n_pos_total\n\n")
}

# T12: score_max do decil i >= score_min do decil i+1 (escores decrescentes)
test_decis_ordem_scores <- function() {
  res <- tabela_decis(dt_grande, var_pred = "predito", var_target = "target")
  for (i in seq_len(nrow(res) - 1)) {
    .assert(res[decil == i, score_min] >= res[decil == i + 1L, score_max] - 1e-6,
            sprintf("Score do decil %d deve ser >= score do decil %d", i, i + 1L))
  }
  cat("PASS: test_decis_ordem_scores\n\n")
}

test_decis_n_linhas()
test_decis_colunas()
test_decis_lift_decil1()
test_decis_captura_total()
test_decis_n_pos_total()
test_decis_ordem_scores()


# ── CURVA_ROC ─────────────────────────────────────────────────────────────────
cat("--- curva_roc() ---\n")

# T13: começa em (0,0) e termina em (1,1)
test_roc_inicio_fim <- function() {
  res <- curva_roc(dt_grande, var_pred = "predito", var_target = "target")
  .assert(res[1, fpr] == 0 && res[1, tpr] == 0,
          "Curva ROC deve começar em (0, 0)")
  .assert(abs(res[.N, fpr] - 1.0) < 0.001 && abs(res[.N, tpr] - 1.0) < 0.001,
          "Curva ROC deve terminar em (1, 1)")
  cat("PASS: test_roc_inicio_fim\n\n")
}

# T14: FPR e TPR são monotonicamente não-decrescentes
test_roc_monotona <- function() {
  res <- curva_roc(dt_grande, var_pred = "predito", var_target = "target")
  .assert(all(diff(res$fpr) >= -1e-9), "FPR deve ser não-decrescente")
  .assert(all(diff(res$tpr) >= -1e-9), "TPR deve ser não-decrescente")
  cat("PASS: test_roc_monotona\n\n")
}

# T15: classificador perfeito — a curva deve "dobrar" pelo canto superior esquerdo
test_roc_perfeito <- function() {
  res <- curva_roc(dt_perfeito, var_pred = "predito", var_target = "target")
  # No clasificador perfeito, tpr=1 deve ser alcançado antes fpr>0
  .assert(any(res$fpr == 0 & res$tpr == 1),
          "Classificador perfeito deve atingir TPR=1 com FPR=0")
  cat("PASS: test_roc_perfeito\n\n")
}

# T16: retorna data.table com colunas fpr e tpr
test_roc_estrutura <- function() {
  res <- curva_roc(dt_grande, var_pred = "predito", var_target = "target")
  .assert(data.table::is.data.table(res),    "Resultado deve ser data.table")
  .assert("fpr" %in% names(res),             "Deve ter coluna fpr")
  .assert("tpr" %in% names(res),             "Deve ter coluna tpr")
  .assert(all(res$fpr >= 0 & res$fpr <= 1),  "FPR deve estar em [0,1]")
  .assert(all(res$tpr >= 0 & res$tpr <= 1),  "TPR deve estar em [0,1]")
  cat("PASS: test_roc_estrutura\n\n")
}

test_roc_inicio_fim()
test_roc_monotona()
test_roc_perfeito()
test_roc_estrutura()

cat("====================================\n")
cat("✅ Todos os testes de reporting_metrics passaram!\n\n")
