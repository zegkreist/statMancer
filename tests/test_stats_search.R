# ============================================================================
# test_stats_search.R
# Testes TDD para stats_search() de R/stats/stats_search.R
# ============================================================================

suppressPackageStartupMessages(library(data.table))

source(file.path(dirname(dirname(rstudioapi::getSourceEditorContext()$path)),
                 "R", "stats", "stats_search.R"))
source(file.path(dirname(rstudioapi::getSourceEditorContext()$path),
                 "helpers", "synthetic_data.R"))

.assert <- function(condicao, msg) {
  if (!condicao) stop(paste0("[FAIL] ", msg))
  cat(paste0("  [OK] ", msg, "\n"))
}

cat("\n======= TESTES: stats_search =======\n\n")


# ── TARGET DE CLASSIFICAÇÃO ───────────────────────────────────────────────────
cat("--- target = classificacao ---\n")

# T1: variáveis com sinal real devem aparecer no topo (p-valor menor)
test_classificacao_ranking_correto <- function() {
  dt  <- sintetico_binario(n_pos = 300, n_neg = 700)
  res <- suppressMessages(
    stats_search(dt, var_target = "target", vars_excluir = "id")
  )

  top3 <- res[1:3, variavel]
  .assert("score_a" %in% top3 || "score_b" %in% top3,
          "Variáveis com sinal real (score_a ou score_b) devem estar no top-3")
  .assert(!("score_ruido" %in% res[1:2, variavel]),
          "Ruído não deve aparecer nas 2 primeiras posições")
  cat("PASS: test_classificacao_ranking_correto\n\n")
}

# T2: retorna data.table com colunas obrigatórias
test_classificacao_estrutura_resultado <- function() {
  dt  <- sintetico_binario()
  res <- suppressMessages(
    stats_search(dt, var_target = "target", vars_excluir = "id")
  )

  cols_esperadas <- c("variavel", "tipo_variavel", "teste", "estatistica",
                      "p_valor", "relevancia", "significativa")

  .assert(data.table::is.data.table(res),                  "Resultado deve ser data.table")
  .assert(all(cols_esperadas %in% names(res)),              "Deve ter todas as colunas esperadas")
  .assert(nrow(res) > 0,                                    "Resultado não pode ser vazio")
  .assert(all(res$p_valor >= 0 & res$p_valor <= 1),         "p_valores devem estar em [0,1]")
  .assert(all(res$relevancia >= 0, na.rm = TRUE),           "Relevância deve ser >= 0")
  cat("PASS: test_classificacao_estrutura_resultado\n\n")
}

# T3: detecta tipo do target automaticamente
test_classificacao_deteccao_automatica <- function() {
  dt  <- sintetico_binario()
  msg <- capture.output(
    res <- stats_search(dt, var_target = "target", vars_excluir = "id"),
    type = "message"
  )
  .assert(any(grepl("classificacao", msg, ignore.case = TRUE)),
          "Deve detectar target como classificação automaticamente")
  cat("PASS: test_classificacao_deteccao_automatica\n\n")
}

# T4: multiclasse
test_classificacao_multiclasse <- function() {
  dt  <- sintetico_multiclasse(n_por_classe = 200, n_classes = 3)
  res <- suppressMessages(
    stats_search(dt, var_target = "target", vars_excluir = "id",
                 tipo_target = "classificacao")
  )
  testes_ks <- res[grepl("KS", teste), variavel]

  .assert(nrow(res) > 0,                     "Deve retornar resultados para multiclasse")
  .assert("score_a" %in% testes_ks,          "score_a deve ser testado via KS multiclasse")
  cat("PASS: test_classificacao_multiclasse\n\n")
}

# T5: variável categórica usa ChiSq
test_classificacao_categorica_usa_chisq <- function() {
  dt  <- sintetico_binario()
  # Adicionar variável categórica com sinal
  set.seed(1)
  dt[target == 1, cat_sinal := sample(c("A", "B"), .N, replace = TRUE, prob = c(0.8, 0.2))]
  dt[target == 0, cat_sinal := sample(c("A", "B"), .N, replace = TRUE, prob = c(0.3, 0.7))]

  res <- suppressMessages(
    stats_search(dt, var_target = "target", vars_excluir = "id")
  )
  row_cat <- res[variavel == "cat_sinal"]

  .assert(nrow(row_cat) == 1,                          "cat_sinal deve aparecer no resultado")
  .assert(grepl("ChiSq", row_cat$teste),               "Deve usar ChiSq para variável categórica")
  cat("PASS: test_classificacao_categorica_usa_chisq\n\n")
}

test_classificacao_ranking_correto()
test_classificacao_estrutura_resultado()
test_classificacao_deteccao_automatica()
test_classificacao_multiclasse()
test_classificacao_categorica_usa_chisq()


# ── TARGET DE REGRESSÃO ───────────────────────────────────────────────────────
cat("--- target = regressao ---\n")

# T6: variáveis com correlação real no topo
test_regressao_ranking_correto <- function() {
  dt  <- sintetico_regressao(n = 800)
  res <- suppressMessages(
    stats_search(dt, var_target = "target", vars_excluir = "id",
                 tipo_target = "regressao")
  )

  top2 <- res[1:2, variavel]
  .assert("x1" %in% top2 || "x2" %in% top2,
          "x1 e x2 (maior efeito) devem estar no top-2")
  .assert(!("ruido" %in% res[1:3, variavel]),
          "Variável ruído não deve aparecer no top-3")
  cat("PASS: test_regressao_ranking_correto\n\n")
}

# T7: usa Pearson/Spearman para numéricas vs target numérico
test_regressao_usa_correlacao <- function() {
  dt  <- sintetico_regressao()
  res <- suppressMessages(
    stats_search(dt, var_target = "target", vars_excluir = "id",
                 tipo_target = "regressao")
  )
  testes_cor <- res[variavel == "x1", teste]

  .assert(grepl("Pearson|Spearman", testes_cor),
          "x1 deve ser testado via Pearson ou Spearman")
  cat("PASS: test_regressao_usa_correlacao\n\n")
}

test_regressao_ranking_correto()
test_regressao_usa_correlacao()


# ── EDGE CASES ────────────────────────────────────────────────────────────────
cat("--- edge cases ---\n")

# T8: erro se var_target não encontrada
test_erro_target_inexistente <- function() {
  dt  <- sintetico_binario()
  err <- tryCatch(
    stats_search(dt, var_target = "coluna_que_nao_existe"),
    error = function(e) e$message
  )
  .assert(is.character(err), "Deve lançar erro se var_target não existe")
  cat("PASS: test_erro_target_inexistente\n\n")
}

# T9: variáveis com muitos NAs são ignoradas graciosamente
test_variaveis_com_muitos_nas <- function() {
  dt <- sintetico_binario()
  dt[, var_nula := NA_real_]

  res <- suppressMessages(
    stats_search(dt, var_target = "target", vars_excluir = "id")
  )
  # var_nula pode retornar erro interno mas não deve quebrar toda a busca
  .assert(nrow(res) > 0, "A busca deve continuar mesmo com variáveis all-NA")
  cat("PASS: test_variaveis_com_muitos_nas\n\n")
}

test_erro_target_inexistente()
test_variaveis_com_muitos_nas()

cat("====================================\n")
cat("✅ Todos os testes de stats_search passaram!\n\n")
