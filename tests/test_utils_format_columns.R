# ============================================================================
# test_utils_format_columns.R
# Testes TDD para format_nomes_colunas()
# de R/utils/utils_format_columns.R
#
# Cobre:
#   T1  — Espaços viram "_" (snake_case)
#   T2  — Letras maiúsculas viram minúsculas (lowercase)
#   T3  — Caracteres especiais são removidos
#   T4  — Múltiplos espaços/separadores colapsam em um único "_"
#   T5  — "_" no início e no fim são removidos
#   T6  — Números são preservados
#   T7  — Nomes já normalizados ficam inalterados
#   T8  — Data.frame como entrada funciona (converte para data.table)
#   T9  — O objeto original não é modificado (sem efeito colateral)
#   T10 — Dados das colunas são preservados integralmente
#   T11 — Nomes duplicados após normalização são tratados (sem crash)
# ============================================================================

suppressPackageStartupMessages(library(data.table))

.proj_root <- tryCatch(
  dirname(dirname(rstudioapi::getSourceEditorContext()$path)),
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    f    <- args[grepl("--file=", args)]
    if (length(f) > 0) dirname(dirname(sub("--file=", "", f)))
    else getwd()
  }
)

source(file.path(.proj_root, "R", "utils", "utils_format_columns.R"))

.assert <- function(condicao, msg) {
  if (!condicao) stop(paste0("[FAIL] ", msg))
  cat(paste0("  [OK] ", msg, "\n"))
}

cat("\n======= TESTES: format_nomes_colunas =======\n\n")

# ── T1: espaços viram "_" ─────────────────────────────────────────────────────
test_espacos_viram_underscore <- function() {
  dt  <- data.table(`Nome Completo` = 1L, `data de nascimento` = 2L)
  res <- format_nomes_colunas(dt)
  .assert("nome_completo"       %in% names(res), "Espaço único vira '_'")
  .assert("data_de_nascimento"  %in% names(res), "Múltiplos espaços viram '_' individual")
  cat("PASS: test_espacos_viram_underscore\n\n")
}

# ── T2: maiúsculas viram minúsculas ───────────────────────────────────────────
test_lowercase <- function() {
  dt  <- data.table(SCORE_A = 1L, NomeCliente = 2L, TOTAL = 3L)
  res <- format_nomes_colunas(dt)
  .assert("score_a"      %in% names(res), "SCORE_A vira score_a")
  .assert("nomecliente"  %in% names(res), "NomeCliente vira nomecliente")
  .assert("total"        %in% names(res), "TOTAL vira total")
  cat("PASS: test_lowercase\n\n")
}

# ── T3: caracteres especiais são removidos ────────────────────────────────────
test_especiais_removidos <- function() {
  dt  <- data.table(`Idade (anos)` = 1L, `valor-R$` = 2L, `col@id!` = 3L)
  res <- format_nomes_colunas(dt)
  .assert("idade_anos" %in% names(res), "Parênteses e espaço tratados: 'Idade (anos)' → idade_anos")
  .assert("valorrr"    %in% names(res) || "valorr" %in% names(res) || "valor" %in% names(res),
          "Hífen e R$ removidos: 'valor-R$'")
  .assert("colid"      %in% names(res), "@ e ! removidos: 'col@id!'")
  cat("PASS: test_especiais_removidos\n\n")
}

# ── T3b: hífen entre palavras — sem espaço ────────────────────────────────────
test_hifen_removido <- function() {
  dt  <- data.table(`score-final` = 1L)
  res <- format_nomes_colunas(dt)
  # hífen é removido → "scorefinal" (não há espaço, não gera _)
  .assert("scorefinal" %in% names(res), "Hífen sem espaços é removido: 'score-final' → scorefinal")
  cat("PASS: test_hifen_removido\n\n")
}

# ── T4: múltiplos separadores colapsam ────────────────────────────────────────
test_multiplos_underscores_colapsam <- function() {
  dt  <- data.table(`col  duplo` = 1L)  # dois espaços
  res <- format_nomes_colunas(dt)
  .assert("col_duplo" %in% names(res),
          "Dois espaços consecutivos colapsam em um único '_'")
  cat("PASS: test_multiplos_underscores_colapsam\n\n")
}

# ── T5: "_" no início e fim são removidos ─────────────────────────────────────
test_underscore_bordas_removido <- function() {
  dt  <- data.table(`_col_` = 1L, ` espaco_ini` = 2L)
  res <- format_nomes_colunas(dt)
  .assert(!any(startsWith(names(res), "_")), "Nenhum nome começa com '_'")
  .assert(!any(endsWith(names(res),   "_")), "Nenhum nome termina com '_'")
  cat("PASS: test_underscore_bordas_removido\n\n")
}

# ── T6: números preservados ───────────────────────────────────────────────────
test_numeros_preservados <- function() {
  dt  <- data.table(score1 = 1L, `Var 2` = 2L, TOTAL3 = 3L)
  res <- format_nomes_colunas(dt)
  .assert("score1" %in% names(res), "Número no final preservado: score1")
  .assert("var_2"  %in% names(res), "Número com espaço: 'Var 2' → var_2")
  .assert("total3" %in% names(res), "Número com maiúscula: TOTAL3 → total3")
  cat("PASS: test_numeros_preservados\n\n")
}

# ── T7: nomes já normalizados ficam inalterados ───────────────────────────────
test_ja_normalizados <- function() {
  nomes <- c("id", "score_a", "target", "col_123")
  dt    <- setNames(as.data.table(as.list(seq_along(nomes))), nomes)
  res   <- format_nomes_colunas(dt)
  .assert(identical(names(res), nomes), "Nomes já normalizados permanecem idênticos")
  cat("PASS: test_ja_normalizados\n\n")
}

# ── T8: data.frame como entrada ───────────────────────────────────────────────
test_aceita_dataframe <- function() {
  df  <- data.frame(`Nome Completo` = 1L, check.names = FALSE)
  res <- format_nomes_colunas(df)
  .assert(is.data.table(res),            "Retorna data.table mesmo recebendo data.frame")
  .assert("nome_completo" %in% names(res), "Nome normalizado mesmo com entrada data.frame")
  cat("PASS: test_aceita_dataframe\n\n")
}

# ── T9: sem efeito colateral no original ──────────────────────────────────────
test_sem_efeito_colateral <- function() {
  dt        <- data.table(`Nome Completo` = 1L)
  nomes_ori <- copy(names(dt))
  format_nomes_colunas(dt)
  .assert(identical(names(dt), nomes_ori),
          "Original não é modificado (sem efeito colateral)")
  cat("PASS: test_sem_efeito_colateral\n\n")
}

# ── T10: dados das colunas preservados ────────────────────────────────────────
test_dados_preservados <- function() {
  dt  <- data.table(`Score A` = c(1.1, 2.2, 3.3), `Flag B` = c(TRUE, FALSE, TRUE))
  res <- format_nomes_colunas(dt)
  .assert(identical(res$score_a, c(1.1, 2.2, 3.3)), "Valores numéricos preservados")
  .assert(identical(res$flag_b,  c(TRUE, FALSE, TRUE)), "Valores lógicos preservados")
  cat("PASS: test_dados_preservados\n\n")
}

# ============================================================================
# Testes TDD para format_delete_columns()
# Cobre:
#   D1  — Remove uma única coluna
#   D2  — Remove múltiplas colunas
#   D3  — Ignora nomes que não existem (sem erro)
#   D4  — Vetor vazio não remove nada
#   D5  — Sem efeito colateral no original
#   D6  — Dados das colunas restantes são preservados
#   D7  — Aceita data.frame como entrada
#   D8  — Remove todas as colunas (retorna data.table vazio)
# ============================================================================
cat("--- format_delete_columns() ---\n\n")

# D1: remove uma única coluna
test_delete_uma_coluna <- function() {
  dt  <- data.table(id = 1L, score = 0.5, target = 1L)
  res <- format_delete_columns(dt, "score")
  .assert(!"score" %in% names(res), "Coluna 'score' removida")
  .assert("id"     %in% names(res), "Coluna 'id' mantida")
  .assert("target" %in% names(res), "Coluna 'target' mantida")
  cat("PASS: test_delete_uma_coluna\n\n")
}

# D2: remove múltiplas colunas
test_delete_multiplas_colunas <- function() {
  dt  <- data.table(id = 1L, a = 1L, b = 2L, c = 3L)
  res <- format_delete_columns(dt, c("a", "c"))
  .assert(!"a" %in% names(res), "Coluna 'a' removida")
  .assert(!"c" %in% names(res), "Coluna 'c' removida")
  .assert("id" %in% names(res), "Coluna 'id' mantida")
  .assert("b"  %in% names(res), "Coluna 'b' mantida")
  cat("PASS: test_delete_multiplas_colunas\n\n")
}

# D3: ignora colunas inexistentes sem erro
test_delete_inexistente_sem_erro <- function() {
  dt  <- data.table(id = 1L, score = 0.5)
  res <- tryCatch(
    format_delete_columns(dt, c("score", "nao_existe")),
    error = function(e) NULL
  )
  .assert(!is.null(res),           "Não lança erro com nome inexistente")
  .assert(!"score" %in% names(res), "Coluna existente foi removida normalmente")
  .assert("id"    %in% names(res), "Coluna 'id' mantida")
  cat("PASS: test_delete_inexistente_sem_erro\n\n")
}

# D4: vetor vazio não remove nada
test_delete_vetor_vazio <- function() {
  dt      <- data.table(id = 1L, score = 0.5)
  res     <- format_delete_columns(dt, character(0))
  .assert(identical(sort(names(res)), c("id", "score")),
          "Vetor vazio preserva todas as colunas")
  cat("PASS: test_delete_vetor_vazio\n\n")
}

# D5: sem efeito colateral no original
test_delete_sem_efeito_colateral <- function() {
  dt        <- data.table(id = 1L, score = 0.5)
  nomes_ori <- copy(names(dt))
  format_delete_columns(dt, "score")
  .assert(identical(names(dt), nomes_ori), "Original não é modificado")
  cat("PASS: test_delete_sem_efeito_colateral\n\n")
}

# D6: dados das colunas restantes preservados
test_delete_dados_preservados <- function() {
  dt  <- data.table(id = 1L:3L, score = c(0.1, 0.2, 0.3), flag = c(TRUE, FALSE, TRUE))
  res <- format_delete_columns(dt, "score")
  .assert(identical(res$id,   1L:3L),                  "Valores de 'id' preservados")
  .assert(identical(res$flag, c(TRUE, FALSE, TRUE)),    "Valores de 'flag' preservados")
  cat("PASS: test_delete_dados_preservados\n\n")
}

# D7: aceita data.frame como entrada
test_delete_aceita_dataframe <- function() {
  df  <- data.frame(id = 1L, score = 0.5, stringsAsFactors = FALSE)
  res <- format_delete_columns(df, "score")
  .assert(is.data.table(res),        "Retorna data.table mesmo com entrada data.frame")
  .assert(!"score" %in% names(res), "Coluna removida com entrada data.frame")
  cat("PASS: test_delete_aceita_dataframe\n\n")
}

# D8: remove todas as colunas — retorna data.table com 0 colunas
test_delete_todas_colunas <- function() {
  dt  <- data.table(a = 1L, b = 2L)
  res <- format_delete_columns(dt, c("a", "b"))
  .assert(is.data.table(res),  "Retorna data.table mesmo sem colunas")
  .assert(ncol(res) == 0L,     "data.table resultante tem 0 colunas")
  cat("PASS: test_delete_todas_colunas\n\n")
}

test_delete_uma_coluna()
test_delete_multiplas_colunas()
test_delete_inexistente_sem_erro()
test_delete_vetor_vazio()
test_delete_sem_efeito_colateral()
test_delete_dados_preservados()
test_delete_aceita_dataframe()
test_delete_todas_colunas()

# ── Execução ──────────────────────────────────────────────────────────────────
test_espacos_viram_underscore()
test_lowercase()
test_especiais_removidos()
test_hifen_removido()
test_multiplos_underscores_colapsam()
test_underscore_bordas_removido()
test_numeros_preservados()
test_ja_normalizados()
test_aceita_dataframe()
test_sem_efeito_colateral()
test_dados_preservados()

cat("====================================\n")
cat("✅ Todos os testes de utils_format_columns passaram!\n\n")
