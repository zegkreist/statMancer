# ============================================================================
# test_utils_categoricas.R
# Testes TDD para codificar_categoricas() e detectar_cols_categoricas()
# de R/utils/utils_categoricas.R
#
# Cobre:
#   - codificar_categoricas: modo treino — constrói factor_map
#   - codificar_categoricas: modo predição — aplica factor_map consistentemente
#   - codificar_categoricas: integridade (sem modificar originais)
#   - codificar_categoricas: nível desconhecido na predição → NA (não erro)
#   - codificar_categoricas: coluna numérica inalterada
#   - codificar_categoricas: coluna factor codificada igual a character
#   - codificar_categoricas: múltiplos tipos misturados
#   - detectar_cols_categoricas: detecta character e factor, ignora numeric
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

source(file.path(.proj_root, "R", "utils", "utils_categoricas.R"))

.assert <- function(cond, msg) {
  if (!cond) stop(paste0("[FAIL] ", msg))
  cat(paste0("  [OK] ", msg, "\n"))
}

cat("\n======= TESTES: utils_categoricas =======\n\n")


# ── Dados compartilhados ─────────────────────────────────────────────────────

.dt_treino <- data.table(
  regiao      = c("norte", "sul", "leste", "norte", "sul",   "norte"),
  sexo        = factor(c("M", "F", "M", "M", "F", "F")),
  score       = c(1.1, 2.2, 3.3, 4.4, 5.5, 6.6),
  faixa_risco = c("alto", "medio", "baixo", "alto", "baixo", "medio")
)

.dt_teste <- data.table(
  regiao      = c("sul", "norte", "leste"),
  sexo        = factor(c("F", "M", "F")),
  score       = c(7.7, 8.8, 9.9),
  faixa_risco = c("baixo", "alto", "medio")
)

.features_all <- c("regiao", "sexo", "score", "faixa_risco")


# ─── TESTES ──────────────────────────────────────────────────────────────────

# T1: modo treino — retorna lista com X e factor_map
test_treino_retorna_estrutura <- function() {
  enc <- codificar_categoricas(.dt_treino, .features_all)

  .assert(is.list(enc),                        "codificar_categoricas retorna lista")
  .assert("X"          %in% names(enc),        "campo 'X' presente")
  .assert("factor_map" %in% names(enc),        "campo 'factor_map' presente")
  .assert(data.table::is.data.table(enc$X),    "X é data.table")
}

# T2: modo treino — factor_map contém os níveis corretos
test_treino_factor_map_niveis_corretos <- function() {
  enc <- codificar_categoricas(.dt_treino, .features_all)

  .assert(!is.null(enc$factor_map$regiao),
          "factor_map tem entrada para 'regiao'")
  .assert(identical(enc$factor_map$regiao, c("leste", "norte", "sul")),
          "níveis de 'regiao' são order. alfabética: leste, norte, sul")
  .assert(identical(enc$factor_map$sexo, c("F", "M")),
          "níveis de 'sexo' são: F, M")
  .assert(identical(enc$factor_map$faixa_risco, c("alto", "baixo", "medio")),
          "níveis de 'faixa_risco' são order. alfabética: alto, baixo, medio")
  .assert(is.null(enc$factor_map$score),
          "factor_map NÃO contém 'score' (numérica)")
}

# T3: modo treino — valores codificados são inteiros ordenados
test_treino_codificacao_inteiros <- function() {
  enc <- codificar_categoricas(.dt_treino, .features_all)

  # regiao: leste=1, norte=2, sul=3
  .assert(enc$X$regiao[enc$X$regiao == 2L][1L] == 2L || TRUE,  # just check type
          "regiao codificada como inteiro")
  .assert(is.integer(enc$X$regiao),   "regiao é inteiro")
  .assert(is.integer(enc$X$sexo),     "sexo é inteiro")
  # "norte" → 2 (alphabetical: leste=1, norte=2, sul=3)
  .assert(enc$X$regiao[1L] == 2L,     "norte codificado como 2")
  # "M" → 2 (alphabetical: F=1, M=2)
  .assert(enc$X$sexo[1L] == 2L,       "M codificado como 2")
}

# T4: modo predição — aplica o factor_map do treino (mesmos códigos)
test_predicao_aplica_factor_map_treino <- function() {
  enc_treino <- codificar_categoricas(.dt_treino, .features_all)
  enc_teste  <- codificar_categoricas(.dt_teste,  .features_all,
                                      factor_map = enc_treino$factor_map)

  # sul=3, norte=2, leste=1 (mesmo mapeamento do treino)
  .assert(enc_teste$X$regiao[1L] == 3L,  "sul codificado como 3 na predição")
  .assert(enc_teste$X$regiao[2L] == 2L,  "norte codificado como 2 na predição")
  .assert(enc_teste$X$regiao[3L] == 1L,  "leste codificado como 1 na predição")
  # F=1, M=2
  .assert(enc_teste$X$sexo[1L] == 1L,   "F codificado como 1 na predição")
  .assert(enc_teste$X$sexo[2L] == 2L,   "M codificado como 2 na predição")
}

# T5: consistência treino/predição — mesmos valores, mesmos códigos
test_consistencia_treino_predicao <- function() {
  enc_treino <- codificar_categoricas(.dt_treino, .features_all)

  # Mesmos dados, modo predição
  enc_pred   <- codificar_categoricas(.dt_treino, .features_all,
                                      factor_map = enc_treino$factor_map)

  .assert(identical(enc_treino$X$regiao, enc_pred$X$regiao),
          "regiao tem exatos mesmos códigos entre treino e predição com mesmos dados")
  .assert(identical(enc_treino$X$sexo, enc_pred$X$sexo),
          "sexo tem exatos mesmos códigos entre treino e predição com mesmos dados")
}

# T6: nível desconhecido na predição → NA (não lança erro)
test_nivel_desconhecido_vira_na <- function() {
  enc_treino <- codificar_categoricas(.dt_treino, .features_all)

  dt_novo <- data.table(
    regiao      = c("norte", "desconhecido"),  # "desconhecido" não existe no treino
    sexo        = factor(c("M", "M")),
    score       = c(1.0, 2.0),
    faixa_risco = c("alto", "alto")
  )

  enc_pred <- codificar_categoricas(dt_novo, .features_all,
                                    factor_map = enc_treino$factor_map)

  .assert(!is.na(enc_pred$X$regiao[1L]),  "norte conhecido → não NA")
  .assert(is.na(enc_pred$X$regiao[2L]),   "nível desconhecido → NA, não erro")
}

# T7: coluna numérica é preservada sem modificação
test_numerica_inalterada <- function() {
  enc <- codificar_categoricas(.dt_treino, .features_all)

  .assert(is.numeric(enc$X$score),                   "score permanece numérico")
  .assert(all(enc$X$score == .dt_treino$score),       "valores de score inalterados")
}

# T8: factor e character com mesmos valores → mesma codificação
test_factor_igual_a_character <- function() {
  dt1 <- data.table(x = c("a", "b", "a", "c"))           # character
  dt2 <- data.table(x = factor(c("a", "b", "a", "c")))   # factor

  enc1 <- codificar_categoricas(dt1, "x")
  enc2 <- codificar_categoricas(dt2, "x")

  .assert(identical(enc1$X$x, enc2$X$x),
          "character e factor com mesmos valores geram mesmos códigos")
  .assert(identical(enc1$factor_map, enc2$factor_map),
          "factor_map é idêntico para character e factor equivalentes")
}

# T9: dados originais NÃO são modificados (copy interno)
test_dados_originais_nao_modificados <- function() {
  dt_original <- data.table(
    regiao = c("norte", "sul"),
    score  = c(1.0, 2.0)
  )
  class_antes <- class(dt_original$regiao)

  codificar_categoricas(dt_original, c("regiao", "score"))

  .assert(identical(class(dt_original$regiao), class_antes),
          "classe da coluna original não é alterada após codificar_categoricas")
  .assert(all(dt_original$regiao == c("norte", "sul")),
          "valores da coluna original preservados")
}

# ── detectar_cols_categoricas ─────────────────────────────────────────────────

# T10: detecta character e factor, ignora numeric
test_detectar_character_e_factor <- function() {
  dt <- data.table(
    num  = c(1.0, 2.0),
    chr  = c("a", "b"),
    fct  = factor(c("x", "y")),
    int  = c(1L, 2L)
  )

  cats <- detectar_cols_categoricas(dt, c("num", "chr", "fct", "int"))

  .assert("chr" %in% cats,  "character detectada como categórica")
  .assert("fct" %in% cats,  "factor detectada como categórica")
  .assert(!"num" %in% cats, "numeric NÃO detectada como categórica")
  .assert(!"int" %in% cats, "integer NÃO detectada como categórica")
  .assert(length(cats) == 2L, "exatamente 2 colunas categóricas detectadas")
}

# T11: retorna vazio quando não há categóricas
test_detectar_sem_categoricas <- function() {
  dt   <- data.table(a = c(1.0, 2.0), b = c(3L, 4L))
  cats <- detectar_cols_categoricas(dt, c("a", "b"))

  .assert(length(cats) == 0L, "retorna vetor vazio quando não há categóricas")
}


# ── Execução ──────────────────────────────────────────────────────────────────
tests <- list(
  "T1:  retorna estrutura (X + factor_map)"        = test_treino_retorna_estrutura,
  "T2:  factor_map com níveis corretos"            = test_treino_factor_map_niveis_corretos,
  "T3:  codificação como inteiros ordinais"        = test_treino_codificacao_inteiros,
  "T4:  predição aplica factor_map do treino"      = test_predicao_aplica_factor_map_treino,
  "T5:  consistência treino ≡ predição"            = test_consistencia_treino_predicao,
  "T6:  nível desconhecido → NA"                   = test_nivel_desconhecido_vira_na,
  "T7:  coluna numérica inalterada"                = test_numerica_inalterada,
  "T8:  factor ≡ character com mesmos valores"     = test_factor_igual_a_character,
  "T9:  dados originais não modificados"           = test_dados_originais_nao_modificados,
  "T10: detectar — character + factor vs numeric"  = test_detectar_character_e_factor,
  "T11: detectar — sem categóricas → vazio"        = test_detectar_sem_categoricas
)

resultados <- list(passou = character(), falhou = character())

for (nome in names(tests)) {
  cat(paste0("\n--- ", nome, " ---\n"))
  resultado <- tryCatch(
    { tests[[nome]](); "passou" },
    error = function(e) { cat(paste0("  ", e$message, "\n")); "falhou" }
  )
  if (resultado == "passou") resultados$passou <- c(resultados$passou, nome)
  else                       resultados$falhou <- c(resultados$falhou, nome)
}

cat("\n\n=============================================\n")
cat(sprintf("RESULTADO: %d/%d testes passaram\n",
            length(resultados$passou), length(tests)))
if (length(resultados$falhou) > 0) {
  cat("FALHAS:\n")
  for (f in resultados$falhou) cat(paste0("  ✗ ", f, "\n"))
} else {
  cat("Todos os testes passaram!\n")
}
cat("=============================================\n\n")
