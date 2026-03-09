# ============================================================================
# test_sampling_split.R
# Testes TDD para train_test_split() de R/sampling/sampling_split.R
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

source(file.path(.proj_root, "R", "sampling", "sampling_split.R"))
source(file.path(.proj_root, "tests", "helpers", "synthetic_data.R"))

.assert <- function(condicao, msg) {
  if (!condicao) stop(paste0("[FAIL] ", msg))
  cat(paste0("  [OK] ", msg, "\n"))
}

cat("\n======= TESTES: sampling_split =======\n\n")


# T1: proporção de IDs no treino próxima do alvo
test_split_proporcao_ids <- function() {
  dt  <- sintetico_binario(n_pos = 200, n_neg = 800)
  res <- train_test_split(dt, var_id = "id", prop_treino = 0.8, seed = 1)

  n_ids_total  <- length(unique(dt$id))
  n_ids_treino <- length(unique(res$treino$id))
  prop_real    <- n_ids_treino / n_ids_total

  .assert(abs(prop_real - 0.8) < 0.02, "Proporção de IDs no treino deve ser ~0.8")
  cat("PASS: test_split_proporcao_ids\n\n")
}

# T2: sem vazamento de IDs entre treino e teste
test_split_sem_vazamento <- function() {
  dt  <- sintetico_binario()
  res <- train_test_split(dt, var_id = "id", prop_treino = 0.75, seed = 2)

  ids_treino <- unique(res$treino$id)
  ids_teste  <- unique(res$teste$id)

  .assert(length(intersect(ids_treino, ids_teste)) == 0,
          "Nenhum ID deve aparecer em treino E teste simultaneamente")
  cat("PASS: test_split_sem_vazamento\n\n")
}

# T3: união de treino + teste = dados originais
test_split_cobertura_total <- function() {
  dt  <- sintetico_binario()
  res <- train_test_split(dt, var_id = "id", prop_treino = 0.8, seed = 3)

  ids_orig   <- sort(unique(dt$id))
  ids_unidos <- sort(unique(c(res$treino$id, res$teste$id)))

  .assert(identical(ids_orig, ids_unidos),
          "Union de treino + teste deve cobrir todos os IDs originais")
  cat("PASS: test_split_cobertura_total\n\n")
}

# T4: estratificação mantém proporção do target em treino e teste
test_split_estratificado_target <- function() {
  dt  <- sintetico_binario(n_pos = 200, n_neg = 800)
  res <- train_test_split(dt, var_id = "id", prop_treino = 0.8,
                          var_estratificacao = "target", seed = 4)

  prop_treino <- res$treino[, mean(target == 1)]
  prop_teste  <- res$teste[,  mean(target == 1)]
  prop_orig   <- dt[, mean(target == 1)]

  .assert(abs(prop_treino - prop_orig) < 0.03,
          "Proporção de positivos no treino deve ser próxima da original")
  .assert(abs(prop_teste - prop_orig) < 0.04,
          "Proporção de positivos no teste deve ser próxima da original")
  cat("PASS: test_split_estratificado_target\n\n")
}

# T5: múltiplos registros por ID — todos os registros do ID ficam no mesmo conjunto
test_split_multiplos_registros_por_id <- function() {
  dt  <- sintetico_multiplos_registros(n_ids = 200, max_registros = 4)
  res <- train_test_split(dt, var_id = "id", prop_treino = 0.8, seed = 5)

  ids_treino <- unique(res$treino$id)
  ids_teste  <- unique(res$teste$id)

  .assert(length(intersect(ids_treino, ids_teste)) == 0,
          "IDs com múltiplos registros não devem vazar entre treino e teste")
  cat("PASS: test_split_multiplos_registros_por_id\n\n")
}

# T6: reprodutibilidade via seed
test_split_seed_reproduzivel <- function() {
  dt  <- sintetico_binario()
  r1  <- train_test_split(dt, var_id = "id", prop_treino = 0.8, seed = 77)
  r2  <- train_test_split(dt, var_id = "id", prop_treino = 0.8, seed = 77)

  .assert(identical(sort(r1$treino$id), sort(r2$treino$id)),
          "Mesma seed deve produzir o mesmo split")
  cat("PASS: test_split_seed_reproduzivel\n\n")
}

# T7: erro se prop_treino fora do intervalo válido
test_split_erro_prop_invalida <- function() {
  dt  <- sintetico_binario()
  err <- tryCatch(
    train_test_split(dt, var_id = "id", prop_treino = 1.5),
    error = function(e) e$message
  )
  .assert(is.character(err), "Deve lançar erro para prop_treino >= 1")
  cat("PASS: test_split_erro_prop_invalida\n\n")
}

# ─────────────────────────────────────────────────────────────────────────────
# T8–T12: garantias fortes de isolamento de ID entre treino e teste
# ─────────────────────────────────────────────────────────────────────────────

# T8: verificação row-level — nenhuma linha de teste tem var_id presente no treino
test_split_isolamento_row_level <- function() {
  dt  <- sintetico_binario(n_pos = 300, n_neg = 700)
  res <- train_test_split(dt, var_id = "id", prop_treino = 0.8, seed = 8)

  # Merge row-level: se algum ID do teste aparecer no treino o merge retorna linhas
  ids_treino    <- res$treino$id
  linhas_vazadas <- res$teste[id %in% ids_treino]

  .assert(nrow(linhas_vazadas) == 0L,
          "Nenhuma linha do teste deve ter ID presente no treino (row-level)")
  cat("PASS: test_split_isolamento_row_level\n\n")
}

# T9: conservação de linhas — treino + teste == dataset original (sem perdas nem duplicatas)
test_split_conservacao_linhas <- function() {
  dt  <- sintetico_binario(n_pos = 400, n_neg = 600)
  res <- train_test_split(dt, var_id = "id", prop_treino = 0.8, seed = 9)

  .assert(nrow(res$treino) + nrow(res$teste) == nrow(dt),
          "nrow(treino) + nrow(teste) deve igualar nrow(dt) original")
  cat("PASS: test_split_conservacao_linhas\n\n")
}

# T10: sem duplicatas dentro de cada partição
test_split_sem_duplicatas_internas <- function() {
  dt  <- sintetico_binario()
  res <- train_test_split(dt, var_id = "id", prop_treino = 0.8, seed = 10)

  .assert(nrow(unique(res$treino)) == nrow(res$treino),
          "Treino nao deve conter linhas duplicadas")
  .assert(nrow(unique(res$teste)) == nrow(res$teste),
          "Teste nao deve conter linhas duplicadas")
  cat("PASS: test_split_sem_duplicatas_internas\n\n")
}

# T11: split estratificado também garante isolamento de IDs
test_split_estratificado_sem_vazamento <- function() {
  dt  <- sintetico_binario(n_pos = 200, n_neg = 800)
  res <- train_test_split(dt, var_id = "id", prop_treino = 0.8,
                          var_estratificacao = "target", seed = 11)

  ids_treino <- unique(res$treino$id)
  ids_teste  <- unique(res$teste$id)

  .assert(length(intersect(ids_treino, ids_teste)) == 0L,
          "Split estratificado: nenhum ID deve aparecer em treino E teste")

  # Row-level também
  linhas_vazadas <- res$teste[id %in% ids_treino]
  .assert(nrow(linhas_vazadas) == 0L,
          "Split estratificado: nenhuma linha do teste deve ter ID no treino (row-level)")
  cat("PASS: test_split_estratificado_sem_vazamento\n\n")
}

# T12: ID com muitos registros — todos os registros ficam no mesmo conjunto
test_split_id_com_muitos_registros <- function() {
  # Cria dataset onde ID=1 tem 50 linhas e todos os outros têm 1 linha
  dt_base <- sintetico_binario(n_pos = 100, n_neg = 100)
  dt_base[, id := id + 1000L]   # desloca IDs para não colidir
  id_especial <- data.table::data.table(
    id     = rep(1L, 50),
    target = rep(1L, 50),
    score_a = runif(50)
  )
  # Adiciona colunas restantes do sintetico_binario para compatibilidade
  for (col in setdiff(names(dt_base), names(id_especial)))
    id_especial[[col]] <- dt_base[[col]][1L]
  dt_full <- data.table::rbindlist(list(dt_base, id_especial), fill = TRUE)

  res <- train_test_split(dt_full, var_id = "id", prop_treino = 0.8, seed = 12)

  # O ID especial deve estar inteiramente em UM dos conjuntos
  linhas_id_treino <- res$treino[id == 1L]
  linhas_id_teste  <- res$teste[id  == 1L]
  total_id         <- linhas_id_treino[, .N] + linhas_id_teste[, .N]

  .assert(total_id == 50L,
          "Todas as 50 linhas do ID especial devem aparecer em treino ou teste")
  .assert(linhas_id_treino[, .N] == 0L || linhas_id_teste[, .N] == 0L,
          "As linhas do ID especial nao podem ser divididas entre treino e teste")
  cat("PASS: test_split_id_com_muitos_registros\n\n")
}

test_split_proporcao_ids()
test_split_sem_vazamento()
test_split_cobertura_total()
test_split_estratificado_target()
test_split_multiplos_registros_por_id()
test_split_seed_reproduzivel()
test_split_erro_prop_invalida()
test_split_isolamento_row_level()
test_split_conservacao_linhas()
test_split_sem_duplicatas_internas()
test_split_estratificado_sem_vazamento()
test_split_id_com_muitos_registros()

cat("====================================\n")
cat("✅ Todos os testes de sampling_split passaram!\n\n")
