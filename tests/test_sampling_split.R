# ============================================================================
# test_sampling_split.R
# Testes TDD para train_test_split() de R/sampling/sampling_split.R
# ============================================================================

suppressPackageStartupMessages(library(data.table))

source(file.path(dirname(dirname(rstudioapi::getSourceEditorContext()$path)),
                 "R", "sampling", "sampling_split.R"))
source(file.path(dirname(rstudioapi::getSourceEditorContext()$path),
                 "helpers", "synthetic_data.R"))

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

test_split_proporcao_ids()
test_split_sem_vazamento()
test_split_cobertura_total()
test_split_estratificado_target()
test_split_multiplos_registros_por_id()
test_split_seed_reproduzivel()
test_split_erro_prop_invalida()

cat("====================================\n")
cat("✅ Todos os testes de sampling_split passaram!\n\n")
