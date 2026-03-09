# ============================================================================
# test_sampling_split_kfold.R
# Testes TDD para kfold_split() de R/sampling/sampling_split.R
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

cat("\n======= TESTES: kfold_split =======\n\n")


# T1: retorna lista de comprimento k
test_kfold_retorna_k_folds <- function() {
  dt  <- sintetico_binario()
  res <- kfold_split(dt, var_id = "id", k = 5, seed = 1)

  .assert(is.list(res),              "Resultado deve ser uma lista")
  .assert(length(res) == 5,         "Lista deve ter exatamente k=5 elementos")
  .assert(all(sapply(res, function(f) all(c("treino", "teste", "fold") %in% names(f)))),
          "Cada fold deve ter $treino, $teste e $fold")
  cat("PASS: test_kfold_retorna_k_folds\n\n")
}

# T2: sem vazamento — IDs de treino e teste são disjuntos dentro de cada fold
test_kfold_sem_vazamento_intra_fold <- function() {
  dt  <- sintetico_binario()
  res <- kfold_split(dt, var_id = "id", k = 5, seed = 2)

  vazamento <- sapply(res, function(f) {
    length(intersect(unique(f$treino$id), unique(f$teste$id))) > 0
  })

  .assert(!any(vazamento),
          "Nenhum fold deve ter IDs em comum entre treino e teste")
  cat("PASS: test_kfold_sem_vazamento_intra_fold\n\n")
}

# T3: cada ID aparece em exatamente UM fold como teste
test_kfold_cada_id_em_um_fold_teste <- function() {
  dt  <- sintetico_binario()
  res <- kfold_split(dt, var_id = "id", k = 5, seed = 3)

  ids_teste_todos <- unlist(lapply(res, function(f) unique(f$teste$id)))
  ids_orig        <- unique(dt$id)

  .assert(length(ids_teste_todos) == length(unique(ids_teste_todos)),
          "Cada ID deve aparecer em exatamente 1 fold como teste (sem repetição)")
  .assert(setequal(ids_teste_todos, ids_orig),
          "A união dos testes de todos os folds deve cobrir todos os IDs")
  cat("PASS: test_kfold_cada_id_em_um_fold_teste\n\n")
}

# T4: union de treino + teste de um fold = todos os IDs
test_kfold_cobertura_por_fold <- function() {
  dt  <- sintetico_binario()
  res <- kfold_split(dt, var_id = "id", k = 5, seed = 4)

  ids_orig <- sort(unique(dt$id))
  cobertura_ok <- sapply(res, function(f) {
    ids_fold <- sort(unique(c(f$treino$id, f$teste$id)))
    identical(ids_fold, ids_orig)
  })

  .assert(all(cobertura_ok),
          "Treino + Teste de cada fold deve cobrir todos os IDs originais")
  cat("PASS: test_kfold_cobertura_por_fold\n\n")
}

# T5: tamanhos dos folds são aproximadamente iguais
test_kfold_tamanho_equilibrado <- function() {
  dt  <- sintetico_binario(n_pos = 500, n_neg = 500)  # 1000 IDs
  res <- kfold_split(dt, var_id = "id", k = 5, seed = 5)

  n_por_fold <- sapply(res, function(f) length(unique(f$teste$id)))
  # Com 1000 IDs e k=5, cada fold deve ter ~200 IDs
  .assert(all(abs(n_por_fold - 200) <= 1),
          "Cada fold de teste deve ter ~200 IDs (1000 / 5)")
  cat("PASS: test_kfold_tamanho_equilibrado\n\n")
}

# T6: estratificação mantém proporção do target em cada fold
test_kfold_estratificado_target <- function() {
  dt       <- sintetico_binario(n_pos = 200, n_neg = 800)
  prop_orig <- dt[, mean(target == 1)]
  res      <- kfold_split(dt, var_id = "id", k = 5,
                           var_estratificacao = "target", seed = 6)

  props_teste <- sapply(res, function(f) f$teste[, mean(target == 1)])

  .assert(all(abs(props_teste - prop_orig) < 0.06),
          "Proporção de positivos em cada fold de teste deve ser próxima da original")
  cat("PASS: test_kfold_estratificado_target\n\n")
}

# T7: múltiplos registros por ID — não vaza registros entre treino e teste
test_kfold_multiplos_registros_por_id <- function() {
  dt  <- sintetico_multiplos_registros(n_ids = 200, max_registros = 4)
  res <- kfold_split(dt, var_id = "id", k = 5, seed = 7)

  vazamento <- sapply(res, function(f) {
    ids_treino <- unique(f$treino$id)
    ids_teste  <- unique(f$teste$id)
    length(intersect(ids_treino, ids_teste)) > 0
  })

  .assert(!any(vazamento),
          "Com múltiplos registros por ID, nenhum ID deve aparecer em treino e teste")
  cat("PASS: test_kfold_multiplos_registros_por_id\n\n")
}

# T8: reprodutibilidade via seed
test_kfold_seed_reproduzivel <- function() {
  dt  <- sintetico_binario()
  r1  <- kfold_split(dt, var_id = "id", k = 5, seed = 88)
  r2  <- kfold_split(dt, var_id = "id", k = 5, seed = 88)

  ids_iguais <- all(sapply(seq_along(r1), function(i) {
    identical(sort(r1[[i]]$teste$id), sort(r2[[i]]$teste$id))
  }))

  .assert(ids_iguais, "Mesma seed deve produzir os mesmos folds")
  cat("PASS: test_kfold_seed_reproduzivel\n\n")
}

# T9: erro quando k < 2
test_kfold_erro_k_invalido <- function() {
  dt  <- sintetico_binario()
  err <- tryCatch(
    kfold_split(dt, var_id = "id", k = 1),
    error = function(e) e$message
  )
  .assert(is.character(err), "Deve lançar erro para k < 2")
  cat("PASS: test_kfold_erro_k_invalido\n\n")
}

# T10: erro quando n_ids < k
test_kfold_erro_poucos_ids <- function() {
  dt  <- data.table::data.table(id = 1:3, target = c(0, 1, 0), x = c(1, 2, 3))
  err <- tryCatch(
    kfold_split(dt, var_id = "id", k = 10),
    error = function(e) e$message
  )
  .assert(is.character(err), "Deve lançar erro se n_ids < k")
  cat("PASS: test_kfold_erro_poucos_ids\n\n")
}

test_kfold_retorna_k_folds()
test_kfold_sem_vazamento_intra_fold()
test_kfold_cada_id_em_um_fold_teste()
test_kfold_cobertura_por_fold()
test_kfold_tamanho_equilibrado()
test_kfold_estratificado_target()
test_kfold_multiplos_registros_por_id()
test_kfold_seed_reproduzivel()
test_kfold_erro_k_invalido()
test_kfold_erro_poucos_ids()

cat("====================================\n")
cat("✅ Todos os testes de kfold_split passaram!\n\n")
