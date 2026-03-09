# ============================================================================
# test_sampling_balance.R
# Testes TDD para upsample() e downsample() de R/sampling/sampling_balance.R
# ============================================================================

suppressPackageStartupMessages(library(data.table))

# ── Setup: carregar funções e dados ──────────────────────────────────────────
source(file.path(dirname(dirname(rstudioapi::getSourceEditorContext()$path)),
                 "R", "sampling", "sampling_balance.R"))
source(file.path(dirname(rstudioapi::getSourceEditorContext()$path),
                 "helpers", "synthetic_data.R"))

# Helper de asserção com mensagem
.assert <- function(condicao, msg) {
  if (!condicao) stop(paste0("[FAIL] ", msg))
  cat(paste0("  [OK] ", msg, "\n"))
}

cat("\n======= TESTES: sampling_balance =======\n\n")


# ── UPSAMPLE ─────────────────────────────────────────────────────────────────
cat("--- upsample() ---\n")

# T1: balanceia classes ao tamanho da majoritária
test_upsample_balanceia_classes <- function() {
  dt  <- sintetico_binario(n_pos = 200, n_neg = 800)
  res <- upsample(dt, var_id = "id", var_target = "target", seed = 1)
  cnt <- res[, .N, by = target]

  .assert(nrow(cnt) == 2,                         "Deve retornar exatamente 2 classes")
  .assert(diff(range(cnt$N)) == 0,                "Ambas as classes devem ter o mesmo N")
  .assert(cnt[target == 1, N] == 800,             "Classe minoritária upsampled para 800")
  cat("PASS: test_upsample_balanceia_classes\n\n")
}

# T2: respeita n_por_classe customizado
test_upsample_n_customizado <- function() {
  dt  <- sintetico_binario(n_pos = 200, n_neg = 800)
  res <- upsample(dt, var_id = "id", var_target = "target", n_por_classe = 300, seed = 1)
  cnt <- res[, .N, by = target]

  .assert(all(cnt$N == 300), "Todas as classes devem ter N = 300")
  cat("PASS: test_upsample_n_customizado\n\n")
}

# T3: estratificação mantém proporção de regiao dentro de cada classe
test_upsample_estratificado <- function() {
  dt  <- sintetico_binario(n_pos = 200, n_neg = 800)
  res <- upsample(dt, var_id = "id", var_target = "target",
                  n_por_classe = 400, var_estratificacao = "regiao", seed = 1)
  cnt <- res[, .N, by = target]

  .assert(nrow(cnt) == 2,  "Deve retornar 2 classes")
  .assert(all(cnt$N > 0),  "Nenhuma classe pode ser vazia")
  cat("PASS: test_upsample_estratificado\n\n")
}

# T4: múltiplos registros por ID — todos os registros do ID são retornados
test_upsample_multiplos_registros_por_id <- function() {
  dt  <- sintetico_multiplos_registros(n_ids = 100, prop_positivos = 0.2)
  res <- upsample(dt, var_id = "id", var_target = "target", n_por_classe = 80, seed = 1)

  ids_posit <- unique(res[target == 1, id])
  .assert(length(ids_posit) <= 80 * 5,            "Quantidade razoável de registros positivos")
  .assert(nrow(res) > 0,                          "Resultado não pode ser vazio")
  cat("PASS: test_upsample_multiplos_registros_por_id\n\n")
}

# T5: reprodutibilidade via seed
test_upsample_seed_reproduzivel <- function() {
  dt  <- sintetico_binario()
  r1  <- upsample(dt, var_id = "id", var_target = "target", seed = 99)
  r2  <- upsample(dt, var_id = "id", var_target = "target", seed = 99)

  .assert(identical(r1$id, r2$id), "Resultados com mesma seed devem ser idênticos")
  cat("PASS: test_upsample_seed_reproduzivel\n\n")
}

# T6: erro se var_id inválida
test_upsample_erro_var_id <- function() {
  dt  <- sintetico_binario()
  err <- tryCatch(
    upsample(dt, var_id = "coluna_inexistente", var_target = "target"),
    error = function(e) e$message
  )
  .assert(is.character(err), "Deve lançar erro se var_id não existe")
  cat("PASS: test_upsample_erro_var_id\n\n")
}

test_upsample_balanceia_classes()
test_upsample_n_customizado()
test_upsample_estratificado()
test_upsample_multiplos_registros_por_id()
test_upsample_seed_reproduzivel()
test_upsample_erro_var_id()


# ── DOWNSAMPLE ───────────────────────────────────────────────────────────────
cat("--- downsample() ---\n")

# T7: balanceia classes ao tamanho da minoritária
test_downsample_balanceia_classes <- function() {
  dt  <- sintetico_binario(n_pos = 200, n_neg = 800)
  res <- downsample(dt, var_id = "id", var_target = "target", seed = 1)
  cnt <- res[, .N, by = target]

  .assert(nrow(cnt) == 2,              "Deve retornar 2 classes")
  .assert(diff(range(cnt$N)) == 0,    "Ambas as classes devem ter o mesmo N")
  .assert(cnt[target == 1, N] == 200, "N de cada classe deve ser 200 (minoritária)")
  cat("PASS: test_downsample_balanceia_classes\n\n")
}

# T8: respeita n_por_classe customizado
test_downsample_n_customizado <- function() {
  dt  <- sintetico_binario(n_pos = 200, n_neg = 800)
  res <- downsample(dt, var_id = "id", var_target = "target", n_por_classe = 150, seed = 1)
  cnt <- res[, .N, by = target]

  .assert(all(cnt$N == 150), "Todas as classes devem ter N = 150")
  cat("PASS: test_downsample_n_customizado\n\n")
}

# T9: estratificação
test_downsample_estratificado <- function() {
  dt  <- sintetico_binario(n_pos = 200, n_neg = 800)
  res <- downsample(dt, var_id = "id", var_target = "target",
                    n_por_classe = 100, var_estratificacao = "regiao", seed = 1)
  cnt <- res[, .N, by = target]

  .assert(nrow(cnt) == 2,  "Deve retornar 2 classes")
  .assert(all(cnt$N > 0),  "Nenhuma classe pode ser vazia")
  cat("PASS: test_downsample_estratificado\n\n")
}

# T10: IDs originais nunca duplicados no downsample
test_downsample_sem_duplicacao_ids <- function() {
  dt  <- sintetico_binario(n_pos = 200, n_neg = 800)
  res <- downsample(dt, var_id = "id", var_target = "target", seed = 1)

  ids_unicos <- unique(res$id)
  .assert(length(ids_unicos) == nrow(res), "IDs não devem ser duplicados no downsample")
  cat("PASS: test_downsample_sem_duplicacao_ids\n\n")
}

test_downsample_balanceia_classes()
test_downsample_n_customizado()
test_downsample_estratificado()
test_downsample_sem_duplicacao_ids()

cat("====================================\n")
cat("✅ Todos os testes de sampling_balance passaram!\n\n")
