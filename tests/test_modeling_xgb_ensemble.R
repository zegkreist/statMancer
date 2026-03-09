# ============================================================================
# test_modeling_xgb_ensemble.R
# Testes TDD para xgb_treino_ensemble() de R/modeling/modeling_xgb_ensemble.R
#
# Cobre:
#   - Estrutura do retorno
#   - Número de modelos treinados
#   - Tipos dos modelos (xgb.Booster)
#   - Métodos de reamostragem: upsample, downsample, bootstrap
#   - Persistência em disco (arquivos .xgb e .RDS)
#   - Estrutura da importância agregada
#   - Conteúdo do metadata (training_config)
#   - vars_excluir exclui corretamente
#   - Compatibilidade do metadata com xgb_predict_ensemble()
#   - Validação de erros de entrada
# ============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(xgboost)
})

# ── Path resolution (RStudio interativo ou Rscript) ──────────────────────────
.proj_root <- tryCatch(
  dirname(dirname(rstudioapi::getSourceEditorContext()$path)),
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    f    <- args[grepl("--file=", args)]
    if (length(f) > 0) dirname(dirname(sub("--file=", "", f)))
    else getwd()
  }
)

source(file.path(.proj_root, "R", "utils",    "utils_categoricas.R"))
source(file.path(.proj_root, "R", "utils",    "utils_modelagem.R"))
source(file.path(.proj_root, "R", "sampling", "sampling_balance.R"))
# modeling_xgb_ensemble.R define xgb_treino_ensemble (sobrescreve a versão S3 de utils_modelagem)
source(file.path(.proj_root, "R", "modeling", "modeling_xgb_ensemble.R"))
source(file.path(.proj_root, "tests", "helpers", "synthetic_data.R"))

.assert <- function(condicao, msg) {
  if (!condicao) stop(paste0("[FAIL] ", msg))
  cat(paste0("  [OK] ", msg, "\n"))
}

cat("\n======= TESTES: modeling_xgb_ensemble =======\n\n")


# ── Dados e parâmetros compartilhados ─────────────────────────────────────────
# Dados pequenos para testes rápidos (nrounds=10 garante < 2s por modelo)
.dt <- sintetico_binario(n_pos = 150L, n_neg = 600L, seed = 7L)

.params <- list(
  parametros = list(
    objective        = "binary:logistic",
    eval_metric      = "auc",
    eta              = 0.15,
    max_leaves       = 8L,
    gamma            = 0.1,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 1,
    grow_policy      = "lossguide",
    tree_method      = "hist"
  ),
  nrounds = 10L   # poucos rounds — testes rápidos
)


# ─── TESTES ──────────────────────────────────────────────────────────────────

# T1: retorna lista com todos os campos esperados
test_ensemble_retorna_estrutura <- function() {
  res <- xgb_treino_ensemble(
    dt = .dt, var_id = "id", var_target = "target",
    parametros_treino     = .params,
    n_models              = 3L,
    metodo_reamostragem   = "upsample",
    early_stopping_rounds = 3L,
    validation_split      = 0.1,
    seed = 1L, log_every = 99L, nthreads = 1L
  )
  .assert(is.list(res),                     "Deve retornar uma lista")
  .assert("modelos"      %in% names(res),   "Deve conter 'modelos'")
  .assert("metadata"     %in% names(res),   "Deve conter 'metadata'")
  .assert("importance"   %in% names(res),   "Deve conter 'importance'")
  .assert("folder_saida" %in% names(res),   "Deve conter 'folder_saida'")
  cat("PASS: test_ensemble_retorna_estrutura\n\n")
}

# T2: número de modelos no resultado === n_models (sem falhas esperadas)
test_ensemble_n_models_correto <- function() {
  res <- xgb_treino_ensemble(
    dt = .dt, var_id = "id", var_target = "target",
    parametros_treino     = .params,
    n_models              = 5L,
    metodo_reamostragem   = "upsample",
    early_stopping_rounds = 3L,
    validation_split      = 0.1,
    seed = 2L, log_every = 99L, nthreads = 1L
  )
  .assert(length(res$modelos) == 5L, "Deve retornar exatamente 5 modelos")
  cat("PASS: test_ensemble_n_models_correto\n\n")
}

# T3: todos os modelos retornados são objetos xgb.Booster
test_ensemble_modelos_sao_xgb_booster <- function() {
  res <- xgb_treino_ensemble(
    dt = .dt, var_id = "id", var_target = "target",
    parametros_treino     = .params,
    n_models              = 3L,
    metodo_reamostragem   = "upsample",
    early_stopping_rounds = 3L,
    validation_split      = 0.1,
    seed = 3L, log_every = 99L, nthreads = 1L
  )
  todos_xgb <- all(sapply(res$modelos, function(m) inherits(m, "xgb.Booster")))
  .assert(todos_xgb, "Todos os modelos devem ser objetos xgb.Booster")
  cat("PASS: test_ensemble_modelos_sao_xgb_booster\n\n")
}

# T4: metodo_reamostragem = "downsample" treina sem erro
test_ensemble_metodo_downsample <- function() {
  res <- xgb_treino_ensemble(
    dt = .dt, var_id = "id", var_target = "target",
    parametros_treino     = .params,
    n_models              = 3L,
    metodo_reamostragem   = "downsample",
    early_stopping_rounds = 3L,
    validation_split      = 0.1,
    seed = 4L, log_every = 99L, nthreads = 1L
  )
  .assert(length(res$modelos) == 3L, "downsample deve produzir 3 modelos")
  cat("PASS: test_ensemble_metodo_downsample\n\n")
}

# T5: metodo_reamostragem = "bootstrap" treina sem erro
test_ensemble_metodo_bootstrap <- function() {
  res <- xgb_treino_ensemble(
    dt = .dt, var_id = "id", var_target = "target",
    parametros_treino     = .params,
    n_models              = 3L,
    metodo_reamostragem   = "bootstrap",
    early_stopping_rounds = 3L,
    validation_split      = 0.1,
    seed = 5L, log_every = 99L, nthreads = 1L
  )
  .assert(length(res$modelos) >= 1L, "bootstrap deve produzir ao menos 1 modelo")
  cat("PASS: test_ensemble_metodo_bootstrap\n\n")
}

# T6: modelos e metadados persistidos em disco
test_ensemble_arquivos_persistidos <- function() {
  tmp_dir <- file.path(tempdir(), paste0("test_ens_disco_", as.integer(Sys.time())))
  on.exit(if (dir.exists(tmp_dir)) unlink(tmp_dir, recursive = TRUE))

  res <- xgb_treino_ensemble(
    dt = .dt, var_id = "id", var_target = "target",
    parametros_treino     = .params,
    n_models              = 3L,
    metodo_reamostragem   = "upsample",
    folder_saida          = tmp_dir,
    early_stopping_rounds = 3L,
    validation_split      = 0.1,
    seed = 6L, log_every = 99L, nthreads = 1L
  )

  arqs_xgb <- list.files(tmp_dir, pattern = "\\.xgb$")
  arqs_rds <- list.files(tmp_dir, pattern = "\\.RDS$")

  .assert(length(arqs_xgb) >= 3L,
          "Deve haver pelo menos 3 arquivos .xgb no folder_saida")
  .assert("ensemble_metadata.RDS" %in% arqs_rds,
          "Deve existir ensemble_metadata.RDS")
  .assert("importancia_modelos.RDS" %in% arqs_rds,
          "Deve existir importancia_modelos.RDS")
  cat("PASS: test_ensemble_arquivos_persistidos\n\n")
}

# T7: importance retorna data.table com Feature e Gain_medio
test_ensemble_importance_estrutura <- function() {
  res <- xgb_treino_ensemble(
    dt = .dt, var_id = "id", var_target = "target",
    parametros_treino     = .params,
    n_models              = 3L,
    metodo_reamostragem   = "upsample",
    early_stopping_rounds = 3L,
    validation_split      = 0.1,
    seed = 7L, log_every = 99L, nthreads = 1L
  )
  imp <- res$importance
  .assert(data.table::is.data.table(imp), "importance deve ser um data.table")
  .assert("Feature"     %in% names(imp),  "importance deve ter coluna 'Feature'")
  .assert("Gain_medio"  %in% names(imp),  "importance deve ter coluna 'Gain_medio'")
  .assert(nrow(imp) > 0L,                 "importance deve ter pelo menos uma feature")
  cat("PASS: test_ensemble_importance_estrutura\n\n")
}

# T8: importance ordenada decrescente por Gain_medio
test_ensemble_importance_ordenada <- function() {
  res <- xgb_treino_ensemble(
    dt = .dt, var_id = "id", var_target = "target",
    parametros_treino     = .params,
    n_models              = 3L,
    metodo_reamostragem   = "upsample",
    early_stopping_rounds = 3L,
    validation_split      = 0.1,
    seed = 8L, log_every = 99L, nthreads = 1L
  )
  gains <- res$importance$Gain_medio
  .assert(all(diff(gains) <= 0), "importance deve estar ordenada decrescente por Gain_medio")
  cat("PASS: test_ensemble_importance_ordenada\n\n")
}

# T9: metadata contém training_config com var_id, var_target e colunas_features
test_ensemble_metadata_training_config <- function() {
  res <- xgb_treino_ensemble(
    dt = .dt, var_id = "id", var_target = "target",
    parametros_treino     = .params,
    n_models              = 3L,
    metodo_reamostragem   = "upsample",
    early_stopping_rounds = 3L,
    validation_split      = 0.1,
    seed = 9L, log_every = 99L, nthreads = 1L
  )
  cfg <- res$metadata$training_config
  .assert(cfg$var_id              == "id",       "metadata deve registrar var_id = 'id'")
  .assert(cfg$var_target          == "target",   "metadata deve registrar var_target = 'target'")
  .assert(cfg$metodo_reamostragem == "upsample", "metadata deve registrar metodo_reamostragem")
  .assert("colunas_features" %in% names(cfg),    "metadata deve registrar colunas_features")
  .assert(length(cfg$colunas_features) > 0L,     "colunas_features deve ser não-vazio")
  cat("PASS: test_ensemble_metadata_training_config\n\n")
}

# T10: vars_excluir remove colunas das features
test_ensemble_vars_excluir <- function() {
  res <- xgb_treino_ensemble(
    dt = .dt, var_id = "id", var_target = "target",
    vars_excluir          = "regiao",
    parametros_treino     = .params,
    n_models              = 3L,
    metodo_reamostragem   = "upsample",
    early_stopping_rounds = 3L,
    validation_split      = 0.1,
    seed = 10L, log_every = 99L, nthreads = 1L
  )
  feats <- res$metadata$training_config$colunas_features
  .assert(!"regiao" %in% feats,  "'regiao' deve ser excluída das features")
  .assert(!"id"     %in% feats,  "'id' deve ser excluído das features")
  .assert(!"target" %in% feats,  "'target' deve ser excluído das features")
  cat("PASS: test_ensemble_vars_excluir\n\n")
}

# T11: metadata compatível com xgb_predict_ensemble() — campo colunas_features presente
test_ensemble_metadata_compativel_predict <- function() {
  res <- xgb_treino_ensemble(
    dt = .dt, var_id = "id", var_target = "target",
    parametros_treino     = .params,
    n_models              = 3L,
    metodo_reamostragem   = "upsample",
    early_stopping_rounds = 3L,
    validation_split      = 0.1,
    seed = 11L, log_every = 99L, nthreads = 1L
  )
  # xgb_predict_ensemble acessa metadata$training_config$colunas_features
  feats <- res$metadata$training_config$colunas_features
  .assert(!is.null(feats) && is.character(feats),
          "colunas_features deve ser vetor de character")

  # Verificar que os modelos podem fazer predição com as features registradas
  dados_pred <- .dt[, ..feats]
  dados_num  <- converter_para_numerico_seguro(dados_pred)
  dmatrix    <- xgboost::xgb.DMatrix(as.matrix(dados_num))
  pred       <- predict(res$modelos[[1]], dmatrix)
  .assert(length(pred) == nrow(.dt), "Predição com o primeiro modelo deve retornar N scores")
  .assert(all(pred >= 0 & pred <= 1),  "Scores devem ser probabilidades [0, 1]")
  cat("PASS: test_ensemble_metadata_compativel_predict\n\n")
}

# T12: erro quando parametros_treino = NULL
test_ensemble_erro_params_null <- function() {
  capturado <- tryCatch({
    xgb_treino_ensemble(
      dt = .dt, var_id = "id", var_target = "target",
      parametros_treino = NULL,
      n_models = 2L, nthreads = 1L
    )
    FALSE
  }, error = function(e) TRUE)
  .assert(capturado, "Deve lançar erro quando parametros_treino = NULL")
  cat("PASS: test_ensemble_erro_params_null\n\n")
}

# T13: erro quando var_id não existe no data.table
test_ensemble_erro_var_id_invalido <- function() {
  capturado <- tryCatch({
    xgb_treino_ensemble(
      dt = .dt, var_id = "coluna_inexistente", var_target = "target",
      parametros_treino     = .params,
      n_models = 2L, nthreads = 1L
    )
    FALSE
  }, error = function(e) TRUE)
  .assert(capturado, "Deve lançar erro quando var_id não existe no data.table")
  cat("PASS: test_ensemble_erro_var_id_invalido\n\n")
}

# T14: seed garante reproducibilidade — mesmos modelos com mesma seed
test_ensemble_reproducibilidade_seed <- function() {
  tmp1 <- file.path(tempdir(), paste0("ens_seed_a_", as.integer(Sys.time())))
  tmp2 <- file.path(tempdir(), paste0("ens_seed_b_", as.integer(Sys.time()) + 1L))
  on.exit({
    if (dir.exists(tmp1)) unlink(tmp1, recursive = TRUE)
    if (dir.exists(tmp2)) unlink(tmp2, recursive = TRUE)
  })

  res1 <- xgb_treino_ensemble(
    dt = .dt, var_id = "id", var_target = "target",
    parametros_treino = .params, n_models = 2L,
    metodo_reamostragem = "upsample", folder_saida = tmp1,
    early_stopping_rounds = 3L, validation_split = 0.1,
    seed = 77L, log_every = 99L, nthreads = 1L
  )
  res2 <- xgb_treino_ensemble(
    dt = .dt, var_id = "id", var_target = "target",
    parametros_treino = .params, n_models = 2L,
    metodo_reamostragem = "upsample", folder_saida = tmp2,
    early_stopping_rounds = 3L, validation_split = 0.1,
    seed = 77L, log_every = 99L, nthreads = 1L
  )

  # As predições dos primeiros modelos devem ser idênticas
  feats  <- res1$metadata$training_config$colunas_features
  dados_num <- converter_para_numerico_seguro(.dt[, ..feats])
  dmat   <- xgboost::xgb.DMatrix(as.matrix(dados_num))
  p1 <- predict(res1$modelos[[1]], dmat)
  p2 <- predict(res2$modelos[[1]], dmat)
  .assert(isTRUE(all.equal(p1, p2)),
          "Mesma seed deve produzir modelos idênticos")
  cat("PASS: test_ensemble_reproducibilidade_seed\n\n")
}


# ── Executar todos os testes ──────────────────────────────────────────────────
test_ensemble_retorna_estrutura()
test_ensemble_n_models_correto()
test_ensemble_modelos_sao_xgb_booster()
test_ensemble_metodo_downsample()
test_ensemble_metodo_bootstrap()
test_ensemble_arquivos_persistidos()
test_ensemble_importance_estrutura()
test_ensemble_importance_ordenada()
test_ensemble_metadata_training_config()
test_ensemble_vars_excluir()
test_ensemble_metadata_compativel_predict()
test_ensemble_erro_params_null()
test_ensemble_erro_var_id_invalido()
test_ensemble_reproducibilidade_seed()

cat("\n======= TODOS OS TESTES PASSARAM: modeling_xgb_ensemble =======\n")
