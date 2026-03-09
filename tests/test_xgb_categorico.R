# ============================================================================
# test_xgb_categorico.R
# Testes TDD para suporte a variáveis categóricas (factor/character)
# no pipeline xgb_train → xgb_predict → xgb_treino_ensemble → xgb_prever_ensemble
#
# Cobre:
#   xgb_train:
#     - Treina corretamente com features character e factor
#     - Retorna factor_map com os campos esperados
#     - Modelos treinados com/sem categoricas são reproduzíveis
#   xgb_predict:
#     - Prediz usando o factor_map do modelo (sem NAs)
#     - Consistência: mesmos dados → mesmos scores via xgb_train+xgb_predict
#     - Nível desconhecido na predição → NA, não quebra o pipeline
#   xgb_treino_ensemble:
#     - metadata$training_config$factor_map é salvo
#     - factor_map tem os níveis corretos
#   xgb_prever_ensemble:
#     - Retorna predições sem NAs para dados com categoricas
#     - Round-trip: scores do ensemble batem (sem variação por re-encoding)
#     - Aceita factor_map nulo/vazio (dados puramente numéricos)
#   stats_search (item 8):
#     - Coluna factor COM sinal é detectada como categórica → ChiSq_CramerV
#     - Coluna character COM sinal é detectada como categórica → ChiSq_CramerV
#     - Variáveis categóricas significativas aparecem acima de ruído
# ============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(xgboost)
})

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
source(file.path(.proj_root, "R", "modeling", "modeling_xgb_train.R"))
source(file.path(.proj_root, "R", "modeling", "modeling_xgb_predict.R"))
source(file.path(.proj_root, "R", "modeling", "modeling_xgb_ensemble.R"))
source(file.path(.proj_root, "R", "stats",    "stats_search.R"))
source(file.path(.proj_root, "tests", "helpers", "synthetic_data.R"))

.assert <- function(cond, msg) {
  if (!cond) stop(paste0("[FAIL] ", msg))
  cat(paste0("  [OK] ", msg, "\n"))
}

cat("\n======= TESTES: xgb com variáveis categóricas =======\n\n")


# ── Dados e parâmetros ────────────────────────────────────────────────────────
.dt <- sintetico_binario_categorico(n_pos = 150L, n_neg = 600L, seed = 11L)

.params_rapidos <- list(
  parametros = list(
    objective = "binary:logistic", eval_metric = "auc",
    eta = 0.15, max_leaves = 8L, gamma = 0.1,
    subsample = 0.8, colsample_bytree = 0.8,
    grow_policy = "lossguide", tree_method = "hist"
  ),
  nrounds = 10L
)


# ─── xgb_train com categóricas ───────────────────────────────────────────────

# T1: xgb_train treina sem erro com colunas factor e character
test_xgb_train_com_categoricas <- function() {
  modelo <- xgb_train(
    dt_treino    = .dt,
    var_target   = "target",
    vars_excluir = c("id"),
    params       = .params_rapidos$parametros,
    nrounds      = .params_rapidos$nrounds,
    verbose      = FALSE
  )

  .assert(is.list(modelo),                        "xgb_train retorna lista")
  .assert(inherits(modelo$modelo, "xgb.Booster"), "modelo é xgb.Booster")
  .assert(!is.null(modelo$factor_map),            "factor_map presente no modelo")
  .assert(is.list(modelo$factor_map),             "factor_map é lista")
}

# T2: factor_map contém entradas para cada coluna categórica nas features
test_factor_map_colunas_corretas <- function() {
  modelo <- xgb_train(
    dt_treino = .dt, var_target = "target",
    vars_excluir = c("id"),
    params   = .params_rapidos$parametros,
    nrounds  = .params_rapidos$nrounds, verbose = FALSE
  )

  cats <- detectar_cols_categoricas(.dt, modelo$features)

  .assert(length(cats) > 0,
          "há colunas categóricas nas features")
  for (col in cats) {
    .assert(!is.null(modelo$factor_map[[col]]),
            paste0("factor_map tem entrada para coluna '", col, "'"))
  }
  .assert(is.null(modelo$factor_map$score_a),
          "factor_map NÃO tem entrada para 'score_a' (numérica)")
}

# T3: xgb_predict não gera NAs nas predições com colunas categóricas
test_xgb_predict_sem_nas <- function() {
  set.seed(42)
  idx     <- sample(nrow(.dt), 200L)
  treino  <- .dt[idx]
  teste   <- .dt[-idx]

  modelo <- xgb_train(
    dt_treino = treino, var_target = "target",
    vars_excluir = c("id"),
    params  = .params_rapidos$parametros,
    nrounds = .params_rapidos$nrounds, verbose = FALSE
  )

  preds <- xgb_predict(modelo, teste, var_id = "id")

  .assert(data.table::is.data.table(preds),     "resultado é data.table")
  .assert("predito" %in% names(preds),          "coluna 'predito' presente")
  .assert(!any(is.na(preds$predito)),            "sem NAs nas predições")
  .assert(all(preds$predito >= 0 & preds$predito <= 1),
          "predições no intervalo [0,1]")
}

# T4: predições consistentes — mesmos dados, mesmo modelo → mesmas predições
test_xgb_predict_consistente <- function() {
  modelo <- xgb_train(
    dt_treino = .dt, var_target = "target",
    vars_excluir = c("id"),
    params  = .params_rapidos$parametros,
    nrounds = .params_rapidos$nrounds, verbose = FALSE
  )

  preds1 <- xgb_predict(modelo, .dt[1:50])
  preds2 <- xgb_predict(modelo, .dt[1:50])

  .assert(identical(preds1$predito, preds2$predito),
          "predições são idênticas para os mesmos dados e modelo")
}

# T5: nível desconhecido na predição → NA na feature, não encerra com erro
test_xgb_predict_nivel_desconhecido <- function() {
  dt_treino <- data.table::copy(.dt[1:500])
  dt_novo   <- data.table::copy(.dt[1:10])
  dt_novo[1, sexo := factor("X")]  # nível nunca visto no treino

  modelo <- xgb_train(
    dt_treino = dt_treino, var_target = "target",
    vars_excluir = c("id"),
    params  = .params_rapidos$parametros,
    nrounds = .params_rapidos$nrounds, verbose = FALSE
  )

  # XGBoost lida com NA como nó especial — não deve falhar
  preds <- tryCatch(
    xgb_predict(modelo, dt_novo),
    error = function(e) NULL
  )

  .assert(!is.null(preds), "xgb_predict não lança erro com nível desconhecido")
  .assert(nrow(preds) == 10L, "retorna predição para todas as obs")
}


# ─── xgb_treino_ensemble com categóricas ─────────────────────────────────────

# T6: xgb_treino_ensemble armazena factor_map no metadata
test_ensemble_salva_factor_map <- function() {
  tmp <- file.path(tempdir(), "test_cat_ensemble")
  on.exit(unlink(tmp, recursive = TRUE), add = TRUE)

  res <- xgb_treino_ensemble(
    dt = .dt, var_id = "id", var_target = "target",
    parametros_treino   = .params_rapidos,
    n_models            = 2L,
    metodo_reamostragem = "upsample",
    folder_saida        = tmp,
    validation_split    = 0.0,
    seed                = 1L, log_every = 99L, nthreads = 1L
  )

  fm <- res$metadata$training_config$factor_map

  .assert(!is.null(fm),      "metadata$training_config$factor_map presente")
  .assert(is.list(fm),       "factor_map é lista")
  cats <- detectar_cols_categoricas(.dt, res$metadata$training_config$colunas_features)
  for (col in cats) {
    .assert(!is.null(fm[[col]]),
            paste0("factor_map contém níveis para '", col, "'"))
  }
}

# T7: factor_map carregado do disco é idêntico ao original
test_ensemble_factor_map_persistido <- function() {
  tmp <- file.path(tempdir(), "test_cat_ens_persist")
  on.exit(unlink(tmp, recursive = TRUE), add = TRUE)

  res <- xgb_treino_ensemble(
    dt = .dt, var_id = "id", var_target = "target",
    parametros_treino = .params_rapidos, n_models = 2L,
    folder_saida = tmp, validation_split = 0.0,
    seed = 2L, log_every = 99L, nthreads = 1L
  )

  loaded <- carregar_ensemble(tmp)
  fm_orig   <- res$metadata$training_config$factor_map
  fm_loaded <- loaded$metadata$training_config$factor_map

  .assert(identical(fm_orig, fm_loaded),
          "factor_map carregado do disco é idêntico ao original")
}


# ─── xgb_prever_ensemble com categóricas ─────────────────────────────────────

# T8: xgb_prever_ensemble retorna predições sem NAs para dados com categóricas
test_prever_ensemble_sem_nas <- function() {
  tmp <- file.path(tempdir(), "test_cat_prever")
  on.exit(unlink(tmp, recursive = TRUE), add = TRUE)

  res <- xgb_treino_ensemble(
    dt = .dt, var_id = "id", var_target = "target",
    parametros_treino = .params_rapidos, n_models = 3L,
    folder_saida = tmp, validation_split = 0.0,
    seed = 3L, log_every = 99L, nthreads = 1L
  )

  dt_novo <- sintetico_binario_categorico(n_pos = 30L, n_neg = 120L, seed = 55L)

  preds <- xgb_prever_ensemble(
    dados_novos = dt_novo, ensemble_obj = res,
    retornar_com_id = FALSE, nthreads = 1L
  )

  .assert(is.numeric(preds),                         "retorna vetor numérico")
  .assert(length(preds) == nrow(dt_novo),            "uma predição por observação")
  .assert(!any(is.na(preds)),                        "sem NAs nas predições")
  .assert(all(preds >= 0 & preds <= 1),              "predições em [0,1]")
}

# T9: xgb_prever_ensemble com retornar_com_id=TRUE retorna data.table(id, predicao)
test_prever_ensemble_com_id <- function() {
  tmp <- file.path(tempdir(), "test_cat_prever_id")
  on.exit(unlink(tmp, recursive = TRUE), add = TRUE)

  res <- xgb_treino_ensemble(
    dt = .dt, var_id = "id", var_target = "target",
    parametros_treino = .params_rapidos, n_models = 2L,
    folder_saida = tmp, validation_split = 0.0,
    seed = 4L, log_every = 99L, nthreads = 1L
  )

  dt_novo <- sintetico_binario_categorico(n_pos = 20L, n_neg = 80L, seed = 66L)

  preds <- xgb_prever_ensemble(
    dados_novos = dt_novo, ensemble_obj = res,
    retornar_com_id = TRUE, nthreads = 1L
  )

  .assert(data.table::is.data.table(preds),  "retorna data.table com id")
  .assert("id" %in% names(preds),            "coluna 'id' presente")
  .assert("predicao" %in% names(preds),      "coluna 'predicao' presente")
  .assert(nrow(preds) == nrow(dt_novo),      "uma linha por observação")
}

# T10: xgb_prever_ensemble round-trip — loaded ensemble == in-memory
test_prever_ensemble_roundtrip_categorico <- function() {
  tmp_orig <- file.path(tempdir(), "test_cat_rt_orig")
  tmp_dest <- file.path(tempdir(), "test_cat_rt_dest")
  on.exit({
    unlink(tmp_orig, recursive = TRUE)
    unlink(tmp_dest, recursive = TRUE)
  }, add = TRUE)

  res <- xgb_treino_ensemble(
    dt = .dt, var_id = "id", var_target = "target",
    parametros_treino = .params_rapidos, n_models = 3L,
    folder_saida = tmp_orig, validation_split = 0.0,
    seed = 5L, log_every = 99L, nthreads = 1L
  )

  salvar_ensemble(res, tmp_dest)
  loaded <- carregar_ensemble(tmp_dest)

  dt_novo <- sintetico_binario_categorico(n_pos = 25L, n_neg = 100L, seed = 77L)

  p_mem   <- xgb_prever_ensemble(dt_novo, res,    retornar_com_id = FALSE, nthreads = 1L)
  p_disco <- xgb_prever_ensemble(dt_novo, loaded, retornar_com_id = FALSE, nthreads = 1L)

  .assert(all(abs(p_mem - p_disco) < 1e-6),
          "xgb_prever_ensemble: in-memory == loaded do disco (com categóricas)")
}

# T11: xgb_prever_ensemble erro com objeto inválido
test_prever_ensemble_erro_objeto_invalido <- function() {
  err <- tryCatch(
    xgb_prever_ensemble(.dt, list(x = 1)),
    error = function(e) e$message
  )
  .assert(grepl("ERRO", err), "xgb_prever_ensemble emite [ERRO] com objeto inválido")
}


# ─── stats_search — item 8: categórica vs classificação ──────────────────────

# T12: factor COM sinal → detectado como "categorica" e testado via ChiSq_CramerV
test_stats_factor_com_sinal_classificacao <- function() {
  dt <- sintetico_binario_categorico(n_pos = 300L, n_neg = 700L, seed = 9L)

  res <- suppressMessages(
    stats_search(dt, var_target = "target", vars_excluir = "id",
                 tipo_target = "classificacao")
  )

  row_sexo  <- res[variavel == "sexo"]
  row_canal <- res[variavel == "canal"]

  .assert(nrow(row_sexo) == 1L,          "sexo aparece nos resultados")
  .assert(row_sexo$tipo_variavel == "categorica",
          "sexo detectada como 'categorica'")
  .assert(grepl("ChiSq", row_sexo$teste),
          "sexo (factor com sinal) testado via ChiSq_CramerV")
  .assert(row_sexo$p_valor < 0.05,       "sexo é significativa (p < 0.05)")

  .assert(nrow(row_canal) == 1L,         "canal aparece nos resultados")
  .assert(grepl("ChiSq", row_canal$teste),
          "canal (factor com sinal) testado via ChiSq_CramerV")
  .assert(row_canal$p_valor < 0.05,      "canal é significativa (p < 0.05)")
}

# T13: character COM sinal → detectado como "categorica" e testado via ChiSq_CramerV
test_stats_character_com_sinal_classificacao <- function() {
  dt <- sintetico_binario_categorico(n_pos = 300L, n_neg = 700L, seed = 9L)

  res <- suppressMessages(
    stats_search(dt, var_target = "target", vars_excluir = "id",
                 tipo_target = "classificacao")
  )

  row_fr <- res[variavel == "faixa_risco"]

  .assert(nrow(row_fr) == 1L,           "faixa_risco aparece nos resultados")
  .assert(row_fr$tipo_variavel == "categorica",
          "faixa_risco (character) detectada como 'categorica'")
  .assert(grepl("ChiSq", row_fr$teste),
          "faixa_risco testada via ChiSq_CramerV")
  .assert(row_fr$p_valor < 0.05,        "faixa_risco é significativa (p < 0.05)")
}

# T14: categóricas com sinal rankiam acima de regiao (sem sinal)
test_stats_categoricas_rankeiam_acima_de_ruido <- function() {
  dt <- sintetico_binario_categorico(n_pos = 400L, n_neg = 800L, seed = 13L)

  res <- suppressMessages(
    stats_search(dt, var_target = "target", vars_excluir = "id",
                 tipo_target = "classificacao")
  )

  pos_regiao  <- which(res$variavel == "regiao")
  pos_sexo    <- which(res$variavel == "sexo")
  pos_canal   <- which(res$variavel == "canal")
  pos_fr      <- which(res$variavel == "faixa_risco")

  .assert(pos_sexo  < pos_regiao,
          "sexo (sinal) rankia acima de regiao (sem sinal)")
  .assert(pos_canal < pos_regiao,
          "canal (sinal) rankia acima de regiao (sem sinal)")
  .assert(pos_fr    < pos_regiao,
          "faixa_risco (sinal) rankia acima de regiao (sem sinal)")
}

# T15: target factor/character ainda funciona com ChiSq
test_stats_target_como_factor <- function() {
  dt <- sintetico_binario_categorico(n_pos = 200L, n_neg = 400L, seed = 17L)
  dt[, target_f := factor(ifelse(target == 1, "positivo", "negativo"))]

  res <- suppressMessages(
    stats_search(dt, var_target = "target_f",
                 vars_excluir = c("id", "target"),
                 tipo_target = "classificacao")
  )

  .assert(nrow(res) > 0,                     "resultado não vazio com target factor")
  .assert("sexo" %in% res$variavel,          "sexo aparece quando target é factor")
  row_sexo <- res[variavel == "sexo"]
  .assert(row_sexo$p_valor < 0.05,
          "sexo significativa mesmo com target como factor")
}


# ── Execução ──────────────────────────────────────────────────────────────────
tests <- list(
  "T1:  xgb_train treina com factor/character"            = test_xgb_train_com_categoricas,
  "T2:  factor_map colunas corretas"                      = test_factor_map_colunas_corretas,
  "T3:  xgb_predict sem NAs com categóricas"              = test_xgb_predict_sem_nas,
  "T4:  xgb_predict consistente (mesmos dados=mesma pred)" = test_xgb_predict_consistente,
  "T5:  nível desconhecido → NA, não erro"                = test_xgb_predict_nivel_desconhecido,
  "T6:  ensemble salva factor_map no metadata"            = test_ensemble_salva_factor_map,
  "T7:  factor_map persistido identicamente"              = test_ensemble_factor_map_persistido,
  "T8:  xgb_prever_ensemble sem NAs"                      = test_prever_ensemble_sem_nas,
  "T9:  xgb_prever_ensemble retorna id+predicao"          = test_prever_ensemble_com_id,
  "T10: round-trip categorico (mem==disco)"               = test_prever_ensemble_roundtrip_categorico,
  "T11: xgb_prever_ensemble erro objeto inválido"         = test_prever_ensemble_erro_objeto_invalido,
  "T12: stats_search factor COM sinal → ChiSq"           = test_stats_factor_com_sinal_classificacao,
  "T13: stats_search character COM sinal → ChiSq"        = test_stats_character_com_sinal_classificacao,
  "T14: categóricas com sinal > regiao sem sinal"         = test_stats_categoricas_rankeiam_acima_de_ruido,
  "T15: target como factor ainda funciona"                = test_stats_target_como_factor
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
