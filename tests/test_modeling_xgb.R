# ============================================================================
# test_modeling_xgb.R
# Testes TDD para xgb_train(), xgb_predict() e xgb_importancia()
# de R/modeling/modeling_xgb_train.R e modeling_xgb_predict.R
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
source(file.path(.proj_root, "R", "modeling", "modeling_xgb_train.R"))
source(file.path(.proj_root, "R", "modeling", "modeling_xgb_predict.R"))
source(file.path(.proj_root, "tests", "helpers", "synthetic_data.R"))

.assert <- function(condicao, msg) {
  if (!condicao) stop(paste0("[FAIL] ", msg))
  cat(paste0("  [OK] ", msg, "\n"))
}

cat("\n======= TESTES: modeling_xgb =======\n\n")


# ── XGB_TRAIN ────────────────────────────────────────────────────────────────
cat("--- xgb_train() ---\n")

# T1: retorna estrutura esperada
test_train_estrutura_resultado <- function() {
  dt   <- sintetico_binario(n_pos = 200, n_neg = 200)
  obj  <- xgb_train(dt, var_target = "target", vars_excluir = "id", nrounds = 10)

  .assert(is.list(obj),                                  "Resultado deve ser uma lista")
  .assert(all(c("modelo", "features", "params",
                "nrounds", "var_target") %in% names(obj)), "Deve ter todos os campos esperados")
  .assert(inherits(obj$modelo, "xgb.Booster"),           "Modelo deve ser xgb.Booster")
  .assert(!"id" %in% obj$features,                       "var_id não deve estar nas features")
  .assert(!"target" %in% obj$features,                   "target não deve estar nas features")
  cat("PASS: test_train_estrutura_resultado\n\n")
}

# T2: aceita parâmetros customizados
test_train_params_customizados <- function() {
  dt  <- sintetico_binario(n_pos = 150, n_neg = 150)
  params_custom <- list(
    objective   = "binary:logistic",
    eval_metric = "logloss",
    eta         = 0.1,
    max_leaves  = 8L,
    grow_policy = "lossguide",
    tree_method = "hist"
  )
  obj <- xgb_train(dt, var_target = "target", vars_excluir = "id",
                   params = params_custom, nrounds = 5)

  .assert(obj$params$eta == 0.1,           "Parâmetro eta deve ser o customizado")
  .assert(obj$nrounds == 5,               "nrounds deve ser 5")
  cat("PASS: test_train_params_customizados\n\n")
}

# T3: erro se var_target ausente
test_train_erro_target_ausente <- function() {
  dt  <- sintetico_binario()
  err <- tryCatch(
    xgb_train(dt, var_target = "coluna_inexistente"),
    error = function(e) e$message
  )
  .assert(is.character(err), "Deve lançar erro se var_target não existe")
  cat("PASS: test_train_erro_target_ausente\n\n")
}

# T4: erro se nenhuma feature restante após exclusões
test_train_erro_sem_features <- function() {
  dt  <- data.table::data.table(id = 1:5, target = c(0,1,0,1,0))
  err <- tryCatch(
    xgb_train(dt, var_target = "target", vars_excluir = "id"),
    error = function(e) e$message
  )
  .assert(is.character(err), "Deve lançar erro se não há features")
  cat("PASS: test_train_erro_sem_features\n\n")
}

test_train_estrutura_resultado()
test_train_params_customizados()
test_train_erro_target_ausente()
test_train_erro_sem_features()


# ── XGB_PREDICT ──────────────────────────────────────────────────────────────
cat("--- xgb_predict() ---\n")

# T5: predições com formato correto
test_predict_formato <- function() {
  dt   <- sintetico_binario(n_pos = 200, n_neg = 200)
  obj  <- xgb_train(dt, var_target = "target", vars_excluir = "id", nrounds = 10)
  pred <- xgb_predict(obj, dt, var_id = "id")

  .assert(data.table::is.data.table(pred),           "Resultado deve ser data.table")
  .assert("id" %in% names(pred),                     "Deve conter coluna id")
  .assert("predito" %in% names(pred),                "Deve conter coluna predito")
  .assert(nrow(pred) == nrow(dt),                    "N de predições deve igualar N de dados")
  .assert(all(pred$predito >= 0 & pred$predito <= 1),"Predições binárias devem estar em [0,1]")
  cat("PASS: test_predict_formato\n\n")
}

# T6: modelo treinado tem score maior nos positivos
test_predict_discriminacao <- function() {
  set.seed(1)
  dt  <- sintetico_binario(n_pos = 400, n_neg = 400)
  obj <- xgb_train(dt, var_target = "target", vars_excluir = "id", nrounds = 50)
  pred <- xgb_predict(obj, dt)
  dt_pred <- cbind(dt[, .(target)], pred)

  media_pos <- dt_pred[target == 1, mean(predito)]
  media_neg <- dt_pred[target == 0, mean(predito)]

  .assert(media_pos > media_neg,
          "Score médio dos positivos deve ser maior que dos negativos")
  cat("PASS: test_predict_discriminacao\n\n")
}

# T7: erro se feature ausente nos novos dados
test_predict_erro_feature_faltando <- function() {
  dt   <- sintetico_binario(n_pos = 200, n_neg = 200)
  obj  <- xgb_train(dt, var_target = "target", vars_excluir = "id", nrounds = 5)

  dt_incompleto <- dt[, .(id, score_a)]  # remove features
  err <- tryCatch(
    xgb_predict(obj, dt_incompleto),
    error = function(e) e$message
  )
  .assert(is.character(err), "Deve lançar erro se features estão faltando")
  cat("PASS: test_predict_erro_feature_faltando\n\n")
}

# T8: predição sem var_id funciona
test_predict_sem_id <- function() {
  dt   <- sintetico_binario(n_pos = 100, n_neg = 100)
  obj  <- xgb_train(dt, var_target = "target", vars_excluir = "id", nrounds = 10)
  pred <- xgb_predict(obj, dt)  # sem var_id

  .assert("predito" %in% names(pred),  "Deve ter coluna predito mesmo sem var_id")
  .assert(nrow(pred) == nrow(dt),      "N predições deve igualar N dados")
  cat("PASS: test_predict_sem_id\n\n")
}

test_predict_formato()
test_predict_discriminacao()
test_predict_erro_feature_faltando()
test_predict_sem_id()


# ── XGB_IMPORTANCIA ──────────────────────────────────────────────────────────
cat("--- xgb_importancia() ---\n")

# T9: importância retorna features conhecidas
test_importancia_features_corretas <- function() {
  dt   <- sintetico_binario(n_pos = 300, n_neg = 300)
  obj  <- xgb_train(dt, var_target = "target", vars_excluir = "id", nrounds = 30)
  imp  <- xgb_importancia(obj)

  .assert(data.table::is.data.table(imp),                         "Deve ser data.table")
  .assert("Feature" %in% names(imp),                              "Deve ter coluna Feature")
  .assert("Gain" %in% names(imp),                                 "Deve ter coluna Gain")
  .assert(all(imp$Feature %in% obj$features),                     "Features devem pertencer ao modelo")
  .assert(imp[1, Gain] >= imp[.N, Gain],                          "Deve estar ordenado por Gain desc")
  cat("PASS: test_importancia_features_corretas\n\n")
}

# T10: top_n limita o resultado
test_importancia_top_n <- function() {
  dt   <- sintetico_binario(n_pos = 200, n_neg = 200)
  obj  <- xgb_train(dt, var_target = "target", vars_excluir = "id", nrounds = 20)
  imp  <- xgb_importancia(obj, top_n = 3)

  .assert(nrow(imp) <= 3, "top_n = 3 deve retornar no máximo 3 linhas")
  cat("PASS: test_importancia_top_n\n\n")
}

test_importancia_features_corretas()
test_importancia_top_n()

cat("====================================\n")
cat("✅ Todos os testes de modeling_xgb passaram!\n\n")
