# ============================================================================
# test_ensemble_io.R
# Testes TDD para carregar_ensemble() e salvar_ensemble()
# definidas em R/modeling/modeling_xgb_ensemble.R
#
# Cobre:
#   - Arquivos salvos automaticamente por xgb_treino_ensemble()
#   - carregar_ensemble: estrutura de retorno
#   - carregar_ensemble: número e tipo dos modelos carregados
#   - carregar_ensemble: metadata compatível com xgb_predict_ensemble()
#   - carregar_ensemble: round-trip de predições
#   - carregar_ensemble: erros esperados (pasta / metadata ausentes)
#   - salvar_ensemble: cópia de arquivos para novo destino
#   - salvar_ensemble: criação automática do folder_destino
#   - salvar_ensemble: validação do argumento resultado
#   - salvar_ensemble + carregar_ensemble: round-trip completo
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
source(file.path(.proj_root, "R", "modeling", "modeling_xgb_ensemble.R"))
source(file.path(.proj_root, "tests", "helpers", "synthetic_data.R"))

.assert <- function(condicao, msg) {
  if (!condicao) stop(paste0("[FAIL] ", msg))
  cat(paste0("  [OK] ", msg, "\n"))
}

cat("\n======= TESTES: ensemble IO (carregar/salvar) =======\n\n")


# ── Dados e parâmetros compartilhados ────────────────────────────────────────
.dt <- sintetico_binario(n_pos = 120L, n_neg = 480L, seed = 99L)

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
  nrounds = 10L
)

# Treina um ensemble base uma vez para todos os testes
.tmp_base <- file.path(tempdir(), "ensemble_io_base")
.res_base <- xgb_treino_ensemble(
  dt                    = .dt,
  var_id                = "id",
  var_target            = "target",
  parametros_treino     = .params,
  n_models              = 3L,
  metodo_reamostragem   = "upsample",
  folder_saida          = .tmp_base,
  early_stopping_rounds = 3L,
  validation_split      = 0.1,
  seed                  = 7L,
  log_every             = 99L,
  nthreads              = 1L
)


# ─── TESTES ──────────────────────────────────────────────────────────────────

# T1: xgb_treino_ensemble salva arquivos .xgb no folder_saida
test_treino_salva_xgb <- function() {
  xgb_files <- list.files(.tmp_base, pattern = "\\.xgb$")
  .assert(length(xgb_files) == 3L,
          paste0("xgb_treino_ensemble deve salvar 3 arquivos .xgb (encontrados: ",
                 length(xgb_files), ")"))
}

# T2: xgb_treino_ensemble salva ensemble_metadata.RDS
test_treino_salva_metadata_rds <- function() {
  .assert(file.exists(file.path(.tmp_base, "ensemble_metadata.RDS")),
          "ensemble_metadata.RDS deve existir no folder_saida")
}

# T3: xgb_treino_ensemble salva importancia_modelos.RDS
test_treino_salva_importancia_rds <- function() {
  .assert(file.exists(file.path(.tmp_base, "importancia_modelos.RDS")),
          "importancia_modelos.RDS deve existir no folder_saida")
}

# T4: carregar_ensemble retorna lista com campos corretos
test_carregar_retorna_estrutura <- function() {
  loaded <- carregar_ensemble(.tmp_base)
  .assert(is.list(loaded),                         "carregar_ensemble retorna lista")
  .assert("modelos"     %in% names(loaded),         "campo 'modelos' presente")
  .assert("metadata"    %in% names(loaded),         "campo 'metadata' presente")
  .assert("folder_saida" %in% names(loaded),        "campo 'folder_saida' presente")
  .assert(loaded$folder_saida == .tmp_base,         "folder_saida corresponde ao parâmetro")
}

# T5: carregar_ensemble carrega o número correto de modelos como xgb.Booster
test_carregar_numero_e_tipo_modelos <- function() {
  loaded <- carregar_ensemble(.tmp_base)
  .assert(length(loaded$modelos) == 3L,
          paste0("carregar_ensemble deve carregar 3 modelos (carregados: ",
                 length(loaded$modelos), ")"))
  todos_booster <- all(sapply(loaded$modelos,
                              function(m) inherits(m, "xgb.Booster")))
  .assert(todos_booster, "todos os modelos carregados devem ser xgb.Booster")
}

# T6: metadata carregada do disco contém training_config idêntico ao original
test_carregar_metadata_compativel <- function() {
  loaded <- carregar_ensemble(.tmp_base)
  orig_cfg <- .res_base$metadata$training_config
  load_cfg <- loaded$metadata$training_config

  .assert(!is.null(load_cfg),
          "metadata carregada deve ter training_config")
  .assert(identical(orig_cfg$colunas_features, load_cfg$colunas_features),
          "colunas_features da metadata carregada igual à original")
  .assert(identical(orig_cfg$var_id,  load_cfg$var_id),
          "var_id da metadata carregada igual à original")
  .assert(identical(orig_cfg$var_target, load_cfg$var_target),
          "var_target da metadata carregada igual à original")
}

# T7: Round-trip de predições — in-memory == loaded from disk
test_roundtrip_predicoes <- function() {
  loaded <- carregar_ensemble(.tmp_base)
  dt_new <- sintetico_binario(n_pos = 30L, n_neg = 120L, seed = 55L)

  preds_mem <- xgb_predict_ensemble(
    dados_novos       = dt_new,
    modelos_xgb       = .res_base$modelos,
    metadata_ensemble = .res_base$metadata,
    metodo_combinacao = "media",
    retornar_com_id   = FALSE,
    verbose           = FALSE,
    nthreads          = 1L
  )

  preds_disk <- xgb_predict_ensemble(
    dados_novos       = dt_new,
    modelos_xgb       = loaded$modelos,
    metadata_ensemble = loaded$metadata,
    metodo_combinacao = "media",
    retornar_com_id   = FALSE,
    verbose           = FALSE,
    nthreads          = 1L
  )

  .assert(length(preds_mem) == length(preds_disk),
          "número de predições igual entre in-memory e loaded")
  .assert(all(abs(preds_mem - preds_disk) < 1e-6),
          "predições in-memory e loaded do disco são numericamente idênticas")
}

# T8: carregar_ensemble emite erro quando pasta não existe
test_carregar_erro_pasta_inexistente <- function() {
  err <- tryCatch(
    carregar_ensemble("/tmp/pasta_que_nao_existe_xyz_abc"),
    error = function(e) e$message
  )
  .assert(grepl("ERRO", err),
          "carregar_ensemble emite [ERRO] quando pasta não existe")
}

# T9: carregar_ensemble emite erro quando ensemble_metadata.RDS está ausente
test_carregar_erro_sem_metadata <- function() {
  tmp_sem_meta <- file.path(tempdir(), "ensemble_sem_meta")
  on.exit(unlink(tmp_sem_meta, recursive = TRUE), add = TRUE)
  dir.create(tmp_sem_meta, recursive = TRUE, showWarnings = FALSE)
  # Apenas um .xgb falso, sem metadata
  writeLines("fake", file.path(tmp_sem_meta, "xgb_model_001.xgb"))

  err <- tryCatch(
    carregar_ensemble(tmp_sem_meta),
    error = function(e) e$message
  )
  .assert(grepl("ERRO", err),
          "carregar_ensemble emite [ERRO] quando ensemble_metadata.RDS ausente")
}

# T10: salvar_ensemble copia todos os arquivos para um novo folder
test_salvar_copia_arquivos <- function() {
  tmp_dest <- file.path(tempdir(), "ensemble_io_salvar_copia")
  on.exit(unlink(tmp_dest, recursive = TRUE), add = TRUE)

  salvar_ensemble(.res_base, tmp_dest)

  orig_files <- sort(basename(list.files(.tmp_base)))
  dest_files <- sort(basename(list.files(tmp_dest)))

  .assert(dir.exists(tmp_dest),
          "salvar_ensemble cria o folder_destino")
  .assert(identical(orig_files, dest_files),
          "salvar_ensemble copia todos os arquivos do folder_saida")
}

# T11: salvar_ensemble cria folder_destino automaticamente se não existir
test_salvar_cria_folder_automaticamente <- function() {
  tmp_novo <- file.path(tempdir(), "ensemble_io_novo_auto")
  on.exit(unlink(tmp_novo, recursive = TRUE), add = TRUE)

  if (dir.exists(tmp_novo)) unlink(tmp_novo, recursive = TRUE)
  .assert(!dir.exists(tmp_novo), "pasta destino não deve existir antes")

  salvar_ensemble(.res_base, tmp_novo)
  .assert(dir.exists(tmp_novo), "salvar_ensemble cria pasta inexistente automaticamente")
}

# T12: salvar_ensemble emite erro quando resultado é inválido
test_salvar_erro_resultado_invalido <- function() {
  tmp_dest <- file.path(tempdir(), "ensemble_io_invalido")
  on.exit(unlink(tmp_dest, recursive = TRUE), add = TRUE)

  err_lista_errada <- tryCatch(
    salvar_ensemble(list(x = 1, y = 2), tmp_dest),
    error = function(e) e$message
  )
  .assert(grepl("ERRO", err_lista_errada),
          "salvar_ensemble emite [ERRO] com lista sem campos corretos")

  err_nao_lista <- tryCatch(
    salvar_ensemble("nao_eh_lista", tmp_dest),
    error = function(e) e$message
  )
  .assert(grepl("ERRO", err_nao_lista),
          "salvar_ensemble emite [ERRO] quando resultado não é lista")
}

# T13: salvar_ensemble + carregar_ensemble — round-trip completo de predições
test_salvar_carregar_roundtrip_completo <- function() {
  tmp_perm <- file.path(tempdir(), "ensemble_io_roundtrip_perm")
  on.exit(unlink(tmp_perm, recursive = TRUE), add = TRUE)

  salvar_ensemble(.res_base, tmp_perm)
  loaded2 <- carregar_ensemble(tmp_perm)

  dt_new2 <- sintetico_binario(n_pos = 25L, n_neg = 100L, seed = 77L)

  preds_orig <- xgb_predict_ensemble(
    dados_novos       = dt_new2,
    modelos_xgb       = .res_base$modelos,
    metadata_ensemble = .res_base$metadata,
    metodo_combinacao = "media",
    retornar_com_id   = FALSE,
    verbose           = FALSE,
    nthreads          = 1L
  )

  preds_rt <- xgb_predict_ensemble(
    dados_novos       = dt_new2,
    modelos_xgb       = loaded2$modelos,
    metadata_ensemble = loaded2$metadata,
    metodo_combinacao = "media",
    retornar_com_id   = FALSE,
    verbose           = FALSE,
    nthreads          = 1L
  )

  .assert(all(abs(preds_orig - preds_rt) < 1e-6),
          "salvar → carregar → predict produz predições idênticas ao original")
}


# ── Execução ─────────────────────────────────────────────────────────────────
tests <- list(
  "T1:  treino salva arquivos .xgb"                = test_treino_salva_xgb,
  "T2:  treino salva ensemble_metadata.RDS"        = test_treino_salva_metadata_rds,
  "T3:  treino salva importancia_modelos.RDS"      = test_treino_salva_importancia_rds,
  "T4:  carregar retorna estrutura"                = test_carregar_retorna_estrutura,
  "T5:  carregar número e tipo de modelos"         = test_carregar_numero_e_tipo_modelos,
  "T6:  carregar metadata compatível"              = test_carregar_metadata_compativel,
  "T7:  round-trip predições (in-mem vs disco)"    = test_roundtrip_predicoes,
  "T8:  carregar erro pasta inexistente"           = test_carregar_erro_pasta_inexistente,
  "T9:  carregar erro sem metadata.RDS"            = test_carregar_erro_sem_metadata,
  "T10: salvar copia todos os arquivos"            = test_salvar_copia_arquivos,
  "T11: salvar cria folder automaticamente"        = test_salvar_cria_folder_automaticamente,
  "T12: salvar erro resultado inválido"            = test_salvar_erro_resultado_invalido,
  "T13: salvar+carregar round-trip completo"       = test_salvar_carregar_roundtrip_completo
)

resultados <- list(passou = character(0), falhou = character(0))

for (nome in names(tests)) {
  cat(paste0("\n--- ", nome, " ---\n"))
  result <- tryCatch(
    { tests[[nome]](); "passou" },
    error = function(e) {
      cat(paste0("  ", e$message, "\n"))
      "falhou"
    }
  )
  if (result == "passou") {
    resultados$passou <- c(resultados$passou, nome)
  } else {
    resultados$falhou <- c(resultados$falhou, nome)
  }
}

cat("\n\n=============================================\n")
cat(sprintf("RESULTADO: %d/%d testes passaram\n",
            length(resultados$passou),
            length(tests)))
if (length(resultados$falhou) > 0) {
  cat("FALHAS:\n")
  for (f in resultados$falhou) cat(paste0("  ✗ ", f, "\n"))
} else {
  cat("Todos os testes passaram!\n")
}
cat("=============================================\n\n")

# Limpeza
unlink(.tmp_base, recursive = TRUE)
