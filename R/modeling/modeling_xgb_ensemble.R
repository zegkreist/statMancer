# ============================================================================
# modeling_xgb_ensemble.R
#
# Ensemble de XGBoost com reamostragem por ID.
# Versão local (sem dependência de S3), orientada ao padrão statMancer.
#
# Estratégias disponíveis:
#   "upsample"   — usa upsample()  de sampling_balance.R (minoritária sobe à maioritária)
#   "downsample" — usa downsample() de sampling_balance.R (maioritária desce à minoritária)
#   "bootstrap"  — reamostragem de IDs com reposição preservando distribuição natural
#
# Dependências (devem ser carregadas antes deste arquivo):
#   source("R/utils/utils_modelagem.R")   → converter_para_numerico_seguro,
#                                           gerar_parametros_estocasticos, rnorm_t
#   source("R/sampling/sampling_balance.R") → upsample, downsample
# ============================================================================


# ─── Funções internas ─────────────────────────────────────────────────────────

#' Reamostragem de um data.table ao nível do ID
#'
#' @keywords internal
.amostrar_ensemble <- function(dt, var_id, var_target, metodo, n_por_classe, seed) {

  if (metodo == "upsample") {
    return(upsample(dt,
                    var_id       = var_id,
                    var_target   = var_target,
                    n_por_classe = n_por_classe,
                    seed         = seed))
  }

  if (metodo == "downsample") {
    return(downsample(dt,
                      var_id       = var_id,
                      var_target   = var_target,
                      n_por_classe = n_por_classe,
                      seed         = seed))
  }

  # metodo == "bootstrap"
  # Amostra IDs com reposição (sem estratificação por classe).
  # Preserva a distribuição natural de classes — comportamento de bootstrap clássico.
  # n_por_classe aqui é reutilizado como n_total de IDs a amostrar.
  set.seed(seed)
  all_ids <- unique(dt[[var_id]])
  n_boot  <- if (!is.null(n_por_classe)) as.integer(n_por_classe) else length(all_ids)

  ids_sampled <- sample(all_ids, n_boot, replace = TRUE)

  # Expande: cada ocorrência de um ID traz todos os seus registros
  idx_dt <- data.table::data.table(.boot_seq = seq_along(ids_sampled))
  idx_dt[, (var_id) := ids_sampled]

  result <- merge(idx_dt, dt, by = var_id, allow.cartesian = TRUE)
  result[, .boot_seq := NULL]
  data.table::setDT(result)
  return(result)
}


# ─── xgb_treino_ensemble ─────────────────────────────────────────────────────

#' @title Treino Ensemble de XGBoost com Reamostragem
#'
#' @description Treina \code{n_models} modelos XGBoost em amostras independentes
#' do conjunto de treino, combinando:
#' \itemize{
#'   \item Reamostragem ao nível do ID (upsample, downsample ou bootstrap) para
#'         garantir diversidade e balanceamento por iteração.
#'   \item Perturbação estocástica dos hiperparâmetros (via
#'         \code{gerar_parametros_estocasticos}) para diversificar as árvores.
#'   \item Validação interna por fold com early stopping opcional.
#' }
#' Os modelos são salvos localmente em \code{folder_saida} (sem dependência de S3).
#' O objeto retornado é compatível com \code{xgb_predict_ensemble()} usando os campos
#' \code{$modelos} e \code{$metadata}.
#'
#' @param dt                   \code{data.table} com os dados de treino.
#' @param var_id               Nome da coluna de ID único (character).
#' @param var_target           Nome da coluna target binária 0/1 (character).
#' @param vars_excluir         Vetor de colunas adicionais a excluir das features.
#'                             DEFAULT: \code{NULL}.
#' @param parametros_treino    Lista \code{list(parametros = list(...), nrounds = N)}
#'                             retornada por \code{xgb_select_params} ou equivalente.
#' @param n_models             Número de modelos no ensemble. DEFAULT: \code{25L}.
#' @param metodo_reamostragem  Estratégia de reamostragem por iteração:
#'   \itemize{
#'     \item \code{"upsample"}: sobreamostra IDs da classe minoritária até a maioritária.
#'     \item \code{"downsample"}: subamostra IDs da classe maioritária até a minoritária.
#'     \item \code{"bootstrap"}: reamostra IDs com reposição preservando distribuição
#'           natural (sem balanceamento forçado).
#'   }
#'   DEFAULT: \code{"upsample"}.
#' @param n_por_classe         IDs a amostrar por classe por iteração.
#'                             DEFAULT: \code{NULL} (tamanho da classe maioritária
#'                             para upsample/bootstrap; minoritária para downsample).
#' @param folder_saida         Pasta local para persistir modelos (.xgb) e metadados (.RDS).
#'                             DEFAULT: subpasta temporária com timestamp.
#' @param early_stopping_rounds Parada antecipada usando validação interna. DEFAULT: \code{50L}.
#' @param validation_split     Fração de IDs reservada para validação interna por iteração.
#'                             DEFAULT: \code{0.15}.
#' @param seed                 Semente base; cada iteração \code{j} usa \code{seed + j}.
#'                             DEFAULT: \code{123L}.
#' @param log_every            Frequência de log a cada N modelos. DEFAULT: \code{5L}.
#' @param nthreads             Threads para XGBoost. DEFAULT: \code{detectCores() - 1}.
#'
#' @return Lista com quatro campos:
#'   \itemize{
#'     \item \code{modelos}:      Lista de \code{xgb.Booster} (carregados em memória).
#'     \item \code{metadata}:     Lista com \code{training_config}, \code{performance_metrics},
#'                                \code{dataset_info}, etc. Compatível com
#'                                \code{xgb_predict_ensemble()}.
#'     \item \code{importance}:   \code{data.table} com importância média das features
#'                                agregada de todos os modelos.
#'     \item \code{folder_saida}: Caminho da pasta com os arquivos persistidos.
#'   }
#'
#' @examples
#' \dontrun{
#' library(data.table)
#' library(xgboost)
#' source("R/utils/utils_modelagem.R")
#' source("R/sampling/sampling_balance.R")
#' source("R/modeling/modeling_xgb_ensemble.R")
#'
#' params <- list(
#'   parametros = list(
#'     objective = "binary:logistic", eval_metric = "auc",
#'     eta = 0.1, max_leaves = 32L, gamma = 0.1,
#'     subsample = 0.8, colsample_bytree = 0.8,
#'     grow_policy = "lossguide", tree_method = "hist"
#'   ),
#'   nrounds = 150L
#' )
#'
#' resultado <- xgb_treino_ensemble(
#'   dt                  = meus_dados,
#'   var_id              = "id_beneficiario",
#'   var_target          = "target",
#'   vars_excluir        = c("data_referencia", "regiao"),
#'   parametros_treino   = params,
#'   n_models            = 25L,
#'   metodo_reamostragem = "upsample",
#'   folder_saida        = "output/ensemble_dcv"
#' )
#'
#' # Predição com o ensemble
#' preds <- xgb_predict_ensemble(
#'   dados_novos       = dados_novos,
#'   modelos_xgb       = resultado$modelos,
#'   metadata_ensemble = resultado$metadata
#' )
#' }
#'
#' @import data.table xgboost
#' @export
xgb_treino_ensemble <- function(
    dt,
    var_id,
    var_target,
    vars_excluir          = NULL,
    parametros_treino,
    n_models              = 25L,
    metodo_reamostragem   = c("upsample", "downsample", "bootstrap"),
    n_por_classe          = NULL,
    folder_saida          = NULL,
    early_stopping_rounds = 50L,
    validation_split      = 0.15,
    seed                  = 123L,
    log_every             = 5L,
    nthreads              = max(1L, parallel::detectCores() - 1L)
) {
  metodo_reamostragem <- match.arg(metodo_reamostragem)

  requireNamespace("data.table")
  requireNamespace("xgboost")

  # ── Validações de entrada ──────────────────────────────────────────────────
  if (!data.table::is.data.table(dt))
    dt <- data.table::as.data.table(dt)

  if (!var_id %in% names(dt))
    stop(paste0("[ERRO] Coluna var_id nao encontrada: '", var_id, "'"))

  if (!var_target %in% names(dt))
    stop(paste0("[ERRO] Coluna var_target nao encontrada: '", var_target, "'"))

  if (is.null(parametros_treino) ||
      !all(c("parametros", "nrounds") %in% names(parametros_treino)))
    stop("[ERRO] parametros_treino deve ser list(parametros = list(...), nrounds = N)")

  if (n_models < 1L)
    stop("[ERRO] n_models deve ser >= 1")

  if (validation_split < 0 || validation_split >= 1)
    stop("[ERRO] validation_split deve estar em [0, 1)")

  # ── Definição de features ──────────────────────────────────────────────────
  cols_excluir_total <- unique(c(var_id, var_target, vars_excluir))
  colunas_features   <- setdiff(names(dt), cols_excluir_total)

  if (length(colunas_features) == 0L)
    stop("[ERRO] Nenhuma feature disponível após exclusões.")

  cat(paste0("[ENSEMBLE] Método: ", metodo_reamostragem,
             " | Modelos: ", n_models,
             " | Features: ", length(colunas_features), "\n"))
  cat(paste0("[ENSEMBLE] Dataset: ", nrow(dt), " obs | ",
             data.table::uniqueN(dt[[var_id]]), " IDs únicos\n"))
  cat(paste0("[ENSEMBLE] Distribuição target: "))
  print(table(dt[[var_target]]))

  # ── Pasta de saída ─────────────────────────────────────────────────────────
  if (is.null(folder_saida)) {
    ts           <- format(Sys.time(), "%Y%m%d_%H%M%S")
    folder_saida <- file.path(tempdir(), paste0("xgb_ensemble_", ts))
  }
  if (!dir.exists(folder_saida))
    dir.create(folder_saida, recursive = TRUE, showWarnings = FALSE)

  cat(paste0("[ENSEMBLE] Salvando em: ", folder_saida, "\n\n"))

  # ── Estruturas de resultado ────────────────────────────────────────────────
  modelos          <- vector("list", n_models)
  importance_list  <- vector("list", n_models)
  performance_list <- vector("list", n_models)

  maximize_metric <- isTRUE(
    parametros_treino$parametros$eval_metric %in% c("auc", "aucpr", "ndcg", "map")
  )

  t0 <- Sys.time()

  # ── Loop principal ─────────────────────────────────────────────────────────
  for (j in seq_len(n_models)) {

    if (j %% log_every == 0L || j == 1L) {
      elapsed <- round(as.numeric(Sys.time() - t0, units = "mins"), 1)
      cat(paste0("[ENSEMBLE] Modelo ", j, "/", n_models, " | ", elapsed, " min\n"))
    }

    # ── Reamostragem ao nível do ID ──────────────────────────────────────────
    amostra <- tryCatch(
      .amostrar_ensemble(dt, var_id, var_target, metodo_reamostragem, n_por_classe, seed + j),
      error = function(e) {
        cat(paste0("[WARN] Reamostragem falhou no modelo ", j, ": ", e$message, "\n"))
        NULL
      }
    )
    if (is.null(amostra) || nrow(amostra) == 0L) next

    # ── Divisão treino / validação interna ────────────────────────────────────
    if (validation_split > 0) {
      ids_unicos <- unique(amostra[[var_id]])
      n_val_ids  <- max(1L, floor(length(ids_unicos) * validation_split))
      set.seed(seed + j + 10000L)
      ids_val    <- sample(ids_unicos, n_val_ids)
      treino_dt  <- amostra[!get(var_id) %in% ids_val]
      val_dt     <- amostra[ get(var_id) %in% ids_val]
    } else {
      treino_dt  <- amostra
      val_dt     <- NULL
    }

    if (nrow(treino_dt) == 0L) next

    # ── Preparação para XGBoost ───────────────────────────────────────────────
    treino_num <- converter_para_numerico_seguro(treino_dt[, ..colunas_features])
    y_treino   <- as.numeric(treino_dt[[var_target]])

    dtrain    <- xgboost::xgb.DMatrix(as.matrix(treino_num), label = y_treino)
    watchlist <- list(train = dtrain)

    if (!is.null(val_dt) && nrow(val_dt) > 0L) {
      val_num <- converter_para_numerico_seguro(val_dt[, ..colunas_features])
      y_val   <- as.numeric(val_dt[[var_target]])
      dval    <- xgboost::xgb.DMatrix(as.matrix(val_num), label = y_val)
      watchlist$eval <- dval
    }

    # ── Perturbação estocástica dos hiperparâmetros ───────────────────────────
    tmp_params            <- gerar_parametros_estocasticos(parametros_treino$parametros)
    tmp_params$base_score <- mean(y_treino)

    # ── Treino ───────────────────────────────────────────────────────────────
    md <- tryCatch(
      xgboost::xgb.train(
        data                  = dtrain,
        params                = tmp_params,
        nrounds               = parametros_treino$nrounds,
        nthread               = nthreads,
        verbose               = 0,
        watchlist             = watchlist,
        early_stopping_rounds = early_stopping_rounds,
        maximize              = maximize_metric
      ),
      error = function(e) {
        cat(paste0("[WARN] Treino falhou no modelo ", j, ": ", e$message, "\n"))
        NULL
      }
    )
    if (is.null(md)) next

    # ── Persistir modelo ──────────────────────────────────────────────────────
    fname <- file.path(folder_saida, sprintf("xgb_model_%03d.xgb", j))
    xgboost::xgb.save(md, fname = fname)
    modelos[[j]] <- md

    # ── Importância das features ──────────────────────────────────────────────
    tryCatch({
      importance_list[[j]] <- data.table::as.data.table(
        xgboost::xgb.importance(model = md)
      )
    }, error = function(e) invisible(NULL))

    # ── Métricas de validação interna ─────────────────────────────────────────
    if (!is.null(val_dt) && nrow(val_dt) > 0L) {
      eval_log <- md$evaluation_log
      if (!is.null(eval_log) && nrow(eval_log) > 0L) {
        best_iter <- if (!is.null(md$best_iteration) && md$best_iteration > 0L)
          md$best_iteration else nrow(eval_log)
        performance_list[[j]] <- list(
          model_id       = j,
          best_iteration = best_iter,
          n_treino       = nrow(treino_dt),
          n_val          = nrow(val_dt)
        )
      }
    }

    # ── Limpeza de memória ────────────────────────────────────────────────────
    rm(md, dtrain)
    if (exists("dval")) rm(dval)
    invisible(gc(verbose = FALSE, reset = TRUE))
  }

  # ── Filtrar modelos válidos ────────────────────────────────────────────────
  modelos_validos <- modelos[!sapply(modelos, is.null)]
  cat(paste0("\n[ENSEMBLE] Modelos válidos: ",
             length(modelos_validos), "/", n_models, "\n"))

  # ── Agregar importâncias ──────────────────────────────────────────────────
  imp_validas <- importance_list[!sapply(importance_list, is.null)]
  importance_agregada <- if (length(imp_validas) > 0L) {
    imp_all <- data.table::rbindlist(imp_validas, use.names = TRUE, fill = TRUE)
    imp_all[, .(
      Gain_medio   = mean(Gain,   na.rm = TRUE),
      Gain_mediana = median(Gain, na.rm = TRUE),
      frequencia   = .N
    ), by = Feature][order(-Gain_medio)]
  } else {
    data.table::data.table()
  }

  # ── Metadata ─────────────────────────────────────────────────────────────
  # Estrutura compatível com xgb_predict_ensemble()
  metadata <- list(
    n_models            = length(modelos_validos),
    training_config     = list(
      var_id                = var_id,
      var_target            = var_target,
      vars_excluir          = vars_excluir,
      colunas_features      = colunas_features,
      metodo_reamostragem   = metodo_reamostragem,
      n_por_classe          = n_por_classe,
      early_stopping_rounds = early_stopping_rounds,
      validation_split      = validation_split,
      seed                  = seed,
      nthreads              = nthreads,
      # Alias para compatibilidade com xgb_predict_ensemble()
      coluna_id             = var_id,
      colunas_excluir       = cols_excluir_total
    ),
    parametros_treino   = parametros_treino,
    performance_metrics = performance_list[!sapply(performance_list, is.null)],
    dataset_info        = list(
      n_obs       = nrow(dt),
      n_ids       = data.table::uniqueN(dt[[var_id]]),
      n_features  = length(colunas_features),
      target_dist = table(dt[[var_target]])
    ),
    folder_saida        = folder_saida,
    tempo_total_min     = round(as.numeric(Sys.time() - t0, units = "mins"), 2),
    timestamp           = Sys.time()
  )

  # ── Persistir metadados e importâncias ────────────────────────────────────
  base::saveRDS(metadata,        file = file.path(folder_saida, "ensemble_metadata.RDS"))
  base::saveRDS(importance_list, file = file.path(folder_saida, "importancia_modelos.RDS"))

  cat(paste0("[ENSEMBLE] Concluído em ", metadata$tempo_total_min, " min\n"))
  cat(paste0("[ENSEMBLE] Arquivos em: ", folder_saida, "\n"))

  return(list(
    modelos      = modelos_validos,
    metadata     = metadata,
    importance   = importance_agregada,
    folder_saida = folder_saida
  ))
}


# ─── carregar_ensemble ────────────────────────────────────────────────────────

#' @title Carrega um ensemble do disco
#'
#' @description Lê todos os arquivos \code{.xgb} e o \code{ensemble_metadata.RDS}
#' de uma pasta gerada por \code{xgb_treino_ensemble()} e reconstrói o objeto
#' compatível com \code{xgb_predict_ensemble()}.
#'
#' @param folder_saida Caminho da pasta que contém os arquivos do ensemble.
#'
#' @return Lista com \code{modelos} (lista de \code{xgb.Booster}),
#'   \code{metadata} e \code{folder_saida}.
#'
#' @export
carregar_ensemble <- function(folder_saida) {
  if (!dir.exists(folder_saida))
    stop(paste0("[ERRO] Pasta do ensemble não encontrada: ", folder_saida))

  meta_path <- file.path(folder_saida, "ensemble_metadata.RDS")
  if (!file.exists(meta_path))
    stop(paste0("[ERRO] ensemble_metadata.RDS não encontrado em: ", folder_saida))

  metadata <- base::readRDS(meta_path)

  xgb_files <- sort(list.files(folder_saida, pattern = "\\.xgb$", full.names = TRUE))
  if (length(xgb_files) == 0L)
    warning("[WARN] Nenhum arquivo .xgb encontrado em: ", folder_saida)

  modelos <- lapply(xgb_files, function(f) {
    tryCatch(
      xgboost::xgb.load(f),
      error = function(e) {
        warning(paste0("[WARN] Falha ao carregar: ", basename(f), " — ", e$message))
        NULL
      }
    )
  })
  modelos <- modelos[!sapply(modelos, is.null)]

  cat(paste0("[ENSEMBLE] Carregados ", length(modelos),
             " modelo(s) de: ", folder_saida, "\n"))

  list(modelos = modelos, metadata = metadata, folder_saida = folder_saida)
}


# ─── salvar_ensemble ─────────────────────────────────────────────────────────

#' @title Copia um ensemble treinado para uma nova pasta
#'
#' @description Copia todos os arquivos da pasta original do ensemble
#' (\code{resultado$folder_saida}) para \code{folder_destino}, criando a pasta
#' se necessário. Útil para mover o ensemble de um diretório temporário para
#' um local permanente.
#'
#' @param resultado      Objeto retornado por \code{xgb_treino_ensemble()}.
#' @param folder_destino Caminho da pasta de destino.
#'
#' @return Invisível: \code{folder_destino}.
#'
#' @export
salvar_ensemble <- function(resultado, folder_destino) {
  if (!is.list(resultado) ||
      !all(c("modelos", "metadata", "folder_saida") %in% names(resultado)))
    stop("[ERRO] 'resultado' deve ser o retorno de xgb_treino_ensemble()")

  if (!dir.exists(folder_destino))
    dir.create(folder_destino, recursive = TRUE, showWarnings = FALSE)

  arquivos <- list.files(resultado$folder_saida, full.names = TRUE)
  if (length(arquivos) == 0L) {
    warning("[WARN] Nenhum arquivo encontrado em folder_saida: ", resultado$folder_saida)
    return(invisible(folder_destino))
  }

  copiados <- file.copy(
    arquivos,
    file.path(folder_destino, basename(arquivos)),
    overwrite = TRUE
  )

  n_ok <- sum(copiados)
  cat(paste0("[ENSEMBLE] ", n_ok, "/", length(arquivos),
             " arquivo(s) copiado(s) para: ", folder_destino, "\n"))

  invisible(folder_destino)
}
