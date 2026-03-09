#' @title Preparação e renderização de relatórios de modelo
#'
#' @description Funções para empacotar os resultados de um modelo treinado
#' em um arquivo RDS padronizado e renderizar o relatório Quarto em HTML.
#'
#' @section Funções disponíveis:
#' \describe{
#'   \item{\code{preparar_dados_relatorio}}{Coleta métricas e salva RDS para o template.}
#'   \item{\code{renderizar_relatorio}}{Renderiza o template .qmd em HTML.}
#' }


# ─── preparar_dados_relatorio ─────────────────────────────────────────────────

#' @title Preparar dados para o relatório de modelo
#'
#' @description Recebe os objetos do pipeline (treino, teste, modelo, predições)
#' e calcula todas as métricas necessárias. Salva um único arquivo \code{.rds}
#' que será lido pelo template Quarto durante a renderização.
#'
#' @param titulo             Título do relatório (character).
#' @param descricao          Descrição curta exibida no cabeçalho. DEFAULT: \code{""}.
#' @param dt_treino          \code{data.table} de treino (original, sem balanceamento).
#' @param dt_teste           \code{data.table} de teste.
#' @param var_target         Nome da coluna target nos dados.
#' @param var_id             Nome da coluna ID. DEFAULT: \code{NULL}.
#' @param busca_estatistica  Resultado de \code{stats_search()}. DEFAULT: \code{NULL}.
#' @param modelo_obj         Objeto retornado por \code{xgb_train()}.
#' @param predicoes_teste    \code{data.table} com colunas \code{predito} e
#'                           \code{var_target} (resultado do merge entre
#'                           \code{xgb_predict()} e os targets do teste).
#' @param ensemble_obj       Objeto retornado por \code{xgb_treino_ensemble()}.
#'                           Quando fornecido, calcula a importância média de
#'                           features em todos os modelos do ensemble.
#'                           DEFAULT: \code{NULL}.
#' @param caminho_saida      Caminho completo para salvar o \code{.rds}.
#'                           DEFAULT: \code{"dados_relatorio.rds"}.
#'
#' @return Invisível: o caminho do arquivo \code{.rds} gerado.
#'
#' @import data.table
#' @export
preparar_dados_relatorio <- function(titulo,
                                     descricao          = "",
                                     dt_treino,
                                     dt_teste,
                                     var_target,
                                     var_id             = NULL,
                                     busca_estatistica  = NULL,
                                     modelo_obj,
                                     predicoes_teste,
                                     ensemble_obj       = NULL,
                                     caminho_saida      = "dados_relatorio.rds") {

  data.table::setDT(dt_treino)
  data.table::setDT(dt_teste)
  data.table::setDT(predicoes_teste)

  # ── Distribuição do target (normaliza nome da coluna para "target") ────────
  dist_treino <- dt_treino[, .N, by = eval(var_target)]
  data.table::setnames(dist_treino, var_target, "target")
  dist_treino[, prop := round(N / sum(N), 4)]

  dist_teste <- dt_teste[, .N, by = eval(var_target)]
  data.table::setnames(dist_teste, var_target, "target")
  dist_teste[, prop := round(N / sum(N), 4)]

  # ── Métricas ───────────────────────────────────────────────────────────────
  metricas <- metricas_binario(predicoes_teste,
                               var_pred   = "predito",
                               var_target = var_target)

  # ── Tabela de decis ────────────────────────────────────────────────────────
  tabela_d <- tabela_decis(predicoes_teste,
                           var_pred   = "predito",
                           var_target = var_target)

  # ── Curva ROC ──────────────────────────────────────────────────────────────
  curva_r <- curva_roc(predicoes_teste,
                       var_pred   = "predito",
                       var_target = var_target)

  # ── Importância das features ───────────────────────────────────────────────
  if (!requireNamespace("xgboost", quietly = TRUE))
    stop("Pacote xgboost necessário para extrair importância das features.")

  imp <- xgboost::xgb.importance(
    model         = modelo_obj$modelo,
    feature_names = modelo_obj$features
  )
  data.table::setDT(imp)
  imp <- imp[order(-Gain)]

  # ── Importância ensemble (média sobre todos os modelos) ────────────────────
  importancia_ensemble <- NULL
  if (!is.null(ensemble_obj)) {
    feat_names <- ensemble_obj$metadata$training_config$colunas_features
    all_imp <- data.table::rbindlist(
      lapply(ensemble_obj$modelos, function(m) {
        i <- xgboost::xgb.importance(model = m, feature_names = feat_names)
        data.table::setDT(i)
        i
      }),
      idcol = "modelo_idx"
    )
    importancia_ensemble <- all_imp[, .(
      Gain_media  = mean(Gain,       na.rm = TRUE),
      Gain_sd     = stats::sd(Gain,  na.rm = TRUE),
      Num_Modelos = .N
    ), by = Feature]
    importancia_ensemble[, Gain_cv := ifelse(
      Gain_media > 0,
      round(Gain_sd / Gain_media, 4),
      NA_real_
    )]
    importancia_ensemble[, Gain_media := round(Gain_media, 6)]
    importancia_ensemble[, Gain_sd    := round(Gain_sd,    6)]
    importancia_ensemble <- importancia_ensemble[order(-Gain_media)]
  }

  # ── Métricas por corte ─────────────────────────────────────────────────────
  metricas_cortes <- metricas_por_cortes(
    predicoes_teste,
    var_pred   = "predito",
    var_target = var_target
  )

  # ── Montar lista final ─────────────────────────────────────────────────────
  dados <- list(
    titulo             = titulo,
    descricao          = descricao,
    data_geracao       = Sys.time(),

    n_treino           = nrow(dt_treino),
    n_ids_treino       = if (!is.null(var_id)) length(unique(dt_treino[[var_id]])) else nrow(dt_treino),
    n_teste            = nrow(dt_teste),
    n_ids_teste        = if (!is.null(var_id)) length(unique(dt_teste[[var_id]])) else nrow(dt_teste),

    dist_target_treino = dist_treino,
    dist_target_teste  = dist_teste,
    taxa_evento_treino = dist_treino[target == 1, prop],
    taxa_evento_teste  = dist_teste[target  == 1, prop],

    busca_estatistica  = busca_estatistica,

    metricas              = metricas,
    metricas_por_cortes   = metricas_cortes,
    tabela_decis          = tabela_d,
    curva_roc             = curva_r,
    importancia           = imp,
    importancia_ensemble  = importancia_ensemble,

    params_modelo      = modelo_obj$params,
    nrounds            = modelo_obj$nrounds,
    features           = modelo_obj$features,
    var_target         = var_target,

    predicoes          = predicoes_teste
  )

  dir_saida <- dirname(caminho_saida)
  if (!dir.exists(dir_saida) && nzchar(dir_saida) && dir_saida != ".")
    dir.create(dir_saida, recursive = TRUE)

  saveRDS(dados, caminho_saida)

  message(paste0("[reporting] Dados salvos em: ", normalizePath(caminho_saida)))
  invisible(caminho_saida)
}


# ─── renderizar_relatorio ────────────────────────────────────────────────────

#' @title Renderizar relatório Quarto em HTML
#'
#' @description Chama \code{quarto::quarto_render()} passando o caminho do
#' arquivo \code{.rds} como parâmetro. O template lê o \code{.rds} e gera
#' um HTML autossuficiente (\code{embed-resources: true}).
#'
#' @param template_qmd Caminho para o arquivo \code{.qmd} do template.
#' @param dados_rds    Caminho para o \code{.rds} gerado por
#'                     \code{preparar_dados_relatorio()}. Será convertido em
#'                     caminho absoluto automaticamente.
#' @param output_file  Nome do arquivo HTML de saída.
#'                     DEFAULT: \code{"relatorio_modelo.html"}.
#' @param output_dir   Diretório de saída. Criado se não existir.
#'                     DEFAULT: \code{"output"} relativo ao projeto.
#'
#' @return Invisível: caminho completo do HTML gerado.
#'
#' @export
renderizar_relatorio <- function(template_qmd,
                                 dados_rds,
                                 output_file = "relatorio_modelo.html",
                                 output_dir  = "output") {

  if (!file.exists(template_qmd))
    stop(paste0("Template não encontrado: ", template_qmd))

  if (!file.exists(dados_rds))
    stop(paste0("Arquivo de dados não encontrado: ", dados_rds))

  if (!requireNamespace("quarto", quietly = TRUE))
    stop("Pacote 'quarto' necessário para renderizar o relatório.")

  dados_rds_abs <- normalizePath(dados_rds, mustWork = TRUE)
  output_dir    <- normalizePath(output_dir, mustWork = FALSE)

  if (!dir.exists(output_dir))
    dir.create(output_dir, recursive = TRUE)

  # Copiar o template para um diretório temporário isolado.
  # O Quarto coloca o HTML no mesmo diretório do .qmd (comportamento padrão).
  # Não passamos output_file ao quarto_render para evitar que o R package
  # resolva o caminho relativo ao CWD, o que causaria path traversal e
  # falhas no embed-resources (bug do quarto R pkg ≤ 1.4 com paths absolutos).
  tmp_dir <- file.path(tempdir(), paste0("qmd_render_", as.integer(Sys.time())))
  dir.create(tmp_dir, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  tmp_qmd <- file.path(tmp_dir, basename(template_qmd))
  file.copy(template_qmd, tmp_qmd, overwrite = TRUE)

  quarto::quarto_render(
    input          = tmp_qmd,
    execute_params = list(dados_rds = dados_rds_abs)
  )

  # O arquivo gerado tem o mesmo nome do .qmd mas com extensão .html
  gerado_nome <- sub("\\.[^.]+$", ".html", basename(template_qmd))
  gerado      <- file.path(tmp_dir, gerado_nome)

  if (!file.exists(gerado))
    stop(paste0("[reporting] Arquivo gerado não encontrado em: ", gerado))

  destino <- file.path(output_dir, output_file)
  file.copy(gerado, destino, overwrite = TRUE)

  message(paste0("[reporting] Relatório gerado: ", destino))
  invisible(destino)
}
