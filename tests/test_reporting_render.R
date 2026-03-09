# ============================================================================
# test_reporting_render.R
# Testes TDD para preparar_dados_relatorio() e renderizar_relatorio()
# de R/reporting/reporting_render.R
#
# Cobre:
#   preparar_dados_relatorio:
#     T1  — Cria o arquivo .rds no caminho indicado
#     T2  — RDS contém todos os campos obrigatórios
#     T3  — Campos numéricos das métricas estão em range válido
#     T4  — tabela_decis é data.table com n_decis linhas
#     T5  — curva_roc começa em (0,0) e termina em (1,1)
#     T6  — importancia é data.table com Feature e Gain
#     T7  — Cria o diretório de saída automaticamente se não existir
#     T8  — Erro quando var_target não existe em predicoes_teste
#
#   renderizar_relatorio:
#     T9  — Arquivo HTML é criado no output_dir
#     T10 — HTML gerado não é vazio (> 0 bytes)
#     T11 — HTML gerado contém o título passado via dados$titulo
#     T12 — Lança erro quando template .qmd não existe
#     T13 — Lança erro quando dados_rds não existe
#     T14 — Cria output_dir automaticamente se não existir
#     T15 — output_file customizado → HTML com o nome correto
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

source(file.path(.proj_root, "R", "utils",     "utils_categoricas.R"))
source(file.path(.proj_root, "R", "modeling",   "modeling_xgb_train.R"))
source(file.path(.proj_root, "R", "reporting",  "reporting_metrics.R"))
source(file.path(.proj_root, "R", "reporting",  "reporting_render.R"))
source(file.path(.proj_root, "tests", "helpers", "synthetic_data.R"))

.assert <- function(condicao, msg) {
  if (!condicao) stop(paste0("[FAIL] ", msg))
  cat(paste0("  [OK] ", msg, "\n"))
}

cat("\n======= TESTES: reporting_render =======\n\n")

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures compartilhadas
# ─────────────────────────────────────────────────────────────────────────────

# Template minimal para testes de renderização (não depende de ggplot2/knitr)
.template_minimal <- file.path(.proj_root, "tests", "fixtures", "template_minimal.qmd")

# Dados sintéticos pequenos
set.seed(42)
.dt_full <- sintetico_binario(n_pos = 200L, n_neg = 400L, seed = 42L)

# Split manual 80/20 (sem usar sampling_split para isolar dependências)
.n       <- nrow(.dt_full)
.idx_tr  <- sample(.n, floor(.n * 0.8))
.dt_tr   <- .dt_full[ .idx_tr]
.dt_te   <- .dt_full[-.idx_tr]

# Treinar modelo mínimo
.modelo_obj <- xgb_train(
  dt_treino    = data.table::copy(.dt_tr),
  var_target   = "target",
  vars_excluir = c("id", "regiao"),
  params       = list(
    objective        = "binary:logistic",
    eval_metric      = "auc",
    eta              = 0.15,
    max_leaves       = 8L,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    grow_policy      = "lossguide",
    tree_method      = "hist"
  ),
  nrounds  = 15L,
  nthreads = 1L
)

# Predições de teste
source(file.path(.proj_root, "R", "modeling", "modeling_xgb_predict.R"))
.preds_raw <- xgb_predict(.modelo_obj, .dt_te, var_id = "id")
.predicoes <- merge(.preds_raw, .dt_te[, .(id, target)], by = "id")


# ─────────────────────────────────────────────────────────────────────────────
# preparar_dados_relatorio()
# ─────────────────────────────────────────────────────────────────────────────
cat("--- preparar_dados_relatorio() ---\n")

# T1: cria o arquivo .rds no caminho indicado
test_preparar_cria_rds <- function() {
  tmp_rds <- tempfile(fileext = ".rds")
  on.exit(if (file.exists(tmp_rds)) unlink(tmp_rds))

  preparar_dados_relatorio(
    titulo          = "Teste T1",
    dt_treino       = data.table::copy(.dt_tr),
    dt_teste        = data.table::copy(.dt_te),
    var_target      = "target",
    var_id          = "id",
    modelo_obj      = .modelo_obj,
    predicoes_teste = data.table::copy(.predicoes),
    caminho_saida   = tmp_rds
  )

  .assert(file.exists(tmp_rds), "Arquivo .rds deve ser criado no caminho indicado")
  cat("PASS: test_preparar_cria_rds\n\n")
}

# T2: RDS contém todos os campos obrigatórios
test_preparar_campos_obrigatorios <- function() {
  tmp_rds <- tempfile(fileext = ".rds")
  on.exit(if (file.exists(tmp_rds)) unlink(tmp_rds))

  preparar_dados_relatorio(
    titulo          = "Teste T2",
    descricao       = "Descrição de teste",
    dt_treino       = data.table::copy(.dt_tr),
    dt_teste        = data.table::copy(.dt_te),
    var_target      = "target",
    var_id          = "id",
    modelo_obj      = .modelo_obj,
    predicoes_teste = data.table::copy(.predicoes),
    caminho_saida   = tmp_rds
  )

  dados <- readRDS(tmp_rds)
  campos <- c("titulo", "descricao", "data_geracao",
               "n_treino", "n_teste",
               "dist_target_treino", "dist_target_teste",
               "metricas", "tabela_decis", "curva_roc",
               "importancia", "params_modelo", "features",
               "var_target", "predicoes")

  for (campo in campos) {
    .assert(campo %in% names(dados), paste0("RDS deve conter campo '", campo, "'"))
  }
  .assert(dados$titulo    == "Teste T2",        "titulo deve ser preservado")
  .assert(dados$descricao == "Descrição de teste", "descricao deve ser preservada")
  cat("PASS: test_preparar_campos_obrigatorios\n\n")
}

# T3: campos numéricos das métricas em range válido
test_preparar_metricas_em_range <- function() {
  tmp_rds <- tempfile(fileext = ".rds")
  on.exit(if (file.exists(tmp_rds)) unlink(tmp_rds))

  preparar_dados_relatorio(
    titulo          = "Teste T3",
    dt_treino       = data.table::copy(.dt_tr),
    dt_teste        = data.table::copy(.dt_te),
    var_target      = "target",
    modelo_obj      = .modelo_obj,
    predicoes_teste = data.table::copy(.predicoes),
    caminho_saida   = tmp_rds
  )

  m <- readRDS(tmp_rds)$metricas
  .assert(m$auc  >= 0 && m$auc  <= 1,    "AUC deve estar em [0, 1]")
  .assert(m$ks   >= 0 && m$ks   <= 1,    "KS deve estar em [0, 1]")
  .assert(m$gini >= -1 && m$gini <= 1,   "Gini deve estar em [-1, 1]")
  cat("PASS: test_preparar_metricas_em_range\n\n")
}

# T4: tabela_decis é data.table com 10 linhas
test_preparar_tabela_decis_estrutura <- function() {
  tmp_rds <- tempfile(fileext = ".rds")
  on.exit(if (file.exists(tmp_rds)) unlink(tmp_rds))

  preparar_dados_relatorio(
    titulo          = "Teste T4",
    dt_treino       = data.table::copy(.dt_tr),
    dt_teste        = data.table::copy(.dt_te),
    var_target      = "target",
    modelo_obj      = .modelo_obj,
    predicoes_teste = data.table::copy(.predicoes),
    caminho_saida   = tmp_rds
  )

  td <- readRDS(tmp_rds)$tabela_decis
  .assert(data.table::is.data.table(td), "tabela_decis deve ser data.table")
  .assert(nrow(td) == 10L,               "tabela_decis deve ter 10 linhas (decis)")
  .assert("lift"          %in% names(td), "tabela_decis deve conter 'lift'")
  .assert("captura_acum"  %in% names(td), "tabela_decis deve conter 'captura_acum'")
  cat("PASS: test_preparar_tabela_decis_estrutura\n\n")
}

# T5: curva_roc começa em (0,0) e termina em (1,1)
test_preparar_curva_roc_extremos <- function() {
  tmp_rds <- tempfile(fileext = ".rds")
  on.exit(if (file.exists(tmp_rds)) unlink(tmp_rds))

  preparar_dados_relatorio(
    titulo          = "Teste T5",
    dt_treino       = data.table::copy(.dt_tr),
    dt_teste        = data.table::copy(.dt_te),
    var_target      = "target",
    modelo_obj      = .modelo_obj,
    predicoes_teste = data.table::copy(.predicoes),
    caminho_saida   = tmp_rds
  )

  roc <- readRDS(tmp_rds)$curva_roc
  n   <- nrow(roc)
  .assert(data.table::is.data.table(roc), "curva_roc deve ser data.table")
  .assert(roc$fpr[1] == 0 && roc$tpr[1] == 0, "Curva ROC deve iniciar em (0, 0)")
  .assert(roc$fpr[n] == 1 && roc$tpr[n] == 1, "Curva ROC deve terminar em (1, 1)")
  cat("PASS: test_preparar_curva_roc_extremos\n\n")
}

# T6: importancia é data.table com Feature e Gain
test_preparar_importancia_estrutura <- function() {
  tmp_rds <- tempfile(fileext = ".rds")
  on.exit(if (file.exists(tmp_rds)) unlink(tmp_rds))

  preparar_dados_relatorio(
    titulo          = "Teste T6",
    dt_treino       = data.table::copy(.dt_tr),
    dt_teste        = data.table::copy(.dt_te),
    var_target      = "target",
    modelo_obj      = .modelo_obj,
    predicoes_teste = data.table::copy(.predicoes),
    caminho_saida   = tmp_rds
  )

  imp <- readRDS(tmp_rds)$importancia
  .assert(data.table::is.data.table(imp), "importancia deve ser data.table")
  .assert("Feature" %in% names(imp),      "importancia deve ter coluna 'Feature'")
  .assert("Gain"    %in% names(imp),      "importancia deve ter coluna 'Gain'")
  .assert(nrow(imp) > 0L,                 "importancia deve conter ao menos uma feature")
  cat("PASS: test_preparar_importancia_estrutura\n\n")
}

# T7: cria diretório de saída automaticamente se não existir
test_preparar_cria_diretorio <- function() {
  tmp_dir <- file.path(tempdir(), paste0("rpt_sub_", as.integer(Sys.time())))
  tmp_rds <- file.path(tmp_dir, "dados.rds")
  on.exit(if (dir.exists(tmp_dir)) unlink(tmp_dir, recursive = TRUE))

  preparar_dados_relatorio(
    titulo          = "Teste T7",
    dt_treino       = data.table::copy(.dt_tr),
    dt_teste        = data.table::copy(.dt_te),
    var_target      = "target",
    modelo_obj      = .modelo_obj,
    predicoes_teste = data.table::copy(.predicoes),
    caminho_saida   = tmp_rds
  )

  .assert(dir.exists(tmp_dir),   "Diretório de saída deve ser criado automaticamente")
  .assert(file.exists(tmp_rds),  "Arquivo .rds deve existir dentro do novo diretório")
  cat("PASS: test_preparar_cria_diretorio\n\n")
}

# T8: erro quando var_target não existe em predicoes_teste
test_preparar_erro_target_ausente <- function() {
  pred_sem_target <- data.table::copy(.predicoes)[, target := NULL]

  capturado <- tryCatch({
    preparar_dados_relatorio(
      titulo          = "Teste T8",
      dt_treino       = data.table::copy(.dt_tr),
      dt_teste        = data.table::copy(.dt_te),
      var_target      = "target",
      modelo_obj      = .modelo_obj,
      predicoes_teste = pred_sem_target,
      caminho_saida   = tempfile(fileext = ".rds")
    )
    FALSE
  }, error = function(e) TRUE)

  .assert(capturado, "Deve lançar erro quando var_target não está em predicoes_teste")
  cat("PASS: test_preparar_erro_target_ausente\n\n")
}

test_preparar_cria_rds()
test_preparar_campos_obrigatorios()
test_preparar_metricas_em_range()
test_preparar_tabela_decis_estrutura()
test_preparar_curva_roc_extremos()
test_preparar_importancia_estrutura()
test_preparar_cria_diretorio()
test_preparar_erro_target_ausente()


# ─────────────────────────────────────────────────────────────────────────────
# renderizar_relatorio()
# ─────────────────────────────────────────────────────────────────────────────
cat("--- renderizar_relatorio() ---\n")

# Fixture compartilhada: RDS completo salvo uma vez para todos os testes de render
.tmp_rds_render <- tempfile(fileext = ".rds")
preparar_dados_relatorio(
  titulo          = "Relatório de Teste Quarto",
  descricao       = "Gerado automaticamente pelo suite de testes TDD",
  dt_treino       = data.table::copy(.dt_tr),
  dt_teste        = data.table::copy(.dt_te),
  var_target      = "target",
  var_id          = "id",
  modelo_obj      = .modelo_obj,
  predicoes_teste = data.table::copy(.predicoes),
  caminho_saida   = .tmp_rds_render
)
on.exit(if (file.exists(.tmp_rds_render)) unlink(.tmp_rds_render), add = TRUE)

# T9: arquivo HTML é criado no output_dir
test_render_html_criado <- function() {
  tmp_out <- file.path(tempdir(), paste0("render_t9_", as.integer(Sys.time())))
  on.exit(if (dir.exists(tmp_out)) unlink(tmp_out, recursive = TRUE))

  renderizar_relatorio(
    template_qmd = .template_minimal,
    dados_rds    = .tmp_rds_render,
    output_file  = "relatorio.html",
    output_dir   = tmp_out
  )

  html_path <- file.path(tmp_out, "relatorio.html")
  .assert(file.exists(html_path), "Arquivo HTML deve ser criado no output_dir")
  cat("PASS: test_render_html_criado\n\n")
}

# T10: HTML gerado não está vazio (> 0 bytes)
test_render_html_nao_vazio <- function() {
  tmp_out <- file.path(tempdir(), paste0("render_t10_", as.integer(Sys.time())))
  on.exit(if (dir.exists(tmp_out)) unlink(tmp_out, recursive = TRUE))

  renderizar_relatorio(
    template_qmd = .template_minimal,
    dados_rds    = .tmp_rds_render,
    output_file  = "relatorio.html",
    output_dir   = tmp_out
  )

  html_path  <- file.path(tmp_out, "relatorio.html")
  html_bytes <- file.info(html_path)$size
  .assert(html_bytes > 0L, paste0("HTML não deve ser vazio (", html_bytes, " bytes)"))
  cat("PASS: test_render_html_nao_vazio\n\n")
}

# T11: HTML gerado contém o título passado nos dados
test_render_html_contem_titulo <- function() {
  tmp_out <- file.path(tempdir(), paste0("render_t11_", as.integer(Sys.time())))
  on.exit(if (dir.exists(tmp_out)) unlink(tmp_out, recursive = TRUE))

  renderizar_relatorio(
    template_qmd = .template_minimal,
    dados_rds    = .tmp_rds_render,
    output_file  = "relatorio.html",
    output_dir   = tmp_out
  )

  html_path <- file.path(tmp_out, "relatorio.html")
  html_text <- paste(readLines(html_path, warn = FALSE), collapse = " ")
  .assert(grepl("Relatório de Teste Quarto", html_text, fixed = TRUE),
          "HTML deve conter o título 'Relatório de Teste Quarto'")
  cat("PASS: test_render_html_contem_titulo\n\n")
}

# T12: lança erro quando template .qmd não existe
test_render_erro_template_ausente <- function() {
  capturado <- tryCatch({
    renderizar_relatorio(
      template_qmd = "/nao/existe/template.qmd",
      dados_rds    = .tmp_rds_render,
      output_dir   = tempdir()
    )
    FALSE
  }, error = function(e) TRUE)

  .assert(capturado, "Deve lançar erro quando o template .qmd não existe")
  cat("PASS: test_render_erro_template_ausente\n\n")
}

# T13: lança erro quando dados_rds não existe
test_render_erro_rds_ausente <- function() {
  capturado <- tryCatch({
    renderizar_relatorio(
      template_qmd = .template_minimal,
      dados_rds    = "/nao/existe/dados.rds",
      output_dir   = tempdir()
    )
    FALSE
  }, error = function(e) TRUE)

  .assert(capturado, "Deve lançar erro quando dados_rds não existe")
  cat("PASS: test_render_erro_rds_ausente\n\n")
}

# T14: cria output_dir automaticamente se não existir
test_render_cria_output_dir <- function() {
  tmp_out <- file.path(tempdir(), paste0("render_novo_dir_", as.integer(Sys.time())))
  on.exit(if (dir.exists(tmp_out)) unlink(tmp_out, recursive = TRUE))

  .assert(!dir.exists(tmp_out), "output_dir não deve existir antes do teste")

  renderizar_relatorio(
    template_qmd = .template_minimal,
    dados_rds    = .tmp_rds_render,
    output_file  = "relatorio.html",
    output_dir   = tmp_out
  )

  .assert(dir.exists(tmp_out),
          "output_dir deve ter sido criado automaticamente")
  .assert(file.exists(file.path(tmp_out, "relatorio.html")),
          "HTML deve existir dentro do diretório criado automaticamente")
  cat("PASS: test_render_cria_output_dir\n\n")
}

# T15: output_file customizado gera HTML com o nome correto
test_render_output_file_customizado <- function() {
  tmp_out  <- file.path(tempdir(), paste0("render_t15_", as.integer(Sys.time())))
  nome_html <- paste0("relatorio_custom_", as.integer(Sys.time()), ".html")
  on.exit(if (dir.exists(tmp_out)) unlink(tmp_out, recursive = TRUE))

  renderizar_relatorio(
    template_qmd = .template_minimal,
    dados_rds    = .tmp_rds_render,
    output_file  = nome_html,
    output_dir   = tmp_out
  )

  html_path <- file.path(tmp_out, nome_html)
  .assert(file.exists(html_path),
          paste0("HTML com nome customizado '", nome_html, "' deve existir"))
  cat("PASS: test_render_output_file_customizado\n\n")
}

test_render_html_criado()
test_render_html_nao_vazio()
test_render_html_contem_titulo()
test_render_erro_template_ausente()
test_render_erro_rds_ausente()
test_render_cria_output_dir()
test_render_output_file_customizado()


cat("\n======= TODOS OS TESTES PASSARAM: reporting_render =======\n")
