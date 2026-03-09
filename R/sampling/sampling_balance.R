#' @title Balanceamento de amostras por ID para modelagem
#'
#' @description Ferramentas de up & down sampling operando ao nível do var_id,
#' garantindo que todos os registros de um mesmo ID fiquem no mesmo conjunto
#' amostrado — evitando vazamento de dados entre amostras.
#'
#' A unidade de amostragem é sempre o ID. Para cada ID, deve existir um único
#' valor de target. Se houver múltiplos registros por ID (ex: múltiplos eventos),
#' todos os registros do ID amostrado são retornados.
#'
#' @section Funções disponíveis:
#' \describe{
#'   \item{\code{upsample}}{Sobreamostragem da classe minoritária com reposição.}
#'   \item{\code{downsample}}{Subamostragem da classe majoritária sem reposição.}
#' }


# ─── upsample ────────────────────────────────────────────────────────────────

#' @title Upsample — sobreamostragem da classe minoritária
#'
#' @description Amostra IDs com reposição até que cada classe possua
#' \code{n_por_classe} IDs únicos, depois retorna todos os registros
#' correspondentes. Suporta estratificação adicional por outras variáveis.
#'
#' @param dt            Um \code{data.table} com os dados.
#' @param var_id        Nome da coluna de identificação única (character).
#' @param var_target    Nome da coluna target (character). Deve ter um valor
#'                      único por ID.
#' @param n_por_classe  Número de IDs amostrados por classe. DEFAULT: tamanho
#'                      da classe majoritária (balanceia sem perda de dados).
#' @param var_estratificacao  Vetor de nomes de colunas para estratificação
#'                            proporcional dentro de cada classe.
#'                            DEFAULT: \code{NULL}.
#' @param seed          Semente para reprodutibilidade. DEFAULT: \code{NULL}.
#'
#' @return Um \code{data.table} com todos os registros dos IDs amostrados.
#'
#' @import data.table
#' @export
upsample <- function(dt,
                     var_id,
                     var_target,
                     n_por_classe       = NULL,
                     var_estratificacao = NULL,
                     seed               = NULL) {

  data.table::setDT(dt)
  dt <- data.table::copy(dt)

  stopifnot(
    "var_id não encontrado"    = var_id    %in% names(dt),
    "var_target não encontrado" = var_target %in% names(dt)
  )

  if (!is.null(seed)) set.seed(seed)

  # Tabela de IDs únicos com target (e colunas de estratificação se houver)
  cols_id <- unique(c(var_id, var_target, var_estratificacao))
  dt_ids  <- unique(dt[, cols_id, with = FALSE], by = var_id)

  # Validar unicidade de target por ID
  target_por_id <- dt_ids[, .N, by = var_id]
  if (any(target_por_id$N > 1))
    stop("Existem IDs com mais de um valor de target. Corrija antes de amostrar.")

  # Número de IDs por classe
  contagem_classes <- dt_ids[, .N, by = var_target]

  if (is.null(n_por_classe))
    n_por_classe <- max(contagem_classes$N)

  ids_amostrados <- .amostrar_ids(
    dt_ids             = dt_ids,
    var_id             = var_id,
    var_target         = var_target,
    n_por_classe       = n_por_classe,
    var_estratificacao = var_estratificacao,
    replace            = TRUE
  )

  # Retornar todos os registros dos IDs amostrados (com duplicatas se houver repeat)
  result <- data.table::rbindlist(
    lapply(ids_amostrados, function(id_val) dt[get(var_id) == id_val]),
    use.names = TRUE
  )

  return(result)
}


# ─── downsample ──────────────────────────────────────────────────────────────

#' @title Downsample — subamostragem da classe majoritária
#'
#' @description Amostra IDs sem reposição até que cada classe possua
#' \code{n_por_classe} IDs, depois retorna todos os registros correspondentes.
#' Suporta estratificação adicional por outras variáveis.
#'
#' @param dt            Um \code{data.table} com os dados.
#' @param var_id        Nome da coluna de identificação única (character).
#' @param var_target    Nome da coluna target (character). Deve ter um valor
#'                      único por ID.
#' @param n_por_classe  Número de IDs amostrados por classe. DEFAULT: tamanho
#'                      da classe minoritária (balanceia sem perda de dados).
#' @param var_estratificacao  Vetor de nomes de colunas para estratificação
#'                            proporcional dentro de cada classe.
#'                            DEFAULT: \code{NULL}.
#' @param seed          Semente para reprodutibilidade. DEFAULT: \code{NULL}.
#'
#' @return Um \code{data.table} com todos os registros dos IDs amostrados.
#'
#' @import data.table
#' @export
downsample <- function(dt,
                       var_id,
                       var_target,
                       n_por_classe       = NULL,
                       var_estratificacao = NULL,
                       seed               = NULL) {

  data.table::setDT(dt)
  dt <- data.table::copy(dt)

  stopifnot(
    "var_id não encontrado"    = var_id    %in% names(dt),
    "var_target não encontrado" = var_target %in% names(dt)
  )

  if (!is.null(seed)) set.seed(seed)

  cols_id <- unique(c(var_id, var_target, var_estratificacao))
  dt_ids  <- unique(dt[, cols_id, with = FALSE], by = var_id)

  target_por_id <- dt_ids[, .N, by = var_id]
  if (any(target_por_id$N > 1))
    stop("Existem IDs com mais de um valor de target. Corrija antes de amostrar.")

  contagem_classes <- dt_ids[, .N, by = var_target]

  if (is.null(n_por_classe))
    n_por_classe <- min(contagem_classes$N)

  classes_insuf <- contagem_classes[N < n_por_classe, get(var_target)]
  if (length(classes_insuf) > 0)
    warning(paste0(
      "Classes com menos IDs do que n_por_classe (serão amostradas com reposição): ",
      paste(classes_insuf, collapse = ", ")
    ))

  ids_amostrados <- .amostrar_ids(
    dt_ids             = dt_ids,
    var_id             = var_id,
    var_target         = var_target,
    n_por_classe       = n_por_classe,
    var_estratificacao = var_estratificacao,
    replace            = FALSE
  )

  result <- dt[get(var_id) %in% ids_amostrados]

  return(result)
}


# ─── helper interno ───────────────────────────────────────────────────────────

#' Amostragem interna de IDs por classe (com ou sem estratificação)
#'
#' @noRd
.amostrar_ids <- function(dt_ids,
                           var_id,
                           var_target,
                           n_por_classe,
                           var_estratificacao,
                           replace) {

  if (is.null(var_estratificacao)) {
    # Amostragem simples por classe
    ids_por_classe <- dt_ids[, {
      ids_classe <- get(var_id)
      n_disp     <- length(ids_classe)
      rep_flag   <- replace || (n_disp < n_por_classe)
      list(ids = sample(ids_classe, n_por_classe, replace = rep_flag))
    }, by = var_target]

    return(ids_por_classe$ids)
  }

  # Amostragem estratificada: manter proporção dos estratos dentro de cada classe
  strat_vars      <- c(var_target, var_estratificacao)
  contagem_strat  <- dt_ids[, .N, by = strat_vars]
  contagem_strat[, prop   := N / sum(N), by = var_target]
  contagem_strat[, n_alvo := pmax(1L, as.integer(round(prop * n_por_classe)))]

  ids_list <- lapply(seq_len(nrow(contagem_strat)), function(i) {
    filtro <- contagem_strat[i]

    # Filtrar dt_ids pelo estrato
    subset_ids <- dt_ids
    for (col in strat_vars) {
      val        <- filtro[[col]]
      subset_ids <- subset_ids[get(col) == val]
    }

    n_alvo   <- filtro$n_alvo
    n_disp   <- nrow(subset_ids)
    if (n_disp == 0 || n_alvo == 0) return(character(0))

    rep_flag <- replace || (n_disp < n_alvo)
    sample(subset_ids[[var_id]], n_alvo, replace = rep_flag)
  })

  return(unlist(ids_list))
}
