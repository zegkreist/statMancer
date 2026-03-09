as_date <- function(dates, format = "%Y-%m-%d", origin = "1970-01-01", ...){
  if (is(dates, "Date")) return(dates)
  
  if (length(dates) == 0) return(as.Date(character(0)))
  
  if (is.integer(dates) | is.numeric(dates))
    return(as.Date(dates, origin = origin, ...))
  else
    if (is.character(dates))
      return(as.Date(lubridate::fast_strptime(dates, format = format, ...)))
  else
    return(as.Date(dates, ...))
}

tolower_column <- function(dt, nome_col) {
  dt <- data.table::copy(dt)
  data.table::setDT(dt)
  dt[, (nome_col) := tolower(get(nome_col))]
  return(dt)
}

toupper_column <- function(dt, nome_col) {
  dt <- data.table::copy(dt)
  data.table::setDT(dt)
  dt[, (nome_col) := toupper(get(nome_col))]
  return(dt)
}

format_integer <- function(dt, nome_col) {
  dt <- data.table::copy(dt)
  data.table::setDT(dt)
  dt[, (nome_col) := as.integer(get(nome_col))]
  return(dt)
}

format_date <- function(dt, nome_col) {
  dt <- data.table::copy(dt)
  data.table::setDT(dt)
  if (all(is.na(dt[, get(nome_col)]))){
    dt[, (nome_col) := as_date(NA)]
  } else {
    
    dt[, (nome_col) := as_date(get(nome_col),
                               format = lubridate::guess_formats(x = get(nome_col),
                                                                 orders = c("%d/%m/%Y", "%Y-%m-%d", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S")) |>  unique())]
  }
  return(dt)
}

#' @title Normalizar nomes de colunas de um data.table
#'
#' @description Converte todos os nomes de colunas para snake_case lowercase,
#' substituindo espaços e separadores por "_" e removendo qualquer caractere
#' que não seja letra, número ou underscore.
#'
#' Regras aplicadas em ordem:
#'   1. Converte para minúsculas
#'   2. Substitui espaços por "_"
#'   3. Remove caracteres que não sejam letras, números ou "_"
#'   4. Colapsa múltiplos "_" consecutivos em um único "_"
#'   5. Remove "_" no início e no fim
#'
#' @param dt data.table (ou data.frame) cujos nomes serão normalizados.
#'   O objeto original não é modificado (copy interno).
#'
#' @return data.table com os nomes de colunas normalizados.
#'
#' @examples
#' dt <- data.table::data.table(`Nome Completo` = 1, `Idade (anos)` = 2, SCORE_A = 3)
#' normalizar_nomes_colunas(dt)
#' # => colunas: nome_completo, idade_anos, score_a
#'
#' @export
format_nomes_colunas <- function(dt) {
  dt <- data.table::copy(dt)
  data.table::setDT(dt)

  novos_nomes <- names(dt)
  novos_nomes <- tolower(novos_nomes)
  novos_nomes <- gsub(" ", "_", novos_nomes, fixed = TRUE)
  novos_nomes <- gsub("[^a-z0-9_]", "", novos_nomes)
  novos_nomes <- gsub("_+", "_", novos_nomes)
  novos_nomes <- gsub("^_|_$", "", novos_nomes)

  data.table::setnames(dt, names(dt), novos_nomes)
  return(dt)
}

#' @title Deletar colunas de um data.table
#'
#' @description Remove as colunas indicadas pelo nome. Colunas inexistentes
#' são silenciosamente ignoradas. O objeto original não é modificado.
#'
#' @param dt data.table (ou data.frame) do qual remover colunas.
#' @param colunas Vetor de caracteres com os nomes das colunas a remover.
#'   Nomes que não existam no dt são ignorados sem erro.
#'
#' @return data.table sem as colunas removidas.
#'
#' @examples
#' dt <- data.table::data.table(id = 1L, score = 0.5, ruido = 99L)
#' format_delete_columns(dt, c("ruido", "nao_existe"))
#' # => colunas: id, score
#'
#' @export
format_delete_columns <- function(dt, colunas) {
  dt <- data.table::copy(dt)
  data.table::setDT(dt)

  para_remover <- intersect(colunas, names(dt))
  if (length(para_remover) > 0L)
    dt[, (para_remover) := NULL]

  return(dt)
}
