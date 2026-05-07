#' @title Separação de dados em Treino e Teste
#'
#' @description Ferramentas para separar dados em conjuntos de treino e teste
#' operando ao nível do \code{var_id} — todos os registros de um mesmo ID
#' ficam obrigatoriamente no mesmo conjunto, evitando vazamento de dados.
#'
#' @section Funções disponíveis:
#' \describe{
#'   \item{\code{train_test_split}}{Split simples ou estratificado por ID.}
#' }


# ─── train_test_split ─────────────────────────────────────────────────────────

#' @title Split treino/teste baseado em ID
#'
#' @description Divide os dados em treino e teste sortendo IDs únicos, de modo
#' que todos os registros de um mesmo ID fiquem no mesmo conjunto. Suporta
#' estratificação por uma variável (ex: target) para manter proporções.
#'
#' Quando \code{var_id} contém mais de uma coluna, cada coluna é tratada como
#' uma dimensão de restrição **independente**: nenhum valor de qualquer dessas
#' colunas pode aparecer simultaneamente em treino e em teste. A divisão é
#' feita sobre **componentes conexos** entre os IDs — se \code{short_id=A}
#' pertence a \code{cpf=1} e \code{short_id=B} também pertence a \code{cpf=1},
#' então A e B são colocados no mesmo conjunto, independentemente de seus valores.
#' O algoritmo usa propagação de rótulos (label propagation) sobre \code{data.table},
#' que converge em poucas iterações mesmo para tabelas grandes.
#'
#' @param dt                  Um \code{data.table} com os dados.
#' @param var_id              Nome (ou vetor de nomes) da(s) coluna(s) de
#'                            identificação. Com uma coluna: comportamento original.
#'                            Com múltiplas colunas: restrição independente por
#'                            dimensão — nenhum valor de qualquer coluna vaza
#'                            entre treino e teste.
#' @param prop_treino         Proporção de IDs destinados ao treino (0 < x < 1).
#'                            DEFAULT: \code{0.8}.
#' @param var_estratificacao  Nome de uma variável para estratificação do split.
#'                            Garante que a distribuição dessa variável seja
#'                            proporcional em treino e teste. DEFAULT: \code{NULL}.
#' @param seed                Semente para reprodutibilidade. DEFAULT: \code{NULL}.
#'
#' @return Uma lista nomeada com dois \code{data.table}:
#'   \item{\code{$treino}}{Dados de treino.}
#'   \item{\code{$teste}}{Dados de teste.}
#'
#' @import data.table
#' @export
train_test_split <- function(dt,
                             var_id,
                             prop_treino        = 0.8,
                             var_estratificacao = NULL,
                             seed               = NULL) {

  data.table::setDT(dt)
  dt <- data.table::copy(dt)

  stopifnot(
    "Todas as colunas de var_id devem existir no data.table" = all(var_id %in% names(dt)),
    "prop_treino deve estar entre 0 e 1"                     = prop_treino > 0 & prop_treino < 1
  )

  if (!is.null(var_estratificacao) && !var_estratificacao %in% names(dt))
    stop(paste0("Coluna de estratificação '", var_estratificacao, "' não encontrada."))

  if (!is.null(seed)) set.seed(seed)

  if (length(var_id) == 1L) {
    # ── Caminho rápido: ID único — comportamento original ─────────────────
    if (is.null(var_estratificacao)) {
      ids_treino <- .split_simples(dt, var_id, prop_treino)
    } else {
      ids_treino <- .split_estratificado(dt, var_id, prop_treino, var_estratificacao)
    }
    treino <- dt[get(var_id) %in%  ids_treino]
    teste  <- dt[!get(var_id) %in% ids_treino]
  } else {
    # ── Múltiplos IDs: componentes conexos ────────────────────────────────
    # Cada coluna de var_id é uma restrição independente: nenhum valor de
    # qualquer dimensão pode aparecer em treino E em teste ao mesmo tempo.
    # Registros ligados transitivamente por qualquer ID compartilhado são
    # agrupados no mesmo componente e alocados juntos.
    result <- .split_multi_id(dt, var_id, prop_treino, var_estratificacao)
    treino <- result$treino
    teste  <- result$teste
  }

  return(list(treino = treino, teste = teste))
}


# ─── helpers internos ────────────────────────────────────────────────────────

#' Split multi-ID via componentes conexos (label propagation)
#'
#' Para cada linha em dt, os valores das colunas var_ids estão "ligados".
#' O algoritmo propaga iterativamente o menor rótulo (string) de componente
#' através de cada coluna de ID até convergência — equivalente a encontrar
#' componentes conexos num grafo bipartido entre os IDs.
#' A divisão treino/teste é feita sobre os componentes resultantes.
#' @noRd
.split_multi_id <- function(dt, var_ids, prop_treino, var_estratificacao) {
  COMP <- ".comp_."

  # Combinações únicas de todos os IDs presentes nos dados
  dt_u <- unique(dt[, var_ids, with = FALSE])

  # Inicializa rótulo de componente como string do ID primário
  dt_u[, (COMP) := as.character(get(var_ids[1L]))]

  # Propagação de rótulos: em cada iteração, para cada coluna de ID,
  # substitui o rótulo de cada linha pelo menor rótulo do mesmo grupo.
  # Repete até não haver mais mudanças (convergência garantida).
  converged <- FALSE
  while (!converged) {
    prev <- dt_u[[COMP]]
    for (col in var_ids) {
      dt_u[, (COMP) := min(get(COMP)), by = col]
    }
    converged <- identical(prev, dt_u[[COMP]])
  }

  componentes <- unique(dt_u[[COMP]])

  # ── Estratificação ───────────────────────────────────────────────────────
  if (!is.null(var_estratificacao)) {
    # Associa o ID primário ao seu rótulo de componente e ao target;
    # usa classe dominante (moda) por componente para estratificação.
    dt_join <- merge(
      unique(dt_u[, c(var_ids[1L], COMP), with = FALSE]),
      unique(dt[, c(var_ids[1L], var_estratificacao), with = FALSE]),
      by    = var_ids[1L],
      all.x = TRUE
    )
    dt_comp_strat <- dt_join[, .(
      estrato = {
        tbl <- table(get(var_estratificacao))
        if (length(tbl) == 0L) NA_character_
        else names(tbl)[which.max(tbl)]
      }
    ), by = COMP]

    comps_treino <- dt_comp_strat[, {
      comps <- get(COMP)
      n_t   <- pmax(1L, round(length(comps) * prop_treino))
      list(sel = sample(comps, n_t, replace = FALSE))
    }, by = estrato][, sel]
  } else {
    n_t          <- round(length(componentes) * prop_treino)
    comps_treino <- sample(componentes, n_t, replace = FALSE)
  }

  # IDs primários de cada partição
  ids_treino <- unique(dt_u[get(COMP) %in%  comps_treino, get(var_ids[1L])])

  list(
    treino = dt[get(var_ids[1L]) %in%  ids_treino],
    teste  = dt[!get(var_ids[1L]) %in% ids_treino]
  )
}

#' Split simples: sorteia IDs sem estratificação
#' @noRd
.split_simples <- function(dt, var_id, prop_treino) {
  ids_unicos <- unique(dt[[var_id]])
  n_treino   <- round(length(ids_unicos) * prop_treino)
  sample(ids_unicos, n_treino, replace = FALSE)
}

#' Split estratificado: mantém proporção da var_estratificacao em treino e teste
#' @noRd
.split_estratificado <- function(dt, var_id, prop_treino, var_estratificacao) {

  # Um registro por ID para determinar o estrato
  dt_ids <- unique(dt[, c(var_id, var_estratificacao), with = FALSE], by = var_id)

  # Sortear dentro de cada estrato
  ids_treino <- dt_ids[, {
    ids_estrato <- get(var_id)
    n_treino    <- pmax(1L, round(length(ids_estrato) * prop_treino))
    list(id_sel = sample(ids_estrato, n_treino, replace = FALSE))
  }, by = var_estratificacao][, id_sel]

  return(ids_treino)
}


# ─── kfold_split ─────────────────────────────────────────────────────────────

#' @title K-Fold Cross-Validation baseado em ID
#'
#' @description Divide os IDs em k folds mutuamente exclusivos. Retorna uma
#' lista de k elementos, cada um com \code{$treino} e \code{$teste}. O fold
#' \code{i} usa as observações do fold \code{i} como teste e o restante como
#' treino. Todos os registros de um mesmo ID ficam no mesmo conjunto.
#'
#' @param dt                  Um \code{data.table} com os dados.
#' @param var_id              Nome da coluna de identificação única (character).
#' @param k                   Número de folds. DEFAULT: \code{5}.
#' @param var_estratificacao  Nome de uma variável para estratificação dos folds
#'                            (ex: target). Mantém distribuição proporcional em
#'                            cada fold. DEFAULT: \code{NULL}.
#' @param seed                Semente para reprodutibilidade. DEFAULT: \code{NULL}.
#'
#' @return Lista de \code{k} elementos, cada um com:
#'   \item{\code{$treino}}{Dados de treino do fold.}
#'   \item{\code{$teste}}{Dados de teste do fold.}
#'   \item{\code{$fold}}{Índice do fold (1..k).}
#'
#' @import data.table
#' @export
kfold_split <- function(dt,
                        var_id,
                        k                  = 5L,
                        var_estratificacao = NULL,
                        seed               = NULL) {

  data.table::setDT(dt)
  dt <- data.table::copy(dt)

  stopifnot(
    "var_id não encontrado"    = var_id %in% names(dt),
    "k deve ser >= 2"          = is.numeric(k) && k >= 2
  )
  k <- as.integer(k)

  if (!is.null(var_estratificacao) && !var_estratificacao %in% names(dt))
    stop(paste0("Coluna de estratificação '", var_estratificacao, "' não encontrada."))

  if (!is.null(seed)) set.seed(seed)

  ids_unicos <- unique(dt[[var_id]])
  n_ids      <- length(ids_unicos)

  if (n_ids < k)
    stop(paste0("Número de IDs (", n_ids, ") é menor que k (", k, ")."))

  if (is.null(var_estratificacao)) {
    fold_assign <- .assign_folds_simples(ids_unicos, k, var_id)
  } else {
    dt_ids      <- unique(dt[, c(var_id, var_estratificacao), with = FALSE], by = var_id)
    fold_assign <- .assign_folds_estratificado(dt_ids, var_id, var_estratificacao, k)
  }

  lapply(seq_len(k), function(i) {
    ids_teste  <- fold_assign[fold == i,  get(var_id)]
    ids_treino <- fold_assign[fold != i,  get(var_id)]
    list(
      treino = dt[get(var_id) %in% ids_treino],
      teste  = dt[get(var_id) %in% ids_teste],
      fold   = i
    )
  })
}


# ─── helpers internos de kfold ───────────────────────────────────────────────

#' @noRd
.assign_folds_simples <- function(ids, k, var_id) {
  n   <- length(ids)
  dt  <- data.table::data.table(v_ = sample(ids),
                                 fold = rep_len(seq_len(k), n))
  data.table::setnames(dt, "v_", var_id)
  dt
}

#' @noRd
.assign_folds_estratificado <- function(dt_ids, var_id, var_estratificacao, k) {
  estratos <- unique(dt_ids[[var_estratificacao]])

  fold_list <- lapply(estratos, function(estrato) {
    ids_estrato <- dt_ids[get(var_estratificacao) == estrato, get(var_id)]
    n_e         <- length(ids_estrato)
    data.table::data.table(v_ = sample(ids_estrato),
                           fold = rep_len(seq_len(k), n_e))
  })

  result <- data.table::rbindlist(fold_list)
  data.table::setnames(result, "v_", var_id)
  result
}
