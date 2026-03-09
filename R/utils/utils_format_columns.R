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
