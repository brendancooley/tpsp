eu15sy <- 1995
eu15ey <- 2003
EU15 <- c('AUT', 'BEL', 'BNL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD',
          'POR', 'SWE')

eu25sy <- 2004
eu25ey <- 2006
EU25 <- c('AUT', 'BEL', 'BNL', 'CYP', 'CZE', 'DEU', 'DNK', 'ELL', 'ESP', 'EST', 'FIN', 'FRA', 'GBR', 'GRC', 'HUN',
          'IRL', 'ITA', 'LTU', 'LUX', 'LVA', 'MLT', 'NLD', 'POL', 'PRT', 'SVK', 'SVN', 'SWE')

eu27sy <- 2007
eu27ey <- 2013
EU27 <- c('AUT', 'BEL', 'BGR', 'BNL', 'CYP', 'CZE', 'DEU', 'DNK', 'ELL', 'ESP', 'EST', 'FIN', 'FRA', 'GBR', 'GRC', 'HUN',
          'IRL', 'ITA', 'LTU', 'LUX', 'LVA', 'NLD', 'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'SWE')

eu28sy <- 2014
eu28ey <- 2018
EU28 <- c('AUT', 'BEL', 'BGR', 'BNL', 'CYP', 'CZE', 'DEU', 'DNK', 'ELL', 'ESP', 'EST', 'FIN', 'FRA', 'GBR', 'GRC', 'HRV', 'HUN',
          'IRL', 'ITA', 'LTU', 'LUX', 'LVA', 'NLD', 'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'SWE')

mapEU <- function(country, year) {
  Y <- unique(year)
  out <- country
  
  for (i in Y) {
    if (i %in% seq(eu15sy, eu15ey)) {
      out <- ifelse(out %in% EU15 & year == i, 'EU', out)
    }
    if (i %in% seq(eu25sy, eu25ey)) {
      out <- ifelse(out %in% EU25 & year == i, 'EU', out)
    }
    if (i %in% seq(eu27sy, eu27ey)) {
      out <- ifelse(out %in% EU27 & year == i, 'EU', out)
    }
    if (i %in% seq(eu28sy, eu28ey)) {
      out <- ifelse(out %in% EU27 & year == i, 'EU', out)
    }
  }
  # for (i in seq(eu15sy, eu15ey)) {
  #   out <- ifelse(out %in% EU15 & year == i, 'EU', out)
  # }
  # for (i in seq(eu25sy, eu25ey)) {
  #   out <- ifelse(out %in% EU25 & year == i, 'EU', out)
  # }
  # for (i in seq(eu27sy, eu27ey)) {
  #   out <- ifelse(out %in% EU27 & year == i, 'EU', out)
  # }
  return(out)
}

