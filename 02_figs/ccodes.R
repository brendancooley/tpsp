# helperPath <- "~/Dropbox (Princeton)/14_Software/R/"
# helperFiles <- list.files(helperPath)
# for (i in helperFiles) {
#   source(paste0(helperPath, i))
# }
# 
# libs <- c("tidyverse", "countrycode", "knitr", "kableExtra", "latex2exp")
# ipak(libs)
# 
# tpspPath <- "~/Dropbox (Princeton)/1_Papers/tpsp/01_data/"
# dataPath <- paste0(tpspPath, "tpsp_data_mini/")
# resultsPath <- paste0(tpspPath, "results_mini/")
# 
# ccodes <- read_csv(paste0(dataPath, "ccodes.csv"), col_names=FALSE)
# b_tilde <- as_tibble(sample(seq(0, 1, .1), N))

ccodesT <- ccodes
N <- nrow(ccodes)
colnames(ccodesT) <- c("iso3")
ccodesT$`Country Name` <- countrycode(ccodesT$iso3, "iso3c", "country.name")
ccodesT$`Country Name` <- ifelse(ccodesT$iso3=="EU", "European Union", ccodesT$`Country Name`)
ccodesT$`Country Name` <- ifelse(ccodesT$iso3=="ROW", "Rest of World", ccodesT$`Country Name`)


b_estsT <- cbind(ccodesT, b_tilde)
colnames(b_estsT) <- c("iso3", "Country Name", "$\\tilde{\\boldsymbol{b}}_i$")
