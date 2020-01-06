# helperPath <- "~/Dropbox (Princeton)/14_Software/R/"
# helperFiles <- list.files(helperPath)
# for (i in helperFiles) {
#   source(paste0(helperPath, i))
# }
# 
# libs <- c("tidyverse", "countrycode", "knitr", "kableExtra")
# ipak(libs)

# ccodes <- read_csv(paste0(dataPath, "ccodes.csv"), col_names=FALSE)

ccodesT <- ccodes
colnames(ccodesT) <- c("iso3")
ccodesT$`Country Name` <- countrycode(ccodesT$iso3, "iso3c", "country.name")
ccodesT$`Country Name` <- ifelse(ccodesT$iso3=="EU", "European Union", ccodesT$`Country Name`)
ccodesT$`Country Name` <- ifelse(ccodesT$iso3=="ROW", "Rest of World", ccodesT$`Country Name`)
