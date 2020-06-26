sourceDir <- paste0("../source/R/")
sourceFiles <- list.files(sourceDir)
for (i in sourceFiles) {
  source(paste0(sourceDir, i))
}

libs <- c("tidyverse", "reticulate", 'patchwork', 'reshape2', 'knitr', "kableExtra", "countrycode", "magick")
ipak(libs)

use_virtualenv("python3")
c_setup <- import_from_path("c_setup", path=".")
setup <- c_setup$setup("local", "mid/")

ccodes <- read_csv(setup$ccodes_path, col_names=FALSE)

ccodesT <- ccodes
N <- nrow(ccodes)
colnames(ccodesT) <- c("iso3")
ccodesT$`Country Name` <- countrycode(ccodesT$iso3, "iso3c", "country.name")
ccodesT$`Country Name` <- ifelse(ccodesT$iso3=="EU", "European Union", ccodesT$`Country Name`)
ccodesT$`Country Name` <- ifelse(ccodesT$iso3=="RoW", "Rest of World", ccodesT$`Country Name`)

ccodes_table <- kable(ccodesT, "latex", booktabs = T, caption = "In-Sample Countries", escape = FALSE) %>% kable_styling(position = "center", latex_options=c("striped"))


save_kable(ccodes_table, setup$f_ccodes_path)
