sourceDir <- paste0("../source/R/")
sourceFiles <- list.files(sourceDir)
for (i in sourceFiles) {
  source(paste0(sourceDir, i))
}

libs <- c("tidyverse")
ipak(libs)

projectFiles <- "~/Dropbox (Princeton)/1_Papers/tpsp/01_data/"