sourceDir <- paste0("../source/R/")
sourceFiles <- list.files(sourceDir)
for (i in sourceFiles) {
  source(paste0(sourceDir, i))
}

libs <- c("tidyverse")
ipak(libs)

projectFiles <- "~/Dropbox (Princeton)/1_Papers/tpsp/01_data/"
size <- "mid_RUS/"
data_dir_base <- paste0(projectFiles, "data/")
dataPath <- paste0(data_dir_base, size)

ccodes <- read_csv(paste0(dataPath, "ccodes.csv"), col_names=F) %>% pull(.)
tau <- read_csv(paste0(dataPath, "tau.csv"), col_names=F)
M <- read_csv(paste0(dataPath, "M.csv"), col_names=F)
Y <- read_csv(paste0(dataPath, "y.csv"), col_names=F)
W <- read_csv(paste0(dataPath, "cDists.csv"), col_names=F)

colnames(tau) <- colnames(W) <- ccodes

### RESHAPE ###

tau <- cbind(ccodes, tau)
colnames(tau)[1] <- c("j_iso3")
data <- tau %>% pivot_longer(-j_iso3, names_to="i_iso3", values_to="tau")


