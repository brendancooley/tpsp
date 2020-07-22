sourceDir <- paste0("../source/R/")
sourceFiles <- list.files(sourceDir)
for (i in sourceFiles) {
  source(paste0(sourceDir, i))
}

libs <- c("tidyverse", "reticulate", 'patchwork', 'reshape2')
ipak(libs)

use_virtualenv("python3")
c_setup <- import_from_path("c_setup", path=".")
setup <- c_setup$setup("local", "mid/")

ccodes <- read_csv(setup$ccodes_path, col_names=FALSE) %>% pull(.)
N <- length(ccodes)
year <- read_csv(paste0(setup$data_path, "year.csv"), col_names=FALSE) %>% pull(.)

pp_star <- read_csv(setup$quantiles_peace_probs_path, col_names=FALSE) %>% as.matrix()

pp_star <- cbind(expand.grid(ccodes, ccodes), pp_star %>% t()) %>% as_tibble()
colnames(pp_star) <- c("i_iso3", "j_iso3", "q025", "q500", "q975")

### COUNTERFACTUAL 2 ###

pp_prime2 <- read_csv(paste0(setup$cfct_china_path, "pp.csv"), col_names=F)

### COUNTERFACTUAL 3 ###

pp_prime3 <- read_csv(paste0(setup$cfct_us_path, "pp.csv"), col_names=F)

### COUNTERFACTUAL 4 ###

pp_prime4 <- read_csv(paste0(setup$cfct_china_v_path, "pp.csv"), col_names=F)
