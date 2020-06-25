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

q_rcv <- read_csv(setup$quantiles_rcv_path, col_names=FALSE)

quantiles_rcv <- expand.grid(ccodes, ccodes)
quantiles_rcv <- quantiles_rcv %>% cbind(q_rcv %>% t()) %>% as_tibble()  # i's value for conquering j
colnames(quantiles_rcv) <- c("Var2", "Var1", "rcv_q025", "rcv_q500", "rcv_q975")

quantiles_rcv$value <- ifelse(quantiles_rcv$rcv_q500 > 1, quantiles_rcv$rcv_q500 - 1, 0)

rcv_melted <- quantiles_rcv %>% select(Var1, Var2, value)

min_val <- min(rcv_melted$value, na.rm=T)
max_val <- max(rcv_melted$value, na.rm=T)

rcv_hm <- hm(rcv_melted, min_val, max_val, x="Attacker", y="Defender", plot_title="Point Estimates: Positive Equilibrium War Values")
