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

quantiles_peace_probs <- read_csv(setup$quantiles_peace_probs_path, col_names=FALSE)

quantiles_pp <- expand.grid(ccodes, ccodes)
quantiles_pp <- quantiles_pp %>% cbind(quantiles_peace_probs %>% t()) %>% as_tibble()  # i's value for conquering j
colnames(quantiles_pp) <- c("Var2", "Var1", "pp_q025", "pp_q500", "pp_q975")

quantiles_pp$value <- 1 - quantiles_pp$pp_q500

# for (i in 1:N) {
#   war_probs[i, i] = NA
# }
# 
# rownames(war_probs) <- colnames(war_probs) <- ccodes
# war_probs_melted <- melt(war_probs) %>% as_tibble()
war_probs_melted <- quantiles_pp %>% select(Var1, Var2, value)

min_val <- min(war_probs_melted$value, na.rm=T)
max_val <- max(war_probs_melted$value, na.rm=T)

war_probs_hm <- hm(war_probs_melted, min_val, max_val, x="Attacker", y="Defender", plot_title="Point Estimates: Equilibrium War Probabilities")
