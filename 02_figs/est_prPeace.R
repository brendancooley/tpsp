helperPath <- "~/Dropbox (Princeton)/14_Software/R/"
helperFiles <- list.files(helperPath)
for (i in helperFiles) {
  source(paste0(helperPath, i))
}

libs <- c("tidyverse", "patchwork", "reshape2")
ipak(libs)

tpspPath <- "~/Dropbox (Princeton)/1_Papers/tpsp/01_data/"
dataPath <- paste0(tpspPath, "data/mid/")
resultsPath <- paste0(tpspPath, "results/mid/")
estimatesPath <- paste0(resultsPath, "estimates/")

ccodes <- read_csv(paste0(dataPath, "ccodes.csv"), col_names=FALSE) %>% pull(.)
N <- length(ccodes)
year <- read_csv(paste0(dataPath, "year.csv"), col_names=FALSE) %>% pull(.)

peace_probs <- read_csv(paste0(estimatesPath, "peace_probs.csv"), col_names=FALSE) %>% as.matrix()
war_probs <- 1 - peace_probs

for (i in 1:N) {
  war_probs[i, i] = NA
}

rownames(war_probs) <- colnames(war_probs) <- ccodes
war_probs_melted <- melt(war_probs) %>% as_tibble()

min_val <- min(war_probs_melted$value, na.rm=T)
max_val <- max(war_probs_melted$value, na.rm=T)

war_probs_hm <- hm(war_probs_melted, min_val, max_val, x="Attacker", y="Defender", plot_title="Equilibrium War Probabilities")
