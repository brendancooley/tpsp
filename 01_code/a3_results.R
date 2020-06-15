sourceDir <- paste0("../source/R/")
sourceFiles <- list.files(sourceDir)
for (i in sourceFiles) {
  source(paste0(sourceDir, i))
}

source("00_params.R")

libs <- c("tidyverse")
ipak(libs)

#### DATA ####

quantiles_alpha1 <- read_csv(paste0(estimatesPath, "quantiles_alpha1.csv"), col_names=F)
quantiles_alpha2 <- read_csv(paste0(estimatesPath, "quantiles_alpha2.csv"), col_names=F)
quantiles_gamma <- read_csv(paste0(estimatesPath, "quantiles_gamma.csv"), col_names=F)
quantiles_v <- read_csv(paste0(estimatesPath, "quantiles_v.csv"), col_names=F)

ccodes <- read_csv(paste0(dataPath, "ccodes.csv"), col_names=FALSE)

#### CLEAN ####

mil_params <- rbind(quantiles_alpha1 %>% t(), quantiles_alpha2 %>% t(), quantiles_gamma %>% t()) %>% as_tibble()
colnames(mil_params) <- c("q025", "q500", "q975")
mil_params$coef <- c("alpha1", "alpha2", "gamma")
mil_params$coef_name <- c("Distance", "Attacker GDP", "Mil Capability Ratio")

mil_params$coef_name <- fct_reorder(mil_params$coef_name, mil_params$q500)

pref_params <- quantiles_v %>% t() %>% as_tibble()
colnames(pref_params) <- c("q025", "q500", "q975")
pref_params$coef <- ccodes %>% pull()

pref_params$coef <- fct_reorder(pref_params$coef, pref_params$q500)

### PLOTS ####

pref_plot <- ggplot(pref_params, aes(x=q500, y=coef)) +
  geom_point() +
  geom_vline(xintercept=1, lty=2) +
  geom_segment(aes(xend=q025, x=q975, yend=coef, y=coef)) +
  theme_classic() +
  labs(x="Estimate", y="Country", title="Preferences for Protectionism", subtitle="Point estimates and 95 percent confidence interval")

mil_plot <- ggplot(mil_params, aes(x=q500, y=coef_name)) +
  geom_point() +
  geom_vline(xintercept=0, lty=2) +
  geom_segment(aes(xend=q025, x=q975, yend=coef_name, y=coef_name)) +
  theme_classic() +
  labs(x="Estimate", y="Coefficient", title="War Costs", subtitle="Point estimates and 95 percent confidence intervals")
