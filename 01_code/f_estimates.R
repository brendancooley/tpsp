sourceDir <- paste0("../source/R/")
sourceFiles <- list.files(sourceDir)
for (i in sourceFiles) {
  source(paste0(sourceDir, i))
}

libs <- c("tidyverse", "reticulate")
ipak(libs)

use_virtualenv("python3")
c_setup <- import_from_path("c_setup", path=".")
setup <- c_setup$setup("local", "mid/")

setup2 <- c_setup$setup("local", "mid/", mil_off=TRUE)

#### DATA ####

quantiles_alpha1 <- read_csv(setup$quantiles_alpha1_path, col_names=F)
quantiles_alpha2 <- read_csv(setup$quantiles_alpha2_path, col_names=F)
quantiles_gamma <- read_csv(setup$quantiles_gamma_path, col_names=F)
quantiles_v <- read_csv(setup$quantiles_v_path, col_names=F)
quantiles_v_mil_off <- read_csv(setup2$quantiles_v_path, col_names=F)

ccodes <- read_csv(setup$ccodes_path, col_names=FALSE)

#### CLEAN ####

mil_params <- rbind(quantiles_alpha1 %>% t(), quantiles_alpha2 %>% t(), quantiles_gamma %>% t()) %>% as_tibble()
colnames(mil_params) <- c("q025", "q500", "q975")
mil_params$coef <- c("alpha1", "alpha2", "gamma")
mil_params$coef_name <- c("Distance", "Attacker GDP", "Mil Capability Ratio")

mil_params$coef_name <- fct_reorder(mil_params$coef_name, mil_params$q500)

pref_params <- quantiles_v %>% t() %>% as_tibble()
colnames(pref_params) <- c("q025", "q500", "q975")
pref_params$coef <- ccodes %>% pull()
# pref_params$model <- "mil_on"

pref_params$coef <- fct_reorder(pref_params$coef, rev(pref_params$coef))

pref_params_mo <- quantiles_v_mil_off %>% t() %>% as_tibble()
colnames(pref_params_mo) <- c("mo_q025", "mo_q500", "mo_q975")
pref_params_mo$coef <- ccodes %>% pull()
# pref_params_mo$model <- "mil_off"

pref_params_mo <- left_join(pref_params, pref_params_mo, by=c("coef"))
pref_params_mo$coef <- fct_reorder(pref_params_mo$coef, rev(pref_params_mo$coef))

### PLOTS ####

pref_plot <- ggplot(pref_params, aes(x=q500, y=coef)) +
  geom_point() +
  geom_vline(xintercept=1, lty=2) +
  geom_segment(aes(xend=q025, x=q975, yend=coef, y=coef)) +
  theme_classic() +
  labs(x="Estimate", y="Country", title="Preferences for Protectionism", subtitle="Point estimates and 95 percent confidence interval")

ggsave(setup$f_estimates_pref_path, width=7, height=3.5)

pref_plot_mil_off <- ggplot(pref_params_mo, aes(x=q500, y=coef)) +
  geom_point(aes(color='black')) +
  geom_vline(xintercept=1, lty=2) +
  geom_segment(aes(xend=q025, x=q975, yend=coef, y=coef, color="black")) +
  geom_point(aes(x=mo_q500, y=coef, color="other"), position=position_nudge(y=-.25)) +
  geom_segment(aes(xend=mo_q025, x=mo_q975, yend=coef, y=coef, color="other"), position=position_nudge(y=-.25)) +
  scale_colour_manual(name='Model', values =c('other'=bcOrange,'black'='black'), labels=c('Coercion','Coercion-Free')) +
  theme_classic() +
  labs(x="Estimate", y="Country", title="Preferences for Protectionism", subtitle="Point estimates and 95 percent confidence interval")

ggsave(setup$f_estimates_pref_mo_path, width=7, height=3.5)

mil_plot <- ggplot(mil_params, aes(x=q500, y=coef_name)) +
  geom_point() +
  geom_vline(xintercept=0, lty=2) +
  geom_segment(aes(xend=q025, x=q975, yend=coef_name, y=coef_name)) +
  theme_classic() +
  labs(x="Estimate", y="Coefficient", title="War Costs", subtitle="Point estimates and 95 percent confidence intervals")

ggsave(setup$f_estimates_mil_path, width=7, height=3.5)

