#### SETUP ####

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

#### DATA ####

ccodes <- read_csv(setup$ccodes_path, col_names=F) %>% pull(.)
tau <- read_csv(setup$tau_path, col_names=F)

q_tau <- read_csv(setup$quantiles_tau_path, col_names=F)

#### CLEAN ####

colnames(tau) <- ccodes

tau <- cbind(ccodes, tau)
colnames(tau)[1] <- c("j_iso3")
tau_long <- tau %>% pivot_longer(-j_iso3, names_to="i_iso3", values_to="tau")

quantiles_tau <- expand.grid(ccodes, ccodes)
quantiles_tau <- quantiles_tau %>% cbind(q_tau %>% t()) %>% as_tibble()  # i's value for conquering j
colnames(quantiles_tau) <- c("i_iso3", "j_iso3", "tau_q025", "tau_q500", "tau_q975")

quantiles_tau <- quantiles_tau %>% left_join(tau_long) %>% filter(i_iso3 != j_iso3)
quantiles_tau %>% print(n=100)

### PLOTS ###

tau_max <- max(c(quantiles_tau$tau_q500, quantiles_tau$tau))
tau_min <- min(c(quantiles_tau$tau_q025, quantiles_tau$tau))
tau_cor <- lm(data=quantiles_tau, tau_q500 ~ tau)

# 45 degree
ggplot(data=quantiles_tau, aes(x=tau, y=tau_q500)) +
  geom_segment(aes(xend=tau, x=tau, yend=tau_q975, y=tau_q025), color="lightgray", size=.5) +
  geom_point(size=1) +
  geom_abline(slope=1, intercept=0) +
  geom_smooth(method="lm", se=FALSE, color="red", size=.5) +
  geom_vline(xintercept=1, lty=2) +
  geom_hline(yintercept=1, lty=2) +
  theme_classic() +
  lims(x=c(tau_min, tau_max), y=c(tau_min, tau_max)) +
  labs(x="Barriers to Trade: Data", y="Barriers to Trade: Model (Point estimates and 95% CIs)", title="Model Fit", subtitle="Correlation between empirical trade barriers and model predictions") +
  theme(aspect.ratio=1)

# epsilons by magnitude

quantiles_tau$epsilon <- quantiles_tau$tau_q500 - quantiles_tau$tau
quantiles_tau$ddyad <- paste0(quantiles_tau$j_iso3, "-", quantiles_tau$i_iso3)
# quantiles_tau <- quantiles_tau %>% arrange(epsilon)
quantiles_tau$ddyad <- fct_reorder(quantiles_tau$ddyad, quantiles_tau$epsilon)

ggplot(data=quantiles_tau, aes(x=epsilon, y=ddyad)) +
  geom_point() +
  geom_segment(aes(x=epsilon, xend=0, y=ddyad, yend=ddyad)) +
  geom_vline(xintercept=0, lty=2) +
  labs(x="Barriers to Trade: Median Predictive Error (Model Prediction - Data)", y="Importer-Exporter", title="Model Fit", subtitle="Median Predictive Error") +
  theme_bw()

  