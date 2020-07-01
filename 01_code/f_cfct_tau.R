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

q_tau <- read_csv(setup$quantiles_tau_path, col_names=F)

tau_prime1 <- read_csv(paste0(setup$cfct_demilitarization_path, "tau.csv"), col_names=F)
tau_prime2 <- read_csv(paste0(setup$cfct_china_path, "tau.csv"), col_names=F)

#### CLEAN ####

quantiles_tau <- expand.grid(ccodes, ccodes)
quantiles_tau <- quantiles_tau %>% cbind(q_tau %>% t()) %>% as_tibble()  # i's value for conquering j
colnames(quantiles_tau) <- c("i_iso3", "j_iso3", "tau_q025", "tau_q500", "tau_q975")

quantiles_tau$tau_prime1 <- tau_prime1 %>% as.matrix() %>% t() %>% as.vector()
quantiles_tau$tau_prime2 <- tau_prime2 %>% as.matrix() %>% t() %>% as.vector()

### COUNTERFACTUAL 2 ###

tau_pp_china <- quantiles_tau %>% ggplot(aes(x=tau_q500, y=i_iso3, color="black")) + 
  geom_point(size=1) +
  geom_point(aes(x=tau_prime2, y=i_iso3, color="red"), alpha=0) +
  scale_x_continuous(limits=c(min_val, max_val)) +
  scale_color_manual("Equilibrium", values=c("black", "red"), labels=c("Coercive", "Non-Coercive"), guide="legend") +
  theme_classic() +
  theme(axis.ticks.x=element_blank(),
        axis.text.x=element_blank()) +
  labs(x="Trade Flows", y="Trade Partner", title="Effect of Coercion on International Trade", subtitle=paste0("Change in (Log) Imports")) +
  facet_wrap(~Var2, nrow=2)