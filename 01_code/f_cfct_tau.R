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
tau_prime3 <- read_csv(paste0(setup$cfct_us_path, "tau.csv"), col_names=F)
tau_prime4 <- read_csv(paste0(setup$cfct_china_v_path, "tau.csv"), col_names=F)

#### CLEAN ####

quantiles_tau <- expand.grid(ccodes, ccodes)
quantiles_tau <- quantiles_tau %>% cbind(q_tau %>% t()) %>% as_tibble()  # i's value for conquering j
colnames(quantiles_tau) <- c("i_iso3", "j_iso3", "tau_q025", "tau_q500", "tau_q975")

quantiles_tau$tau_prime1 <- tau_prime1 %>% as.matrix() %>% t() %>% as.vector()
quantiles_tau$tau_prime2 <- tau_prime2 %>% as.matrix() %>% t() %>% as.vector()
quantiles_tau$tau_prime3 <- tau_prime3 %>% as.matrix() %>% t() %>% as.vector()
quantiles_tau$tau_prime4 <- tau_prime4 %>% as.matrix() %>% t() %>% as.vector()
quantiles_tau <- quantiles_tau %>% filter(i_iso3!=j_iso3)

# quantiles_tau %>% filter(j_iso3=="EU")

### COUNTERFACTUAL 1 ###

min_val <- min(c(quantiles_tau$tau_prime1, quantiles_tau$tau_q500), na.rm=T)
max_val <- max(c(quantiles_tau$tau_prime1, quantiles_tau$tau_q500), na.rm=T)

tau_pp_demilitarization <- quantiles_tau %>% ggplot(aes(x=tau_q500, y=i_iso3, color="black")) + 
  geom_point(size=1) +
  geom_point(aes(x=tau_prime1, y=i_iso3, color=bcOrange), alpha=0) +
  geom_vline(xintercept=1, lty=2) +
  scale_x_continuous(limits=c(min_val, max_val)) +
  scale_color_manual("Equilibrium", values=c(bcOrange, "black"), labels=c("Coercion-Free", "Baseline"), guide="legend") +
  theme_classic() +
  theme(axis.ticks.x=element_blank(),
        axis.text.x=element_blank()) +
  labs(x="Trade Policy", y="Trade Partner", title="Effect of Coercion on Trade Policies ", subtitle=paste0("Changes in Protectionism")) +
  facet_wrap(~j_iso3, nrow=2) +
  scale_x_continuous(limits=c(min_val, max_val)) +
  geom_point(aes(x=tau_prime1, y=i_iso3, color=bcOrange), size=1) +
  geom_segment(aes(x=tau_prime1, xend=tau_q500, y=i_iso3, yend=i_iso3, color=bcOrange))

### COUNTERFACTUAL 2 ###

min_val <- min(c(quantiles_tau$tau_prime2, quantiles_tau$tau_q500), na.rm=T)
max_val <- max(c(quantiles_tau$tau_prime2, quantiles_tau$tau_q500), na.rm=T)

quantiles_tau_china <- quantiles_tau %>% filter(j_iso3=="CHN")
# quantiles_tau_china <- quantiles_tau
tau_pp_china <- quantiles_tau_china %>% ggplot(aes(x=tau_q500, y=i_iso3, color="black")) + 
  geom_point(size=1) +
  geom_point(aes(x=tau_prime2, y=i_iso3, color=bcOrange), alpha=0) +
  geom_vline(xintercept=1, lty=2) +
  scale_x_continuous(limits=c(min_val, max_val)) +
  scale_color_manual("Equilibrium", values=c(bcOrange, "black"), labels=c("2030 Projected Military Capability", "Baseline"), guide="legend") +
  theme_classic() +
  theme(axis.ticks.x=element_blank(),
        axis.text.x=element_blank()) +
  labs(x="Trade Policy", y="Trade Partner", title="Effect of Multipolarization on Chinese Trade Policies ", subtitle=paste0("Changes in Protectionism")) +
  # facet_wrap(~j_iso3, nrow=2) +
  theme(aspect.ratio=1) +
  scale_x_continuous(limits=c(min_val, max_val)) +
  geom_point(aes(x=tau_prime2, y=i_iso3, color=bcOrange), size=1) +
  geom_segment(aes(x=tau_prime2, xend=tau_q500, y=i_iso3, yend=i_iso3), color=bcOrange)

ggsave(setup$f_cfact_china_tau_path)


### COUNTERFACTUAL 3 ###

min_val <- min(c(quantiles_tau$tau_prime3, quantiles_tau$tau_q500), na.rm=T)
max_val <- max(c(quantiles_tau$tau_prime3, quantiles_tau$tau_q500), na.rm=T)

tau_pp_us <- quantiles_tau %>% ggplot(aes(x=tau_q500, y=i_iso3, color="black")) + 
  geom_point(size=1) +
  geom_point(aes(x=tau_prime3, y=i_iso3, color="red"), alpha=0) +
  geom_vline(xintercept=1, lty=2) +
  scale_x_continuous(limits=c(min_val, max_val)) +
  scale_color_manual("Equilibrium", values=c("black", "red"), labels=c("Baseline", "U.S. Retrenchment"), guide="legend") +
  theme_classic() +
  theme(axis.ticks.x=element_blank(),
        axis.text.x=element_blank()) +
  labs(x="Trade Policy", y="Trade Partner", title="Effect of U.S. Retrenchment on Trade Policies ", subtitle=paste0("Changes in Protectionism")) +
  facet_wrap(~j_iso3, nrow=2) +
  scale_x_continuous(limits=c(min_val, max_val)) +
  geom_point(aes(x=tau_prime3, y=i_iso3, color="red"), size=1) +
  geom_segment(aes(x=tau_prime3, xend=tau_q500, y=i_iso3, yend=i_iso3))

### COUNTERFACTUAL 4 ###

min_val <- min(c(quantiles_tau$tau_prime4, quantiles_tau$tau_q500), na.rm=T)
max_val <- max(c(quantiles_tau$tau_prime4, quantiles_tau$tau_q500), na.rm=T)

tau_pp_china_v <- quantiles_tau %>% ggplot(aes(x=tau_q500, y=i_iso3, color="black")) + 
  geom_point(size=1) +
  geom_point(aes(x=tau_prime4, y=i_iso3, color="red"), alpha=0) +
  geom_vline(xintercept=1, lty=2) +
  scale_x_continuous(limits=c(min_val, max_val)) +
  scale_color_manual("Equilibrium", values=c("black", "red"), labels=c("Baseline", "Chinese Preference Liberalization"), guide="legend") +
  theme_classic() +
  theme(axis.ticks.x=element_blank(),
        axis.text.x=element_blank()) +
  labs(x="Trade Policy", y="Trade Partner", title="Effect of Chinese Preference Liberalization on Trade Policies ", subtitle=paste0("Changes in Protectionism")) +
  facet_wrap(~j_iso3, nrow=2) +
  scale_x_continuous(limits=c(min_val, max_val)) +
  geom_point(aes(x=tau_prime4, y=i_iso3, color="red"), size=1) +
  geom_segment(aes(x=tau_prime4, xend=tau_q500, y=i_iso3, yend=i_iso3))

