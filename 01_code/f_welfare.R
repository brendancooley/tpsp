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

q_G <- read_csv(setup$quantiles_Ghat_path, col_names=FALSE)
q_U <- read_csv(setup$quantiles_Uhat1_path, col_names=FALSE)

quantiles_G <- data.frame(ccodes)

quantiles_G <- quantiles_G %>% cbind(q_G %>% t()) %>% as_tibble()
colnames(quantiles_G) <- c("iso3", "G_q025", "G_q500", "G_q975")
quantiles_U <- q_U %>% t() %>% as_tibble()
colnames(quantiles_U) <- c("U_q025", "U_q500", "U_q975")

welfare <- cbind(quantiles_G, quantiles_U)

### COUNTERFACTUAL 1 ###

G_prime <- read_csv(paste0(setup$cfct_demilitarization_path, "G_hat.csv"), col_names=FALSE)
colnames(G_prime) <- "G_prime"
U_prime <- read_csv(paste0(setup$cfct_demilitarization_path, "U_hat.csv"), col_names=FALSE)
colnames(U_prime) <- "U_prime"

welfare <- cbind(welfare, G_prime)
welfare <- cbind(welfare, U_prime)

welfare$G_frac <- welfare$G_q500 / welfare$G_prime
welfare$U_frac <- welfare$U_q500 / welfare$U_prime


deltaG1 <- ggplot(data=welfare, aes(x=iso3, y=G_frac, ymin=1, ymax=G_frac)) +
    geom_linerange(color=bcOrange, size=.75, lty=2) +
    geom_point(color=bcOrange, size=3) +
    geom_hline(yintercept=1, lty=2) +
    theme_classic() +
    labs(title=paste0("Effects of Military Coercion on Government Welfare, ", year), x="", y="Fractional Change in Gov. Welfare") +
    scale_y_continuous(limit=c(min(welfare$G_frac), max(welfare$G_frac))) +
    theme(axis.text.x=element_text(angle=60, hjust=1))

ggsave(setup$f_cfact_demilitarization_G_path)

deltaU1 <- ggplot(data=welfare, aes(x=iso3, y=U_frac, ymin=1, ymax=U_frac)) +
  geom_linerange(color=bcOrange, size=.75, lty=2) +
  geom_point(color=bcOrange, size=3) +
  geom_hline(yintercept=1, lty=2) +
  theme_classic() +
  labs(title=paste0("Effects of Military Coercion on Consumer Welfare, ", year), x="", y="Fractional Change in Con. Welfare") +
  scale_y_continuous(limit=c(1, max(welfare$U_frac))) +
  theme(axis.text.x=element_text(angle=60, hjust=1))

ggsave(setup$f_cfact_demilitarization_U_path)
