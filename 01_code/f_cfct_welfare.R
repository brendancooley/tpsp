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

welfare <- cbind(quantiles_G, quantiles_U) %>% as_tibble()

### COUNTERFACTUAL 1 ###

G_prime <- read_csv(paste0(setup$cfct_demilitarization_path, "G_hat.csv"), col_names=FALSE)
colnames(G_prime) <- "G_prime"
U_prime <- read_csv(paste0(setup$cfct_demilitarization_path, "U_hat.csv"), col_names=FALSE)
colnames(U_prime) <- "U_prime"

welfare1 <- cbind(welfare, G_prime)
welfare1 <- cbind(welfare1, U_prime)

welfare1$G_frac <- welfare1$G_q500 / welfare1$G_prime
welfare1$U_frac <- welfare1$U_q500 / welfare1$U_prime
welfare1 <- welfare1 %>% as_tibble()


deltaG1 <- ggplot(data=welfare1, aes(x=iso3, y=G_frac, ymin=1, ymax=G_frac)) +
    geom_linerange(color=bcOrange, size=.75, lty=2) +
    geom_point(color=bcOrange, size=3) +
    geom_hline(yintercept=1, lty=2) +
    theme_classic() +
    labs(title="Effects of Military Coercion on Government Welfare", x="", y="Fractional Change in Gov. Welfare") +
    scale_y_continuous(limit=c(min(welfare1$G_frac), max(welfare1$G_frac))) +
    theme(axis.text.x=element_text(angle=60, hjust=1))

ggsave(setup$f_cfact_demilitarization_G_path)

deltaU1 <- ggplot(data=welfare1, aes(x=iso3, y=U_frac, ymin=1, ymax=U_frac)) +
  geom_linerange(color=bcOrange, size=.75, lty=2) +
  geom_point(color=bcOrange, size=3) +
  geom_hline(yintercept=1, lty=2) +
  theme_classic() +
  labs(subtitle="Effects of Military Coercion on Consumer Welfare", x="", y="Fractional Change in Cons. Welfare") +
  scale_y_continuous(limit=c(1, max(welfare1$U_frac))) +
  theme(axis.text.x=element_text(angle=60, hjust=1))

ggsave(setup$f_cfact_demilitarization_U_path)

### COUNTERFACTUAL 2

G_prime <- read_csv(paste0(setup$cfct_china_path, "G_hat.csv"), col_names=FALSE)
colnames(G_prime) <- "G_prime"
U_prime <- read_csv(paste0(setup$cfct_china_path, "U_hat.csv"), col_names=FALSE)
colnames(U_prime) <- "U_prime"

welfare2 <- cbind(welfare, G_prime)
welfare2 <- cbind(welfare2, U_prime)

welfare2$G_frac <- welfare2$G_q500 / welfare2$G_prime
welfare2$U_frac <- welfare2$U_q500 / welfare2$U_prime
welfare2 <- welfare2 %>% as_tibble()

# negative values for AUS make these plots not particularly sensible
