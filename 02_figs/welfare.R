helperPath <- "~/Dropbox (Princeton)/14_Software/R/"
helperFiles <- list.files(helperPath)
for (i in helperFiles) {
  source(paste0(helperPath, i))
}

libs <- c("tidyverse", "latex2exp")
ipak(libs)

tpspPath <- "~/Dropbox (Princeton)/1_Papers/tpsp/01_data/"
dataPath <- paste0(tpspPath, "tpsp_data_mini/")
resultsPath <- paste0(tpspPath, "results_rcv_ft/")
counterfactualsPath <- paste0(resultsPath, "counterfactuals/")

ccodes <- read_csv(paste0(dataPath, "ccodes.csv"), col_names=FALSE)
Y <- read_csv(paste0(dataPath, "year.csv"), col_names=FALSE) %>% pull(.)

G_star <- read_csv(paste0(counterfactualsPath, "G_star.csv"), col_names=FALSE)
G_prime <- read_csv(paste0(counterfactualsPath, "G_prime.csv"), col_names=FALSE)
V_star <- read_csv(paste0(counterfactualsPath, "V_star.csv"), col_names=FALSE)
V_prime <- read_csv(paste0(counterfactualsPath, "V_prime.csv"), col_names=FALSE)

# Ghatft <- read_csv(paste0(resultsPath, "Ghatft.csv"), col_names=FALSE)

G <- bind_cols(ccodes, G_star, G_prime)
colnames(G) <- c("ccode", "G_star", "G_prime")
G$frac <- G$G_star / G$G_prime
G$base <- 1
Gmin <- min(G$frac)
Gmax <- max(G$frac)

V <- bind_cols(ccodes, V_star, V_prime)
colnames(V) <- c("ccode", "V_star", "V_prime")
V$frac <- V$V_star / V$V_prime
V$base <- 1
Vmin <- min(V$frac)
Vmax <- max(V$frac)

# Ghat <- bind_cols(ccodes, Ghatft) %>% as_tibble()
# Ghat_base <- rep(1, nrow(Ghat)) %>% as_tibble()
# Ghat <- bind_cols(Ghat, Ghat_base)
# 
# colnames(Ghat) <- c("ccode", "ghat", "ghat_base")
# 
# Ghat <- Ghat %>% arrange(desc(ghat)) %>%
#   mutate(ccode = factor(ccode, unique(ccode)))
# 
# ghat_max <- ceiling(max(Ghat$ghat))

# GhatftFig <- ggplot(data=Ghat, aes(x=ccode, y=ghat,ymin= ghat_base, ymax=ghat)) +
#   geom_linerange(color=bcOrange, size=.75, lty=2) +
#   geom_point(color=bcOrange, size=3) +
#   geom_hline(yintercept=1, lty=2) +
#   theme_classic() +
#   labs(title=paste0("Welfare Effects of Free Trade, ", Y), x="", y="Change in Consumer Welfare") +
#   scale_y_continuous(breaks=seq(0, ghat_max), limit=c(0, ghat_max)) +
#   theme(axis.text.x=element_text(angle=60, hjust=1))

deltaG <- ggplot(data=G, aes(x=ccode, y=frac, ymin=base, ymax=frac)) +
    geom_linerange(color=bcOrange, size=.75, lty=2) +
    geom_point(color=bcOrange, size=3) +
    geom_hline(yintercept=1, lty=2) +
    theme_classic() +
    labs(title=paste0("Effects of Military Coercion on Government Welfare, ", Y), x="", y="Fractional Change in Gov. Welfare") +
    scale_y_continuous(limit=c(Gmin, Gmax)) +
    theme(axis.text.x=element_text(angle=60, hjust=1))

deltaV <- ggplot(data=V, aes(x=ccode, y=frac, ymin=base, ymax=frac)) +
  geom_linerange(color=bcOrange, size=.75, lty=2) +
  geom_point(color=bcOrange, size=3) +
  geom_hline(yintercept=1, lty=2) +
  theme_classic() +
  labs(title=paste0("Effects of Military Coercion on Consumer Welfare, ", Y), x="", y="Fractional Change in Con. Welfare") +
  scale_y_continuous(limit=c(Vmin, Vmax)) +
  theme(axis.text.x=element_text(angle=60, hjust=1))
