helperPath <- "~/Dropbox (Princeton)/14_Software/R/"
helperFiles <- list.files(helperPath)
for (i in helperFiles) {
  source(paste0(helperPath, i))
}

libs <- c("tidyverse", "latex2exp")
ipak(libs)

analysisPath <- "../working/analysis/"
dataPath <- paste0(analysisPath, "tpsp_data_mini/")
resultsPath <- paste0(analysisPath, "results_rcv_ft/")

ccodes <- read_csv(paste0(dataPath, "ccodes.csv"), col_names=FALSE)
Y <- read_csv(paste0(dataPath, "year.csv"), col_names=FALSE) %>% pull(.)
# Ghatft <- read_csv(paste0(resultsPath, "Ghatft.csv"), col_names=FALSE)

Ghat <- bind_cols(ccodes, Ghatft) %>% as_tibble()
Ghat_base <- rep(1, nrow(Ghat)) %>% as_tibble()
Ghat <- bind_cols(Ghat, Ghat_base)

colnames(Ghat) <- c("ccode", "ghat", "ghat_base")

Ghat <- Ghat %>% arrange(desc(ghat)) %>%
  mutate(ccode = factor(ccode, unique(ccode)))

ghat_max <- ceiling(max(Ghat$ghat))

GhatftFig <- ggplot(data=Ghat, aes(x=ccode, y=ghat,ymin= ghat_base, ymax=ghat)) +
  geom_linerange(color=bcOrange, size=.75, lty=2) +
  geom_point(color=bcOrange, size=3) +
  geom_hline(yintercept=1, lty=2) +
  theme_classic() +
  labs(title=paste0("Welfare Effects of Free Trade, ", Y), x="", y="Change in Consumer Welfare") +
  scale_y_continuous(breaks=seq(0, ghat_max), limit=c(0, ghat_max)) +
  theme(axis.text.x=element_text(angle=60, hjust=1))

