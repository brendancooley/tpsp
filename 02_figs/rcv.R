# helperPath <- "~/Dropbox (Princeton)/14_Software/R/"
# helperFiles <- list.files(helperPath)
# for (i in helperFiles) {
#   source(paste0(helperPath, i))
# }
# 
# libs <- c("tidyverse", "patchwork")
# ipak(libs)
# 
# tpspPath <- "~/Dropbox (Princeton)/1_Papers/tpsp/01_data/"
# dataPath <- paste0(tpspPath, "data/mid/")
# resultsPath <- paste0(tpspPath, "results/mid/")
# estimatesPath <- paste0(resultsPath, "estimates/")
# 
# ccodes <- read_csv(paste0(dataPath, "ccodes.csv"), col_names=FALSE)
# N <- nrow(ccodes)
# rcv_eq <- read_csv(paste0(estimatesPath, "rcv_eq.csv"), col_names=FALSE)

for (i in 1:N) {
  for (j in 1:N) {
    if (rcv_eq[i, j] < 1) {
      rcv_eq[i, j] <- 1
    }
  }
}
for (i in 1:N) {
  rcv_eq[i, i] <- NA
}

hmColors <- colorRampPalette(c("white", bcOrange))(30)
naColor <- "#D3D3D3"

mint <- min(rcv_eq, na.rm=T)
maxt <- max(rcv_eq, na.rm=T)

rcvhm <- function(rcv, minTau, maxTau) {
  colnames(rcv) <- ccodes %>% pull(.)
  rcv <- bind_cols(rcv, ccodes) 
  colnames(rcv)[colnames(rcv)=="X1"] <- "j_iso3"
  rcvDF <- rcv %>% gather("i_iso3", "rcv_ji", -j_iso3)
  rcvDF <- rcvDF %>% filter(i_iso3 != "RoW", j_iso3 != "RoW")
  rcvDF$rcv_ji <- ifelse(rcvDF$rcv_ji==0, NA, rcvDF$rcv_ji)
  rcvDF$rcv_ji <- as.numeric(rcvDF$rcv_ji)

  ggplot(rcvDF, aes(x=i_iso3, y=j_iso3, fill=rcv_ji)) +
    geom_tile(colour="white", width=.9, height=.9) +
    scale_fill_gradient(low=hmColors[1], high=hmColors[length(hmColors)], 
                        breaks=c(minTau, maxTau), limits=c(minTau, maxTau), labels=c("Low", "High"), 
                        guide="colorbar", na.value=naColor) +
    labs(x='Attacker', y='Defender', title=paste0("Conquest Values")) +
    labs(fill="Value") +
    theme_classic() +
    coord_fixed() +
    theme(axis.text.x=element_text(angle=60, hjust=1),
          axis.ticks.y=element_blank(),
          axis.ticks.x=element_blank(),
          axis.line.y=element_blank(),
          axis.line.x=element_blank(),
          legend.position="none")
}

# rcvhm(rcv_eq, mint, maxt)
# rcvhm(rcv0, 0, mint, maxt) + rcvhm(rcv1, 1, mint, maxt)
# coords = which(rcv0 == max(rcv0), arr.ind = TRUE)
# coords[2]
