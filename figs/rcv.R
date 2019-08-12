# helperPath <- "~/Dropbox (Princeton)/14_Software/R/"
# helperFiles <- list.files(helperPath)
# for (i in helperFiles) {
#   source(paste0(helperPath, i))
# }
# 
# libs <- c("tidyverse", "patchwork")
# ipak(libs)
# 
# analysisPath <- "../working/analysis/"
# dataPath <- paste0(analysisPath, "tpsp_data/")
# resultsPath <- paste0(analysisPath, "results/")
# 
# ccodes <- read_csv(paste0(dataPath, "ccodes.csv"), col_names=FALSE)
# Y <- read_csv(paste0(dataPath, "year.csv"), col_names=FALSE) %>% pull(.)
# rcv0 <- read_csv(paste0(resultsPath, "rcv0.csv"), col_names=FALSE)
# rcv1 <- read_csv(paste0(resultsPath, "rcv1.csv"), col_names=FALSE)

hmColors <- colorRampPalette(c("white", bcOrange))(30)
naColor <- "#D3D3D3"

mint <- .99
maxt <- max(max(rcv0), max(rcv1))

rcvhm <- function(rcv, b, minTau, maxTau) {
  colnames(rcv) <- ccodes %>% pull(.)
  rcv <- bind_cols(rcv, ccodes) 
  colnames(rcv)[colnames(rcv)=="X1"] <- "j_iso3"
  rcvDF <- rcv %>% gather("i_iso3", "rcv_ji", -j_iso3)
  rcvDF$rcv_ji <- ifelse(rcvDF$rcv_ji==0, NA, rcvDF$rcv_ji)
  rcvDF$rcv_ji <- as.numeric(rcvDF$rcv_ji)

  ggplot(rcvDF, aes(x=i_iso3, y=j_iso3, fill=rcv_ji)) +
    geom_tile(colour="white", width=.9, height=.9) +
    scale_fill_gradient(low=hmColors[1], high=hmColors[length(hmColors)], 
                        breaks=c(minTau, maxTau), limits=c(minTau, maxTau), labels=c("Low", "High"), 
                        guide="colorbar", na.value=naColor) +
    labs(x='Defender', y='Attacker', title=paste0("Regime Change Values, b=", as.character(b))) +
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

# rcvhm(rcv0, 0, mint, maxt) + rcvhm(rcv1, 1, mint, maxt)
# coords = which(rcv0 == max(rcv0), arr.ind = TRUE)
# coords[2]
